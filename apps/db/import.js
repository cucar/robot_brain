/**
 * Database import utility — bulk-loads a backup folder of CSVs into MySQL via
 * `LOAD DATA LOCAL INFILE`. The CSV layout is the one produced by the Backup
 * class (and apps/db/export.js), so any backup folder can be round-tripped.
 *
 * Usage:
 *   node apps/db/import.js <backup-folder>
 *
 * <backup-folder> is the path to a single timestamped folder, e.g.
 * apps/stocks/jobs/test/backups/2026-04-25_12-34-56.
 *
 * All target tables are truncated first — this is a full restore, not a merge.
 *
 * Bulk-load tuning: for each table the secondary indexes are dropped before the
 * load and rebuilt afterwards. Maintaining several index B-trees row-by-row
 * during a multi-GB load is the dominant cost (random page I/O once the trees
 * outgrow the buffer pool); loading into a PK-only table and then building each
 * index in one sorted pass is far cheaper. Durability is also relaxed for the
 * duration (innodb_flush_log_at_trx_commit=2) and restored at the end.
 */
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import getMySQLConnection from './database.js';

const TABLES = [
	{ file: 'channels.csv',     table: 'channels',     columns: '(id, name)' },
	{ file: 'dimensions.csv',   table: 'dimensions',   columns: '(id, name)' },
	{ file: 'neurons.csv',      table: 'neurons',      columns: '(id, temporal_level, spatial_level)' },
	{ file: 'base_neurons.csv', table: 'base_neurons', columns: '(neuron_id, channel_id, type, dimension_id, val)' },
	{ file: 'connections.csv',  table: 'connections',  columns: '(from_neuron_id, to_neuron_id, distance, strength, reward)' },
	{ file: 'patterns.csv',     table: 'patterns',     columns: '(pattern_neuron_id, parent_neuron_id, strength)' },
	{ file: 'contexts.csv', table: 'contexts', columns: '(pattern_neuron_id, context_neuron_id, context_age, strength)' },
	{ file: 'neuron_error_stats.csv', table: 'neuron_error_stats', columns: '(neuron_id, age, n, mean, m2)' }
];

/**
 * Read a table's secondary (non-PRIMARY) indexes from information_schema so we
 * can drop and later rebuild them exactly as defined — no hardcoded DDL that
 * could drift from db.sql. Each entry carries the index name, whether it is
 * UNIQUE, and the column list (with prefix lengths preserved). The PRIMARY key
 * is deliberately excluded: it is the clustered index that defines row order and
 * uniqueness, and the load depends on it.
 */
async function getSecondaryIndexes(conn, table) {
	const [rows] = await conn.query(
		`SELECT INDEX_NAME, NON_UNIQUE, COLUMN_NAME, SUB_PART
		   FROM information_schema.STATISTICS
		  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = ? AND INDEX_NAME <> 'PRIMARY'
		  ORDER BY INDEX_NAME, SEQ_IN_INDEX`,
		[table]
	);
	const byName = new Map();
	for (const r of rows) {
		if (!byName.has(r.INDEX_NAME)) {
			byName.set(r.INDEX_NAME, { name: r.INDEX_NAME, unique: r.NON_UNIQUE === 0, cols: [] });
		}
		byName.get(r.INDEX_NAME).cols.push(r.SUB_PART ? `\`${r.COLUMN_NAME}\`(${r.SUB_PART})` : `\`${r.COLUMN_NAME}\``);
	}
	return [...byName.values()];
}

async function main() {
	const folder = process.argv[2];
	if (!folder) {
		console.error('Usage: node apps/db/import.js <backup-folder>');
		process.exit(1);
	}
	if (!fs.existsSync(folder)) {
		console.error(`Backup folder not found: ${folder}`);
		process.exit(1);
	}

	const conn = await getMySQLConnection();
	console.log(`📥 Importing backup: ${folder}`);

	// Enable LOAD DATA LOCAL on the server side. Requires the connecting user to
	// have SYSTEM_VARIABLES_ADMIN (or SUPER on older MySQL) — root does by default.
	// Persists for the life of the server, but cheap to re-issue every run.
	await conn.query('SET GLOBAL local_infile = 1');

	// Relax durability and per-row checks for the bulk load. flush_log_at_trx_commit
	// is global-only; capture the original so we can restore it in `finally` even if
	// the load throws. unique/FK checks are session-scoped and reset on disconnect.
	const [origRows] = await conn.query('SELECT @@GLOBAL.innodb_flush_log_at_trx_commit AS f');
	const origFlush = origRows[0].f;
	await conn.query('SET GLOBAL innodb_flush_log_at_trx_commit = 2');
	await conn.query('SET SESSION unique_checks = 0');
	await conn.query('SET SESSION foreign_key_checks = 0');

	try {
		// Truncate in reverse order so child tables go first; FK checks already off.
		for (const { table } of [...TABLES].reverse()) await conn.query(`TRUNCATE ${table}`);

		for (const { file, table, columns } of TABLES) {
			const filepath = path.resolve(folder, file).replace(/\\/g, '/');
			if (!fs.existsSync(filepath)) {
				console.log(`   skip ${table}: ${file} missing`);
				continue;
			}

			// Drop secondary indexes so the load maintains only the clustered PK,
			// then rebuild them in one sorted ALTER after the rows are in. Wrapped in
			// try/finally so a failed load still restores the indexes.
			const indexes = await getSecondaryIndexes(conn, table);
			if (indexes.length)
				await conn.query(`ALTER TABLE ${table} ${indexes.map(i => `DROP INDEX \`${i.name}\``).join(', ')}`);

			try {
				// LOAD DATA LOCAL INFILE: client streams the file to the server. Requires
				// `localInfile: true` on the connection and `local_infile=ON` server-side
				// (settable dynamically via `SET GLOBAL local_infile = 1;`).
				const sql = `LOAD DATA LOCAL INFILE '${filepath}' INTO TABLE ${table} FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\\n' ${columns}`;
				const t = process.hrtime.bigint();
				const [result] = await conn.query(sql);
				console.log(`   ${table}: ${result.affectedRows} rows in ${((Number(process.hrtime.bigint() - t)) / 1e9).toFixed(1)}s`);
			}
			finally {
				if (indexes.length) {
					const add = indexes.map(i => `ADD ${i.unique ? 'UNIQUE ' : ''}INDEX \`${i.name}\` (${i.cols.join(', ')})`).join(', ');
					const t = process.hrtime.bigint();
					await conn.query(`ALTER TABLE ${table} ${add}`);
					console.log(`   ${table}: rebuilt ${indexes.length} index(es) in ${((Number(process.hrtime.bigint() - t)) / 1e9).toFixed(1)}s`);
				}
			}
		}
	} finally {
		// Restore durability regardless of how the load ended.
		await conn.query(`SET GLOBAL innodb_flush_log_at_trx_commit = ${origFlush}`).catch(() => {});
		await conn.end();
	}
	console.log('✅ Import complete');
}

main().catch(err => { console.error(err); process.exit(1); });
