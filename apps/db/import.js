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
 */
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import getMySQLConnection from './database.js';

const TABLES = [
	{ file: 'channels.csv',     table: 'channels',     columns: '(id, name)' },
	{ file: 'dimensions.csv',   table: 'dimensions',   columns: '(id, name)' },
	{ file: 'neurons.csv',      table: 'neurons',      columns: '(id, level)' },
	{ file: 'base_neurons.csv', table: 'base_neurons', columns: '(neuron_id, channel_id, type, dimension_id, val)' },
	{ file: 'connections.csv',  table: 'connections',  columns: '(from_neuron_id, to_neuron_id, distance, strength, reward)' },
	{ file: 'patterns.csv',     table: 'patterns',     columns: '(pattern_neuron_id, parent_neuron_id, strength)' },
	{ file: 'pattern_past.csv', table: 'pattern_past', columns: '(pattern_neuron_id, context_neuron_id, context_age, strength)' }
];

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

	// Truncate in reverse order so child tables go first; FK checks off as a belt-and-braces.
	await conn.query('SET FOREIGN_KEY_CHECKS = 0');
	for (const { table } of [...TABLES].reverse()) await conn.query(`TRUNCATE ${table}`);

	for (const { file, table, columns } of TABLES) {
		const filepath = path.resolve(folder, file).replace(/\\/g, '/');
		if (!fs.existsSync(filepath)) {
			console.log(`   skip ${table}: ${file} missing`);
			continue;
		}

		// LOAD DATA LOCAL INFILE: client streams the file to the server. Requires
		// `localInfile: true` on the connection and `local_infile=ON` server-side
		// (settable dynamically via `SET GLOBAL local_infile = 1;`).
		const sql = `LOAD DATA LOCAL INFILE '${filepath}' INTO TABLE ${table} FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' LINES TERMINATED BY '\\n' ${columns}`;
		const [result] = await conn.query(sql);
		console.log(`   ${table}: ${result.affectedRows} rows`);
	}

	await conn.query('SET FOREIGN_KEY_CHECKS = 1');
	await conn.end();
	console.log('✅ Import complete');
}

main().catch(err => { console.error(err); process.exit(1); });
