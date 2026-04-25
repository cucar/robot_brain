/**
 * Database export utility — dumps the MySQL brain tables to a fresh backup
 * folder (`./backups/<timestamp>/`) in the current working directory. The
 * user can then move the folder under any job's `backups/` directory to make
 * it loadable via `--load`.
 *
 * Usage:
 *   node apps/db/export.js
 *
 * Output layout matches the Backup class (no header rows, comma-separated):
 *   channels.csv, dimensions.csv, neurons.csv, base_neurons.csv,
 *   connections.csv, patterns.csv, pattern_past.csv
 */
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import getMySQLConnection from './database.js';

// Table order is irrelevant for file output (no FKs in flat CSVs), but kept
// consistent with the Backup writer for readability and easy diffing. ORDER BY
// id on the parent tables produces deterministic output across runs; the leaf
// tables (connections, patterns, pattern_past) are large enough that we skip
// the sort to save query cost.
const TABLES = [
	{ file: 'channels.csv',     query: 'SELECT id, name FROM channels ORDER BY id' },
	{ file: 'dimensions.csv',   query: 'SELECT id, name FROM dimensions ORDER BY id' },
	{ file: 'neurons.csv',      query: 'SELECT id, level FROM neurons ORDER BY id' },
	{ file: 'base_neurons.csv', query: 'SELECT neuron_id, channel_id, type, dimension_id, val FROM base_neurons ORDER BY neuron_id' },
	{ file: 'connections.csv',  query: 'SELECT from_neuron_id, to_neuron_id, distance, strength, reward FROM connections' },
	{ file: 'patterns.csv',     query: 'SELECT pattern_neuron_id, parent_neuron_id, strength FROM patterns' },
	{ file: 'pattern_past.csv', query: 'SELECT pattern_neuron_id, context_neuron_id, context_age, strength FROM pattern_past' }
];

/**
 * Build a `YYYY-MM-DD_HH-mm-ss` timestamp for the export folder name. Local time
 * (not UTC) is used so the folder name matches the user's shell history. Format
 * mirrors Backup.save() so folders produced here are drop-in compatible with
 * `--load` once moved under a job's backups/ directory.
 */
function formatTimestamp(d) {
	const pad = n => String(n).padStart(2, '0');
	return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}_${pad(d.getHours())}-${pad(d.getMinutes())}-${pad(d.getSeconds())}`;
}

/**
 * Escape a single CSV cell. Same quoting rules as brain/src/backup.js: wrap in
 * double-quotes (and double up embedded quotes) when the value contains a
 * delimiter, quote, or newline. NULL / undefined coerce to the empty string so
 * MySQL NULL round-trips as an empty CSV cell.
 */
function escapeField(value) {
	const s = value === null || value === undefined ? '' : String(value);
	if (s.includes(',') || s.includes('"') || s.includes('\n')) return `"${s.replace(/"/g, '""')}"`;
	return s;
}

/**
 * Open a MySQL connection, dump every brain table to a fresh timestamped folder
 * under `./backups/` in cwd, and close the connection. One CSV per table, no
 * header rows, layout matching brain/src/backup.js so the output is loadable
 * via `--load` after being moved under a job dir.
 */
async function main() {
	// Open a single connection and reuse it for every table — cheaper than
	// reconnecting and keeps all reads in the same transactional snapshot if the
	// session has REPEATABLE READ (the MySQL default).
	const conn = await getMySQLConnection();

	// Folder layout: `./backups/<timestamp>/` under cwd. Caller chooses cwd, so
	// running this from a job dir lands the export right next to that job's
	// existing backups.
	const folder = path.join(process.cwd(), 'backups', formatTimestamp(new Date()));
	fs.mkdirSync(folder, { recursive: true });
	console.log(`📤 Exporting brain to: ${folder}`);

	for (const { file, query } of TABLES) {
		// `fields` carries the column metadata in result order — we use the names
		// to index into each row object so the CSV column order matches the SELECT.
		const [rows, fields] = await conn.query(query);
		const colNames = fields.map(f => f.name);

		// One CSV line per row. Each cell is escaped before joining so commas in
		// values can't break the column count.
		const lines = rows.map(row => colNames.map(c => escapeField(row[c])).join(','));

		// Trailing newline only when there's content — empty tables produce a
		// zero-byte file, which `LOAD DATA INFILE` and `wc -l` both handle cleanly.
		const filepath = path.join(folder, file);
		fs.writeFileSync(filepath, lines.length ? lines.join('\n') + '\n' : '');
		console.log(`   ${file}: ${rows.length} rows`);
	}

	// Explicit close — the mysql2 connection holds the event loop open otherwise.
	await conn.end();
	console.log('✅ Export complete');
}

// Top-level error handler: log and exit non-zero so shell pipelines and CI can
// detect failures. No stack trimming — this is a dev tool, full trace is fine.
main().catch(err => { console.error(err); process.exit(1); });
