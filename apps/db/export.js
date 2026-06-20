/**
 * Database export utility — dumps the MySQL brain tables to a backup folder.
 *
 * Usage:
 *   node apps/db/export.js [output-path]
 *
 * If output-path is provided, writes directly to that folder (creating it if
 * needed). Otherwise falls back to `./backups/<timestamp>/` in cwd.
 *
 * Output layout matches the Backup class (no header rows, comma-separated):
 *   channels.csv, dimensions.csv, neurons.csv, base_neurons.csv,
 *   connections.csv, patterns.csv, contexts.csv
 *
 * Each table is streamed row-by-row from MySQL straight to its CSV file: the
 * query result is consumed off the wire one row at a time (mysql2 `.stream()`)
 * and piped through a Transform into a file write stream. The full result set is
 * never materialized client-side, so multi-GB tables (connections, contexts)
 * export in bounded memory instead of OOMing on the buffered `conn.query()`.
 */
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import getMySQLConnection from './database.js';

// Table order is irrelevant for file output (no FKs in flat CSVs), but kept
// consistent with the Backup writer for readability and easy diffing. ORDER BY
// id on the parent tables produces deterministic output across runs; the leaf
// tables (connections, patterns, contexts) are large enough that we skip
// the sort to save query cost.
const TABLES = [
	{ file: 'channels.csv',     query: 'SELECT id, name FROM channels ORDER BY id' },
	{ file: 'dimensions.csv',   query: 'SELECT id, name FROM dimensions ORDER BY id' },
	{ file: 'neurons.csv',      query: 'SELECT id, temporal_level, spatial_level FROM neurons ORDER BY id' },
	{ file: 'base_neurons.csv', query: 'SELECT neuron_id, channel_id, type, dimension_id, val FROM base_neurons ORDER BY neuron_id' },
	{ file: 'connections.csv',  query: 'SELECT from_neuron_id, to_neuron_id, distance, strength, reward FROM connections' },
	{ file: 'patterns.csv',     query: 'SELECT pattern_neuron_id, parent_neuron_id, strength FROM patterns' },
	{ file: 'contexts.csv', query: 'SELECT pattern_neuron_id, context_neuron_id, context_age, strength FROM contexts' },
	{ file: 'neuron_error_stats.csv', query: 'SELECT neuron_id, age, n, mean, m2 FROM neuron_error_stats ORDER BY neuron_id, age' }
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

	const folder = process.argv[2]
		? path.resolve(process.argv[2])
		: path.join(process.cwd(), 'backups', formatTimestamp(new Date()));
	fs.mkdirSync(folder, { recursive: true });
	console.log(`📤 Exporting brain to: ${folder}`);

	for (const { file, query } of TABLES) {
		const filepath = path.join(folder, file);

		// Stream rows off the wire one at a time via the underlying core connection
		// (`conn.connection`) — the promise wrapper's `conn.query()` buffers the
		// whole result set, which is what OOMs on the large tables. A plain
		// unordered SELECT streams from the storage engine, so neither the server
		// nor the client holds the full table.
		const queryStream = conn.connection.query(query).stream();

		// `fields` fires before the first row and carries the column metadata in
		// result order. Fall back to the row's own key order (mysql2 builds row
		// objects in column order) if it hasn't arrived for the first chunk.
		let colNames = null;
		queryStream.on('fields', fields => { colNames = fields.map(f => f.name); });

		// One CSV line per row, each cell escaped so commas/quotes/newlines in
		// values can't break the column count. Object-in, string-out transform.
		let rowCount = 0;
		const toCsv = new Transform({
			writableObjectMode: true,
			transform(row, _enc, cb) {
				const cols = colNames ?? Object.keys(row);
				rowCount++;
				cb(null, cols.map(c => escapeField(row[c])).join(',') + '\n');
			}
		});

		// pipeline wires up backpressure and error propagation end to end. Empty
		// tables emit no rows, leaving a zero-byte file — which `LOAD DATA INFILE`
		// and `wc -l` both handle cleanly.
		await pipeline(queryStream, toCsv, fs.createWriteStream(filepath));
		console.log(`   ${file}: ${rowCount} rows`);
	}

	// Explicit close — the mysql2 connection holds the event loop open otherwise.
	await conn.end();
	console.log('✅ Export complete');
}

// Top-level error handler: log and exit non-zero so shell pipelines and CI can
// detect failures. No stack trimming — this is a dev tool, full trace is fine.
main().catch(err => { console.error(err); process.exit(1); });
