import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';
import mysql from 'mysql2/promise';

// Load .env from apps/db/ regardless of cwd — these utilities are typically run
// from the repo root (`node apps/db/import.js ...`), so the default cwd lookup
// would miss the file that lives next to this module.
dotenv.config({ path: path.join(path.dirname(fileURLToPath(import.meta.url)), '.env') });

/**
 * MySQL connection helper for the apps/db utility jobs (import / export).
 *
 * Connection settings come from .env (see .env.example). `infileStreamFactory`
 * enables `LOAD DATA LOCAL INFILE` (mysql2 v2+ requires it as opt-in for the
 * client side); the server must also have `local_infile=ON`, which the import
 * job sets via `SET GLOBAL local_infile = 1` before issuing the LOAD.
 */
export default () => mysql.createConnection({
	host: process.env.DB_HOST ?? 'localhost',
	port: process.env.DB_PORT ? Number(process.env.DB_PORT) : 3306,
	user: process.env.DB_USER ?? 'root',
	password: process.env.DB_PASSWORD ?? '',
	database: process.env.DB_NAME ?? 'machine_intelligence',
	// Whitelist: only let the server pull files our import path explicitly asked
	// for. The path the server requests is the same path we put in the SQL —
	// rejecting anything else closes the LOCAL INFILE attack surface.
	infileStreamFactory: filepath => fs.createReadStream(filepath),
	waitForConnections: true,
	connectionLimit: 10,
	queueLimit: 0,
	namedPlaceholders: true
});
