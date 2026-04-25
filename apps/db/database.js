import 'dotenv/config';
import mysql from 'mysql2/promise';

/**
 * MySQL connection helper for the apps/db utility jobs (import / export).
 *
 * Connection settings come from .env (see .env.example). `localInfile: true`
 * enables `LOAD DATA LOCAL INFILE` so the import job can bulk-load CSVs from
 * a local backup folder — the MySQL server must also have `local_infile=ON`
 * for the client flag to take effect.
 */
export default () => mysql.createConnection({
	host: process.env.DB_HOST ?? 'localhost',
	port: process.env.DB_PORT ? Number(process.env.DB_PORT) : 3306,
	user: process.env.DB_USER ?? 'root',
	password: process.env.DB_PASSWORD ?? '',
	database: process.env.DB_NAME ?? 'machine_intelligence',
	localInfile: true,
	waitForConnections: true,
	connectionLimit: 10,
	queueLimit: 0,
	namedPlaceholders: true
});
