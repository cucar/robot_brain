import mysql from 'mysql2/promise';

export default () => mysql.createConnection({
	host: 'localhost',
	user: 'root',
	password: 'deneme',
	database: 'machine_intelligence',
	waitForConnections: true,
	connectionLimit: 10,
	queueLimit: 0
});