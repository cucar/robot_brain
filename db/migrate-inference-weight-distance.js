import getConnection from './db.js';

async function migrate() {
	const conn = await getConnection();

	try {
		console.log('Starting migration: inference tables to weight_distance...');
		
		// Drop and recreate connection_inference table
		console.log('Recreating connection_inference table...');
		await conn.query('DROP TABLE IF EXISTS connection_inference');
		await conn.query(`
			CREATE TABLE connection_inference (
				level TINYINT,
				connection_id BIGINT,
				weight_distance TINYINT UNSIGNED NOT NULL,
				PRIMARY KEY (level, connection_id),
				INDEX idx_level (level)
			) ENGINE=MEMORY
		`);
		console.log('✓ connection_inference table recreated');
		
		// Drop and recreate pattern_inference table
		console.log('Recreating pattern_inference table...');
		await conn.query('DROP TABLE IF EXISTS pattern_inference');
		await conn.query(`
			CREATE TABLE pattern_inference (
				level TINYINT,
				pattern_neuron_id BIGINT UNSIGNED NOT NULL,
				connection_id BIGINT,
				weight_distance TINYINT UNSIGNED NOT NULL,
				PRIMARY KEY (level, pattern_neuron_id, connection_id),
				INDEX idx_level (level)
			) ENGINE=MEMORY
		`);
		console.log('✓ pattern_inference table recreated');
		
		console.log('\n✓ Migration completed successfully!');
		
	} catch (error) {
		console.error('Migration failed:', error);
		process.exit(1);
	} finally {
		await conn.end();
		process.exit(0);
	}
}

migrate();

