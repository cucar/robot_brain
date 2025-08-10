CREATE DATABASE IF NOT EXISTS machine_intelligence;
USE machine_intelligence;

-- SELECT '{"x":0.11,"y":0.21,"r":1,"g":0,"b":0}' as point_str, (
		-- SELECT CONCAT(neuron_id, '|', distance) as neuron_distance
		-- FROM (
				SELECT neuron_id, dimension_id, SQRT(SUM(
					CASE 
                    WHEN dimension_id = 1 THEN POW( val - 0.11, 2) 
                    WHEN dimension_id = 2 THEN POW( val - 0.21, 2) 
                    WHEN dimension_id = 3 THEN POW(val - 1, 2) 
                    WHEN dimension_id = 4 THEN POW(val - 0, 2) 
                    WHEN dimension_id = 5 THEN POW(val - 0, 2) 
                    END)) AS distance
				FROM coordinates
				WHERE neuron_id IN (39)
				AND dimension_id IN (1,2,3,4,5);
				-- GROUP BY neuron_id
				-- HAVING distance <= 1e-8
				-- ORDER BY distance
				-- LIMIT 1;
		-- ) q;
-- ) as neuron_distance;

truncate active_neurons;
truncate connections;
truncate coordinates;
delete from neurons;
select * from neurons;
select * from coordinates where neuron_id = 40;
select * from connections;
select * from active_neurons;

DROP TABLE IF EXISTS dimensions;
CREATE TABLE IF NOT EXISTS dimensions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) UNIQUE NOT NULL
);

-- these dimensions can be used for visual processing
INSERT INTO dimensions (name) VALUES ('x'), ('y'), ('r'), ('g'), ('b');

-- these dimensions are used for forecasting timeseries changes - slopes can go from -100% to 100% (rate of change)
-- INSERT INTO dimensions (name, min_value, max_value) VALUES ('gold_slope', -100, 100), ('aem_slope', -100, 100);

-- DROP TABLE IF EXISTS neurons;
CREATE TABLE IF NOT EXISTS neurons (
    id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
    creation_time DATETIME NOT NULL DEFAULT NOW()
);

DROP TABLE IF EXISTS coordinates;
CREATE TABLE IF NOT EXISTS coordinates (
    neuron_id BIGINT UNSIGNED NOT NULL,
    dimension_id INT NOT NULL,
    val DOUBLE NOT NULL,
    PRIMARY KEY (neuron_id, dimension_id),
    INDEX (dimension_id, val),
    FOREIGN KEY (neuron_id) REFERENCES neurons(id) ON DELETE CASCADE
);

-- connections are directionless - source id is always smaller than target id by convention
DROP TABLE IF EXISTS connections;
CREATE TABLE IF NOT EXISTS connections (
    neuron1_id BIGINT UNSIGNED NOT NULL,
    neuron2_id BIGINT UNSIGNED NOT NULL,
    strength DOUBLE NOT NULL DEFAULT 1,
    PRIMARY KEY (neuron1_id, neuron2_id), -- composite PRIMARY KEY for unique connections
    FOREIGN KEY (neuron1_id) REFERENCES neurons(id) ON DELETE CASCADE,
    FOREIGN KEY (neuron2_id) REFERENCES neurons(id) ON DELETE CASCADE
);

INSERT INTO connections (source_id, target_id, strength)
SELECT s.neuron_id as source_id, t.neuron_id as target_id, 1 / (1 + t.age) as strength -- as the age difference increases, strength decreases
FROM active_neurons s
CROSS JOIN active_neurons t
WHERE s.level = 0 -- get the active neurons in the given level
AND s.age = 0 -- reinforcing connections for the newly activated neurons only
AND t.level = s.level -- reinforcing connections only within the same level
AND (t.neuron_id != s.neuron_id OR t.age != s.age) -- if it's the same neuron, it's gotta be an older one
ON DUPLICATE KEY UPDATE strength = strength + VALUES(strength); -- if connection exists, add on to it

-- **MEMORY Tables (Per-run persistent, cleared at application start or on demand):**

-- neurons currently active within the sliding window
CREATE TABLE IF NOT EXISTS active_neurons (
    neuron_id BIGINT UNSIGNED,
    level INT NOT NULL,
    age INT NOT NULL,
    INDEX idx_neuron_id (neuron_id),
    INDEX idx_level (level),
    INDEX idx_age (age)
) ENGINE=MEMORY;

select * from active_neurons;

-- stores candidates for patterns with their calculated peakiness score
CREATE TABLE IF NOT EXISTS potential_peaks (
    neuron_id BIGINT UNSIGNED PRIMARY KEY,
    peakiness_score FLOAT NOT NULL
) ENGINE=MEMORY;

-- holds neurons that are temporarily suppressed during peak selection within a level
CREATE TABLE IF NOT EXISTS suppressed_neurons (
    neuron_id BIGINT UNSIGNED PRIMARY KEY
) ENGINE=MEMORY;

-- stores the calculated centroids for detected patterns
CREATE TABLE IF NOT EXISTS observed_pattern_centroids (
    pattern_id BIGINT UNSIGNED NOT NULL,
    dimension_id INT NOT NULL,
    value FLOAT NOT NULL,
    PRIMARY KEY (pattern_id, dimension_id)
) ENGINE=MEMORY;

-- maps patterns to their constituent lower-level neurons
CREATE TABLE IF NOT EXISTS observed_pattern_ingredients (
    pattern_id BIGINT UNSIGNED NOT NULL,
    ingredient_id BIGINT UNSIGNED NOT NULL,
    PRIMARY KEY (pattern_id, ingredient_id)
) ENGINE=MEMORY;