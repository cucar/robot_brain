CREATE DATABASE IF NOT EXISTS machine_intelligence;
USE machine_intelligence;

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
truncate neurons;

select * from neurons;

-- DROP TABLE IF EXISTS coordinates;
CREATE TABLE IF NOT EXISTS coordinates (
    neuron_id BIGINT UNSIGNED NOT NULL,
    dimension_id INT NOT NULL,
    value DOUBLE NOT NULL,
    PRIMARY KEY (neuron_id, dimension_id),
    INDEX (dimension_id, value),
    FOREIGN KEY (neuron_id) REFERENCES neurons(id) ON DELETE CASCADE
);

select * from coordinates;
truncate coordinates;

-- DROP TABLE IF EXISTS connections;
CREATE TABLE IF NOT EXISTS connections (
    source_id BIGINT UNSIGNED NOT NULL,
    target_id BIGINT UNSIGNED NOT NULL,
    strength DOUBLE NOT NULL DEFAULT 1,
    PRIMARY KEY (source_id, target_id), -- composite PRIMARY KEY on (source_id, target_id) for unique connections
    INDEX idx_target_source (target_id, source_id), -- for reverse lookups (finding all sources to a target)
    INDEX idx_source_strength (source_id, strength DESC), -- for efficient retrieval of strong outgoing connections
    FOREIGN KEY (source_id) REFERENCES neurons(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES neurons(id) ON DELETE CASCADE
);

select * from connections;
truncate connections;

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

-- holds neurons that are temporarily suppressed during peak selection within a level
CREATE TABLE IF NOT EXISTS suppressed_neurons (
    neuron_id BIGINT UNSIGNED PRIMARY KEY
) ENGINE=MEMORY;

-- stores candidates for patterns with their calculated peakiness score
CREATE TABLE IF NOT EXISTS potential_peaks (
    neuron_id BIGINT UNSIGNED PRIMARY KEY,
    peakiness_score FLOAT NOT NULL,
    INDEX idx_peakiness_score (peakiness_score DESC) -- For efficient sorting
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