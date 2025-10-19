-- Unified Brain Database Schema
-- Single connections table handles ALL relationships: spatial, temporal, hierarchical, predictions
-- Memory tables for active state, persistent tables for learned knowledge
CREATE DATABASE IF NOT EXISTS machine_intelligence;
USE machine_intelligence;

-- drop statements for reset 
-- DROP TABLE IF EXISTS dimensions;
-- DROP TABLE IF EXISTS neurons;
-- DROP TABLE IF EXISTS coordinates;
-- DROP TABLE IF EXISTS connections;
-- DROP TABLE IF EXISTS patterns;
-- DROP TABLE IF EXISTS pattern_peaks;
-- DROP TABLE IF EXISTS active_neurons;
-- DROP TABLE IF EXISTS pattern_inference;
-- DROP TABLE IF EXISTS connection_inference;
-- DROP TABLE IF EXISTS inferred_neurons;
-- DROP TABLE IF EXISTS neuron_rewards;
-- DROP TABLE IF EXISTS active_connections;
-- DROP TABLE IF EXISTS matched_patterns;

-- check state
select * from neurons;
select * from coordinates;
select * from patterns where pattern_neuron_id = 137;
select * from connections;
select * from connections where id > 96 and distance = 2;
select * from active_neurons;
select * from active_connections;

-- dimensions table determines input/output mapping for channels
CREATE TABLE IF NOT EXISTS dimensions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) UNIQUE NOT NULL,
    channel VARCHAR(50) NOT NULL,
    type ENUM('input', 'output') NOT NULL
);

-- these dimensions can be used for visual processing
-- INSERT INTO dimensions (name) VALUES ('x'), ('y'), ('r'), ('g'), ('b');

-- these dimensions are used for forecasting timeseries changes - slopes can go from -100% to 100% (rate of change)
-- INSERT INTO dimensions (name) VALUES ('gold_slope'), ('aem_slope');

-- test dimensions
-- INSERT IGNORE INTO dimensions (name) VALUES ('d0'), ('d1'), ('d2'), ('d3'), ('d4'), ('d5'), ('d6'), ('d7'), ('d8'), ('d9'), ('d10'), ('d11'), ('d12'), ('d13'), ('d14'), ('d15'), ('d16'), ('d17'), ('d18'), ('d19');

-- neurons table is the core representation of concepts - auto increment
CREATE TABLE IF NOT EXISTS neurons (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY
);

-- coordinates for base neurons
CREATE TABLE IF NOT EXISTS coordinates (
    neuron_id BIGINT UNSIGNED NOT NULL,
    dimension_id INT NOT NULL,
    val DOUBLE NOT NULL,
    PRIMARY KEY (neuron_id, dimension_id),
    FOREIGN KEY (neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    INDEX (dimension_id, val)
);

-- connections between neurons within levels - distance=0 is spatial (co-occurrence) - distance > 0 is temporal (sequences) 
CREATE TABLE IF NOT EXISTS connections (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    from_neuron_id BIGINT UNSIGNED NOT NULL,
    to_neuron_id BIGINT UNSIGNED NOT NULL,
    distance TINYINT UNSIGNED NOT NULL,  -- 0=spatial, 1=immediate, 2=next step, etc.
    strength DOUBLE NOT NULL DEFAULT 1.0,
    FOREIGN KEY (from_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    FOREIGN KEY (to_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    UNIQUE INDEX (from_neuron_id, to_neuron_id, distance),
    INDEX idx_from_distance_strength (from_neuron_id, distance, strength DESC),
    INDEX idx_to_distance_strength (to_neuron_id, distance, strength DESC),
    INDEX idx_distance_strength (distance, strength DESC),
    INDEX idx_strength (id, strength)
);

-- pattern neurons definitions - neuron connections between levels
CREATE TABLE IF NOT EXISTS patterns (
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    connection_id BIGINT UNSIGNED NOT NULL,
    strength DOUBLE NOT NULL DEFAULT 1.0,
    PRIMARY KEY (pattern_neuron_id, connection_id),
    FOREIGN KEY (pattern_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    FOREIGN KEY (connection_id) REFERENCES connections(id) ON DELETE CASCADE,
    INDEX idx_connection_strength (connection_id, strength),
    INDEX idx_pattern_strength (pattern_neuron_id, strength)
);

-- pattern peaks - maps each pattern neuron to its peak neuron (the decision node that owns the pattern)
-- patterns are learned by peak neurons to differentiate between sequences leading to them
CREATE TABLE IF NOT EXISTS pattern_peaks (
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    peak_neuron_id BIGINT UNSIGNED NOT NULL,
    PRIMARY KEY (pattern_neuron_id),
    FOREIGN KEY (pattern_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    FOREIGN KEY (peak_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    INDEX idx_peak (peak_neuron_id)
);

-- includes reward factor for neurons based on their performance (default 1.0 = neutral)
CREATE TABLE IF NOT EXISTS neuron_rewards (
    neuron_id BIGINT UNSIGNED NOT NULL,
    reward_factor DOUBLE NOT NULL DEFAULT 1.0,
    PRIMARY KEY (neuron_id),
    FOREIGN KEY (neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    INDEX (reward_factor, neuron_id)
);

-- neurons currently active within the sliding window (MEMORY table)
-- note that it is possible for the same neuron to be active in different ages or levels
CREATE TABLE IF NOT EXISTS active_neurons (
    neuron_id BIGINT UNSIGNED,
    level TINYINT NOT NULL,				         -- clustering level
    age TINYINT UNSIGNED NOT NULL DEFAULT 0,     -- how long the neuron has been active - note that higher levels age slower
    PRIMARY KEY (neuron_id, level, age),
    INDEX idx_level_age (level, age),
    INDEX idx_current_active (age, level)
) ENGINE=MEMORY;

-- stores down-level predictions from patterns - rebuilt fresh each frame
CREATE TABLE pattern_inference (
    level TINYINT,
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    connection_id BIGINT,
    weight_distance TINYINT UNSIGNED NOT NULL,  -- exponentially-rounded distance to age=-1 for weighting
    PRIMARY KEY (level, pattern_neuron_id, connection_id),
    INDEX idx_level (level)
) ENGINE=MEMORY;

-- stores same-level predictions from connections - rebuilt fresh each frame
CREATE TABLE connection_inference (
    level TINYINT,
    connection_id BIGINT,
    weight_distance TINYINT UNSIGNED NOT NULL,  -- exponentially-rounded distance to age=-1 for weighting
    PRIMARY KEY (level, connection_id),
    INDEX idx_level (level)
) ENGINE=MEMORY;

-- prediction/output neurons in t+1 in each level - potential future states (MEMORY table)
CREATE TABLE IF NOT EXISTS inferred_neurons (
    neuron_id BIGINT UNSIGNED NOT NULL,     -- id of the predicting active pattern neuron
    level TINYINT NOT NULL,  					    -- level of the predicting active pattern neuron
    age TINYINT UNSIGNED NOT NULL DEFAULT 0,        -- the age of the prediction - higher levels age slower
    PRIMARY KEY (neuron_id, level, age),
    INDEX idx_level_age (level, age),
    INDEX idx_current_active (age, level)
) ENGINE=MEMORY;

-- mapping table for observed patterns - peak neurons and their connections (MEMORY table)
-- this just a scratch table for faster processing - it temporarily holds the observed patterns for the current level in the frame
CREATE TABLE IF NOT EXISTS observed_patterns (
    peak_neuron_id BIGINT UNSIGNED NOT NULL,
    connection_id BIGINT UNSIGNED NOT NULL,
    INDEX idx_connection (connection_id),
    INDEX idx_peak (peak_neuron_id)
) ENGINE=MEMORY;

-- mapping table for matched patterns - peak neurons and their matched pattern neurons (MEMORY table)
-- this just a scratch table for faster processing - it temporarily holds the matched patterns for the current level in the frame
CREATE TABLE IF NOT EXISTS matched_patterns (
    peak_neuron_id BIGINT UNSIGNED NOT NULL,
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    PRIMARY KEY (peak_neuron_id, pattern_neuron_id),
    INDEX idx_pattern (pattern_neuron_id)
) ENGINE=MEMORY;

-- active connections for fast hierarchical reward propagation (MEMORY table)
-- this just a scratch table for faster processing - it temporarily holds the active connections for the current frame
CREATE TABLE IF NOT EXISTS active_connections (
    connection_id BIGINT UNSIGNED NOT NULL,
    from_neuron_id BIGINT UNSIGNED NOT NULL,
    to_neuron_id BIGINT UNSIGNED NOT NULL,
    level TINYINT NOT NULL,
    age TINYINT NOT NULL DEFAULT 0,
    PRIMARY KEY (connection_id, level),
    INDEX idx_to_neuron_level (to_neuron_id, level),
    INDEX idx_from_neuron_level (from_neuron_id, level),
    INDEX idx_level (level),
    INDEX idx_age (age)
) ENGINE=MEMORY;