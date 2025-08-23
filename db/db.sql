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
-- DROP TABLE IF EXISTS active_neurons;
-- DROP TABLE IF EXISTS predicted_connections;

-- dimensions table determines input/output mapping
CREATE TABLE IF NOT EXISTS dimensions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) UNIQUE NOT NULL
);

-- these dimensions can be used for visual processing
INSERT INTO dimensions (name) VALUES ('x'), ('y'), ('r'), ('g'), ('b');

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
    INDEX idx_strength (strength)
);

-- pattern neurons definitions - neuron connections between levels
CREATE TABLE IF NOT EXISTS patterns (
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    connection_id BIGINT UNSIGNED NOT NULL,
    strength DOUBLE NOT NULL DEFAULT 1.0,
    PRIMARY KEY (pattern_neuron_id, connection_id),
    FOREIGN KEY (pattern_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    FOREIGN KEY (connection_id) REFERENCES connections(id) ON DELETE CASCADE,
    INDEX idx_from_distance_strength (connection_id, strength DESC)
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

-- predicted connections - potential future states (MEMORY table)
CREATE TABLE IF NOT EXISTS predicted_connections (
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,     -- id of the predicting active pattern neuron
    connection_id BIGINT UNSIGNED NOT NULL,         -- predicted connection that did not occur yet
    level TINYINT NOT NULL,  					    -- level of the predicting active pattern neuron
    age TINYINT UNSIGNED NOT NULL DEFAULT 0,        -- the age of the prediction - higher levels age slower
    prediction_strength DOUBLE NOT NULL,            -- confidence of prediction
    PRIMARY KEY (pattern_neuron_id, connection_id), -- different levels/ages predicting the same connection just resets the age and strengthens the connection
    INDEX idx_prediction_aging (age),
    INDEX idx_prediction_validation (connection_id)
) ENGINE=MEMORY;