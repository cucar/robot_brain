CREATE DATABASE IF NOT EXISTS machine_intelligence;
USE machine_intelligence;

-- DROP TABLE IF EXISTS dimensions;
CREATE TABLE IF NOT EXISTS dimensions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) UNIQUE NOT NULL
);

-- these dimensions can be used for visual processing
INSERT INTO dimensions (name) VALUES ('x'), ('y'), ('r'), ('g'), ('b');

-- these dimensions are used for forecasting timeseries changes - slopes can go from -100% to 100% (rate of change)
-- INSERT INTO dimensions (name) VALUES ('gold_slope'), ('aem_slope');

-- DROP TABLE IF EXISTS neurons;
CREATE TABLE IF NOT EXISTS neurons (
    id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
    creation_time DATETIME NOT NULL DEFAULT NOW()
);

-- DROP TABLE IF EXISTS coordinates;
CREATE TABLE IF NOT EXISTS coordinates (
    neuron_id BIGINT UNSIGNED NOT NULL,
    dimension_id INT NOT NULL,
    val DOUBLE NOT NULL,
    PRIMARY KEY (neuron_id, dimension_id),
    INDEX (dimension_id, val),
    FOREIGN KEY (neuron_id) REFERENCES neurons(id) ON DELETE CASCADE
);

-- connections are directionless - source id is always smaller than target id by convention
-- DROP TABLE IF EXISTS connections;
CREATE TABLE IF NOT EXISTS connections (
    neuron1_id BIGINT UNSIGNED NOT NULL,
    neuron2_id BIGINT UNSIGNED NOT NULL,
    strength DOUBLE NOT NULL DEFAULT 1,
    PRIMARY KEY (neuron1_id, neuron2_id), -- composite PRIMARY KEY for unique connections
    FOREIGN KEY (neuron1_id) REFERENCES neurons(id) ON DELETE CASCADE,
    FOREIGN KEY (neuron2_id) REFERENCES neurons(id) ON DELETE CASCADE
);

-- patterns are used for predictions - they store matches from lower-level neurons to higher-level neurons
-- DROP TABLE IF EXISTS patterns;
CREATE TABLE IF NOT EXISTS patterns (
    parent_id BIGINT UNSIGNED NOT NULL,
    child_id BIGINT UNSIGNED NOT NULL,
    strength DOUBLE NOT NULL DEFAULT 1,
    PRIMARY KEY (parent_id, child_id), -- composite PRIMARY KEY for unique connections
    FOREIGN KEY (parent_id) REFERENCES neurons(id) ON DELETE CASCADE,
    FOREIGN KEY (child_id) REFERENCES neurons(id) ON DELETE CASCADE
);

-- neurons currently active within the sliding window (MEMORY Table)
CREATE TABLE IF NOT EXISTS active_neurons (
    neuron_id BIGINT UNSIGNED,
    level INT NOT NULL,
    age INT NOT NULL,
    INDEX idx_neuron_id (neuron_id),
    INDEX idx_level (level),
    INDEX idx_age (age)
) ENGINE=MEMORY;