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
-- DROP TABLE IF EXISTS connection_inference;
-- DROP TABLE IF EXISTS pattern_inferred_neurons;
-- DROP TABLE IF EXISTS connection_inferred_neurons;
-- DROP TABLE IF EXISTS inferred_neurons;
-- DROP TABLE IF EXISTS observed_connections;
-- DROP TABLE IF EXISTS observed_neuron_strengths;
-- DROP TABLE IF EXISTS observed_peaks;
-- DROP TABLE IF EXISTS observed_patterns;
-- DROP TABLE IF EXISTS inferred_connections;
-- DROP TABLE IF EXISTS inferred_neuron_strengths;
-- DROP TABLE IF EXISTS inferred_level_strengths;
-- DROP TABLE IF EXISTS inferred_peaks;
-- DROP TABLE IF EXISTS active_connections;
-- DROP TABLE IF EXISTS matched_peaks;
-- DROP TABLE IF EXISTS matched_patterns;

-- check state
select count(*) from neurons;
select count(*) from coordinates;
select count(*) from connections;
select count(*) from connections where strength > 0;
select * from connections where id > 96 and distance = 2;
select * from active_neurons;
select * from connection_inference;
select count(*) from active_connections;
select count(*) from patterns where strength > 0;

-- dimensions table determines input/output mapping for channels
CREATE TABLE IF NOT EXISTS dimensions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) UNIQUE NOT NULL,
    channel VARCHAR(50) NOT NULL,
    type ENUM('input', 'output') NOT NULL
) ENGINE=MEMORY;

-- these dimensions can be used for visual processing
-- INSERT INTO dimensions (name) VALUES ('x'), ('y'), ('r'), ('g'), ('b');

-- these dimensions are used for forecasting timeseries changes - slopes can go from -100% to 100% (rate of change)
-- INSERT INTO dimensions (name) VALUES ('gold_slope'), ('aem_slope');

-- test dimensions
-- INSERT IGNORE INTO dimensions (name) VALUES ('d0'), ('d1'), ('d2'), ('d3'), ('d4'), ('d5'), ('d6'), ('d7'), ('d8'), ('d9'), ('d10'), ('d11'), ('d12'), ('d13'), ('d14'), ('d15'), ('d16'), ('d17'), ('d18'), ('d19');

-- neurons table is the core representation of concepts - auto increment
CREATE TABLE IF NOT EXISTS neurons (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY
) ENGINE=MEMORY;

-- coordinates for base neurons
CREATE TABLE IF NOT EXISTS coordinates (
    neuron_id BIGINT UNSIGNED NOT NULL,
    dimension_id INT NOT NULL,
    val DOUBLE NOT NULL,
    PRIMARY KEY (neuron_id, dimension_id),
    FOREIGN KEY (neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    INDEX (dimension_id, val)
) ENGINE=MEMORY;

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
    INDEX idx_from_distance_strength (from_neuron_id, distance, strength),
    INDEX idx_to_distance_strength (to_neuron_id, distance, strength),
    INDEX idx_distance_strength (distance, strength),
    INDEX idx_strength (id, strength),
    INDEX idx_strength2 (strength)
) ENGINE=MEMORY;

-- pattern neurons definitions - neuron connections between levels
CREATE TABLE IF NOT EXISTS patterns (
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    connection_id BIGINT UNSIGNED NOT NULL,
    strength DOUBLE NOT NULL DEFAULT 1.0,
    PRIMARY KEY (pattern_neuron_id, connection_id),
    FOREIGN KEY (pattern_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    FOREIGN KEY (connection_id) REFERENCES connections(id) ON DELETE CASCADE,
    INDEX idx_connection_strength (connection_id, strength),
    INDEX idx_pattern_strength (pattern_neuron_id, strength),
    INDEX idx_strength (strength)
) ENGINE=MEMORY;

-- pattern peaks - maps each pattern neuron to its peak neuron (the decision node that owns the pattern)
-- patterns are learned by peak neurons to differentiate between sequences leading to them
CREATE TABLE IF NOT EXISTS pattern_peaks (
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    peak_neuron_id BIGINT UNSIGNED NOT NULL,
    PRIMARY KEY (pattern_neuron_id),
    FOREIGN KEY (pattern_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    FOREIGN KEY (peak_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    INDEX idx_peak (peak_neuron_id)
) ENGINE=MEMORY;

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

-- stores same-level connection predictions - scratch table for validation between frames
CREATE TABLE connection_inference (
    level TINYINT,
    connection_id BIGINT,
    to_neuron_id BIGINT UNSIGNED,
    strength DOUBLE NOT NULL DEFAULT 0.0,
    PRIMARY KEY (level, connection_id),
    INDEX idx_level (level),
    INDEX idx_to_neuron (level, to_neuron_id)
) ENGINE=MEMORY;

-- connection-based predictions before conflict resolution (MEMORY table)
CREATE TABLE IF NOT EXISTS connection_inferred_neurons (
    neuron_id BIGINT UNSIGNED NOT NULL,
    level TINYINT NOT NULL,
    age TINYINT UNSIGNED NOT NULL DEFAULT 0,
    strength DOUBLE NOT NULL DEFAULT 0.0,
    PRIMARY KEY (neuron_id, level, age),
    INDEX idx_level_age (level, age)
) ENGINE=MEMORY;

-- pattern-based predictions before conflict resolution (MEMORY table)
CREATE TABLE IF NOT EXISTS pattern_inferred_neurons (
    neuron_id BIGINT UNSIGNED NOT NULL,
    level TINYINT NOT NULL,
    age TINYINT UNSIGNED NOT NULL DEFAULT 0,
    strength DOUBLE NOT NULL DEFAULT 0.0,
    PRIMARY KEY (neuron_id, level, age),
    INDEX idx_level_age (level, age)
) ENGINE=MEMORY;

-- final resolved predictions after conflict resolution (MEMORY table)
CREATE TABLE IF NOT EXISTS inferred_neurons (
    neuron_id BIGINT UNSIGNED NOT NULL,
    level TINYINT NOT NULL,
    age TINYINT UNSIGNED NOT NULL DEFAULT 0,
    strength DOUBLE NOT NULL DEFAULT 0.0,
    PRIMARY KEY (neuron_id, level, age),
    INDEX idx_level_age (level, age),
    INDEX idx_current_active (age, level)
) ENGINE=MEMORY;

-- scratch table for weighted connection data during peak detection (MEMORY table)
-- stores pre-calculated weighted strengths to avoid rescanning active_connections multiple times
CREATE TABLE IF NOT EXISTS observed_connections (
    to_neuron_id BIGINT UNSIGNED NOT NULL,
    connection_id BIGINT UNSIGNED NOT NULL,
    strength DOUBLE NOT NULL,
    INDEX idx_to_neuron (to_neuron_id),
    INDEX idx_connection (connection_id)
) ENGINE=MEMORY;

-- scratch table for per-neuron aggregates during peak detection (MEMORY table)
-- stores total_strength and connection_count for ALL candidate neurons before filtering for peaks
CREATE TABLE IF NOT EXISTS observed_neuron_strengths (
    to_neuron_id BIGINT UNSIGNED NOT NULL,
    total_strength DOUBLE NOT NULL,
    connection_count INT UNSIGNED NOT NULL,
    PRIMARY KEY (to_neuron_id),
    INDEX idx_strength_count (total_strength, connection_count)
) ENGINE=MEMORY;

-- mapping table for observed peak neurons - just the peaks detected in current frame (MEMORY table)
-- mirrors pattern_peaks design - stores just the peak neurons for fast existence checks
CREATE TABLE IF NOT EXISTS observed_peaks (
    peak_neuron_id BIGINT UNSIGNED NOT NULL,
    total_strength DOUBLE NOT NULL,
    connection_count INT UNSIGNED NOT NULL,
    PRIMARY KEY (peak_neuron_id)
) ENGINE=MEMORY;

-- mapping table for observed patterns - peak neurons and their connections (MEMORY table)
-- this just a scratch table for faster processing - it temporarily holds the observed patterns for the current level in the frame
CREATE TABLE IF NOT EXISTS observed_patterns (
    peak_neuron_id BIGINT UNSIGNED NOT NULL,
    connection_id BIGINT UNSIGNED NOT NULL,
    INDEX idx_peak_connection (peak_neuron_id, connection_id),  -- Composite index for JOINs in matchPatternNeurons and mergeMatchedPatterns
    INDEX idx_connection (connection_id),
    INDEX idx_peak (peak_neuron_id)
) ENGINE=MEMORY;

-- scratch table for candidate connections during inference (MEMORY table)
-- stores pre-calculated weighted strengths for all candidate predictions
CREATE TABLE IF NOT EXISTS inferred_connections (
    level TINYINT NOT NULL,
    connection_id BIGINT UNSIGNED NOT NULL,
    from_neuron_id BIGINT UNSIGNED NOT NULL,
    to_neuron_id BIGINT UNSIGNED NOT NULL,
    strength DOUBLE NOT NULL,
    INDEX idx_level_to_neuron (level, to_neuron_id),
    INDEX idx_connection (connection_id)
) ENGINE=MEMORY;

-- scratch table for per-neuron aggregates during inference (MEMORY table)
-- stores total_strength for ALL candidate predictions before filtering for peaks
CREATE TABLE IF NOT EXISTS inferred_neuron_strengths (
    level TINYINT NOT NULL,
    to_neuron_id BIGINT UNSIGNED NOT NULL,
    total_strength DOUBLE NOT NULL,
    PRIMARY KEY (level, to_neuron_id)
) ENGINE=MEMORY;

-- scratch table for per-level average strengths during inference (MEMORY table)
-- stores average strength per level for peak detection threshold calculation
CREATE TABLE IF NOT EXISTS inferred_level_strengths (
    level TINYINT NOT NULL,
    avg_strength DOUBLE NOT NULL,
    PRIMARY KEY (level)
) ENGINE=MEMORY;

-- mapping table for inferred peak neurons - just the peaks detected during inference (MEMORY table)
-- mirrors observed_peaks design - stores just the peak predictions for fast existence checks
CREATE TABLE IF NOT EXISTS inferred_peaks (
    level TINYINT NOT NULL,
    peak_neuron_id BIGINT UNSIGNED NOT NULL,
    total_strength DOUBLE NOT NULL,
    PRIMARY KEY (level, peak_neuron_id)
) ENGINE=MEMORY;

-- mapping table for matched peaks (MEMORY table)
-- stores just the peak neurons that have at least one pattern match
-- mirrors observed_peaks and pattern_peaks design for fast existence checks
CREATE TABLE IF NOT EXISTS matched_peaks (
    peak_neuron_id BIGINT UNSIGNED NOT NULL,
    PRIMARY KEY (peak_neuron_id)
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
    PRIMARY KEY (connection_id, level, age),  -- Allow same connection at different ages for reward distribution
    INDEX idx_to_neuron_level (to_neuron_id, level),
    INDEX idx_from_neuron_level (from_neuron_id, level),
    INDEX idx_level_age (level, age)  -- Composite index for detectPeaks WHERE clause
) ENGINE=MEMORY;

-- increase the amount of records that can be stored in memory tables
SET GLOBAL max_heap_table_size = 1024 * 1024 * 1024 * 2;