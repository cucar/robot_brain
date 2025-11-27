-- Unified Brain Database Schema
-- Single connections table handles ALL relationships: spatial, temporal, hierarchical, predictions
-- Memory tables for active state, persistent tables for learned knowledge
CREATE DATABASE IF NOT EXISTS machine_intelligence;
USE machine_intelligence;

-- drop statements for reset (used tables only)
-- DROP TABLE IF EXISTS dimensions;
-- DROP TABLE IF EXISTS neurons;
-- DROP TABLE IF EXISTS coordinates;
-- DROP TABLE IF EXISTS connections;
-- DROP TABLE IF EXISTS pattern_past;
-- DROP TABLE IF EXISTS pattern_future;
-- DROP TABLE IF EXISTS pattern_peaks;
-- DROP TABLE IF EXISTS active_neurons;
-- DROP TABLE IF EXISTS inferred_neurons;
-- DROP TABLE IF EXISTS inferred_neurons_resolved;
-- DROP TABLE IF EXISTS active_connections;
-- DROP TABLE IF EXISTS matched_patterns;
-- DROP TABLE IF EXISTS matched_pattern_connections;
-- DROP TABLE IF EXISTS connection_inference_sources;
-- DROP TABLE IF EXISTS pattern_inference_sources;
-- DROP TABLE IF EXISTS exploration_inference_sources;
-- DROP TABLE IF EXISTS inference_log;
-- DROP TABLE IF EXISTS inference_chain;
-- DROP TABLE IF EXISTS unpack_sources;
-- DROP TABLE IF EXISTS unpredicted_connections;
-- DROP TABLE IF EXISTS new_patterns;

select * from inferred_neurons;

SELECT c.to_neuron_id, 0, 0, c.id, c.strength * c.reward * POW(0.9, c.distance - 1)
FROM active_neurons an
JOIN connections c ON c.from_neuron_id = an.neuron_id
WHERE an.level = 0
AND c.to_neuron_id = 4
AND c.distance = an.age + 1
AND c.strength > 0;

SELECT SUM(prediction_strength) as total_strength
FROM exploration_inference_sources
WHERE inferred_neuron_id = 4 AND level = 0 AND age = 0;

-- check state
select count(*) from neurons;
select * from dimensions;
select * from coordinates where neuron_id = 5;
select count(*) from connections;
select count(*) from connections where strength > 0;
select * from connections where from_neuron_id in (SELECT pattern_neuron_id FROM pattern_peaks pp) order by strength desc;
select count(*) from pattern_peaks;
select count(*) from pattern_past;
select count(*) from pattern_future;
select * from unpack_sources;
select * from inferred_neurons;
select * from inferred_neurons_resolved;
select * from inference_log;
select * from inference_chain;
select * from connections;
select * from coordinates where dimension_id = 16;

select * from connection_inference_sources where age = 1 and inferred_neuron_id = 8;
select * from connections where id = 9;
select * from coordinates where neuron_id = 3;
select * from dimensions;

SELECT c.to_neuron_id, 0, 0, c.id, c.strength * c.reward * POW(0.9, c.distance - 1)
FROM active_neurons an
JOIN connections c ON c.from_neuron_id = an.neuron_id
WHERE an.level = 0
AND c.distance = an.age + 1
AND c.strength > 0;

select * from inference_chain;

select * from active_neurons;
select * from active_connections;
select * from connections;

SELECT an.level, an.age, an.neuron_id, c.to_neuron_id, c.id, c.strength * c.reward * POW(0.9, c.distance - 1)
FROM active_neurons an
JOIN connections c ON c.from_neuron_id = an.neuron_id
WHERE an.level = 0
AND c.distance = an.age + 1
AND c.strength > 0;

select * from dimensions;
select * from coordinates where neuron_id in (4);

select * from connections where id in (9, 13);            

select * 
from connection_inference_sources cis -- ON c.id = cis.connection_id
JOIN inference_chain ic ON ic.source_neuron_id = cis.inferred_neuron_id AND ic.age = cis.age
JOIN inference_log il ON il.age = ic.age AND il.type = 'connection'
JOIN inferred_neurons_resolved inf ON inf.neuron_id = ic.base_neuron_id AND inf.level = 0 AND inf.age = ic.age
JOIN coordinates coord ON coord.neuron_id = ic.base_neuron_id AND coord.dimension_id IN (16)
WHERE ic.age > 0 AND ic.age <= 10
AND cis.level = il.level
;

select id from dimensions where type = 'output';
select * from inference_chain;
select * from unpack_sources;

SELECT current_neuron_id, source_neuron_id, 0
FROM unpack_sources
WHERE current_level = 0;

select * 
from connections c
JOIN connection_inference_sources cis ON c.id = cis.connection_id
JOIN inference_chain ic ON ic.source_neuron_id = cis.inferred_neuron_id AND ic.age = cis.age
-- JOIN inference_log il ON il.age = ic.age AND il.type = 'connection'
-- JOIN inferred_neurons_resolved inf ON inf.neuron_id = ic.base_neuron_id AND inf.level = 0 AND inf.age = ic.age
-- JOIN coordinates coord ON coord.neuron_id = ic.base_neuron_id AND coord.dimension_id IN (select id from dimensions where type = 'output')
-- WHERE ic.age > 0 AND ic.age <= 1
-- AND cis.level = il.level
;
                

select * from inferred_neurons_resolved where age = 1;
select * from connection_inference_sources;

SELECT *
FROM inferred_neurons_resolved inf
LEFT JOIN active_neurons an ON inf.neuron_id = an.neuron_id AND an.level = inf.level AND an.age = 0
WHERE inf.age = 1;

SELECT current_neuron_id, source_neuron_id, 0
FROM unpack_sources
WHERE current_level = 0;

select * from inference_chain;

select *
from connections c
JOIN connection_inference_sources cis ON cis.connection_id = c.id
JOIN inferred_neurons_resolved inf ON inf.neuron_id = cis.inferred_neuron_id AND inf.level = cis.level
WHERE cis.level = 0
AND c.strength > 0
AND NOT EXISTS (
	SELECT 1 FROM active_neurons an
	WHERE an.neuron_id = cis.inferred_neuron_id
	AND an.level = cis.level
	AND an.age = 0
);

SELECT *
FROM inferred_neurons_resolved inf
LEFT JOIN active_neurons an ON inf.neuron_id = an.neuron_id AND an.level = inf.level AND an.age = 0
WHERE inf.age = 1;

select * from connection_inference_sources;

SELECT inferred_neuron_id, level, 0, SUM(prediction_strength) as total_strength
FROM connection_inference_sources
WHERE level = 1
GROUP BY inferred_neuron_id, level
HAVING total_strength >= 10;

SELECT src.neuron_id, src.level, src.strength
FROM inferred_neurons src
WHERE src.level = 0
AND src.age = 0;

select * from connection_inference_sources;
select * from active_connections where to_neuron_id = 6;

select * from unpredicted_connections;
select * from new_patterns;

SELECT COUNT(DISTINCT pattern_neuron_id) as total_patterns FROM pattern_peaks;
SELECT FLOOR(strength) as strength_bucket, COUNT(*) as count FROM connections WHERE strength > 0 GROUP BY strength_bucket ORDER BY strength_bucket;
SELECT AVG(strength) as avg_strength, MAX(strength) as max_strength, MIN(strength) as min_strength FROM connections WHERE strength > 0;

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
    reward DOUBLE NOT NULL DEFAULT 1.0,  -- multiplicative reward factor for temporal credit assignment
    FOREIGN KEY (from_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    FOREIGN KEY (to_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    UNIQUE INDEX (from_neuron_id, to_neuron_id, distance),
    INDEX idx_from_distance_strength (from_neuron_id, distance, strength),
    INDEX idx_to_distance_strength (to_neuron_id, distance, strength),
    INDEX idx_distance_strength (distance, strength),
    INDEX idx_strength (strength)
) ENGINE=MEMORY;

-- pattern_past: connections leading TO the peak (for recognition/matching)
CREATE TABLE IF NOT EXISTS pattern_past (
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    connection_id BIGINT UNSIGNED NOT NULL,
    strength DOUBLE NOT NULL DEFAULT 1.0,
    PRIMARY KEY (pattern_neuron_id, connection_id),
    FOREIGN KEY (pattern_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    FOREIGN KEY (connection_id) REFERENCES connections(id) ON DELETE CASCADE,
    INDEX idx_connection_strength (connection_id, strength),
    INDEX idx_pattern_strength (pattern_neuron_id, strength),
    INDEX idx_pattern_connection (pattern_neuron_id, connection_id, strength),
    INDEX idx_strength (strength)
) ENGINE=MEMORY;

-- pattern_future: connections FROM the peak (for inference/unpacking)
CREATE TABLE IF NOT EXISTS pattern_future (
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    connection_id BIGINT UNSIGNED NOT NULL,
    strength DOUBLE NOT NULL DEFAULT 1.0,
    reward DOUBLE NOT NULL DEFAULT 1.0,  -- multiplicative reward factor for temporal credit assignment
    PRIMARY KEY (pattern_neuron_id, connection_id),
    FOREIGN KEY (pattern_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
    FOREIGN KEY (connection_id) REFERENCES connections(id) ON DELETE CASCADE,
    INDEX idx_pattern_strength (pattern_neuron_id, strength),
    INDEX idx_strength (strength)
) ENGINE=MEMORY;

-- pattern peaks - maps each pattern neuron to its peak neuron (the decision node that owns the pattern)
-- patterns are learned by peak neurons to differentiate between sequences leading to them
CREATE TABLE IF NOT EXISTS pattern_peaks (
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    peak_neuron_id BIGINT UNSIGNED NOT NULL,
    strength DECIMAL(10,2) NOT NULL DEFAULT 1.0,
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

-- raw predictions before conflict resolution (MEMORY table)
-- used by both connection and pattern inference (only one has data per frame due to early-return)
CREATE TABLE IF NOT EXISTS inferred_neurons (
    neuron_id BIGINT UNSIGNED NOT NULL,
    level TINYINT NOT NULL,
    age TINYINT UNSIGNED NOT NULL DEFAULT 0,
    strength DOUBLE NOT NULL DEFAULT 0.0,
    PRIMARY KEY (neuron_id, level, age),
    INDEX idx_level_age (level, age)
) ENGINE=MEMORY;

-- final resolved predictions after conflict resolution (MEMORY table)
CREATE TABLE IF NOT EXISTS inferred_neurons_resolved (
    neuron_id BIGINT UNSIGNED NOT NULL,
    level TINYINT NOT NULL,
    age TINYINT UNSIGNED NOT NULL DEFAULT 0,
    strength DOUBLE NOT NULL DEFAULT 0.0,
    PRIMARY KEY (neuron_id, level, age),
    INDEX idx_level_age (level, age),
    INDEX idx_current_active (age, level)
) ENGINE=MEMORY;

-- mapping table for matched patterns - peak neurons and their matched pattern neurons (MEMORY table)
-- this just a scratch table for faster processing - it temporarily holds the matched patterns for the current level in the frame
CREATE TABLE IF NOT EXISTS matched_patterns (
    peak_neuron_id BIGINT UNSIGNED NOT NULL,
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    PRIMARY KEY (peak_neuron_id, pattern_neuron_id),
    INDEX idx_pattern (pattern_neuron_id)
) ENGINE=MEMORY;

-- scratch table for pattern connection analysis (MEMORY table)
-- populated during matchObservedPatterns in a single pass, consumed by mergeMatchedPatterns
-- status: 'common' = in both pattern and active (strengthen), 'novel' = in active only (add), 'missing' = in pattern only (weaken)
CREATE TABLE IF NOT EXISTS matched_pattern_connections (
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    connection_id BIGINT UNSIGNED NOT NULL,
    status ENUM('common', 'novel', 'missing') NOT NULL,
    INDEX idx_pattern (pattern_neuron_id),
    INDEX idx_connection (connection_id),
    INDEX idx_status (status)
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

-- scratch table for tracking connection-based predictions (MEMORY table)
-- populated during connection inference, used for validation and error pattern creation
-- ages with inferred_neurons, deleted when age >= baseNeuronMaxAge
CREATE TABLE IF NOT EXISTS connection_inference_sources (
    inferred_neuron_id BIGINT UNSIGNED NOT NULL,
    level TINYINT NOT NULL,
    age TINYINT UNSIGNED NOT NULL DEFAULT 0,
    connection_id BIGINT UNSIGNED NOT NULL,
    prediction_strength DOUBLE NOT NULL,
    PRIMARY KEY (inferred_neuron_id, level, age, connection_id),
    INDEX idx_connection (connection_id),
    INDEX idx_aggregate (level, inferred_neuron_id, prediction_strength),
    INDEX idx_age (age)
) ENGINE=MEMORY;

-- scratch table for tracking pattern-based predictions (MEMORY table)
-- populated during pattern inference, used for validation and error pattern creation
-- ages with inferred_neurons, deleted when age >= baseNeuronMaxAge
CREATE TABLE IF NOT EXISTS pattern_inference_sources (
    inferred_neuron_id BIGINT UNSIGNED NOT NULL,
    level TINYINT NOT NULL,
    age TINYINT UNSIGNED NOT NULL DEFAULT 0,
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    connection_id BIGINT UNSIGNED NOT NULL,
    prediction_strength DOUBLE NOT NULL,
    PRIMARY KEY (inferred_neuron_id, level, age, pattern_neuron_id, connection_id),
    INDEX idx_pattern (pattern_neuron_id),
    INDEX idx_connection (connection_id),
    INDEX idx_aggregate (level, inferred_neuron_id, prediction_strength),
    INDEX idx_age (age)
) ENGINE=MEMORY;

-- scratch table for tracking exploration-based predictions (MEMORY table)
-- populated during curiosity exploration, used for reward application
-- ages with inferred_neurons, deleted when age >= baseNeuronMaxAge
CREATE TABLE IF NOT EXISTS exploration_inference_sources (
    inferred_neuron_id BIGINT UNSIGNED NOT NULL,
    age TINYINT UNSIGNED NOT NULL DEFAULT 0,
    connection_id BIGINT UNSIGNED NOT NULL,
    prediction_strength DOUBLE NOT NULL,
		PRIMARY KEY (inferred_neuron_id, age, connection_id),
    INDEX idx_connection (connection_id),
    INDEX idx_aggregate (inferred_neuron_id, prediction_strength),
    INDEX idx_age (age)
) ENGINE=MEMORY;

-- scratch table for tracking which inference type was performed at each level for each frame
-- used for temporal credit assignment in applyRewards
-- ages with inferred_neurons, deleted when age >= baseNeuronMaxAge
-- multiple inference types can occur in same frame (e.g., connection + exploration)
CREATE TABLE IF NOT EXISTS inference_log (
    age TINYINT UNSIGNED NOT NULL,
    level TINYINT NOT NULL,
    type ENUM('connection', 'pattern', 'exploration') NOT NULL,
    PRIMARY KEY (age, level, type),
    INDEX idx_age (age)
) ENGINE=MEMORY;

-- scratch table for tracking which high-level source neurons unpacked to which base-level outputs
-- used for channel-specific reward attribution in applyRewards
-- populated during saveInferenceChain: maps base outputs (level 0) to their source-level neurons
-- source_level can be obtained from inference_log by joining on age
-- unpacking is just mechanical translation via pattern_peaks, so we only track endpoints
-- age represents how many frames ago this prediction was made (both source and base age together)
-- ages with inferred_neurons, deleted when age >= baseNeuronMaxAge
CREATE TABLE IF NOT EXISTS inference_chain (
    base_neuron_id BIGINT UNSIGNED NOT NULL,
    source_neuron_id BIGINT UNSIGNED NOT NULL,
    age TINYINT UNSIGNED NOT NULL,
    PRIMARY KEY (base_neuron_id, source_neuron_id, age),
    INDEX idx_base (base_neuron_id, age),
    INDEX idx_source (source_neuron_id, age),
    INDEX idx_age (age)
) ENGINE=MEMORY;

-- scratch table for tracking source neurons during unpacking (MEMORY table)
-- used internally by saveInferenceChain to propagate source info as we unpack level by level
-- truncated at start of saveInferenceChain, populated during unpacking, then copied to inference_chain
CREATE TABLE IF NOT EXISTS unpack_sources (
    current_neuron_id BIGINT UNSIGNED NOT NULL,
    current_level TINYINT NOT NULL,
    source_neuron_id BIGINT UNSIGNED NOT NULL,
    PRIMARY KEY (current_neuron_id, current_level, source_neuron_id),
    INDEX idx_current (current_neuron_id, current_level)
) ENGINE=MEMORY;

-- scratch table for tracking unpredicted connections (MEMORY table)
-- connections that fired but were not predicted
CREATE TABLE IF NOT EXISTS unpredicted_connections (
    connection_id BIGINT UNSIGNED NOT NULL,
    level TINYINT NOT NULL,
    from_neuron_id BIGINT UNSIGNED NOT NULL,
    to_neuron_id BIGINT UNSIGNED NOT NULL,
    strength DOUBLE NOT NULL,
    INDEX idx_connection (connection_id),
    INDEX idx_from (from_neuron_id),
    INDEX idx_level (level)
) ENGINE=MEMORY;

-- scratch table for mapping peak neurons to new pattern neurons (MEMORY table)
-- used during bulk pattern creation to track sequential IDs
CREATE TABLE IF NOT EXISTS new_patterns (
    seq_id INT AUTO_INCREMENT PRIMARY KEY,
    peak_neuron_id BIGINT UNSIGNED NOT NULL UNIQUE,
    pattern_neuron_id BIGINT UNSIGNED,
    INDEX idx_peak (peak_neuron_id)
) ENGINE=MEMORY;

-- increase the amount of records that can be stored in memory tables
SET GLOBAL max_heap_table_size = 1024 * 1024 * 1024 * 2;