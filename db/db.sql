-- Unified Brain Database Schema
CREATE DATABASE IF NOT EXISTS machine_intelligence;
USE machine_intelligence;

-- 1 = up, 2 = down, 3 = buy, 4 = sell (not created yet)
select c.neuron_id, d.name, c.val from coordinates c join dimensions d on d.id = c.dimension_id;
select * from dimensions;
select * from channels;
select * from neurons where id = 100848;
select * from pattern_past where context_neuron_id = 106110;
select * from neurons order by level desc;
select * from base_neurons;
select * from connections;
select parent_neuron_id, pattern_neuron_id, strength from patterns order by parent_neuron_id, pattern_neuron_id;
select * from patterns where pattern_neuron_id = 159;
select * from pattern_past where context_neuron_id = 159;
select * from pattern_past where context_neuron_id = 1632;
select * from patterns where pattern_neuron_id = 1632;

select * from pattern_past order by pattern_neuron_id, context_age;
select * from pattern_past where context_neuron_id not in (select id from neurons);
select * from pattern_peaks where pattern_neuron_id = 8;
select * from pattern_past where pattern_neuron_id in (8, 9) order by pattern_neuron_id, context_age;
select * from pattern_future order by pattern_neuron_id, distance, inferred_neuron_id;

-- channels table for efficient storage (neurons reference by id instead of varchar)
-- IDs come from static class counters in Channel class (not auto-increment)
-- DROP TABLE IF EXISTS channels;
CREATE TABLE IF NOT EXISTS channels (
    id SMALLINT UNSIGNED PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL
);

-- dimensions table defines coordinate space (just names, no type/channel)
-- IDs come from static class counters in Dimension class (not auto-increment)
-- DROP TABLE IF EXISTS dimensions;
CREATE TABLE IF NOT EXISTS dimensions (
    id SMALLINT UNSIGNED PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL
);

-- neurons table is the core representation of concepts - auto increment
-- level is an intrinsic property: base neurons are level 0, pattern neurons are level 1+
-- DROP TABLE IF EXISTS neurons;
CREATE TABLE IF NOT EXISTS neurons (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    level TINYINT UNSIGNED NOT NULL DEFAULT 0,
    INDEX idx_level (level)
);

-- base neuron attributes (level=0 neurons)
-- DROP TABLE IF EXISTS base_neurons;
CREATE TABLE IF NOT EXISTS base_neurons (
    neuron_id BIGINT PRIMARY KEY,
    channel_id SMALLINT UNSIGNED NOT NULL,
    type ENUM('event', 'action') NOT NULL,
    INDEX idx_channel_type (channel_id, type)
);

-- Base neuron coordinates (only level=0 neurons)
-- DROP TABLE IF EXISTS coordinates;
CREATE TABLE IF NOT EXISTS coordinates (
    neuron_id BIGINT UNSIGNED,
    dimension_id SMALLINT UNSIGNED,
    val FLOAT,
    PRIMARY KEY (neuron_id, dimension_id),
    INDEX (dimension_id, val)
);

-- connections between base-level neurons (level=0 to level=0)
-- DROP TABLE IF EXISTS connections;
CREATE TABLE IF NOT EXISTS connections (
    from_neuron_id BIGINT UNSIGNED,
    to_neuron_id BIGINT UNSIGNED,
    distance TINYINT UNSIGNED NOT NULL,
    strength DOUBLE DEFAULT 1.0,
    reward DOUBLE DEFAULT 0,
    PRIMARY KEY (from_neuron_id, to_neuron_id, distance),
    INDEX idx_from_distance_strength (from_neuron_id, distance, strength),
    INDEX idx_to_distance_strength (to_neuron_id, distance, strength),
    INDEX idx_strength (strength)
);

-- patterns - maps each pattern neuron to its parent neuron (the decision node that owns the pattern)
-- patterns are learned by parent neurons to differentiate between sequences leading to them
-- DROP TABLE IF EXISTS patterns;
CREATE TABLE IF NOT EXISTS patterns (
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    parent_neuron_id BIGINT UNSIGNED NOT NULL,
    strength DOUBLE NOT NULL DEFAULT 1.0,
    PRIMARY KEY (pattern_neuron_id),
    INDEX idx_parent (parent_neuron_id)
);

-- pattern_past: pattern contexts for recognition/matching (cross-channel)
-- DROP TABLE IF EXISTS pattern_past;
CREATE TABLE IF NOT EXISTS pattern_past (
    pattern_neuron_id BIGINT UNSIGNED,
    context_neuron_id BIGINT UNSIGNED,
    context_age TINYINT UNSIGNED,
    strength DOUBLE NOT NULL DEFAULT 1.0,
    PRIMARY KEY (pattern_neuron_id, context_neuron_id, context_age),
    INDEX idx_strength (strength)
);

-- pattern_future: cross-level predictions from patterns to base neurons for inference/voting (cross-channel)
-- patterns directly infer base-level neurons (events or actions) at various temporal distances
-- this is a cross-level connection: pattern neuron (level > 0) → base neuron (level 0)
-- distance: temporal distance of prediction (1 = next frame, 2 = 2 frames ahead, etc.)
-- DROP TABLE IF EXISTS pattern_future;
CREATE TABLE IF NOT EXISTS pattern_future (
    pattern_neuron_id BIGINT UNSIGNED,
    inferred_neuron_id BIGINT UNSIGNED,
    distance TINYINT UNSIGNED,
    strength DOUBLE NOT NULL DEFAULT 1.0,
    reward DOUBLE NOT NULL DEFAULT 0,
    PRIMARY KEY (pattern_neuron_id, inferred_neuron_id, distance),
    INDEX idx_pattern_distance_strength (pattern_neuron_id, distance, strength),
    INDEX idx_inferred_distance_strength (inferred_neuron_id, distance, strength),
    INDEX idx_strength (strength)
);