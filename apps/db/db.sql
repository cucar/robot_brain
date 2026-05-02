-- Unified Brain Database Schema
CREATE DATABASE IF NOT EXISTS machine_intelligence;
USE machine_intelligence;

-- 1 = up, 2 = down, 3 = buy, 4 = sell (not created yet)

-- channels table for efficient storage (neurons reference by id instead of varchar)
-- IDs come from static class counters in Channel class (not auto-increment)
-- DROP TABLE IF EXISTS channels;
CREATE TABLE IF NOT EXISTS channels (
    id SMALLINT UNSIGNED PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL
);

-- dimensions table defines coordinate space (just names, no type/channel)
-- IDs are allocated by the Thalamus (nextDimensionId counter) when registering channel specs (not auto-increment)
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
CREATE TABLE base_neurons (
    neuron_id BIGINT PRIMARY KEY,
    channel_id SMALLINT UNSIGNED NOT NULL,
    type ENUM('event','action') NOT NULL,
    dimension_id SMALLINT UNSIGNED NOT NULL,
    val FLOAT NOT NULL,
    INDEX idx_channel_type (channel_id, type),
    INDEX idx_dim_val (dimension_id, val)
);

-- connections between base-level neurons (level=0 to level=0)
-- DROP TABLE IF EXISTS connections;
CREATE TABLE IF NOT EXISTS connections (
    from_neuron_id BIGINT UNSIGNED,
    to_neuron_id BIGINT UNSIGNED,
    distance TINYINT UNSIGNED NOT NULL,
    strength DECIMAL(30,20) DEFAULT 1.0,
    reward DECIMAL(30,20) DEFAULT 0,
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
    strength DECIMAL(30,20) NOT NULL DEFAULT 1.0,
    PRIMARY KEY (pattern_neuron_id),
    INDEX idx_parent (parent_neuron_id),
    INDEX idx_strength (strength)
);

-- contexts: pattern contexts for recognition/matching (cross-channel)
-- DROP TABLE IF EXISTS contexts;
CREATE TABLE IF NOT EXISTS contexts (
    pattern_neuron_id BIGINT UNSIGNED,
    context_neuron_id BIGINT UNSIGNED,
    context_age TINYINT UNSIGNED,
    strength DECIMAL(30,20) NOT NULL DEFAULT 1.0,
    PRIMARY KEY (pattern_neuron_id, context_neuron_id, context_age),
    INDEX idx_strength (strength)
);

-- neuron_error_stats: per-(neuron, age) running mean and variance of observed
-- prediction error rates, maintained online via Welford's algorithm. Used by
-- the dynamic error-correction modes (conservative / neutral / aggressive)
-- to derive each neuron's own threshold for spawning correction patterns.
-- Only neurons that have observed at least one error sample at a given age
-- have a row here — sparse by design.
-- DROP TABLE IF EXISTS neuron_error_stats;
CREATE TABLE IF NOT EXISTS neuron_error_stats (
    neuron_id BIGINT UNSIGNED NOT NULL,
    age TINYINT UNSIGNED NOT NULL,
    n INT UNSIGNED NOT NULL,
    mean DOUBLE NOT NULL,
    m2 DOUBLE NOT NULL,
    PRIMARY KEY (neuron_id, age)
);