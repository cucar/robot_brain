-- Unified Brain Database Schema
-- Single connections table handles ALL relationships: spatial, temporal, hierarchical, predictions
-- Memory tables for active state, persistent tables for learned knowledge
CREATE DATABASE IF NOT EXISTS machine_intelligence;
USE machine_intelligence;

select * from inferred_neurons where age = 0;
select * from inferred_neurons where age = 1;
select * from active_neurons where age in (0,1) order by age, neuron_id;
select * from new_patterns;
select * from new_pattern_future;

select * from neurons where level > 0;
select * from dimensions;

-- connection inferences - 1=up, 2=down, 3=buy, 4=sell
SELECT an.neuron_id as from_neuron, an.age, c.to_neuron_id, c.distance, c.strength, c.reward
-- SELECT c.to_neuron_id, avg(c.reward), sum(c.strength), group_concat(an.neuron_id)
FROM active_neurons an
JOIN neurons nf ON nf.id = an.neuron_id
JOIN connections c ON c.from_neuron_id = an.neuron_id
JOIN neurons n ON n.id = c.to_neuron_id
WHERE c.distance = an.age + 1
and n.type = 'action'
-- group by c.to_neuron_id
-- order by c.to_neuron_id
order by c.to_neuron_id, an.age, an.neuron_id, c.distance, c.to_neuron_id, an.age
;

select if(neuron_id=1,'up','down'), age, strength from active_neurons order by age;

select * from connections;
select * from connections where to_neuron_id = 4 order by from_neuron_id, distance;
select * from connections where from_neuron_id in (5,2) and to_neuron_id in (6, 9) and distance = 1 order by to_neuron_id, distance;
select * from connections where id in (9, 13);            
select * from connections where from_neuron_id in (SELECT pattern_neuron_id FROM pattern_peaks pp) order by strength desc;
select count(*) from connections;
select count(*) from connections where strength > 0;

select * from coordinates;
select * from coordinates where neuron_id = 2;
select * from coordinates where dimension_id = 16;
select * from coordinates where neuron_id in (1,2,11,12,13);
select * from coordinates where neuron_id = 5;

select * from active_neurons;
select * from active_connections;
select * from active_connections where to_neuron_id = 6;

select * from matched_patterns;
select * from matched_pattern_connections;

select * from connections where id in (select connection_id from new_pattern_future);
select * from new_patterns;

select * from pattern_peaks;
select * from connections where id in (select connection_id from pattern_future where pattern_neuron_id = 10);

-- action pattern contexts
select p.pattern_neuron_id, c.to_neuron_id as peak_neuron_id, c.from_neuron_id, c.distance, c.id
from pattern_past p 
join connections c on p.connection_id = c.id 
where pattern_neuron_id in (select id from neurons where level > 0 and type = 'action')
order by p.pattern_neuron_id, c.distance, c.from_neuron_id;

-- action pattern contexts
select p.pattern_neuron_id, c.to_neuron_id as peak_neuron_id, c.from_neuron_id, c.distance, c.id
from pattern_past p 
join connections c on p.connection_id = c.id 
where pattern_neuron_id in (select id from neurons where level > 0 and type = 'action')
order by p.pattern_neuron_id, c.distance, c.from_neuron_id;

select * from neurons where id in (32);
select * from neurons where level > 0;

select * from active_neurons;

select * from pattern_past p join connections c on p.connection_id = c.id where pattern_neuron_id = 10 order by c.distance;
select * from pattern_future where pattern_neuron_id = 10;
select peak_neuron_id, pattern_neuron_id, strength from pattern_peaks order by pattern_neuron_id;
select * from pattern_past;
select * from pattern_future;
select count(*) from pattern_peaks;
select count(*) from pattern_past;
select count(*) from pattern_future;

SELECT c.from_neuron_id, an.neuron_id, c.distance, 'event'
FROM inferred_neurons inf
-- creating event patterns, so the inferred neuron is an event
JOIN neurons n_inferred ON n_inferred.id = inf.neuron_id AND n_inferred.type = 'event'
-- the inference must have been done with a strength above a threshold to trigger pattern creation 
JOIN connections c ON c.to_neuron_id = inf.neuron_id AND inf.age = c.distance AND c.strength >= 0
JOIN active_neurons an_peak ON an_peak.neuron_id = c.from_neuron_id AND an_peak.age = inf.age
-- actions cannot learn patterns - only events can learn action/event patterns
JOIN neurons n_peak ON n_peak.id = c.from_neuron_id AND n_peak.type = 'event'
-- find the active neurons in same channel (what actually happened just now)
-- we are creating patterns for events that happened in the same channel 
JOIN active_neurons an ON an.age = 0 AND an.level = 0
JOIN neurons actual ON actual.id = an.neuron_id AND actual.type = 'event' AND actual.channel_id = n_peak.channel_id
-- the inference did not come true (prediction error)
WHERE inf.neuron_id NOT IN (SELECT neuron_id FROM active_neurons WHERE age = 0)
-- don't create a pattern if there is already an active pattern for this peak 
-- it means the peak recognized this situation - we'll refine it instead in that case
AND NOT EXISTS (
	SELECT 1 FROM active_neurons an_pattern
	JOIN pattern_peaks pp ON pp.pattern_neuron_id = an_pattern.neuron_id
	WHERE pp.peak_neuron_id = c.from_neuron_id
	AND an_pattern.age = c.distance
)
-- if the peak neuron does not have a context (connections TO it from older neurons at inference time)
-- then, we can't create a pattern for it because pattern_past would be empty - need more data - wait
AND EXISTS (
	SELECT 1 FROM active_connections context_ac
	WHERE context_ac.to_neuron_id = c.from_neuron_id
	AND context_ac.age = c.distance
);

-- 14 = out, 13 = own
select c.neuron_id, d.name, c.val 
from coordinates c join dimensions d on d.id = c.dimension_id 
-- where c.neuron_id <= 16
order by c.neuron_id;

SELECT src.neuron_id, src.level, src.strength
FROM inferred_neurons src
WHERE src.level = 0
AND src.age = 0;

-- these dimensions can be used for visual processing
-- INSERT INTO dimensions (name) VALUES ('x'), ('y'), ('r'), ('g'), ('b');

-- these dimensions are used for forecasting timeseries changes - slopes can go from -100% to 100% (rate of change)
-- INSERT INTO dimensions (name) VALUES ('gold_slope'), ('aem_slope');

-- test dimensions
-- INSERT IGNORE INTO dimensions (name) VALUES ('d0'), ('d1'), ('d2'), ('d3'), ('d4'), ('d5'), ('d6'), ('d7'), ('d8'), ('d9'), ('d10'), ('d11'), ('d12'), ('d13'), ('d14'), ('d15'), ('d16'), ('d17'), ('d18'), ('d19');

-- channels table for efficient storage (neurons reference by id instead of varchar)
-- DROP TABLE IF EXISTS channels;
CREATE TABLE IF NOT EXISTS channels (
    id SMALLINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) UNIQUE NOT NULL
) ENGINE=MEMORY;

-- dimensions table defines coordinate space (just names, no type/channel)
-- DROP TABLE IF EXISTS dimensions;
CREATE TABLE IF NOT EXISTS dimensions (
    id SMALLINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) UNIQUE NOT NULL
) ENGINE=MEMORY;

-- neurons table is the core representation of concepts - auto increment
-- level is an intrinsic property: base neurons are level 0, pattern neurons are level 1+
-- DROP TABLE IF EXISTS neurons;
CREATE TABLE IF NOT EXISTS neurons (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    level TINYINT UNSIGNED NOT NULL DEFAULT 0,
    INDEX idx_level (level)
) ENGINE=MEMORY;

-- base neuron attributes (level=0 neurons)
-- DROP TABLE IF EXISTS base_neurons;
CREATE TABLE base_neurons (
    neuron_id BIGINT PRIMARY KEY,
    channel_id SMALLINT UNSIGNED NOT NULL,
    type ENUM('event', 'action') NOT NULL,
    INDEX idx_channel_type (channel_id, type)
) ENGINE=MEMORY;

-- Base neuron coordinates (only level=0 neurons)
-- DROP TABLE IF EXISTS coordinates;
CREATE TABLE coordinates (
    neuron_id BIGINT UNSIGNED,
    dimension_id SMALLINT UNSIGNED,
    val FLOAT,
    PRIMARY KEY (neuron_id, dimension_id),
    INDEX (dimension_id, val)
) ENGINE=MEMORY;

-- connections between base-level neurons (level=0 to level=0)
-- DROP TABLE IF EXISTS connections;
CREATE TABLE IF NOT EXISTS connections (
    from_neuron_id BIGINT UNSIGNED,
    to_neuron_id BIGINT UNSIGNED,
    distance TINYINT UNSIGNED NOT NULL,
    strength FLOAT DEFAULT 1.0,
    reward FLOAT DEFAULT 0,  -- additive reward (0 = neutral, positive = good, negative = bad)
    PRIMARY KEY (from_neuron_id, to_neuron_id, distance),
    INDEX idx_from_distance_strength (from_neuron_id, distance, strength),
    INDEX idx_to_distance_strength (to_neuron_id, distance, strength)
) ENGINE=MEMORY;

-- neurons currently active within the context sliding window (all levels)
-- note that it is possible for the same neuron to be active in different ages
-- DROP TABLE IF EXISTS active_neurons;
CREATE TABLE IF NOT EXISTS active_neurons (
    neuron_id BIGINT UNSIGNED,
    age TINYINT UNSIGNED DEFAULT 0, -- how long the neuron has been active
    PRIMARY KEY (neuron_id, age),
    INDEX idx_level_age (age)
) ENGINE=MEMORY;

-- pattern peaks - maps each pattern neuron to its peak neuron (the decision node that owns the pattern)
-- patterns are learned by peak neurons to differentiate between sequences leading to them
-- DROP TABLE IF EXISTS pattern_peaks;
CREATE TABLE IF NOT EXISTS pattern_peaks (
    pattern_neuron_id BIGINT UNSIGNED NOT NULL,
    peak_neuron_id BIGINT UNSIGNED NOT NULL,
    strength FLOAT NOT NULL DEFAULT 1.0,
    PRIMARY KEY (pattern_neuron_id),
    INDEX idx_peak (peak_neuron_id)
) ENGINE=MEMORY;

-- pattern_past: pattern contexts for recognition/matching (cross-channel)
-- DROP TABLE IF EXISTS pattern_past;
CREATE TABLE IF NOT EXISTS pattern_past (
    pattern_neuron_id BIGINT UNSIGNED,
    context_neuron_id BIGINT UNSIGNED,
    context_age TINYINT UNSIGNED,
    strength FLOAT NOT NULL DEFAULT 1.0,
    PRIMARY KEY (pattern_neuron_id, context_neuron_id, context_age)
) ENGINE=MEMORY;

-- pattern_future: cross-level predictions from patterns to base neurons for inference/voting (cross-channel)
-- patterns directly infer base-level neurons (events or actions) at various temporal distances
-- this is a cross-level connection: pattern neuron (level > 0) → base neuron (level 0)
-- distance: temporal distance of prediction (1 = next frame, 2 = 2 frames ahead, etc.)
-- DROP TABLE IF EXISTS pattern_future;
CREATE TABLE IF NOT EXISTS pattern_future (
    pattern_neuron_id BIGINT UNSIGNED,
    inferred_neuron_id BIGINT UNSIGNED,
    distance TINYINT UNSIGNED,
    strength FLOAT NOT NULL DEFAULT 1.0,
    reward FLOAT NOT NULL DEFAULT 0, -- additive reward (0 = neutral, positive = good, negative = bad)
    PRIMARY KEY (pattern_neuron_id, inferred_neuron_id, distance),
    INDEX idx_pattern_distance_strength (pattern_neuron_id, distance, strength)
) ENGINE=MEMORY;

-- mapping table for matched patterns - peak neurons and their matched pattern neurons (MEMORY table)
-- this just a scratch table for faster processing - it temporarily holds the matched patterns for the current level in the frame
-- DROP TABLE IF EXISTS matched_patterns;
CREATE TABLE IF NOT EXISTS matched_patterns (
    peak_neuron_id BIGINT UNSIGNED,
    pattern_neuron_id BIGINT UNSIGNED,
    PRIMARY KEY (peak_neuron_id, pattern_neuron_id),
    INDEX idx_pattern (pattern_neuron_id)
) ENGINE=MEMORY;

-- scratch table for pattern connection analysis
-- status: 'common' = in both pattern and active (strengthen), 'novel' = in active only (add), 'missing' = in pattern only (weaken)
-- DROP TABLE IF EXISTS matched_pattern_context;
CREATE TABLE IF NOT EXISTS matched_pattern_context (
    pattern_neuron_id BIGINT UNSIGNED,
    context_neuron_id BIGINT UNSIGNED,
    context_age TINYINT UNSIGNED,
    status ENUM('common', 'novel', 'missing') NOT NULL,
    PRIMARY KEY (pattern_neuron_id, context_neuron_id, context_age),
    INDEX idx_pattern_status (pattern_neuron_id, status)
) ENGINE=MEMORY;

-- scratch table for new pattern creation 
-- temporal distance of the inference from the point of pattern activation is always 1 at pattern creation to avoid partial pattern contexts
-- DROP TABLE IF EXISTS new_pattern_future;
CREATE TABLE IF NOT EXISTS new_pattern_future (
    peak_neuron_id BIGINT UNSIGNED, -- the neuron that will be the peak of the new pattern (base event neuron for connection errors, pattern neuron for pattern errors)
    inferred_neuron_id BIGINT UNSIGNED, -- the base neuron that the pattern should infer
    distance TINYINT UNSIGNED,
    PRIMARY KEY (peak_neuron_id, inferred_neuron_id)
) ENGINE=MEMORY;

-- scratch table for mapping peak neurons to new pattern neurons - used during bulk pattern creation to track sequential IDs
-- type: pattern type ('event' or 'action') - one peak can have both types of patterns
-- patterns are always created at distance=1 (full context)
-- DROP TABLE IF EXISTS new_patterns;
CREATE TABLE IF NOT EXISTS new_patterns (
    seq_id INT AUTO_INCREMENT PRIMARY KEY,
    peak_neuron_id BIGINT UNSIGNED,
    pattern_neuron_id BIGINT UNSIGNED,
    UNIQUE INDEX idx_peak_type (peak_neuron_id)
) ENGINE=MEMORY;

-- inferred neurons from connection/pattern inference or exploration
-- used by both connection and pattern inference
-- after conflict resolution, is_winner is set for action neurons (NULL for events)
-- DROP TABLE IF EXISTS inferred_neurons;
CREATE TABLE IF NOT EXISTS inferred_neurons (
    neuron_id BIGINT UNSIGNED,
    strength FLOAT,
    is_winner TINYINT UNSIGNED, -- NULL for events, 1 for winning action votes, 0 for losing action votes
    PRIMARY KEY (neuron_id),
    INDEX idx_is_winner (is_winner)
) ENGINE=MEMORY;

-- increase the amount of records that can be stored in memory tables
SET GLOBAL max_heap_table_size = 1024 * 1024 * 1024 * 2;