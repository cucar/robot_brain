-- Unified Brain Database Schema
-- Single connections table handles ALL relationships: spatial, temporal, hierarchical, predictions
-- Memory tables for active state, persistent tables for learned knowledge
CREATE DATABASE IF NOT EXISTS machine_intelligence;
USE machine_intelligence;

-- 1 = up, 2 = buy, 3 = down, 4 = sell
select c.neuron_id, d.name, c.val from coordinates c join dimensions d on d.id = c.dimension_id;
select * from dimensions;
select * from channels;
select * from neurons order by level desc;
select * from active_neurons order by age;
select * from base_neurons;
select * from connections;
select peak_neuron_id, pattern_neuron_id, strength from pattern_peaks order by peak_neuron_id, pattern_neuron_id;
select * from pattern_past where pattern_neuron_id = 17 order by pattern_neuron_id, context_age;
select * from pattern_future order by pattern_neuron_id, distance, inferred_neuron_id;
select * from new_patterns;
select * from new_pattern_future;
select * from matched_patterns;
select * from matched_pattern_past;
select * from inferred_neurons;

-- new patterns pattern_past entries
SELECT np.pattern_neuron_id, ctx.neuron_id, ctx.age - 1
FROM new_patterns np
JOIN neurons n_peak ON n_peak.id = np.peak_neuron_id
-- capture same-level active, but older neurons: pattern context is determined by same-level context only
JOIN active_neurons ctx ON ctx.age > 1
JOIN neurons ctx_n ON ctx_n.id = ctx.neuron_id AND ctx_n.level = n_peak.level
-- exclude actions from context in base level
-- note that this may be a stock specific thing - actions can learn patterns in robotics context
WHERE NOT EXISTS (SELECT 1 FROM base_neurons b WHERE b.neuron_id = ctx.neuron_id AND b.type = 'action')
order by ctx.age;

-- new observed pattern_future event connections
SELECT ap.neuron_id as pattern_neuron_id, an.neuron_id as inferred_neuron_id, ap.age
FROM active_neurons ap
JOIN neurons n_pattern ON n_pattern.id = ap.neuron_id AND n_pattern.level > 0
JOIN active_neurons an ON an.age = 0
JOIN base_neurons b ON b.neuron_id = an.neuron_id AND b.type = 'event'
WHERE ap.age > 0;

-- new level 1 pattern events
SELECT c.from_neuron_id, an.neuron_id as inferred_neuron_id, 1 as distance
FROM inferred_neurons inf
-- creating event patterns, so the inferred neuron is an event
JOIN base_neurons b_inf ON b_inf.neuron_id = inf.neuron_id AND b_inf.type = 'event'
-- the inference must have been done with a strength above a threshold to trigger pattern creation 
-- only peak neurons at age=1 learn the patterns, to ensure full context
-- at this point (frameNumber >= contextLength + 1), age=1 peaks always have full context
-- they will have additional distances as we observe them as added in when refining patterns
JOIN connections c ON c.to_neuron_id = inf.neuron_id AND c.distance = 1 AND c.strength >= 10
JOIN active_neurons an_peak ON an_peak.neuron_id = c.from_neuron_id AND an_peak.age = 1
-- actions cannot learn patterns - only events can learn action/event patterns - so, peak must be an event
-- note that this may be a stock specific thing - actions can learn patterns in robotics context    
JOIN base_neurons b_peak ON b_peak.neuron_id = c.from_neuron_id AND b_peak.type = 'event'
-- find the active neurons in same channel (what actually happened just now)
-- we are creating patterns for events that happened in ANY channel to be able to generalize 
JOIN active_neurons an ON an.age = 0
JOIN base_neurons b_actual ON b_actual.neuron_id = an.neuron_id AND b_actual.type = 'event' 
-- the inference did not come true (prediction error)
WHERE inf.neuron_id NOT IN (SELECT neuron_id FROM active_neurons WHERE age = 0)
-- don't create a pattern if there is already an active pattern for this peak 
-- it means the peak recognized this situation - we'll refine it instead in that case
AND NOT EXISTS (
	SELECT 1 FROM active_neurons an_pattern
	JOIN pattern_peaks pp ON pp.pattern_neuron_id = an_pattern.neuron_id
	WHERE pp.peak_neuron_id = c.from_neuron_id
	AND an_pattern.age = 1
);

-- new level 1 pattern actions
SELECT * /* c.from_neuron_id, MIN(b_alt.neuron_id) as inferred_neuron_id, 1 as distance */
FROM inferred_neurons inf
-- creating action patterns, so the inferred neuron is an action that was executed (winner)
JOIN base_neurons b_inf ON b_inf.neuron_id = inf.neuron_id AND b_inf.type = 'action'
-- the inference must have been done with a strength above a threshold to trigger pattern creation
-- only peak neurons at age=1 learn the patterns, to ensure full context
JOIN connections c ON c.to_neuron_id = inf.neuron_id AND c.distance = 1 AND c.strength >= 1
JOIN active_neurons an_peak ON an_peak.neuron_id = c.from_neuron_id AND an_peak.age = 1
-- actions cannot learn patterns - only events can learn action/event patterns - so, peak must be an event
-- note that this may be a stock specific thing - actions can learn patterns in robotics context    
JOIN base_neurons b_peak ON b_peak.neuron_id = c.from_neuron_id AND b_peak.type = 'event'
-- find alternative actions in the same channel (what we should try instead)
JOIN base_neurons b_alt ON b_alt.channel_id = b_inf.channel_id AND b_alt.type = 'action'
-- the action was executed (winner) and resulted in pain
WHERE inf.is_winner = 1
AND b_inf.channel_id IN (1)
-- AND b_alt.neuron_id != inf.neuron_id  -- don't suggest the same action that just failed
-- don't create a pattern if there is already an active pattern for this peak 
-- it means the peak recognized this situation - we'll refine it instead in that case
AND NOT EXISTS (
	SELECT 1 FROM active_neurons an_pattern
	JOIN pattern_peaks pp ON pp.pattern_neuron_id = an_pattern.neuron_id
	WHERE pp.peak_neuron_id = c.from_neuron_id
	AND an_pattern.age = 1
)
GROUP BY c.from_neuron_id, b_inf.channel_id;

-- max levels by age - age=1 level=0 - age=0,2,3,4 level=1
select age, max(level) from active_neurons a join neurons n on n.id = a.neuron_id group by age order by age;

-- previous age level
select max(level) from active_neurons a join neurons n on n.id = a.neuron_id where age = 1;

-- inference votes
-- 1 = up, 2 = buy, 3 = down, 4 = sell
SELECT v.dimension_id, v.dimension_name, v.neuron_id as inferred_neuron_id, v.from_neuron_id as inferring_neuron_id, v.distance, v.val, v.type, 
	v.channel_id, v.channel, v.strength, v.reward, v.source_level, v.source_type
FROM inference_votes v
WHERE NOT EXISTS (
	SELECT 1
	FROM inference_votes pv
	JOIN pattern_peaks pp ON pp.pattern_neuron_id = pv.from_neuron_id
	WHERE pv.source_type = 'pattern'
	AND pp.peak_neuron_id = v.from_neuron_id
	AND pv.dimension_id = v.dimension_id
)
and type = 'action'
order by dimension_id, inferred_neuron_id, inferring_neuron_id, distance;

select * from dimensions;

-- connection inference votes
SELECT c.from_neuron_id, c.to_neuron_id, coord.dimension_id, d.name, coord.val,
	   b.type, b.channel_id, ch.name, c.strength, c.reward, c.distance, 0, 'connection'
FROM active_neurons an
JOIN neurons n ON n.id = an.neuron_id AND n.level = 0
JOIN connections c ON c.from_neuron_id = an.neuron_id
JOIN coordinates coord ON coord.neuron_id = c.to_neuron_id
JOIN dimensions d ON d.id = coord.dimension_id
JOIN base_neurons b ON b.neuron_id = c.to_neuron_id
JOIN channels ch ON ch.id = b.channel_id
WHERE c.distance = an.age + 1 AND c.strength > 0
and c.from_neuron_id = 1
order by distance;
            
select * from inferred_neurons where age = 0;
select * from inferred_neurons where age = 1;
select * from active_neurons where age in (0,1) order by age, neuron_id;
select * from neurons where level > 0;

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

select * from pattern_past p where pattern_neuron_id = 5 order by context_age;
select * from pattern_future where pattern_neuron_id = 5 order by distance;
select pattern_neuron_id, peak_neuron_id, strength from pattern_peaks order by strength desc;
select * from pattern_past where pattern_neuron_id = 1554;
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
-- DROP TABLE IF EXISTS matched_pattern_past;
CREATE TABLE IF NOT EXISTS matched_pattern_past (
    pattern_neuron_id BIGINT UNSIGNED,
    context_neuron_id BIGINT UNSIGNED,
    context_age TINYINT UNSIGNED,
    status ENUM('common', 'novel', 'missing') NOT NULL,
    PRIMARY KEY (pattern_neuron_id, context_neuron_id, context_age),
    INDEX idx_pattern_status (pattern_neuron_id, status)
) ENGINE=MEMORY;

-- scratch table for new pattern creation
-- New patterns can have inferences at multiple distances (1 to contextLength-1)
-- to enable learning from exploration at all temporal horizons
-- DROP TABLE IF EXISTS new_pattern_future;
CREATE TABLE IF NOT EXISTS new_pattern_future (
    peak_neuron_id BIGINT UNSIGNED, -- the neuron that will be the peak of the new pattern (base event neuron for connection errors, pattern neuron for pattern errors)
    inferred_neuron_id BIGINT UNSIGNED, -- the base neuron that the pattern should infer
    distance TINYINT UNSIGNED,
    PRIMARY KEY (peak_neuron_id, inferred_neuron_id, distance)
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

-- scratch table for vote collection during inference
-- contains all votes (connection and pattern) with full neuron info for consensus determination
-- pattern votes override their peak's connection votes for dimensions the pattern covers
-- DROP TABLE IF EXISTS inference_votes;
CREATE TABLE IF NOT EXISTS inference_votes (
    from_neuron_id BIGINT UNSIGNED,     -- the voter (base neuron for connections, pattern neuron for patterns)
    neuron_id BIGINT UNSIGNED,          -- the target (always base neuron)
    dimension_id SMALLINT UNSIGNED,     -- dimension of the target neuron
    dimension_name VARCHAR(50),         -- dimension name for grouping
    val FLOAT,                          -- coordinate value in this dimension
    type ENUM('event', 'action'),       -- neuron type for winner selection logic
    channel_id SMALLINT UNSIGNED,       -- channel id for exploration
    channel VARCHAR(50),                -- channel name for debug output
    strength FLOAT,
    reward FLOAT,
    distance TINYINT UNSIGNED,
    source_level TINYINT UNSIGNED,      -- 0 for connections, pattern level for patterns
    source_type ENUM('connection', 'pattern'),
    INDEX idx_from_dim (from_neuron_id, dimension_id),
    INDEX idx_neuron (neuron_id)
) ENGINE=MEMORY;

-- increase the amount of records that can be stored in memory tables
SET GLOBAL max_heap_table_size = 1024 * 1024 * 1024 * 2;