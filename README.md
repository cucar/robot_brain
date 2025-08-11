# Machine Intelligence Engine

This document describes the current implemented design, aligned with the code in `brain.js` and the MySQL schema in `db/db.sql`. It also outlines the planned Prediction workflow and inter-level connections.

## Overview

- The system learns sparse, high-dimensional patterns from input frames by creating and adapting neurons with coordinates over named dimensions (e.g., `x`, `y`, `r`, `g`, `b`).
- Within each frame, neurons that co-activate at the same level form undirected connections whose strengths reflect observation counts adjusted by temporal age.
- Peaks are selected among newly activated neurons using a neighborhood-comparison heuristic. Each peak and its strongly connected neighbors define a cluster. A weighted centroid of cluster members becomes the next-level point for recursive processing.
- New neurons are created for points that have no close-enough match at the current level. Existing neurons adapt their coordinates toward matched observations.
- A periodic forgetting cycle decays connection strengths and prunes zero-strength edges.

## Key Files

- `brain.js`: Core learning and per-frame processing logic.
- `db/db.sql`: MySQL schema for dimensions, neurons, coordinates, undirected connections, inter-level patterns, and active window.
- `main.js`: Example bootstrapping and feeding sample frames.
- `channels/*`: Placeholders for modality-specific preprocessing. Not wired into the main loop yet.

## MySQL Schema (Implemented)

Tables reflect the current code and are different from the early design draft:

- `dimensions(id, name)`
- `neurons(id, creation_time)`
- `coordinates(neuron_id, dimension_id, val)`
- `connections(neuron1_id, neuron2_id, strength)`
  - Undirected by convention: `neuron1_id = LEAST(a,b)`, `neuron2_id = GREATEST(a,b)`
  - `strength` is the accumulated co-activation count (decayed in forgetting)
- `patterns(parent_id, child_id, strength)`
  - Stores inter-level (higher→lower) pattern links for prediction (planned usage)
- `active_neurons(neuron_id, level, age)` ENGINE=MEMORY

Note: Earlier draft tables such as `potential_peaks`, `suppressed_neurons`, or directed `connections(source_id, target_id, observation_count)` are not used in the implementation.

## Hyperparameters (brain.js)

- `neuronMaxAge`: Frames an active neuron remains in the sliding window (default 10)
- `decayRate`: Forget cycle period in frames (default 1000)
- `adaptationSpeed`: Coordinate learning rate toward observed points (default 0.1)
- `maxLevel`: Maximum recursive clustering depth per frame (default 6)
- `maxResolution`: Highest-level matching tolerance exponent (default 2)
- `peakMinStrength`: Minimum connection strength to be considered a peak (default 3)
- `peakMinRatio`: Neighborhood ratio threshold for peaks (default 1.5)

## Per-Frame Processing (Implemented)

Called via `await brain.observeFrame(frame)`, where `frame` is an array of points like `{ x: 0.10, y: 0.20, r: 1, g: 0, b: 0 }`.

1) Active Window Maintenance
- Increment `age` of all active neurons; remove ones older than `neuronMaxAge`.

2) Multi-Level Loop per Frame
- Initialize `level = 0`, `levelPoints = frame`, and `clusters = {}` (map from centroid JSON to ingredient neuron ids of the previous level).
- For each level until no pattern is found:
  - Matching and Creation
    - Compute resolution: `resolution = 10^(level - (maxResolution + maxLevel))`.
    - For each point in `levelPoints`, fetch candidate neurons within `±resolution` per dimension.
    - Pick closest among candidates by Euclidean distance restricted to observed dimensions; accept if `distance <= resolution`.
    - For points without acceptable matches, create new neurons with `coordinates` from the point. Creation is batched and deduplicated per level via a grid keyed by `resolution`.
  - Activation
    - Activate all matched or created neuron ids in `active_neurons` with `{ level, age: 0 }`.
  - Coordinate Adaptation
    - For each matched neuron, update `coordinates.val` toward the mean of observed values across the frame using `adaptationSpeed` with a single CASE-based `UPDATE` per frame.
  - Intra-Level Connection Reinforcement
    - Insert or update undirected `connections(neuron1_id, neuron2_id)` among all active neurons with `age = 0` at the same `level`.
    - Increment by `SUM(1/(1 + target.age))` to reflect co-activation proximity; direction is ignored.
  - Peak Detection and Cluster Centroids
    - Compute each active neuron's total incident strength from `connections` among active neurons at this level.
    - For a neuron to be a peak: `strength >= peakMinStrength` and `strength >= peakMinRatio * avg(neighbor strengths)`.
    - For each peak, form a cluster: the peak plus its directly connected active neighbors. Use undirected connection strengths as soft weights.
    - Compute a weighted centroid across dimensions using a streaming mean. This centroid is the next-level point.
  - Next-Level Seeds
    - If no clusters found, stop. Otherwise set `levelPoints = patterns.map(p => p.centroid)` and `clusters = { centroidJSON: neuron_ids }` for the next level.

3) Forgetting (Periodic)
- Every `decayRate` frames: decrement `connections.strength` by 1 and delete edges with `strength <= 0`.
- Planned: prune isolated neurons after removing edges; currently not implemented.

4) Predictions (Temporarily Stubbed)
- The system collects `predictedNeuronIds` during per-level processing but does not compute or return predictions yet. A log statement exists as a placeholder.

## Channels

The `channels/*` directory contains placeholder classes for `VisionChannel`, `AudioChannel`, `TextChannel`, and `SlopeChannel` that illustrate how raw inputs could be mapped to named dimensions. They are not integrated into the main processing path; `main.js` feeds normalized sample frames directly.

## Differences From Early Design Draft

- Edges are undirected (`connections(neuron1_id, neuron2_id)`), not directed `source→target`.
- No SQL-driven phased temporary tables (`potential_peaks`, `suppressed_neurons`, etc.). Peak detection and centroids are computed with ad-hoc queries and in-memory aggregation in `brain.js`.
- Pattern formation is local to the active set at a single level per frame; no global greedy suppression table is used.
- Interneurons are not typed separately; all neurons share the same `neurons` table and are used across levels. Higher-level “concepts” emerge via centroids and matching resolution, not by explicit neuron types.

## Predictions (Planned)

The following features are planned and partially scaffolded by the `patterns` table and TODOs in `brain.js`:

- Handle predictions end-to-end and return a map `{ neuronId: count }` sorted by descending count; log the predicted ids.
- Strengthen inter-level connections (patterns)
  - When a higher-level neuron is matched/created from a cluster of lower-level neurons, reinforce `patterns(parent=high, child=low)` for that cluster.
  - When a higher-level neuron activates, all its children should activate; normally children activate the parent, but prior learning allows partial evidence to trigger the parent which then fans out to children.
- Generate predictions from inter-level links
  - For activated higher-level neurons, fetch their historical children (from `patterns`). Children not already active in this frame become predictions. Return these as the predicted neuron ids.
- Forgetting for inter-level links
  - Decay `patterns.strength` periodically; before deleting entries whose `strength` decays to zero, capture involved neuron ids to optionally prune orphaned neurons.
- Matching continues to use centroid-to-neuron proximity per level; after matching at level N:
  - First, reinforce inter-level links from level N−1 cluster children to the level N parent neuron.
  - Then, collect other historically linked level N−1 children of that parent; the ones not active are predictions.
- Storage location for inter-level links
  - Use the dedicated `patterns(parent_id, child_id, strength)` table (preferable to overloading `connections`).

## Running the Example

1) Ensure MySQL is running and apply `db/db.sql`.
2) Configure DB access in `db/db.js` if needed.
3) Run: `node main.js`

The sample frames in `main.js` send a couple of simple, normalized RGB points. You should see logs for neuron creation, activation, connection reinforcement, detected patterns, and the (stub) predicted ids list.

## Next Steps

- Implement inter-level `patterns` reinforcement in `activateFrameNeurons` using the `clusters` map from the previous level.
- Implement prediction extraction and aggregation, returning `{ neuronId: count }` and logging sorted ids.
- Implement `patterns` decay and optional orphan-pruning in `runForgetCycle`.
- Integrate channels to produce frames for `observeFrame`.


