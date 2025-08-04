# Machine Intelligence Design

The main function of the brain is to ensure the well-being of the organism and maximize chances of survival for the body and genes. When this involves a complex set of rules and conditions, we call that intelligence. The theory presented here asserts that there is a core intelligence algorithm, which can be implemented by machines or biological beings. This design outlines how to implement the intelligence algorithm in machines.

## Intelligence Algorithm and Brain

The brain has inputs from sensors. These sensors are implemented as senses in biological bodies. In machines they could be implemented as channels of input streams. Ultimately, these sensors take observations from the surroundings in various forms, and they convert them to consistent signals that are the main input lines of the brain, such that same input results in the same signal to the brain.

The brain learns to recognize patterns in these signals and stores the knowledge of the learned patterns in high-level neurons. This pattern recognition is a recursive process, where the brain learns patterns, patterns of patterns, patterns of patterns of patterns, and so on.

The activated neurons signifying patterns are associated with positive/negative reinforcements. In biological brains, this is accomplished by hormones and similar chemicals. The reward/punishment of the state gets activated from biological conditions. In machines this can be implemented by having an extra input that signifies desire for that state. If it’s good, it will be positive. If it’s bad, it will be negative. The brain will record these desires along with the neurons that are active at that time.

The brain also has output neurons. The output works the exact opposite way of the input neurons. The input neurons start with base neurons and build higher level patterns. Output neurons start with higher levels and trickle down to lower levels. Each lower-level output neuron activates a device in a particular way, which may control simple things like moving an arm or leg.

The goal of the brain is to (this is what it does in each frame):

- Observe and learn patterns
- Associate learned patterns with award and punishment
- Predict the next neurons to be activated from currently active patterns
- Activate output patterns that will maximize the chances of getting to award and avoiding punishment
- The output high-level patterns will dissolve into multiple output commands in the base neurons of the output layer

### Example: Vision

For example, for vision, the V1/V2 cortical areas in the brain would be the end of the line for these sensors (eyes). Sensors take an observation; they convert it to a well-known discrete signal and activate the same neuron in the brain’s base input area to convey what was observed. In machines, this could be implemented as multiple streams of inputs. Each of these streams would represent a dimension of observation. If we assume 100x100 screen for a basic RGB camera sending inputs as 5 streams: X, Y, R, G, B. Each frame would have 10k pixels activated, along with their colors in RGB, which would be 10k each, so we would have 40k neurons activated in each frame from this simple camera. RGB activations are a simple selection of a neuron for one value out of 255. XY activations are more complicated. They consider the pixel’s coordinate in the vision, but also the body’s position. As the body moves it automatically keeps track of a rough positioning of self. So, the X is our best estimation of the coordinates relative to the body.

### Neuron Connectedness Graph

It is a world model for an artificial brain that's represented as a weighted graph. Each neuron would correspond to a node. The distances between them would signify their connection strength, which would be initially some random number, doesn't matter what the initial value is. The brain updates the connection strengths based on observation counts. The clustering would be activating a higher neuron/node that represents currently active neurons, which are the nodes with weights. The difference is that the clustering node would be a new node that represents a particular pattern. Like picking objects from an image. The interneurons would correspond to multiple points from base dimension neurons. The distances would be used initially for spatial pooling, then temporal pooling using time distances as weights. The clustering neuron's coordinates would correspond to the centroid of all its ingredients. A neuron can resolve into one cluster in one case and another in another case. It depends on the context. Brain basically warps reality based on observations.

### Core Concepts: Coordinate System and Neuron Representation

The coordinate system is a form of **sparse, high-dimensional representation**.

**Base Dimensions:** For a 100x100 RGB image, we're not representing a pixel at (x,y) by a single neuron, but rather by a combination of neurons, one for each dimension. A neuron at (10, 0, 0, 0, 0) represents the x-coordinate 10. A neuron at (0, 50, 0, 0, 0) represents the y-coordinate 50. This is a form of **distributed representation** where each dimension has its own set of neurons.

Neurons represent points/coordinates in this space. The senses (channels of neurons) determine the original coordinate transformation mechanism. For visual input, each pixel may be represented in 5 dimensions: x, y, r, g, b. If we assume a 100x100 frame and 255 color values for each color ingredient, we will need 100 x base neurons, 100 y neurons, 255 r neurons, etc. So, each base neuron would represent a point on the dimension line, so all other dimension values would be 0, on a 5-dimensional input. In practice, the number of dimensions would be a lot more, like 1560 (like OpenAI). But the interneurons would have non-zero values on multiple dimensions, depending on its ingredients.

**Neurons:** Represented by id, type (base/interneuron), coordinates.

**Base neurons:** They have only values in one dimension. Other dimension values would be 0. Base neurons live on the axes of this high-dimensional space. Directly activated by sensory input.

**Interneurons:** An interneuron's coordinates would be the centroid of the coordinates of its active "ingredient" neurons (the base neurons it is strongly connected to and that are currently firing). For example, if an interneuron detects a diagonal line, its coordinates would be the average of the (x, y, r, g, b) coordinates of all the pixels that make up that line segment. This makes perfect sense for a **feature-based representation**. The interneuron's coordinate is a summary of the features it detects. Learned representations of patterns from lower-level co-activations.

**Connections:** Directed, weighted edges between neurons.

**Activity:** A neuron is active if it's currently relevant to the processing of the input stream. It maintains level (depth of activation within current frame's propagation) and age (frames since last re-activation in a sliding window). Neurons are "active" if their inputs from the current frame's processing push them over a threshold, and they remain active in a "sliding window" for \_t_cycles_.

**Levels:** Hierarchical processing; Level 0 is input, Level 1 processes Level 0, etc. A neuron's level in active_neurons refers to the processing depth _within the current frame_.

**Suppression:** A per-frame mechanism to prevent redundant peak detection, stored in a MEMORY table.

**Peakiness:** A measure of how strongly a neuron represents a unique, active pattern, calculated by comparing its influence to that of its active neighbors.

**Pattern Centroid:** Interneurons learn a geometric centroid in the feature space defined by their constituent ingredient neurons.

**Community/Cluster Detection (via Peakiness):** The algorithm identifies communities within active neurons at a given level. A "peak" neuron represents the core of a detected cluster, and its strongly connected active neighbors form the "ingredients" of the pattern. These detections are based on the common occurrence counts (connection strengths).

### Weights as Observation Counts

Weights are observation counts for that pair of neurons. This is a simple and effective learning rule, like a basic form of Hebbian learning: "neurons that fire together, wire together."

- **Simplicity and Parallelizability:** Using observation counts as weights is ideal for a parallelizable system. Each neuron can simply increment the count for its active connections. There's no complex gradient calculation or backpropagation.
- **Interpretation:** The weight isn't just a static value; it's a measure of the _statistical co-occurrence_ of two neurons' activations. A high weight means they have fired together many times.

### Dynamic Node Creation and Forgetting

New neurons are created when a new pattern is found, something that does not fit any of the known interneurons. There will be a forgetting cycle to prevent the curse of dimensionality. It will be executed regularly. It will simply reduce the observation counts of all connections. If it reaches zero, those neurons will no longer be connected. The same is true for learned pattern ingredients as well. Each interneuron has a set of connections to its ingredients/ancestors. Those will be decremented as well.

This is the most powerful and unique part of this model, addressing a major challenge in unsupervised learning: **the curse of dimensionality and managing network growth.**

**"When a new pattern is found...":** This is where our peak detection algorithm comes in. A "new pattern" is an activation of a cluster of neurons that is not already represented by an existing interneuron.

- **How do you detect this?** You can run the peak detection algorithm. If the detected "peak" and its cluster of strongly connected ingredients (the above normal connected nodes) do not have a strong connection to any existing interneuron, you can declare it a "new pattern."
- **The criterion for a "new pattern" could be:** The newly detected peak cluster is not "density-reachable" from any existing interneuron. Or, the sum of weights from the cluster ingredients to any existing interneuron is below a certain threshold.

**Forgetting Cycle:** This is a solution to the curse of dimensionality and combinatorial explosion.

- **Mechanism:** Periodically, you decrement all weights (observation counts). This is a form of **decay** or **pruning**.
- **Effect:** Weakly-formed connections or patterns that are no longer observed will naturally fade away. This keeps the network sparse and prevents it from becoming a monolithic, overfitted blob.
- **Self-pruning:** If a connection's count reaches zero, it's removed. This is a very clean and simple pruning rule.
- **Forgetting Learned Patterns:** Decrementing the ingredient connections of interneurons is also brilliant. It means a learned pattern can "forget" its ingredients if they are no longer consistently part of the pattern. This allows for adaptability and for neurons to be repurposed.

### Warping Reality (Jacobian Transform)

This is a beautiful and deep interpretation of the network's function.

**Jacobian Matrix:** In a neural network, the Jacobian matrix of a layer's output with respect to its input represents all the partial derivatives. It describes how the output of the network changes for small changes in the input. In essence, it's a **local linear approximation of the transformation**.

**Our Analogy:** In this model, the "interneuron's coordinates would correspond to the centroid of all its ingredients." When an input activates a set of base neurons, these neurons, through their connections, activate a higher-level interneuron. The creation of a new interneuron with a centroid as its coordinate is a **non-linear transformation**. The set of all such transformations across all active interneurons at a given moment can be thought of as a **warping of the input space**.

**Learning as Warping:** As you observe more data, the observation counts (weights) change, the clusters (and thus the ingredients of interneurons) change, and the centroids of the new or updated interneurons shift. This is exactly what a **Jacobian transformation** does: it describes how space is stretched, rotated, and shared. This model is learning a series of these transformations to map raw sensory input to a more abstract, meaningful, and warped representation of "reality."

The core idea is that this “warping” process is matching learned points/neurons/nodes to the observed patterns, and this initially takes a few steps/levels of recursive recognition, but in time as more observations pile up, it reaches the learned point faster, very much like humans do.

## Intelligence Algorithm

### Hyperparameters

Assume alpha, beta, min_conn_strength, match_threshold, activation_threshold, t_cycles, decay_rate are available as session variables or constants.

- **alpha:** Threshold multiplier for Peakiness Score. A threshold multiplier (e.g., 1.5, 2.0). This controls how much stronger a node's connections must be compared to its neighbors to be considered a potential peak.
- **beta:** Minimum Weighted Degree for a potential peak.A minimum absolute weighted degree threshold. A node must have at least this much total connection strength to be considered at all. This helps filter out isolated or weakly connected nodes.
- **min_conn_strength:** Minimum observation_count for a connection to be an "ingredient" of a pattern.
- **match_threshold:** Euclidean distance threshold for matching a new pattern to an existing interneuron.
- **activation_threshold:** Sum of weighted inputs required for a neuron to activate.
- **t_cycles:** Number of frames a neuron remains active in the active_neurons sliding window without re-activation.
- **decay_rate (forget cycle):** Amount to decrement observation_count in the forgetting cycle. The number of observation cycles we will go through before running a forget cycle to reduce the observed weights and prune connections.

### System State

- A set of nodes (neurons), V.
- A set of weighted, directed edges (connections), E, with weights w_{uv} (observation counts).
- Each node v has a coordinate c_v in a high-dimensional feature space.

### Weighted Graph Peak Detection

The key to parallelization is to ensure that computations for different nodes (or groups of nodes) can be performed independently without requiring constant synchronization.

This algorithm uses local Weighted Density & Neighborhood Comparison. It focuses on two main passes that can be highly parallelized.

### Main Loop (Executed on Observation/Input)

**Input Activation:** An input (e.g., an image frame) activates a subset of base neurons. For each activated base neuron, its activation is 1, and for others, 0.

**Activation Propagation:** The activation signal propagates through the network based on connection strengths (weights). You'll need a rule for how activation spreads and aggregates.

### Phase 1: Calculate Node "Peakiness" Score

This phase calculates a score for each node, indicating how "peak-like" it is based on its weighted connections compared to its neighbors. For each active neuron (in parallel):

**Calculate Weighted Degree (WDv​):** For each active neuron v, calculate its Weighted Degree (WD_v) and Avg Neighbor Weighted Degree (AvgNWD_v). Sum of weights of all edges connected to v. WDv​=∑u∈N(v)​wvu​ where N(v) is the set of neighbors of v.

**Calculate Average Neighbor Weighted Degree (AvgNWDv​):** Average of the weighted degrees of its _direct neighbors_. AvgNWDv​=∣N(v)∣1​∑u∈N(v)​WDu​ (if ∣N(v)∣>0, else 0)

**Calculate the Peakiness Score (P_v) of each active neuron:**

1. Initialize Pv​=0.
2. **Condition 1 (Absolute Strength):** If WDv​<β, then v is not strong enough to be a peak, so Pv​=0. Continue to next node.
3. **Condition 2 (Relative Strength):** If WDv​>α⋅AvgNWDv​: This means v's total connection strength is significantly higher than the average of its neighbors' total connection strengths. Set Pv​=WDv​−(α⋅AvgNWDv​) (or simply 1 if it meets the condition, to mark it as a candidate). A continuous score allows for ranking.
4. Else (v is not significantly stronger than its neighbors), Pv​=0.

**Parallelization Notes for Phase 1:**

Each node's WDv​ calculation is independent.

Each node's AvgNWDv​ calculation (using Option A) depends on the WD of its neighbors. This means WD values for _all_ nodes must be computed first before AvgNWDv​ can be computed. This can be done in two sub-steps within Phase 1:

1. **Parallel Compute all WDv​ values.** (Highly parallel: each node processes its own edges)
2. **Parallel Compute all AvgNWDv​ values using the pre-computed WDs.** (Highly parallel: each node looks up its neighbors' WDs)
3. **Parallel Compute Pv​ scores.** (Highly parallel: each node uses its own WDv​ and AvgNWDv​)

### Phase 2: Peak Selection and Clustering

This phase takes the "peakiness" scores and identifies the final peaks and their associated clusters.

First, sort all active neurons with P_v > 0 by score.

Then, iterate through the sorted list:

- Select the highest-scoring neuron p. This is a potential new pattern.
- **Check for Novelty:** Form a candidate cluster C_p around p (the above normal connected neighbors). Check if this cluster's ingredient neurons have a strong pre-existing connection to an existing interneuron.
    - If **NO**, this is a **NEW PATTERN!**
        - **CREATE A NEW INTERNEURON:** Create a new node n_new.
        - **CALCULATE COORDINATES:** Set its coordinate c_{n_new} to be the centroid (average) of the coordinates of all neurons in C_p.
        - **CREATE CONNECTIONS:** Create new edges from each neuron in C_p to n_new. Initialize weights to 1 (or a small value).
        - **SUPPRESS:** Mark p and its neighbors as "assigned" to prevent redundant new patterns.
    - If **YES**, the pattern is already known.
        - **UPDATE CONNECTIONS:** Increment the observation counts for the connections from C_p's ingredients to the existing interneuron.
        - **SUPPRESS:** Mark p and its neighbors as "assigned."

1. **Global Peak Identification:**

Collect all nodes v for which Pv​>0. These are your potential peaks.

Sort these potential peaks by their Pv​ scores in descending order.

Iterate through the sorted list:

- Select the node with the highest Pv​ as a confirmed peak.
- **Neighborhood Suppression:** Mark this peak node and its _direct neighbors_ as "assigned" or "processed." This prevents nearby nodes that are also somewhat strong from being chosen as independent peaks, maximizing differences between chosen peaks.
- Continue to the next highest Pv​ node that has _not_ been marked "assigned." Select it as a confirmed peak and mark it and its neighbors as "assigned."
- Repeat until all potential peaks have been considered or marked.

1. **Clustering "Above Normal Connected" Nodes:**

For each confirmed peak Pk​:

- Form a cluster Ck​ initialized with Pk​.
- Iterate through Pk​'s direct neighbors u∈N(Pk​):
    - **Condition:** If the connection strength wPk​u​ between the peak and neighbor u is "above normal." What is "above normal"?
        - **Option A (Simple Threshold):** $w_{P_k u} > \\text{min_connection_strength_threshold}$
        - **Option B (Relative to Peak's Connections):** $w_{P_k u} > \\gamma \\cdot \\text{Avg_Connection_Strength_of_Peak}$ where γ is another parameter.
        - **Option C (Relative to Local Neighborhood):** $w_{P_k u} > \\text{Avg_outgoing_weight_from_u_to_neighbors}$ (more complex, might require another pass). For simplicity, **Option A or B** are good starting points.
    - If the condition is met, add node u to cluster Ck​.
    - **Avoid Overlap:** A node should ideally belong to only one cluster. If a neighbor u could be added to multiple peak clusters, assign it to the peak with which it has the strongest connection (wPk​u​). This requires a global check for each neighbor. A simpler approach for parallelization might be to process clusters in parallel and then resolve conflicts in a final pass.

**Parallelization Notes for Phase 2:**

- **Global Peak Identification:** The sorting and iterative selection is inherently sequential (or at least hard to fully parallelize without complex locking mechanisms). However, if the number of potential peaks is much smaller than the total nodes, this step might not be the bottleneck.
- **Clustering:** Once peaks are identified, assigning neighbors to clusters can be highly parallelized:
    - Each confirmed peak can independently identify its potential cluster members based on the "above normal" connection criteria.
    - A final, potentially sequential, pass is needed to resolve nodes that could belong to multiple clusters (e.g., assign to the cluster of the strongest connecting peak).

### Forgetting Cycle

This is executed regularly after every x cycle. Decrement all weights (w_{uv}) by a small amount. Remove any edge where w_{uv} falls to zero. This prunes weak connections.

## Implementation: MySQL Schema

### Persistent Tables

#### neurons

- id: BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT
- creation_time: DATETIME

#### dimensions

- id: BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT
- name: VARCHAR(100)

#### neuron_coordinates

- neuron_id: BIGINT UNSIGNED, FOREIGN KEY to neurons.id
- dimension_id: BIGINT UNSIGNED, FOREIGN KEY to dimensions.id
- value: DOUBLE
- PRIMARY KEY (neuron_id, dimension_id)
- INDEX (dimension_id, value)

#### connections

- source_id: BIGINT UNSIGNED, FOREIGN KEY to neurons.id
- target_id: BIGINT UNSIGNED, FOREIGN KEY to neurons.id
- weight: DOUBLE
- PRIMARY KEY (source_id, target_id)

### Memory Tables

#### active_neurons

This table stores active neurons, also known as the context.

- neuron_id: BIGINT UNSIGNED
- level: INT (The level at which this neuron was activated _in the current frame's processing hierarchy_. Reset to 0 for initial input.)
- age: INT (Number of frames this neuron has been active since its last re-activation. Reset to 0 upon any activation.)

PRIMARY KEY is all 3 columns combined.

#### suppressed_neurons

- neuron_id: BIGINT UNSIGNED PRIMARY KEY

#### potential_peaks

- neuron_id: BIGINT UNSIGNED PRIMARY KEY
- peakiness_score: DOUBLE

#### observed_pattern_centroids

- pattern_id: BIGINT UNSIGNED
- dimension_id: TINYINT UNSIGNED
- value: DOUBLE

PRIMARY KEY (pattern_id, dimension_id)

#### observed_pattern_ingredients

- pattern_id: BIGINT UNSIGNED PRIMARY KEY
- ingredient_id: BIGINT UNSIGNED

## Dimensions/Channels Setup

Before the frame processing can be done, input channels and dimensions have to be set up. Something like this:

INSERT INTO dimensions (id, name) VALUES (1, 'x'), (2, 'y'), (3, 'r'), (4, 'g'), (5, 'b');

## Per-Frame Processing

This section outlines the processing of each frame by the brain, once the inputs are received from the sensors.

### Phase A: Frame Initialization & Active Window Management

This phase prepares the active_neurons table for the current frame and clears temporary processing tables.

#### Clear Frame-Specific Temporary Tables

TRUNCATE TABLE suppressed_neurons;

TRUNCATE TABLE potential_peaks;

TRUNCATE TABLE observed_pattern_centroids;

TRUNCATE TABLE observed_pattern_ingredients;

#### Age Existing Active Neurons

UPDATE active_neurons SET age = age + 1;

#### Deactivate Old Neurons

DELETE FROM active_neurons WHERE age > \_t_cycles_;

#### Activate Base Neurons

Based on the current sensory input (e.g., pixel data). This activates neurons at level = 0 and sets their age to 0. Example: Activate base neurons for a pixel (x=10, y=20, r=128). This INSERT block is repeated for each active input dimension/value combination.

SET X = SELECT id FROM dimensions WHERE name = 'x'

SET Y = SELECT id FROM dimensions WHERE name = 'y'

SET R = SELECT id FROM dimensions WHERE name = 'r'

INSERT INTO active_neurons (neuron_id, level, age)

SELECT n.id, 0 AS level, 0 AS age

FROM neurons n

JOIN neuron_coordinates nc_x ON n.id = nc_x.neuron_id AND nc_x.dimension_id = :X AND nc_x.value = 10

JOIN neuron_coordinates nc_y ON n.id = nc_y.neuron_id AND nc_y.dimension_id = :Y AND nc_y.value = 20

JOIN neuron_coordinates nc_r ON n.id = nc_r.neuron_id AND nc_r.dimension_id = :R AND nc_r.value = 128

#### Co-activation Reinforcement

This step handles the reinforcement of connections _between_ neurons that are active in the initial input layer (Level 0) and all other active neurons in the same level. This is crucial for forming spatial and temporal associations. Note that the temporal associations are inversely correlated to age. **NOTE: it may be a good idea to add reverse direction weights as well here.**

INSERT INTO connections (source_id, target_id, weight)

SELECT s.neuron_id as source_id, t.neuron_id as target_id, 1 / (1 + t.age) as weight

FROM active_neurons s

JOIN active_neurons t ON t.neuron_id != s.neuron_id AND t.level = 0 AND t.age != s.age -- exclude the same neurons on the same level and age

WHERE s.level = 0 AND s.age = 0 -- source is only the newly activated neurons

ON DUPLICATE KEY UPDATE weight = weight + VALUES(weight);

### Phase B: Recursive Propagation, Activation, & Connection Reinforcement

This loop drives the hierarchical processing for the current frame. observation_count for connections is updated _as_ propagation occurs.

SET @current_processing_level = 0;

SET @neurons_activated_in_last_propagation = (SELECT COUNT(\*) FROM active_neurons WHERE level = 0); -- Count from Phase A.

\-- Loop as long as neurons were activated in the previous propagation step.

WHILE @neurons_activated_in_last_propagation > 0 DO

\-- Step B.1: Identify Potential Next-Level Activations (Propagation Sums)

\-- Calculate the total weighted input for each neuron from current_processing_level.

\-- This is a temporary view/CTE for the next steps.

WITH propagation_sums AS (

SELECT

c.target_id AS potential_neuron_id,

SUM(c.observation_count) AS total_weighted_input

FROM

connections c

INNER JOIN

active_neurons an ON c.source_id = an.neuron_id AND an.level = @current_processing_level

GROUP BY

c.target_id

)

\-- Step B.2: Activate Next Level Neurons & Update active_neurons

\-- Neurons that receive enough input are activated at the next level, and their age is reset.

INSERT INTO active_neurons (neuron_id, level, age)

SELECT

ps.potential_neuron_id,

@current_processing_level + 1 AS level,

0 AS age -- Reset age upon activation

FROM

propagation_sums ps

WHERE

ps.total_weighted_input > \_activation_threshold_

ON DUPLICATE KEY UPDATE

level = VALUES(level), -- Neuron is now active at this new level for this frame.

age = 0; -- Reset age because it's re-activated.

SET @neurons_activated_in_last_propagation = ROW_COUNT(); -- Count newly inserted/updated rows.

\-- Step B.3: Reinforce Connections Based on Successful Propagation

\-- Increment observation counts for all connections that were part of this successful activation.

INSERT INTO connections (source_id, target_id, observation_count)

SELECT

c.source_id,

c.target_id,

1 -- Increment by 1 for this propagation event.

FROM

connections c

INNER JOIN

active_neurons src_an ON c.source_id = src_an.neuron_id AND src_an.level = @current_processing_level

INNER JOIN

active_neurons tgt_an ON c.target_id = tgt_an.neuron_id AND tgt_an.level = @current_processing_level + 1

ON DUPLICATE KEY UPDATE

observation_count = observation_count + VALUES(observation_count);

\-- Move to the next processing level

SET @current_processing_level = @current_processing_level + 1;

END WHILE; -- End of propagation loop

### Phase C: Peak Detection & Pattern Matching/Creation

This phase identifies distinct patterns from the _entire set_ of active_neurons from the current frame's processing, creating new interneurons if a pattern is novel. This operates on the active_neurons table which now contains all activated neurons across all levels for this frame.

SQL

\-- Phase C.1: Calculate Peakiness Scores for all active neurons from this frame's propagation.

TRUNCATE TABLE potential_peaks;

INSERT INTO potential_peaks (neuron_id, peakiness_score)

WITH active_neuron_wd AS (

SELECT

an.neuron_id,

SUM(c.observation_count) AS weighted_degree

FROM

active_neurons an

INNER JOIN

connections c ON an.neuron_id = c.source_id

GROUP BY

an.neuron_id

),

active_neighbor_avg_wd AS (

SELECT

an.neuron_id,

AVG(anwd.weighted_degree) AS avg_neighbor_wd

FROM

active_neurons an

INNER JOIN

connections c ON an.neuron_id = c.source_id

INNER JOIN

active_neuron_wd anwd ON c.target_id = anwd.neuron_id

GROUP BY

an.neuron_id

)

SELECT

wd.neuron_id,

CASE

WHEN wd.weighted_degree < \_beta_ THEN 0

WHEN wd.weighted_degree > \_alpha_ \* COALESCE(n_avg.avg_neighbor_wd, 0) THEN wd.weighted_degree - (\_alpha_\* COALESCE(n_avg.avg_neighbor_wd, 0))

ELSE 0

END AS peakiness_score

FROM

active_neuron_wd wd

LEFT JOIN -- Use LEFT JOIN to include active neurons with no active neighbors for AvgNWD calc.

active_neighbor_avg_wd n_avg ON wd.neuron_id = n_avg.neuron_id;

\-- Phase C.2: Sequential Peak Selection & Pattern Centroid Calculation (and Suppression)

\-- This uses a cursor to ensure greedy, distinct peak selection.

BEGIN

DECLARE peak_cursor CURSOR FOR

SELECT pp.neuron_id

FROM potential_peaks pp

LEFT JOIN suppressed_neurons sn ON pp.neuron_id = sn.neuron_id

WHERE pp.peakiness_score > 0

AND sn.neuron_id IS NULL -- Exclude already suppressed neurons.

ORDER BY pp.peakiness_score DESC;

DECLARE finished INT DEFAULT 0;

DECLARE current_peak_id BIGINT UNSIGNED;

DECLARE @pattern_id_counter BIGINT UNSIGNED DEFAULT (SELECT COALESCE(MAX(pattern_id), 0) FROM observed_pattern_centroids); -- Ensure unique IDs.

DECLARE CONTINUE HANDLER FOR NOT FOUND SET finished = 1;

OPEN peak_cursor;

peak_loop: LOOP

FETCH peak_cursor INTO current_peak_id;

IF finished THEN

LEAVE peak_loop;

END IF;

\-- Ensure this peak hasn't been suppressed by a prior selection in this loop iteration.

IF EXISTS (SELECT 1 FROM suppressed_neurons WHERE neuron_id = current_peak_id) THEN

ITERATE peak_loop;

END IF;

SET @pattern_id_counter = @pattern_id_counter + 1;

\-- C.2.a: Find Ingredient Neurons for the current peak.

INSERT INTO observed_pattern_ingredients (pattern_id, ingredient_id)

SELECT

@pattern_id_counter,

c.target_id

FROM

connections c

INNER JOIN

active_neurons an ON c.target_id = an.neuron_id -- Ensure ingredient is active in current frame.

WHERE

c.source_id = current_peak_id

AND c.observation_count > \_min_conn_strength_;

\-- C.2.b: Calculate the Centroid of these ingredients.

INSERT INTO observed_pattern_centroids (pattern_id, dimension_id, value)

SELECT

@pattern_id_counter,

nc.dimension_id,

AVG(nc.value) AS centroid_value

FROM

observed_pattern_ingredients opi

INNER JOIN

neuron_coordinates nc ON opi.ingredient_id = nc.neuron_id

WHERE

opi.pattern_id = @pattern_id_counter

GROUP BY

opi.pattern_id, nc.dimension_id;

\-- C.2.c: Suppress the selected peak and its direct, strong neighbors for this frame.

INSERT IGNORE INTO suppressed_neurons (neuron_id)

SELECT current_peak_id AS neuron_id

UNION

SELECT c.target_id AS neuron_id

FROM connections c

WHERE c.source_id = current_peak_id

AND c.observation_count > \_min_conn_strength_; -- Only suppress strongly connected neighbors.

END LOOP peak_loop;

CLOSE peak_cursor;

END;

\-- Phase C.3: Match Observed Patterns to Existing Interneurons or Create New Ones.

\-- This part identifies patterns. Reinforcement happens in Phase B based on successful propagation.

\-- Determine the best match for each observed pattern centroid.

CREATE TEMPORARY TABLE pattern_matches ENGINE=MEMORY AS

WITH pattern_distances AS (

SELECT

opc.pattern_id,

n.id AS interneuron_id,

SQRT(SUM(POW(nc_inter.value - opc.value, 2))) AS distance

FROM

observed_pattern_centroids opc

INNER JOIN

neuron_coordinates nc_inter ON opc.dimension_id = nc_inter.dimension_id

INNER JOIN

neurons n ON nc_inter.neuron_id = n.id

WHERE

n.type = 2 -- Only compare against existing interneurons.

GROUP BY

opc.pattern_id, n.id

),

ranked_matches AS (

SELECT

pd.pattern_id,

pd.interneuron_id,

pd.distance,

RANK() OVER (PARTITION BY pd.pattern_id ORDER BY pd.distance ASC) as rank_num

FROM

pattern_distances pd

)

SELECT

rm.pattern_id,

rm.interneuron_id,

rm.distance,

CASE WHEN rm.distance < \_match_threshold_ THEN 1 ELSE 0 END AS is_match -- 1 if matched, 0 if new.

FROM

ranked_matches rm

WHERE

rm.rank_num = 1; -- Get the closest interneuron for each centroid.

\-- C.3.a: Handle Matched Patterns (No explicit reinforcement here; that happens in Phase B)

\-- Matched patterns simply signify recognition. Their connections were already reinforced.

\-- We might add some other "recognition" signal or log here if needed.

\-- For the core learning, no further DB update is required beyond what Phase B did.

\-- C.3.b: Create New Interneurons for Novel Patterns.

\-- Get a new ID for the new neuron (e.g., from a sequence generator or MAX(ID)+1).

\-- This might need to be done carefully in batches, potentially via a stored procedure.

SET @next_neuron_id = (SELECT COALESCE(MAX(id), 0) + 1 FROM neurons); -- This needs careful concurrency handling in real systems.

INSERT INTO neurons (id, type, creation_time)

SELECT

@next_neuron_id + (ROW_NUMBER() OVER (ORDER BY pm.pattern_id)) - 1, -- Assign unique new IDs

2 AS type, -- Interneuron

NOW() AS creation_time

FROM

pattern_matches pm

WHERE

pm.is_match = 0; -- Only for patterns that did not match existing interneurons.

\-- Insert coordinates for the new neurons.

INSERT INTO neuron_coordinates (neuron_id, dimension_id, value)

SELECT

@next_neuron_id + (ROW_NUMBER() OVER (PARTITION BY opc.pattern_id ORDER BY opc.dimension_id)) - 1,

opc.dimension_id,

opc.value

FROM

pattern_matches pm

INNER JOIN

observed_pattern_centroids opc ON pm.pattern_id = opc.pattern_id

WHERE

pm.is_match = 0;

\-- Create connections from ingredient neurons to the new interneurons.

INSERT INTO connections (source_id, target_id, observation_count)

SELECT

opi.ingredient_id,

(@next_neuron_id + (ROW_NUMBER() OVER (PARTITION BY opi.pattern_id ORDER BY opi.ingredient_id)) - 1) AS new_neuron_id,

1 AS observation_count -- Initial observation count is 1.

FROM

pattern_matches pm

INNER JOIN

observed_pattern_ingredients opi ON pm.pattern_id = opi.pattern_id

WHERE

pm.is_match = 0;

\-- Add newly created interneurons to the active_neurons sliding window.

\-- This ensures they are considered for future propagation and aging.

INSERT INTO active_neurons (neuron_id, level, age)

SELECT

id,

@current_processing_level + 1 AS level, -- Or whatever level makes sense for their creation.

0 AS age -- Start with age 0.

FROM

neurons

WHERE

id >= @next_neuron_id -- Assume new IDs are sequential from @next_neuron_id.

AND type = 2; -- Ensure they are the newly created interneurons.

### Phase D: Forgetting Cycle (Background Process)

This remains a separate, periodic background process.

\-- Decrement all observation counts by a set decay rate (e.g., 0.001)

\-- and prevent them from going below zero.

UPDATE connections

SET observation_count = GREATEST(0, observation_count - \_decay_rate_);

\-- Remove connections that have decayed to zero.

DELETE FROM connections

WHERE observation_count <= 0;

\-- Optional: Prune neurons that have no active connections (become isolated)

\-- This would be a more complex pruning rule, beyond just connection decay.

### Phase E: Prediction and Inference

This phase operates after the entire hierarchy has been activated and new patterns potentially learned for the current frame. It leverages the activated high-level interneurons to infer missing parts of the input or predict likely future components.

1. **Identify the "Predictor" Neurons:** These are the neurons in active_neurons that reached the highest level of activation during the current frame's processing. They represent the system's most abstract interpretation of the current, potentially partial, input.

SQL

CREATE TEMPORARY TABLE highest_level_active_neurons ENGINE=MEMORY AS

SELECT

neuron_id

FROM

active_neurons

WHERE

level = (SELECT MAX(level) FROM active_neurons);

1. **Retrieve the "Complete Patterns" Associated with Predictor Neurons:** When an interneuron was originally created (in Phase C.3.b), its definition included its centroid (neuron_coordinates) and the observed_pattern_ingredients that formed it. To predict, we look up these ingredients. This allows us to "dream up" the full input that _should_ have activated this high-level neuron.

CREATE TEMPORARY TABLE predicted_ingredient_neuron_ids ENGINE=MEMORY AS

SELECT DISTINCT opi.ingredient_id

FROM neurons n

INNER JOIN connections c ON n.id = c.target_id -- Connection to the interneuron

INNER JOIN highest_level_active_neurons hl_an ON c.source_id = hl_an.neuron_id -- From the high-level active neuron

INNER JOIN observed_pattern_ingredients opi ON opi.pattern_id = (

\-- Find the pattern_id that created this interneuron.

\-- This implies storing pattern_id during interneuron creation or linking it differently.

\-- Let's assume interneurons have a creation_pattern_id foreign key for simplicity, or we reconstruct the pattern from its strong connections.

\-- A more robust approach might be to query the original pattern ingredients used to create THIS interneuron ID.

\-- For now, let's just get the \*direct\* ingredients (via connections) of the highest active neurons.

\-- This assumes the interneuron itself captures the 'essence' rather than a specific pattern_id.

SELECT opi_inner.pattern_id

FROM observed_pattern_ingredients opi_inner

WHERE opi_inner.ingredient_id = c.target_id -- This is tricky. Let's simplify.

LIMIT 1 -- Assuming unique relationship between interneuron and its creation pattern

)

WHERE n.type = 2; -- Ensure it's an interneuron that was activated.

**Correction/Refinement on predicted_ingredient_neuron_ids:** The most direct way to predict the "completion" is to identify all neurons that are strongly connected as _ingredients_ to the highest-level active neurons. We can retrieve the coordinates of these "ingredient" neurons.

SQL

CREATE TEMPORARY TABLE predicted_pattern_coordinates ENGINE=MEMORY AS

SELECT

nc.dimension_id,

AVG(nc.value) AS predicted_value -- Average coordinates across all contributing ingredients.

FROM

highest_level_active_neurons hl_an

INNER JOIN

connections c ON hl_an.neuron_id = c.source_id -- Connections from high-level neuron to its ingredients

INNER JOIN

neuron_coordinates nc ON c.target_id = nc.neuron_id -- Get coordinates of these ingredient neurons

WHERE

c.observation_count > \_min_conn_strength_ -- Only consider strong ingredient connections.

GROUP BY

nc.dimension_id;

**Explanation:** This query reconstructs the "expected" coordinates of the input space by looking at all the (strong) ingredients of the highest-level active neurons. If multiple high-level neurons are active, their predicted components are averaged, allowing for a combined prediction.

1. **Output the Prediction:** The predicted_pattern_coordinates table now holds the system's prediction for the complete pattern based on its current observation and learned knowledge. This could be used for:
    - **Filling in missing data:** If the input was sparse (e.g., a few pixels of an image), these coordinates would represent the system's "guess" for the full image.
    - **Anomaly detection:** If the predicted pattern deviates significantly from the actual observed (partial) pattern, it might indicate an anomaly.
    - **Guided search:** In a larger system, these predicted coordinates could guide sensory input to confirm the prediction.