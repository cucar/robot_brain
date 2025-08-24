# Machine Intelligence Engine

This document describes the current implemented spatio-temporal predictive learning architecture, aligned with the code in `brain.js` and the MySQL schema in `db/db.sql`.

## Overview

The Machine Intelligence Engine implements a **Hierarchical Spatio-Temporal Pattern Learning System** that:

- **Learns multi-dimensional patterns** from input frames by creating neurons with coordinates over named dimensions (e.g., `x`, `y`, `r`, `g`, `b`, or any custom dimensions)
- **Forms directed temporal connections** between neurons based on their temporal distance (age difference), enabling sequence learning and prediction
- **Discovers spatial patterns** through peak detection in connection strength neighborhoods, forming higher-level pattern neurons
- **Generates predictions** by decomposing learned patterns back to base neurons and predicting future connections
- **Adapts continuously** through reinforcement learning with positive/negative feedback for accurate/inaccurate predictions
- **Forgets unused patterns** through periodic decay cycles to prevent overfitting and maintain relevance

## Key Files

- `brain.js`: Core learning and per-frame processing logic.
- `db/db.sql`: MySQL schema for dimensions, neurons, coordinates, undirected connections, inter-level patterns, and active window.
- `main.js`: Example bootstrapping and feeding sample frames.
- `channels/*`: Placeholders for modality-specific preprocessing. Not wired into the main loop yet.

## MySQL Schema (Implemented)

The current implementation uses a unified schema optimized for spatio-temporal pattern learning:

### Core Tables
- **`dimensions(id, name)`** - Defines input/output coordinate space (e.g., 'x', 'y', 'r', 'g', 'b')
- **`neurons(id)`** - Universal neuron storage (base neurons and pattern neurons use same table)
- **`coordinates(neuron_id, dimension_id, val)`** - Stores dimensional values for base neurons

### Connection Architecture
- **`connections(id, from_neuron_id, to_neuron_id, distance, strength)`** 
  - **Directed temporal connections**: `from_neuron_id` → `to_neuron_id`
  - **`distance`**: Temporal separation (0=spatial co-occurrence, 1=immediate sequence, 2=next step, etc.)
  - **`strength`**: Connection weight (reinforced by observations, decayed by forgetting)
  
### Pattern Learning
- **`patterns(pattern_neuron_id, connection_id, strength)`**
  - Links pattern neurons to the connection patterns they represent
  - Enables hierarchical decomposition and prediction generation

### Memory Tables (ENGINE=MEMORY)
- **`active_neurons(neuron_id, level, age)`** - Currently active neurons in sliding window
  - **`level`**: Hierarchical level (0=base, 1=patterns of base, 2=patterns of patterns, etc.)
  - **`age`**: Frames since activation (higher levels age slower)
- **`predicted_connections(pattern_neuron_id, connection_id, level, age, prediction_strength)`**
  - Active predictions waiting for validation
  - Enables reinforcement learning through prediction accuracy

## Hyperparameters (brain.js)

- **`baseNeuronMaxAge`**: Base frames a neuron stays active (default 10, higher levels age slower)
- **`forgetCycles`**: Frames between forget cycles (default 1000)
- **`forgetRate`**: Connection strength decay per forget cycle (default 0.1)
- **`positiveLearningRate`**: Pattern strength increase for correct predictions (default 0.1)
- **`negativeLearningRate`**: Pattern strength decrease for failed predictions (default 0.1)
- **`maxLevels`**: Maximum hierarchical depth to prevent infinite recursion (default 6)

## Frame Processing Workflow

The system processes input through `await brain.processFrame(frame)` where `frame` is an array of coordinate objects.

### 1. Active Window Management
- **Age neurons**: Increment age of all active neurons and predictions
- **Sliding window**: Remove neurons/predictions older than `baseNeuronMaxAge * (level + 1)`
- **Prediction validation**: Penalize predictions that didn't occur (negative learning)

### 2. Base Neuron Activation
- **Exact matching**: Find existing neurons with identical coordinate values across all dimensions
- **Neuron creation**: Bulk create new neurons for unmatched input points (with deduplication)
- **Activation**: Add matched/created neurons to `active_neurons` at level 0, age 0
- **Connection reinforcement**: Create directed temporal connections from older neurons (age > 0) to new neurons (age = 0)

### 3. Hierarchical Pattern Discovery
For each level (0 to `maxLevels`):

#### Connection Analysis
- **Get active connections**: Retrieve directed connections flowing into newly activated neurons
- **Prediction validation**: Reward patterns that correctly predicted these connections (positive learning)
- **Strength calculation**: Compute total connection strength for each neuron (bidirectional)

#### Peak Detection & Pattern Formation
- **Neighborhood mapping**: Build bidirectional connectivity graphs
- **Peak identification**: Find neurons whose strength exceeds their neighborhood average
- **Pattern matching**: Match peak connection signatures to existing pattern neurons
- **Pattern creation**: Create new pattern neurons for novel connection patterns
- **Pattern reinforcement**: Strengthen pattern-connection associations for observed patterns

#### Higher-Level Activation
- **Pattern activation**: Activate matching/created pattern neurons at level+1
- **Hierarchical connections**: Form connections between pattern neurons at higher level
- **Prediction generation**: Generate predictions for inactive connections of active patterns

### 4. Prediction System
- **Pattern decomposition**: Recursively decompose pattern neurons to base neurons using CTE queries
- **Temporal organization**: Organize predictions by temporal distance for sequence forecasting
- **Confidence scoring**: Weight predictions by pattern strength and decomposition path
- **Return format**: Structured predictions with coordinates, confidence, and temporal distance

### 5. Forgetting Cycle (Periodic)
- **Connection decay**: Reduce all connection and pattern strengths by `forgetRate`
- **Pruning**: Remove connections/patterns with strength ≤ 0
- **Neuron cleanup**: Delete orphaned neurons with no connections, patterns, or active state

## Channels

The `channels/*` directory contains placeholder classes for `VisionChannel`, `AudioChannel`, `TextChannel`, and `SlopeChannel` that illustrate how raw inputs could be mapped to named dimensions. They are not integrated into the main processing path; `main.js` feeds normalized sample frames directly.

## Key Architecture Changes

The implementation evolved significantly from the original design:

### Connection Model
- **Directed temporal connections** instead of undirected co-occurrence
- **Distance-based encoding** for temporal relationships (0=spatial, 1+=temporal sequence)
- **Unified connection table** handles both spatial and temporal relationships

### Pattern Learning
- **Connection-based patterns** instead of coordinate-based centroids
- **Peak detection** in connection strength neighborhoods
- **Pattern neurons** linked to connection signatures rather than coordinate clusters
- **Hierarchical decomposition** through recursive pattern-connection relationships

### Prediction System
- **Fully implemented end-to-end predictions** with confidence scoring
- **Temporal sequence forecasting** organized by temporal distance
- **Reinforcement learning** through prediction accuracy feedback
- **Pattern decomposition** using recursive SQL queries to find base neurons

### Memory Architecture
- **Sliding window** with level-dependent aging (higher levels age slower)
- **Active predictions table** for tracking and validating forecasts
- **Bulk operations** optimized for high-performance real-time processing

## Usage Examples

### Basic Setup
```bash
# 1. Ensure MySQL is running and apply schema
mysql -u root -p < db/db.sql

# 2. Configure database connection in db/db.js if needed

# 3. Run the basic example
node main.js
```

### Example Applications

The system supports multiple learning scenarios:

#### 1. **Sequence Learning** (Text Processing)
```javascript
// Learn character sequences like "cats"
const letterCoords = {
    'c': {d0: 0.1}, 'a': {d0: 0.2}, 
    't': {d0: 0.3}, 's': {d0: 0.4}
};

// Train on repeated sequences
for (const letter of "cats") {
    const predictions = await brain.processFrame([letterCoords[letter]]);
    // System learns temporal patterns and predicts next letters
}
```

#### 2. **Multi-dimensional Pattern Recognition** (Market Analysis)
```javascript
// Learn market patterns with multiple indicators
const marketFrame = [{
    d0: 0.05,  // price change
    d1: 0.03,  // volume change  
    d2: 0.02   // volatility
}];

const predictions = await brain.processFrame(marketFrame);
// Returns predictions organized by temporal distance
```

#### 3. **Motor Control Sequences**
```javascript
// Learn movement patterns
const reachSequence = [
    { motor0: 0.0, motor1: 0.0, motor2: 0.0 }, // rest
    { motor0: 0.3, motor1: 0.0, motor2: 0.0 }, // extend
    { motor0: 0.3, motor1: 0.4, motor2: 0.0 }, // flex
    { motor0: 0.3, motor1: 0.4, motor2: 0.2 }  // grasp
];

// System learns the sequence and predicts next movements
```

### Prediction Output Format
```javascript
{
    distance: 1,           // temporal distance (frames ahead)
    strength: 0.85,        // average confidence
    predictions: [{
        coordinates: { x: 0.1, y: 0.2 },
        confidence: 0.9    // prediction confidence
    }]
}
```

## Performance Characteristics

- **Real-time processing**: Optimized for high-frequency input streams
- **Hierarchical scaling**: Higher-level patterns emerge automatically
- **Memory efficiency**: Sliding window prevents unbounded growth
- **Adaptive learning**: Continuous reinforcement through prediction accuracy
- **Bulk operations**: Database-optimized for concurrent pattern processing


