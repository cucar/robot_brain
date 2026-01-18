# Machine Intelligence Engine

This document describes the current implemented spatio-temporal predictive learning architecture, aligned with the code in `brain.js` and the MySQL schema in `db/db.sql`.

## Overview

The Machine Intelligence Engine implements a **Hierarchical Spatio-Temporal Pattern Learning System** with an integrated **Channel-Based Architecture** that:

- **Learns multi-dimensional patterns** from input frames by creating neurons with coordinates over named dimensions provided by channels (e.g., visual, audio, motor, financial data)
- **Forms directed temporal connections** between neurons based on their temporal distance (age difference), enabling sequence learning and prediction
- **Discovers spatial patterns** through peak detection in connection strength neighborhoods, forming higher-level pattern neurons
- **Generates predictions and actions** through a sophisticated inference system that predicts future states and executes outputs via channels
- **Adapts continuously** through a reward-based learning system where channels provide feedback that modifies neuron behavior
- **Executes autonomous exploration** when inactive, using curiosity-driven actions to discover new patterns
- **Forgets unused patterns** through periodic decay cycles to prevent overfitting and maintain relevance
- **Integrates multiple modalities** through a job system that coordinates channels representing different sensory and motor systems

## Key Files

- `brain.js`: Core learning engine with frame processing, pattern recognition, inference, and reward systems.
- `db/db.sql`: MySQL schema for neurons, connections, patterns, inference tables, and memory management.
- `run-brain.js`: Job runner entry point - executes specific brain episodes with different channel configurations.
- `jobs/*.js`: Episode definitions that configure channels and run specific learning scenarios (vision, audio, motor control, trading, etc.).
- `channels/*.js`: Fully integrated sensory/motor interfaces that provide input dimensions, execute outputs, and give feedback to the brain.

## MySQL Schema (Implemented)

The current implementation uses a unified schema optimized for spatio-temporal pattern learning:

### Core Tables
- **`dimensions(id, name, channel, type)`** - Defines input/output coordinate space with channel association and type (input/output)
- **`neurons(id)`** - Universal neuron storage (base neurons and pattern neurons use same table)
- **`coordinates(neuron_id, dimension_id, val)`** - Stores dimensional values for base neurons

### Connection Architecture
- **`connections(id, from_neuron_id, to_neuron_id, distance, strength)`**
  - **Cross-level directed connections**: Neurons at any level can connect to neurons at any other level
  - **`distance`**: Encodes both temporal and hierarchical relationships:
    - **Same level**: Time-dilated temporal distance `FLOOR(age / POW(contextLength, level))`
    - **Higher → Lower**: `contextLength - 1` (persistent context from slower timescale)
    - **Lower → Higher**: `0` (instantaneous from faster timescale perspective)
  - **`strength`**: Connection weight (reinforced by observations, decayed by forgetting)

### Pattern Learning
- **`patterns(pattern_neuron_id, connection_id, strength)`**
  - Links pattern neurons to the connection patterns they represent
  - Enables hierarchical decomposition and prediction generation

### Reward System
- **`neuron_rewards(neuron_id, reward_factor)`** - Stores reward factors for neurons based on performance
  - **`reward_factor`**: Multiplier for neuron strength (1.0=neutral, >1.0=boost, <1.0=reduce)

### Memory Tables (ENGINE=MEMORY)
- **`active_neurons(neuron_id, level, age)`** - Currently active neurons in sliding window
  - **`level`**: Hierarchical level (0=base, 1=patterns of base, 2=patterns of patterns, etc.)
  - **`age`**: Frames since activation (higher levels age slower)
- **`inferred_neurons(neuron_id, level, age)`** - Predicted/output neurons for next frame
  - Used for decision execution and temporal separation between prediction and action
- **`pattern_inference(level, pattern_neuron_id, connection_id, age)`** - Pattern-level predictions
  - Tracks predictions from higher-level patterns to lower-level connections
- **`connection_inference(level, connection_id, age)`** - Connection-level predictions
  - Tracks same-level connection predictions for reinforcement learning
- **`observed_patterns(peak_neuron_id, connection_id)`** - Temporary mapping of observed peak patterns
  - Used during pattern matching and creation phases
- **`active_connections(connection_id, from_neuron_id, to_neuron_id, level, strength)`** - Active connections cache
  - Contains only connections that are currently firing (distance matches temporal relationship)
  - Temporary table for fast hierarchical reward propagation and pattern inference

## Hyperparameters (brain.js)

- **`contextLength`**: Base frames a neuron stays active (default 10, higher levels age slower via POW formula)
- **`forgetCycles`**: Frames between forget cycles (default 1000)
- **`connectionForgetRate`**: Connection strength decay per forget cycle (default 0.1)
- **`patternForgetRate`**: Pattern strength decay per forget cycle (default 0.1)
- **`rewardForgetRate`**: Reward factor decay toward neutral per forget cycle (default 0.05)
- **`negativeLearningRate`**: Pattern/connection strength decrease for failed predictions (default 0.1)
- **`maxLevels`**: Maximum hierarchical depth to prevent infinite recursion (default 6)
- **`mergePatternThreshold`**: Minimum overlap for pattern matching (default 0.66)
- **`inactivityThreshold`**: Frames of inactivity before exploration (default 5)

## Frame Processing Workflow

The system processes input through `await brain.processFrame(frame, globalReward)` where `frame` is an array of coordinate objects from channels and `globalReward` is aggregated feedback.

### 1. Reward Application
- **Apply rewards**: Update `neuron_rewards` table for executed decisions (age ≥ 1) with temporal decay
- **Reward propagation**: Recent decisions get full reward, older decisions get proportionally less
- **Multiplicative updates**: Compound reward factors over time for persistent performance patterns

### 2. Active Window Management
- **Age neurons**: Increment age of all `active_neurons` and `inferred_neurons`
- **Sliding window**: Remove neurons older than `contextLength * POW(level + 1)` (higher levels age slower)
- **Memory cleanup**: Maintain temporal context within level-dependent time windows

### 3. Output Execution
- **Execute previous outputs**: Execute `inferred_neurons` with age=1 through their respective channels
- **Curiosity exploration**: If brain is inactive for `inactivityThreshold` frames, execute random exploration actions
- **Channel feedback**: Channels execute actions and update their internal state for next feedback cycle

### 4. Recognition Phase
- **Base neuron activation**: Find/create neurons for input coordinates and activate at level 0, age 0
- **Cross-level connection reinforcement**: Create connections from ALL active neurons (any level) to newly activated neurons
  - Distance encodes hierarchical relationship: same-level uses temporal distance, cross-level uses fixed distances
  - Higher-level neurons provide persistent context, lower-level neurons appear instantaneous
- **Hierarchical pattern discovery**: Recursively detect patterns using peak detection in connection neighborhoods
  - Patterns can now include connections across multiple levels (heterogeneous multi-scale patterns)
- **Pattern activation**: Activate matching pattern neurons at higher levels and create new patterns for novel signatures

### 5. Inference Phase (Reverse Level Order)
For each level from highest to lowest:

#### Connection Predictions
- **Cross-level inference**: Predict connections from any level to the target level using `connection_inference` table
  - Distance matching accounts for source neuron's level relative to target level
- **Pattern predictions**: Generate lower-level predictions from active patterns using `pattern_inference` table
  - Uses `active_connections` table to exclude already-firing connections

#### Strength Optimization
- **Calculate neuron strengths**: Compute weighted connection strength totals for predicted neurons
  - **Distance weighting**: Recent connections (distance=0) weighted at 1.0, distant connections (distance=9) weighted at 0.1
  - Linear interpolation: `weight = (contextLength - distance) / contextLength`
- **Apply reward factors**: Multiply base strengths by `neuron_rewards` factors to bias toward successful neurons
- **Peak detection**: Identify neurons stronger than the average strength across all candidates

#### Output Generation
- **Infer peak neurons**: Insert peak neurons into `inferred_neurons` table for next frame execution
- **Temporal separation**: Decisions made in current frame (age=0) are executed in next frame (age=1)

### 6. Forgetting Cycle (Periodic)
- **Connection decay**: Reduce all connection and pattern strengths by `forgetRate` every `forgetCycles` frames
- **Pruning**: Remove connections/patterns with strength ≤ 0
- **Neuron cleanup**: Delete orphaned neurons with no connections, patterns, or active state

## Channel Architecture

The `channels/*` directory contains fully integrated sensory and motor interfaces that serve as the brain's connection to different modalities:

### Available Channels
- **`VisionChannel`** - Visual processing with saccadic eye movements (input: visual_x/y/r/g/b, output: saccade_x/y)
- **`AudioChannel`** - Audio processing with ear positioning (input: audio frequencies, output: ear movements)
- **`TextChannel`** - Language processing with character generation (input: characters, output: next characters)
- **`ArmChannel`** - Motor control with proprioceptive feedback (input: joint positions/forces, output: muscle activations)
- **`TongueChannel`** - Taste processing with tongue movements (input: taste dimensions, output: tongue position)
- **`StockChannel`** - Financial trading with market data (input: price/volume changes, output: buy/sell/hold actions)

### Channel Integration
- **Dimension registration**: Channels automatically register their input/output dimensions with the brain during initialization
- **Frame processing**: Channels provide input data via `getFrameEvents()` and execute outputs via `executeOutputs()`
- **Feedback loop**: Channels provide reward signals via `getRewards()` based on action outcomes and state changes
- **Exploration**: Channels define valid exploration actions via `getExplorationAction()` for curiosity-driven learning
- **Job coordination**: The job system orchestrates multiple channels for complex multi-modal learning scenarios

## Cross-Level Connection Architecture

The system implements a biologically-inspired cross-level connection architecture where neurons at any hierarchical level can form connections with neurons at any other level. This enables multi-scale pattern recognition and contextual learning.

### Distance Encoding for Cross-Level Connections

Connections encode both temporal and hierarchical relationships through their distance value:

#### Same-Level Connections
- **Formula**: `FLOOR(age / POW(contextLength, level))`
- **Meaning**: Time-dilated temporal distance
- **Example**: At level 0, distance = exact age; at level 1, distance bucketed by 10s; at level 2, bucketed by 100s
- **Purpose**: Captures temporal sequences at appropriate timescales for each level

#### Higher → Lower Level Connections
- **Formula**: `contextLength - 1` (e.g., 9 if contextLength=10)
- **Meaning**: Persistent context from slower timescale
- **Rationale**: Higher-level neurons exist for longer periods, acting as stable context for lower-level processing
- **Example**: A Level 2 market trend neuron provides context for Level 0 price movements

#### Lower → Higher Level Connections
- **Formula**: `0` (zero distance)
- **Meaning**: Instantaneous co-occurrence from higher perspective
- **Rationale**: From a higher level's slow timescale, lower-level events appear to happen simultaneously
- **Example**: Multiple Level 0 neurons activating "at once" from Level 2's perspective

### Benefits of Cross-Level Connections

1. **Multi-Scale Context Integration**: Higher-level patterns provide context for lower-level processing
   - "This intraday pattern only happens during bull markets"
   - "Daily breakouts are more reliable with specific volume patterns"

2. **Skip Connections**: Direct information flow across levels enables faster learning
   - Similar to ResNet skip connections in deep learning
   - Shortcuts for critical long-term dependencies

3. **Heterogeneous Patterns**: Patterns can encode relationships across multiple levels
   - Level 2 pattern = Level 1 connections + Level 0 connections
   - Richer compositional representations

4. **Contextual Disambiguation**: Same low-level pattern means different things in different contexts
   - Pattern X at Level 0 has meaning Y when Level 2 context is Z
   - Enables context-aware predictions

### Distance Weighting in Peak Detection

To prioritize recent information, connection strengths are weighted by temporal proximity during peak detection:

- **Formula**: `weight = (contextLength - distance) / contextLength`
- **Examples** (contextLength=10):
  - distance=0 → weight=1.0 (100% - most recent)
  - distance=1 → weight=0.9 (90%)
  - distance=5 → weight=0.5 (50%)
  - distance=9 → weight=0.1 (10% - oldest)

This linear weighting ensures that more recent connections have greater influence on clustering and peak detection, while still allowing older connections to contribute.

## Core Architecture Systems

### Job System
The job system provides episode-based learning with configurable channel combinations:

- **Job base class**: `jobs/job.js` provides template for brain initialization, channel registration, and frame processing loops
- **Episode management**: Each job defines specific learning scenarios with different channel configurations
- **Brain lifecycle**: Jobs handle brain initialization, memory reset strategies, and cleanup
- **Multi-modal coordination**: Jobs can combine multiple channels for complex sensorimotor learning tasks

### Reward System
The brain implements a sophisticated reward-based learning mechanism:

- **Global reward aggregation**: Channels provide individual feedback that gets multiplied into a global reward factor
- **Temporal decay**: Recent decisions receive full reward impact, older decisions receive proportionally less
- **Neuron reward factors**: Persistent performance tracking via `neuron_rewards` table with multiplicative updates
- **Strength optimization**: Reward factors multiply base connection strengths to bias toward successful patterns
- **Compound learning**: Repeated success/failure compounds over time for stable behavioral adaptation

### Exploration System
Curiosity-driven exploration prevents the brain from getting stuck in local optima:

- **Inactivity detection**: Tracks frames since last meaningful activity across all channels
- **Random exploration**: When inactive beyond threshold, executes random valid actions from channels
- **Channel-aware exploration**: Each channel defines context-appropriate exploration actions
- **State-dependent validity**: Exploration respects channel constraints (e.g., can't sell stocks not owned)

### Inference System
The brain uses a sophisticated two-phase inference system with temporal separation:

- **Pattern inference**: Higher-level patterns predict lower-level connections via `pattern_inference` table
- **Connection inference**: Same-level connection predictions via `connection_inference` table
- **Temporal separation**: Decisions made in frame N (age=0) are executed in frame N+1 (age=1)
- **Peak detection**: Identifies neurons stronger than neighborhood average for output selection
- **Reward optimization**: Multiplies base strengths by reward factors before peak detection


## Usage Examples

### Basic Setup
```bash
# 1. Ensure MySQL is running and apply schema
mysql -u root -p < db/db.sql

# 2. Configure database connection in db/db.js if needed

# 3. Setup job data (downloads/prepares data for the job)
node run-setup.js <job-name>

# 4. Run a specific job/episode
node run-brain.js <job-name>
```

### Job Setup vs Job Execution

The system separates **data preparation** from **job execution**:

#### **Setup (Data Preparation)**
```bash
node run-setup.js <job-name>
```
- Downloads or prepares data required by the job
- Only needs to be run once (or when you want to refresh data)
- Optional - not all jobs require setup
- Example: Downloads historical stock data from Alpha Vantage

#### **Execution (Running the Brain)**
```bash
node run-brain.js <job-name>
```
- Runs the actual learning/training job
- Uses data prepared by setup (if applicable)
- Can be run multiple times with the same data

### Available Jobs

The system includes pre-built learning scenarios:

#### **Vision Learning**
```bash
node run-brain.js vision1
```
Learns visual patterns and saccadic eye movements from sample visual data.

#### **Text Processing**
```bash
node run-brain.js text1
```
Learns character sequences and generates text predictions.

#### **Motor Control**
```bash
node run-brain.js arm1
```
Learns arm movement patterns with proprioceptive feedback.

#### **Financial Trading (Stock Training)**
```bash
# First, download historical stock data
node run-setup.js stock-training

# Then run the training job
node run-brain.js stock-training
```
Trains on historical stock data for KGC (Kinross Gold), GLD (Gold ETF), and SPY (S&P 500).
- **Setup**: Downloads full historical daily data from Alpha Vantage API
- **Training**: Runs 50 episodes through historical data, learning patterns and trading strategies
- **Data location**: `data/stock/KGC.csv`, `data/stock/GLD.csv`, `data/stock/SPY.csv`

#### **Multi-Modal Learning**
```bash
node run-brain.js multisensory1
```
Integrates multiple channels (vision, audio, motor) for complex sensorimotor learning.

### Creating Custom Jobs

```javascript
import Job from './jobs/job.js';
import VisionChannel from './channels/vision.js';
import ArmChannel from './channels/arm.js';

export default class CustomJob extends Job {
    getChannels() {
        return [
            { name: 'vision', channelClass: VisionChannel },
            { name: 'arm', channelClass: ArmChannel }
        ];
    }

    async configureChannels() {
        // Custom channel configuration
        this.brain.channels.get('vision').setTargetObject({x: 0.5, y: 0.5});
    }
}
```

### Channel Output Format
Channels execute outputs and provide feedback automatically. The brain processes:
- **Event dimensions**: Sensory data from channel `getFrameEvents()`
- **Action dimensions**: Motor commands via channel `executeOutputs()`
- **Reward signals**: Performance feedback via channel `getRewards()`

## Performance Characteristics

- **Real-time processing**: Optimized for high-frequency input streams
- **Hierarchical scaling**: Higher-level patterns emerge automatically
- **Memory efficiency**: Sliding window prevents unbounded growth
- **Adaptive learning**: Continuous reinforcement through prediction accuracy
- **Bulk operations**: Database-optimized for concurrent pattern processing


