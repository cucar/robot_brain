# Machine Intelligence Engine

A prediction-driven learning system that learns cause and effect through error correction.

## Overview

The brain is fundamentally a **prediction machine**. Every neuron and pattern exists to predict what comes next. Learning occurs when predictions fail - these failures create patterns that capture the context needed to make better predictions.

**Core Principles:**
- **Prediction drives learning** - neurons form connections that predict future events
- **Failure creates structure** - patterns are only created when confident predictions fail
- **Events learn by association** - connections strengthen when events co-occur
- **Actions learn by reward** - action selection is based on expected reward, not strength
- **Voting enables consensus** - all active neurons vote on predictions, weighted by level and time
- **Patterns override connections** - when a pattern matches, it corrects the connection's prediction

## Key Files

- `brain.js`: Core learning engine with frame processing, pattern recognition, inference, and voting systems
- `db/db.sql`: MySQL schema for neurons, connections, patterns, and memory tables
- `run-brain.js`: Job runner entry point
- `jobs/*.js`: Episode definitions that configure channels and run learning scenarios
- `channels/*.js`: Sensory/motor interfaces that provide inputs, execute outputs, and give rewards

## Documentation

- `docs/architecture.md`: Detailed design document covering voting, patterns, and frame processing
- `docs/error-driven-learning.md`: Deep dive on how patterns are created from prediction errors

## Database Schema

### Core Tables
- **`dimensions`** - Named coordinate dimensions with channel and type (event/action)
- **`neurons`** - All neurons (base and pattern) with level and channel
- **`base_neurons`** - Base neuron metadata (type: event/action)
- **`neuron_coordinates`** - Coordinate values for base neurons

### Connections
- **`connections`** - Links between base neurons with distance, strength, and reward
  - `distance`: Temporal gap between source and target activation
  - `strength`: How often this connection has been observed
  - `reward`: Expected outcome for action connections (exponential smoothing)

### Patterns
- **`pattern_peaks`** - Maps pattern neuron to its peak neuron
- **`pattern_past`** - Context neurons with relative ages (for recognition)
- **`pattern_future`** - Predicted base neurons with distances and rewards

### Memory Tables (ENGINE=MEMORY)
- **`active_neurons`** - Currently active neurons with ages
- **`matched_patterns`** - Patterns that matched in current frame
- **`matched_pattern_past`** - Context analysis (common/novel/missing)
- **`inference_votes`** - All votes before/after pattern override
- **`inferred_neurons`** - Final predictions with winner flags

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| contextLength | 5 | Frames a neuron stays active |
| levelVoteMultiplier | 3 | Weight increase per pattern level |
| rewardExpSmooth | 0.9 | Exponential smoothing for rewards |
| eventErrorMinStrength | 2.0 | Min strength to create error pattern |
| actionRegretMinStrength | 2.0 | Min strength to create regret pattern |
| mergePatternThreshold | 0.5 | Min match ratio for pattern recognition |
| forgetCycles | 100 | Frames between forget cycles |
| connectionForgetRate | 1 | Strength decay per forget cycle |
| maxLevels | 10 | Maximum pattern hierarchy depth |

## Frame Processing Flow

```
1. processFrameIO()
   - Get frame events from all channels
   - Get previous frame's action winners from inferred_neurons
   - Execute actions in channels
   - Get rewards from channels

2. ageNeurons()
   - Increment age of all active neurons

3. processBaseNeurons()
   - Find/create neurons for frame points
   - Insert as active neurons at age 0
   - Reinforce connections (event→event, event→action)
   - Apply rewards to action connections

4. processPatternNeurons()
   - recognizePatterns(): Match and activate patterns level by level
   - refinePatterns(): Update pattern_future based on observations/rewards
   - learnNewPatterns(): Create patterns from prediction errors and action regret

5. deactivateOldNeurons()
   - Remove neurons that aged out of context window

6. inferNeurons()
   - collectVotes(): Gather connection and pattern votes
   - Delete overridden votes (pattern override rule)
   - Aggregate and rank to determine winners
   - applyExploration(): Add exploration actions if no winners
   - saveInferences(): Store for next frame

7. runForgetCycle() (periodic)
   - Decay connection and pattern strengths
   - Delete zero-strength entries
   - Clean up orphaned pattern neurons
```

## Voting System

All active neurons vote on what happens next. Votes are weighted by:

1. **Level weight**: `1 + level * levelVoteMultiplier` (higher patterns have more influence)
2. **Time decay**: `1 - (distance - 1) / contextLength` (recent predictions weighted more)

**Pattern Override Rule**: When a peak neuron has both connection votes AND pattern votes, pattern votes override connection votes. This is how patterns correct connection predictions.

**Winner Selection**:
- Events: highest total strength wins
- Actions: highest weighted reward wins

## Channel Interface

Channels connect the brain to external systems (sensors, actuators, trading APIs).

**Required Methods:**
- `getEventDimensions()` / `getOutputDimensions()`: Define coordinate space
- `getActionNeurons()`: All possible actions (for pre-creation)
- `getFrameEvents()`: Current observations
- `executeOutputs(actions)`: Execute brain's decisions
- `getRewards()`: Feedback on outcomes (0 = neutral)

**Available Channels:**
- `TextChannel` - Character sequence learning
- `StockChannel` - Financial trading with price/volume data
- `VisionChannel` - Visual processing with saccades
- `AudioChannel` - Audio processing
- `ArmChannel` - Motor control

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

#### **Financial Trading (Stock Test)**
```bash
# First, download historical stock data
node run-setup.js stock-test

# Then run the training job (trains on all data except last 50 rows)
node run-brain.js stock-test

# Or test on specific data range using offset and holdout
node run-brain.js stock-test --offset 100 --holdout 50
```
Trains or tests on historical stock data for KGC (Kinross Gold), GLD (Gold ETF), and SPY (S&P 500).
- **Setup**: Downloads full historical daily data from Yahoo Finance API
- **Training**: Runs episodes through historical data, learning patterns and trading strategies
- **Data location**: `data/stock/KGC.csv`, `data/stock/GLD.csv`, `data/stock/SPY.csv`
- **Parameters**:
  - `--holdout N`: Hold out last N rows from training (default: 50)
  - `--offset N`: Skip first N rows (default: 0)
  - `--episodes N`: Number of training episodes (default: 1)

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
        this.brain.thalamus.getChannel('vision').setTargetObject({x: 0.5, y: 0.5});
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
- **Hierarchical scaling**: Higher-level patterns emerge automatically from prediction errors
- **Memory efficiency**: Sliding window prevents unbounded growth
- **Adaptive learning**: Continuous refinement through prediction accuracy and reward feedback
