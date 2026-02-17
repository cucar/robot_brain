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

## Architecture

The system uses an **in-memory architecture** with optional MySQL persistence:

### Core Components

- **`brain/brain.js`** - Main orchestrator: frame processing, pattern recognition, learning, and inference
- **`brain/thalamus.js`** - Relay station: manages neurons, channels, and dimension mappings
- **`brain/memory.js`** - Temporal sliding window: tracks active neurons and inferred predictions
- **`brain/context.js`** - Pattern matching: context representation and matching logic
- **`brain/neuron.js`** - Unified neuron class: handles connections, patterns, voting, and learning
- **`brain/database.js`** - Persistence layer: backup/restore to MySQL (optional)
- **`brain/diagnostics.js`** - Performance tracking and debug output
- **`brain/dump.js`** - Brain state dumps for debugging

### Channels & Jobs

- **`channels/*.js`** - Sensory/motor interfaces that provide inputs, execute outputs, and give rewards
- **`jobs/*.js`** - Episode definitions that configure channels and run learning scenarios
- **`run-brain.js`** - Job runner entry point

### Documentation

- **`docs/architecture.md`** - Detailed design document covering voting, patterns, and frame processing
- **`docs/error-driven-learning.md`** - Deep dive on how patterns are created from prediction errors

## Data Structures

### In-Memory (Primary)

All learning happens in-memory using JavaScript objects:

- **Neurons** (`Neuron` class) - Unified representation for sensory and pattern neurons
  - Level 0 (sensory): have coordinates, channel, type (event/action)
  - Level 1+ (pattern): have peak neuron, context, and predictions
  - All neurons: have connections (Map), patterns (Set), activation strength

- **Memory** (`Memory` class) - Temporal sliding window
  - Active neurons indexed by age (0 = newest, contextLength-1 = oldest)
  - Inferred neurons from previous frame (for execution)
  - Context retrieval for pattern matching and learning

- **Context** (`Context` class) - Pattern context representation
  - Entries: array of {neuron, distance, strength}
  - Fast matching with threshold-based recognition
  - Merge logic for pattern refinement

- **Thalamus** (`Thalamus` class) - Central registry
  - Neuron lookup by ID and coordinates
  - Channel management and action execution
  - Dimension name/ID mappings

### MySQL (Optional Persistence)

Database schema for backup/restore (not used during frame processing):

- **`channels`** - Channel registry with IDs
- **`dimensions`** - Dimension names with IDs
- **`neurons`** - All neurons with level
- **`base_neurons`** - Sensory neuron metadata (channel, type)
- **`coordinates`** - Sensory neuron coordinate values
- **`connections`** - Base neuron connections (distance, strength, reward)
- **`patterns`** - Pattern-to-parent mappings with strength
- **`pattern_past`** - Pattern contexts (context neurons with ages and strengths)
- **`pattern_future`** - Pattern predictions (inferred neurons with distances, strengths, rewards)

## Hyperparameters

Configured in `Neuron`, `Context`, and `Memory` classes:

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| contextLength | 5 | Memory | Frames a neuron stays active |
| maxStrength | 100 | Neuron/Context | Maximum connection/pattern strength |
| minStrength | 0 | Neuron/Context | Minimum strength before deletion |
| levelVoteMultiplier | 3 | Neuron | Weight increase per pattern level |
| rewardSmoothing | 1 | Neuron | Exponential smoothing for rewards (1 = full replacement) |
| eventErrorMinStrength | 2 | Neuron | Min strength to create error pattern |
| actionRegretMinStrength | 2 | Neuron | Min strength to create regret pattern |
| actionRegretMinPain | 0 | Neuron | Min negative reward to trigger regret |
| mergeThreshold | 0.5 | Context | Min match ratio for pattern recognition |
| negativeReinforcement | 0.1 | Context | Weakening rate for missing context |
| connectionForgetRate | 1 | Neuron | Connection strength decay per forget cycle |
| contextForgetRate | 1 | Neuron | Pattern context strength decay per forget cycle |
| patternForgetRate | 1 | Neuron | Pattern prediction strength decay per forget cycle |
| forgetCycles | 100 | Brain | Frames between forget cycles |
| maxLevels | 10 | Brain | Maximum pattern hierarchy depth |

## Frame Processing Flow

The brain processes each frame through a series of steps:

```
1. getFrame()
   - Get frame events from all channels (sensory inputs)
   - Get previous frame's inferred actions from memory
   - Combine into frame points: {coordinates, channel, type}

2. getRewards()
   - Get channel-specific feedback on executed actions
   - Store rewards for connection/pattern reinforcement

3. memory.age()
   - Shift temporal window: age 0 → age 1 → ... → age contextLength
   - Deactivate neurons that aged out of context window

4. activateSensors()
   - Find/create neurons for frame points (via Thalamus)
   - Activate neurons at age 0 in memory
   - Track inference accuracy (diagnostics)

5. recognizePatterns()
   - For each level (0, 1, 2, ...) until no patterns found:
     - Get peaks (age=0) and context (age>0) at this level
     - Match patterns: parent.matchPattern(context)
     - Activate matched patterns at age 0
   - Enables hierarchical pattern recognition

6. updateConnections()
   - For each context neuron (age > 0):
     - Learn connections to newly active neurons (age = 0)
     - Update strength (increment) and reward (exponential smoothing)
     - Create new connections if not exist

7. learnNewPatterns()
   - For each neuron that voted in previous frame:
     - Check for prediction errors (strong prediction, wrong outcome)
     - Check for action regret (negative reward)
     - Create pattern with parent=predictor, context=active neurons, prediction=actual outcome
     - Activate new pattern at parent's age

8. inferNeurons()
   - collectVotes(): Active neurons vote for next frame
     - Suppress votes from parents with activated patterns
     - Weight votes by level and time decay
     - Save votes and context for learning
   - determineConsensus(): Aggregate votes and select winners
     - Events win by highest strength
     - Actions win by highest reward
   - ensureChannelActions(): Add exploration if no action inferred
   - memory.saveInferences(): Store for next frame

9. executeActions()
   - Execute inferred actions via Thalamus
   - Channels coordinate execution (e.g., portfolio allocation)

10. runForgetCycle() (periodic, every forgetCycles frames)
    - neuron.forget(): Decay connection and pattern strengths
    - Delete patterns with no content or references
    - Recursive cleanup of context references
```

## Voting System

All active neurons vote on what happens next. The voting system enables distributed decision-making.

### Vote Collection

Active neurons at age 0 to contextLength-1 cast votes:
- **Connection votes**: From neuron.connections at distance = age + 1
- **Pattern votes**: From pattern neurons via their peak's routing table
- **Suppression**: Peaks with activated patterns don't vote (pattern overrides connection)

### Vote Weighting

Each vote is weighted by two factors:

1. **Level weight**: `1 + level * levelVoteMultiplier`
   - Higher-level patterns have more influence (default: 3x per level)
   - Reflects that patterns represent more context

2. **Time decay**: `1 - age / contextLength`
   - Recent predictions weighted more than distant ones
   - Age 0 gets full weight, older ages decay linearly

**Effective strength** = `levelWeight * timeWeight * rawStrength`

### Consensus Determination

1. **Aggregate votes**: Sum effective strengths per target neuron
2. **Calculate rewards**: Weighted average of vote rewards (for actions)
3. **Select winners per dimension**:
   - Events: highest total strength wins
   - Actions: highest weighted reward wins
4. **Return winners**: Neurons that won in any dimension

### Pattern Override

When a pattern activates on a parent neuron, the parent's connection votes are suppressed. This is implemented by checking `state.activatedPattern` during vote collection - if not null, the neuron doesn't vote. This is how patterns correct connection predictions.

## Channel Interface

Channels are adapters between the brain and external devices (sensors, actuators, trading APIs, etc.).

### Required Methods

- **`getEventDimensions()`** - Returns array of Dimension objects for sensory inputs
- **`getOutputDimensions()`** - Returns array of Dimension objects for motor outputs
- **`getActions()`** - Returns array of all possible action coordinates (for pre-creation)
- **`getFrameEvents()`** - Returns array of current observations as coordinate objects
- **`executeOutputs(actions)`** - Executes brain's action decisions
- **`getRewards(actions)`** - Returns reward signal (0 = neutral, positive = good, negative = bad)

### Optional Methods

- **`static initialize(options)`** - Channel-level initialization (called once)
- **`static resetChannelContext()`** - Reset shared state across instances
- **`static executeChannelActions(channels, actionsMap)`** - Coordinated execution (e.g., portfolio allocation)
- **`static getPortfolioMetrics(channels)`** - Aggregate metrics across channel instances
- **`resetContext()`** - Reset instance state for new episode
- **`calculatePredictionError()`** - Continuous prediction error (e.g., MAPE)
- **`getOutputPerformanceMetrics()`** - Channel-specific performance (e.g., P&L, score)
- **`getMetrics()`** - Diagnostic metrics for reporting

### Available Channels

- **`TextChannel`** - Character sequence learning and text generation
- **`StockChannel`** - Financial trading with price/volume data and position tracking
- **`VisionChannel`** - Visual processing with saccadic eye movements
- **`AudioChannel`** - Audio processing and pattern recognition
- **`ArmChannel`** - Motor control with proprioceptive feedback
- **`TongueChannel`** - Taste processing and tongue movements

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

Jobs extend the `Job` base class and define channels and episode logic:

```javascript
import { Job } from './jobs/job.js';
import { VisionChannel } from './channels/vision.js';
import { ArmChannel } from './channels/arm.js';

export default class CustomJob extends Job {

    // Define which channels to use
    getChannels() {
        return [
            { name: 'vision', channelClass: VisionChannel },
            { name: 'arm', channelClass: ArmChannel }
        ];
    }

    // Optional: Configure channels after initialization
    async configureChannels() {
        this.brain.getChannel('vision').setTargetObject({x: 0.5, y: 0.5});
    }

    // Optional: Custom episode logic
    async executeJob() {
        // Reset context for clean episode start
        this.brain.resetContext();

        // Process frames until done
        while (await this.brain.processFrame()) {
            // Frame processing happens automatically
        }
    }

    // Optional: Show custom results
    async showResults() {
        const summary = this.brain.getEpisodeSummary();
        console.log('Episode completed:', summary);
    }
}
```

### Job Lifecycle

1. **Setup** (optional): `node run-setup.js <job-name>` - Download/prepare data
2. **Execution**: `node run-brain.js <job-name>` - Run the brain
   - `getChannels()` - Define channels
   - `brain.init()` - Initialize brain and channels
   - `configureChannels()` - Custom channel setup
   - `executeJob()` - Main episode logic
   - `showResults()` - Display results
   - `brain.backup()` - Save to MySQL (if --database flag)

## Performance Characteristics

- **In-memory processing**: All learning happens in JavaScript objects (no database queries during frames)
- **Hierarchical scaling**: Higher-level patterns emerge automatically from prediction errors
- **Memory efficiency**: Sliding window prevents unbounded growth (contextLength frames)
- **Adaptive learning**: Continuous refinement through prediction accuracy and reward feedback
- **Optional persistence**: MySQL backup/restore for long-term storage (not used during processing)
- **Forget cycles**: Periodic cleanup prevents curse of dimensionality

## Key Design Decisions

### Why In-Memory?

The original MySQL-based implementation had performance bottlenecks. The current architecture:
- Stores all neurons, connections, and patterns in JavaScript objects
- Uses Maps and Sets for O(1) lookups
- Processes frames without database queries
- Optionally backs up to MySQL between episodes

### Why Thalamus?

The Thalamus acts as a relay station (named after the biological thalamus):
- Centralizes neuron registry and lookup
- Manages channel coordination
- Handles dimension name/ID mappings
- Abstracts reference frame translations

### Why Unified Neuron Class?

Both sensory and pattern neurons share common functionality:
- All neurons have connections (predictions)
- All neurons have patterns (routing table for context-specific predictions)
- All neurons vote with the same weighting logic
- Simplifies code and enables consistent behavior

### Why Context Class?

Pattern matching requires efficient context comparison:
- Fast key-based lookup for matching
- Threshold-based recognition (default 50%)
- Merge logic for pattern refinement
- Shared between observed contexts and known patterns
