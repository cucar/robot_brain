# Artificial Brain Test Plan

This directory contains comprehensive unit tests for the artificial brain implementation. The testing strategy is designed to ensure 100% correctness across all components, as even the slightest bug could result in system failure.

## Test Plan Overview

The test plan is organized into 10 hierarchical sections, starting with foundational components and building up to complete system integration. Each section focuses on critical components that must work perfectly for the system to function correctly.

## Test Sections

### 1. SQL Query Logic & Mathematical Calculations (`tests/1-sql-logic/`)
**Focus**: Complex SQL queries, mathematical formulas, and aggregation logic

**Critical Test Areas**:
- POW function calculations for aging and distance (POW(baseNeuronMaxAge, level + 1))
- Distance calculations using FLOOR(age / POW(...)) formulas
- Temporal decay formulas in reward calculations
- Complex JOIN conditions with multiple table relationships
- GROUP BY and HAVING clauses in pattern detection
- Aggregation logic for strength calculations
- Pattern matching queries with percentage thresholds
- Connection inference and pattern inference query logic

**Key SQL Queries to Test**:
- Aging queries: `DELETE FROM active_neurons WHERE age >= POW(?, level + 1)`
- Distance calculations: `FLOOR(f.age / POW(?, f.level)) as distance`
- Reward decay: `1.0 + (? - 1.0) * (1.0 - inf.age / POW(?, inf.level + 1))`
- Pattern matching: Complex CTEs with overlap percentage calculations
- Connection reinforcement: Multi-table JOINs with temporal conditions

**Why Critical**: These mathematical calculations and complex queries are the brain's core logic. Incorrect formulas would cause learning failures, memory corruption, or system instability.

---

### 2. Brain Initialization & Configuration (`tests/2-initialization/`)
**Focus**: Constructor parameters, hyperparameter validation, channel registration

**Critical Test Areas**:
- Brain constructor and hyperparameter validation
- Database connection initialization
- Dimension creation and loading
- Channel registration and validation
- Reset operations (context vs hard reset)

**Key Methods to Test**:
- `constructor()` - Hyperparameter initialization
- `init()` - Database and dimension setup
- `registerChannel()` - Channel registration
- `resetContext()` - Memory table cleanup
- `resetBrain()` - Complete system reset

**Why Critical**: Incorrect initialization leads to cascading failures throughout the system.

---

### 3. Neuron Management System (`tests/3-neurons/`)
**Focus**: Neuron creation, activation, aging, and lifecycle

**Critical Test Areas**:
- Neuron creation from frame coordinates
- Coordinate matching and tolerance handling
- Neuron activation and deactivation
- Age-based neuron lifecycle management
- Level-based aging (POW function behavior)
- Bulk operations and performance

**Key Methods to Test**:
- `getFrameNeurons()` - Frame to neuron conversion
- `matchNeuronsFromPoints()` - Coordinate matching
- `createBaseNeurons()` - Neuron creation
- `activateNeurons()` - Neuron activation
- `ageNeurons()` - Age progression and cleanup
- `insertActiveNeurons()` - Bulk insertion

**Why Critical**: Neurons are the fundamental units of the brain. Any bugs in neuron management break the entire system.

---

### 4. Connection & Pattern Learning (`tests/4-connections/`)
**Focus**: Connection reinforcement, pattern detection, hierarchical learning

**Critical Test Areas**:
- Connection creation and reinforcement
- Spatial vs temporal connection handling
- Pattern detection and clustering
- Peak detection algorithms
- Hierarchical pattern activation
- Pattern merging and threshold handling

**Key Methods to Test**:
- `reinforceConnections()` - Connection strength updates
- `activatePatternNeurons()` - Hierarchical pattern detection
- `activateLevelPatterns()` - Level-specific pattern processing
- `detectPeaks()` - Peak detection algorithm
- `getActiveConnections()` - Connection retrieval
- `reinforcePatterns()` - Pattern strength updates

**Why Critical**: This is where the brain learns relationships and builds knowledge hierarchies.

---

### 5. Inference & Prediction Engine (`tests/5-inference/`)
**Focus**: Pattern inference, connection inference, prediction generation

**Critical Test Areas**:
- Pattern-based predictions
- Connection-based predictions
- Temporal prediction accuracy
- Level-specific inference behavior
- Prediction aging and expiration
- Strength calculation algorithms

**Key Methods to Test**:
- `inferNeurons()` - Main inference orchestration
- `inferPatterns()` - Pattern-based predictions
- `inferConnections()` - Connection-based predictions
- `getPredictedConnections()` - Prediction retrieval
- `getNeuronStrengths()` - Strength calculations
- `inferPeakNeurons()` - Peak neuron selection

**Why Critical**: This is the brain's "thinking" process - predictions drive all outputs and learning.

---

### 6. Reward & Learning System (`tests/6-rewards/`)
**Focus**: Reward application, temporal decay, learning reinforcement

**Critical Test Areas**:
- Global reward application
- Temporal decay calculations
- Reward factor multiplication
- Negative learning rate application
- Reward optimization algorithms
- Hierarchical reward propagation

**Key Methods to Test**:
- `applyRewards()` - Global reward application
- `optimizeRewards()` - Reward-based optimization
- Temporal decay formulas
- Reward factor storage and retrieval
- Pattern strength adjustments

**Why Critical**: Without proper rewards, the brain cannot learn from experience or improve performance.

---

### 7. Memory Management & Forgetting (`tests/7-memory/`)
**Focus**: Forget cycles, memory cleanup, system optimization

**Critical Test Areas**:
- Forget cycle timing and execution
- Strength decay calculations
- Orphaned neuron cleanup
- Pattern and connection pruning
- Memory table optimization
- Performance under memory pressure

**Key Methods to Test**:
- `runForgetCycle()` - Complete forget cycle
- Strength decay algorithms
- Orphaned data cleanup
- Memory usage optimization
- Forget cycle frequency control

**Why Critical**: Prevents memory bloat and maintains system performance over time.

---

### 8. Channel System Integration (`tests/8-channels/`)
**Focus**: Channel interfaces, input/output processing, feedback loops

**Critical Test Areas**:
- Channel initialization and registration
- Dimension registration and validation
- Frame input processing
- Output execution and feedback
- Exploration action generation
- Multi-channel coordination

**Key Methods to Test**:
- Channel base class functionality
- Specific channel implementations (Stock, Vision, Text, etc.)
- Input/output dimension handling
- Feedback calculation
- Exploration algorithms

**Why Critical**: Channels are how the brain interacts with the world - any interface bugs break learning.

---

### 9. Frame Processing Pipeline (`tests/9-pipeline/`)
**Focus**: Complete frame processing workflow and transaction integrity

**Critical Test Areas**:
- Complete frame processing workflow
- Transaction handling and rollback
- Error recovery and consistency
- Processing order and dependencies
- Performance under load
- Concurrent processing scenarios

**Key Methods to Test**:
- `processFrame()` - Main processing pipeline
- Transaction integrity
- Error handling and recovery
- Processing performance
- State consistency

**Why Critical**: This is the main processing loop that must be bulletproof for system reliability.

---

### 10. Integration & End-to-End Testing (`tests/10-integration/`)
**Focus**: Complete system integration with real scenarios

**Critical Test Areas**:
- Multi-channel learning scenarios
- Long-running learning sessions
- Edge cases and boundary conditions
- Performance under various loads
- Real-world data processing
- System stability and reliability

**Key Scenarios to Test**:
- Stock trading scenarios
- Vision processing scenarios
- Text processing scenarios
- Multi-modal learning
- System recovery scenarios

**Why Critical**: Ensures all components work together correctly in real-world conditions.

## Testing Guidelines

### Test Structure
Each test section should include:
- **Unit tests** for individual methods
- **Integration tests** for component interactions
- **Edge case tests** for boundary conditions
- **Performance tests** for critical paths

### Test Data
- Use controlled, predictable test data
- Include edge cases and boundary conditions
- Test with both small and large datasets
- Validate mathematical calculations precisely

### Test Execution Order
1. Start with Section 1 (SQL Query Logic & Mathematical Calculations)
2. Work sequentially through sections 2-7
3. Test Section 8 (Channels) in parallel with 2-7
4. Complete with Sections 9-10 (Integration testing)

### Success Criteria
- All tests must pass with 100% accuracy
- No memory leaks or resource issues
- Consistent behavior across multiple runs
- Performance within acceptable bounds

## Getting Started

To begin testing:
1. Set up test database environment
2. Install testing dependencies (Jest, MySQL test utilities)
3. Start with Section 1 (SQL Logic) tests - validate all mathematical formulas
4. Work through each section systematically

Each section will have its own subdirectory with specific test files and documentation.
