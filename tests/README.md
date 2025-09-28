# Artificial Brain Test Plan

This directory contains comprehensive unit tests for the artificial brain implementation. The testing strategy is designed to ensure 100% correctness across all components, as even the slightest bug could result in system failure.

## Test Plan Overview

The test plan is organized into 10 hierarchical sections, starting with foundational components and building up to complete system integration. Each section focuses on critical components that must work perfectly for the system to function correctly.

## Test Sections

### 1. SQL Query Logic & Mathematical Calculations (`tests/sql-logic-tests.js`) ✅
**Focus**: Critical mathematical formulas used in SQL queries

**Test File**: Single focused test file with essential validations
**Tests**: 12 tests covering core mathematical expressions

**Critical Areas Tested**:
- POW function calculations: `POW(baseNeuronMaxAge, level + 1)` for aging thresholds
- Distance calculations: `FLOOR(age / POW(baseNeuronMaxAge, level))` for connections
- Reward decay formulas: `1.0 + (globalReward - 1.0) * (1.0 - age/levelMaxAge)` for temporal decay

**Why Critical**: These mathematical calculations are the foundation of the brain's learning logic. Incorrect formulas would cause system-wide failures.

---

### 2. Brain Initialization & Configuration (`tests/brain-initialization-tests.js`) ✅
**Focus**: Database connectivity, channel registration, and system initialization

**Test File**: Single focused test file with integration point validations
**Tests**: 18 tests covering actual failure points

**Critical Areas Tested**:
- Channel registration and dimension provision
- Database connection establishment
- Dimension creation and mapping consistency
- Reset operations (memory vs full reset)

**Why Critical**: These are integration points where real failures occur - database connectivity, dimension creation logic, and SQL operations.

---

### 3. Neuron Management System (Split into 2 focused test files) ✅
**Focus**: Neuron creation, activation, aging, and lifecycle management

#### 3a. Neuron Creation & Coordinate Matching (`tests/neuron-creation-tests.js`) ✅
**Focus**: Frame-to-neuron conversion and coordinate handling
**Tests**: 27 tests covering coordinate matching, neuron creation, and bulk operations

**Critical Areas Tested**:
- `matchFrameNeurons()` - Simplified single-query coordinate matching (refactored from complex UNION)
- `createBaseNeurons()` - Bulk neuron creation with coordinate insertion and deduplication
- `bulkInsertNeurons()` - Auto-increment ID handling and edge cases
- `getFrameNeurons()` - End-to-end frame-to-neuron conversion with matching and creation
- Edge cases: empty frames, duplicate points, coordinate validation

#### 3b. Neuron Lifecycle & Activation (`tests/neuron-lifecycle-tests.js`) ✅
**Focus**: Neuron activation, aging, and lifecycle management
**Tests**: 21 tests covering activation, aging, and connection reinforcement

**Critical Areas Tested**:
- `insertActiveNeurons()` - Neuron activation at different levels
- `ageNeurons()` - Age progression and level-based cleanup (validates POW formulas from Section 1)
- `activateNeurons()` - Integration of activation and connection reinforcement
- `reinforceConnections()` - Direct testing of temporal connection creation with distance calculations
- Connection strength reinforcement and duplicate handling
- Level-based aging thresholds and neuron lifecycle management

**Why Critical**: Neurons are the fundamental units of the brain. Any bugs in neuron management break the entire system.

---

#### 4. Connection & Pattern Learning (`tests/connection-pattern-tests.js`) ✅
**Focus**: Connection reinforcement, pattern detection, hierarchical learning
**Tests**: 118 tests covering connection learning, pattern detection, and hierarchical activation

**Critical Areas Tested**:
- `getActiveConnections()` - Connection retrieval with proper distance calculations
- `detectPeaks()` - Peak detection algorithm with neighborhood strength analysis
- `activateLevelPatterns()` - Level-specific pattern processing and activation
- `activatePatternNeurons()` - Hierarchical pattern detection across multiple levels
- `reinforceConnections()` - Connection strength updates and new connection creation
- `reinforcePatterns()` - Pattern strength reinforcement and pattern-connection relationships
- `matchPatternNeurons()` - Pattern matching and merging logic
- Edge cases: empty inputs, single connections, no active neurons

**Why Critical**: This is where the brain learns relationships and builds knowledge hierarchies.

---

### 5. Inference & Prediction Engine (`tests/inference-prediction-tests.js`) ✅
**Focus**: Pattern inference, connection inference, prediction generation

**Test File**: Single focused test file with comprehensive inference validations
**Tests**: 61 tests across 11 test methods covering all inference and prediction functionality

**Critical Areas Tested**:
- `inferConnections()` - Same-level connection predictions with temporal distance calculations
- `inferPatterns()` - Pattern-based lower-level predictions with aging and lifecycle management
- `getPredictedConnections()` - Unified prediction retrieval from both connection and pattern inference
- `getNeuronStrengths()` - Strength calculations from bidirectional connection sums
- `inferPeakNeurons()` - Peak detection algorithm and inferred neuron insertion
- `inferNeurons()` - Complete orchestration of multi-level inference processing
- `optimizeRewards()` - Reward factor application to base strengths before peak detection
- Temporal prediction accuracy with distance = age + 1 relationships
- Level-specific inference behavior and proper level separation
- Prediction aging, expiration, and lifecycle management
- Edge cases: empty inputs, no connections, no predictions

**Why Critical**: This is the brain's "thinking" process - predictions drive all outputs and learning. Any bugs in inference break the brain's ability to anticipate and respond to future states.

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
1. ✅ **Section 1**: SQL Logic & Mathematical Calculations (COMPLETE - 12 tests)
2. ✅ **Section 2**: Brain Initialization & Configuration (COMPLETE - 18 tests)
3. ✅ **Section 3**: Neuron Management System (COMPLETE - 49 tests total)
   - ✅ 3a: Neuron Creation & Coordinate Matching (27 tests)
   - ✅ 3b: Neuron Lifecycle & Activation (22 tests)
4. ✅ **Section 4**: Connection & Pattern Learning (COMPLETE - 118 tests)
5. ✅ **Section 5**: Inference & Prediction Engine (COMPLETE - 11 test methods)
6. Work sequentially through sections 6-7
7. Test Section 8 (Channels) in parallel with 6-7
8. Complete with Sections 9-10 (Integration testing)

### Success Criteria
- All tests must pass with 100% accuracy
- No memory leaks or resource issues
- Consistent behavior across multiple runs
- Performance within acceptable bounds

## Getting Started

To begin testing:
1. Set up test database environment
2. No external testing dependencies needed - uses plain Node.js with simple assertions
3. Run tests individually: `node tests/sql-logic-tests.js`
4. Work through each section systematically

## Current Test Files

### ✅ Completed
- `tests/sql-logic-tests.js` - 12 tests validating core mathematical formulas
- `tests/brain-initialization-tests.js` - 18 tests validating system initialization
- `tests/neuron-creation-tests.js` - 27 tests for coordinate matching and neuron creation
- `tests/neuron-lifecycle-tests.js` - 22 tests for activation, aging, and lifecycle management
- `tests/connection-pattern-tests.js` - 118 tests for connection learning and pattern detection
- `tests/inference-prediction-tests.js` - 61 comprehensive tests for inference and prediction

**Total: 258 tests covering the foundational brain architecture and core learning systems**

### 📋 Planned
- Sections 4-10 as outlined above

Each test file is self-contained with focused, essential validations of actual failure points.
