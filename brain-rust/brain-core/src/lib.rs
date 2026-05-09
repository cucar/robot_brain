/// Core brain computation engine.
///
/// This crate implements the hierarchical prediction system:
/// neurons, contexts, connections, pattern recognition, and voting.
///
/// Architecture (single-threaded, Phase 1):
/// - Brain: orchestrator — frame loop, learning, inference
/// - Thalamus: neuron registry, channel management, routing
/// - Memory: temporal sliding window of active neurons
/// - Column: neuron partition, batch operations
/// - Region: column partition, deterministic routing
/// - Neuron: connections, children, voting, learning, decay
/// - Context: pattern context matching & merging
/// - Quantizer: scalar-to-bucket discretization
/// - Diagnostics: accuracy tracking
pub mod context;
pub mod neuron;
pub mod types;
