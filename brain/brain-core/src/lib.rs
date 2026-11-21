/// Core brain computation engine: the machine of docs/algorithm.md.
///
/// Architecture:
/// - Brain: orchestrator — the frame, its levels, recognition, the election, covering, connect, infer, selection
/// - Thalamus: the registry, the dictionary, the board, routing, the batch calls
/// - Region: column partition, parallel fan-out
/// - Column: neuron partition
/// - Neuron: its history of rows, its relation index, its connections
pub mod board;
pub mod brain;
pub mod dictionary;
pub mod neuron;
pub mod thalamus;
pub mod types;

mod backup;
mod column;
mod connect;
mod cover;
mod death;
mod election;
mod expansion;
mod recognize;
mod region;
mod selection;
