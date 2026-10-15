//! UCAR — Universal Compression with Actions and Rewards (docs/algorithm.md).
pub mod machine;
pub mod neuron;
pub mod offsets;
pub mod types;

pub use machine::{Machine, Winner, FrameStats};
pub use types::*;
