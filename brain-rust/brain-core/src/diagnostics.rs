/// Diagnostics - per-frame and per-episode stats for the brain core.
///
/// This module is pure data: it accumulates counters and exposes them via getters.
/// It does NOT print, format, or otherwise render. All presentation lives in the
/// host-side renderer (libs/node/src/renderer.js) so that when the brain moves to
/// Rust the counters port cleanly (integers, floats, small structs) while the
/// rendering stays in JS where I/O and app-layer composition belong.

use crate::quantizer::Quantizer;
use crate::types::{ChannelId, Coordinate, DimensionId};
use rustc_hash::FxHashMap;

/// A misprediction record: what was predicted vs what actually happened.
#[derive(Debug, Clone)]
pub struct Misprediction {
    pub channel_id: ChannelId,
    pub predicted: Coordinate,
    pub actual: Coordinate,
}

/// A single inference result item, fully self-contained — Thalamus.get_inference_results
/// pre-resolves correctness, the per-channel actual coord, and the reward, so
/// track_inference_performance is a single pass over a flat array with no further lookups.
#[derive(Debug, Clone)]
pub struct InferenceResultItem {
    pub neuron_type: InferenceType,
    pub is_correct: bool,
    pub channel_id: ChannelId,
    pub coordinate: Option<Coordinate>,
    pub actual_coord: Option<Coordinate>,
    pub reward: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InferenceType {
    Event,
    Action,
}

/// A single per-dimension inference for continuous error tracking.
#[derive(Debug, Clone)]
pub struct DimInference {
    pub dim_id: DimensionId,
    pub kind: InferenceType,
    pub continuous: Option<f64>,
}

/// Episode-to-date stats in a stable shape for host-side rendering.
/// All rates are returned as Options — None means "no data yet", callers should
/// render "N/A" rather than 0.
#[derive(Debug, Clone)]
pub struct DiagnosticStats {
    pub base_accuracy: Option<f64>,
    pub accuracy_correct: u64,
    pub accuracy_total: u64,
    pub avg_reward: Option<f64>,
    pub reward_count: u64,
    pub total_reward: f64,
    pub mape: Option<f64>,
    pub mape_count: u64,
    pub mispredictions: Vec<Misprediction>,
}

pub struct Diagnostics {
    accuracy_correct: u64,
    accuracy_total: u64,
    total_reward: f64,
    reward_count: u64,
    continuous_total_error: f64,
    continuous_count: u64,
    mispredictions: Vec<Misprediction>,
}

impl Diagnostics {
    pub fn new() -> Self {
        Self {
            accuracy_correct: 0,
            accuracy_total: 0,
            total_reward: 0.0,
            reward_count: 0,
            continuous_total_error: 0.0,
            continuous_count: 0,
            mispredictions: Vec::new(),
        }
    }

    /// Reset per-episode counters. Cumulative across an episode but wiped when the
    /// Job kicks off a new one — mispredictions are reset too so retrospective
    /// tooling only sees the current episode.
    pub fn reset_accuracy_stats(&mut self) {
        self.accuracy_correct = 0;
        self.accuracy_total = 0;
        self.total_reward = 0.0;
        self.reward_count = 0;
        self.continuous_total_error = 0.0;
        self.continuous_count = 0;
        self.mispredictions.clear();
    }

    /// Accumulate MAPE (Mean Absolute Percentage Error) from scalar-space inferences.
    /// Compares continuous (score-weighted) event predictions against the actual
    /// input scalars for the same (channel_id, dim_id). Actions are skipped — reward
    /// from next frame is the ground-truth signal for those.
    ///
    /// Dims not registered with the quantizer belong to a legacy channel that tracks
    /// its own prediction error via channel.calculate_prediction_error; skipping here
    /// avoids double-counting.
    pub fn track_continuous_error(&mut self, inferences_by_channel: &FxHashMap<ChannelId, Vec<DimInference>>, inputs: &FxHashMap<ChannelId, FxHashMap<DimensionId, f64>>, quantizer: &Quantizer) {
        for (&channel_id, dim_inferences) in inferences_by_channel {
            let actuals = match inputs.get(&channel_id) {
                Some(a) => a,
                None => continue,
            };
            for inf in dim_inferences {
                if inf.kind != InferenceType::Event { continue; }
                if !quantizer.has(inf.dim_id) { continue; } // channel still owns bucketization - skip to avoid double-counting
                let continuous = match inf.continuous {
                    Some(c) => c,
                    None => continue, // brain had no observed-bucket data to produce a scalar prediction
                };
                let actual = match actuals.get(&inf.dim_id) {
                    Some(&a) if a != 0.0 => a,
                    _ => continue, // skip undefined and avoid divide-by-zero
                };
                self.continuous_total_error += ((actual - continuous) / actual).abs() * 100.0;
                self.continuous_count += 1;
            }
        }
    }

    /// Track event accuracy, action rewards, and misprediction log. Continuous prediction
    /// error is tracked separately in track_continuous_error.
    ///
    /// Each item is fully self-contained — Thalamus.get_inference_results pre-resolves
    /// correctness, the per-channel actual coord, and the reward, so this routine is
    /// a single pass over a flat array with no further lookups.
    pub fn track_inference_performance(&mut self, items: &[InferenceResultItem]) {
        for item in items {
            if item.neuron_type == InferenceType::Event {
                self.accuracy_total += 1;
                if item.is_correct { self.accuracy_correct += 1; }
                else if let (Some(predicted), Some(actual)) = (&item.coordinate, &item.actual_coord) {
                    self.mispredictions.push(Misprediction {
                        channel_id: item.channel_id,
                        predicted: predicted.clone(),
                        actual: actual.clone(),
                    });
                }
            } else if item.neuron_type == InferenceType::Action {
                self.total_reward += item.reward;
                self.reward_count += 1;
            }
        }
    }

    /// Return episode-to-date stats in a stable shape for host-side rendering.
    /// All rates are returned as Options — None means "no data yet", callers should
    /// render "N/A" rather than 0.
    pub fn get_stats(&self) -> DiagnosticStats {
        DiagnosticStats {
            base_accuracy: if self.accuracy_total > 0 { Some(self.accuracy_correct as f64 / self.accuracy_total as f64) } else { None },
            accuracy_correct: self.accuracy_correct,
            accuracy_total: self.accuracy_total,
            avg_reward: if self.reward_count > 0 { Some(self.total_reward / self.reward_count as f64) } else { None },
            reward_count: self.reward_count,
            total_reward: self.total_reward,
            mape: if self.continuous_count > 0 { Some(self.continuous_total_error / self.continuous_count as f64) } else { None },
            mape_count: self.continuous_count,
            mispredictions: self.mispredictions.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Coordinate;

    #[test]
    fn test_initial_stats_are_empty() {
        let d = Diagnostics::new();
        let stats = d.get_stats();
        assert!(stats.base_accuracy.is_none());
        assert!(stats.avg_reward.is_none());
        assert!(stats.mape.is_none());
        assert_eq!(stats.accuracy_total, 0);
    }

    #[test]
    fn test_track_event_accuracy() {
        let mut d = Diagnostics::new();
        d.track_inference_performance(&[
            InferenceResultItem { neuron_type: InferenceType::Event, is_correct: true, channel_id: 0, coordinate: None, actual_coord: None, reward: 0.0 },
            InferenceResultItem { neuron_type: InferenceType::Event, is_correct: true, channel_id: 0, coordinate: None, actual_coord: None, reward: 0.0 },
            InferenceResultItem { neuron_type: InferenceType::Event, is_correct: false, channel_id: 0, coordinate: Some(Coordinate { dim_id: 0, bucket_id: 1 }), actual_coord: Some(Coordinate { dim_id: 0, bucket_id: 2 }), reward: 0.0 },
        ]);
        let stats = d.get_stats();
        assert_eq!(stats.accuracy_correct, 2);
        assert_eq!(stats.accuracy_total, 3);
        assert!((stats.base_accuracy.unwrap() - 2.0 / 3.0).abs() < 1e-10);
        assert_eq!(stats.mispredictions.len(), 1);
    }

    #[test]
    fn test_track_action_rewards() {
        let mut d = Diagnostics::new();
        d.track_inference_performance(&[
            InferenceResultItem { neuron_type: InferenceType::Action, is_correct: false, channel_id: 0, coordinate: None, actual_coord: None, reward: 10.0 },
            InferenceResultItem { neuron_type: InferenceType::Action, is_correct: false, channel_id: 0, coordinate: None, actual_coord: None, reward: -2.0 },
        ]);
        let stats = d.get_stats();
        assert!((stats.total_reward - 8.0).abs() < 1e-10);
        assert_eq!(stats.reward_count, 2);
        assert!((stats.avg_reward.unwrap() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_reset() {
        let mut d = Diagnostics::new();
        d.track_inference_performance(&[
            InferenceResultItem { neuron_type: InferenceType::Event, is_correct: true, channel_id: 0, coordinate: None, actual_coord: None, reward: 0.0 },
        ]);
        assert_eq!(d.get_stats().accuracy_total, 1);
        d.reset_accuracy_stats();
        assert_eq!(d.get_stats().accuracy_total, 0);
        assert!(d.get_stats().base_accuracy.is_none());
    }

    #[test]
    fn test_track_continuous_error() {
        let mut d = Diagnostics::new();
        use crate::quantizer::{QuantizeMode, Quantizer};

        let mut q = Quantizer::new();
        q.register_dimension(0, 3, QuantizeMode::Static, Some(vec![-0.5, 0.5]), None);

        let mut inferences = FxHashMap::default();
        inferences.insert(0, vec![
            DimInference { dim_id: 0, kind: InferenceType::Event, continuous: Some(0.8) },
        ]);

        let mut inputs = FxHashMap::default();
        let mut dim_vals = FxHashMap::default();
        dim_vals.insert(0, 1.0); // actual = 1.0, predicted = 0.8 → |0.2/1.0| * 100 = 20%
        inputs.insert(0, dim_vals);

        d.track_continuous_error(&inferences, &inputs, &q);
        let stats = d.get_stats();
        assert_eq!(stats.mape_count, 1);
        assert!((stats.mape.unwrap() - 20.0).abs() < 1e-10);
    }
}
