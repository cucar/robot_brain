use rustc_hash::FxHashMap;

use crate::types::{ContextEntry, MatchResult, NeuronId, Strength};

/// Spatial context: set of neurons co-active on the current frame with strengths.
/// No distance dimension. Storage is just `neuron_id → strength`.
#[derive(Debug, Clone)]
pub struct SpatialContext {
    entries: FxHashMap<NeuronId, Strength>,
}

impl SpatialContext {
    pub fn new() -> Self {
        Self { entries: FxHashMap::default() }
    }

    /// Number of context entries (one per neuron).
    pub fn size(&self) -> usize {
        self.entries.len()
    }

    /// Add a neuron at a given strength.
    /// Panics if the entry already exists — caller must check first.
    pub fn add_neuron(&mut self, neuron_id: NeuronId, strength: Strength) {
        if self.entries.contains_key(&neuron_id) {
            panic!("SpatialContext entry already exists for neuron {}", neuron_id);
        }
        self.entries.insert(neuron_id, strength);
    }

    /// Check if an entry exists.
    pub fn has_key(&self, neuron_id: NeuronId) -> bool {
        self.entries.contains_key(&neuron_id)
    }

    /// Increment the strength of an existing entry by 1.
    /// Panics if the entry does not exist — callers must check has_key first.
    pub fn strengthen_neuron(&mut self, neuron_id: NeuronId) {
        let strength = self.entries.get_mut(&neuron_id).expect("SpatialContext entry not found for strengthening");
        *strength += 1.0;
    }

    /// Decrement the strength of an existing entry by 1 (no-op if absent). Returns true when the entry has
    /// decayed to ≤ 0 and should be deleted — the caller removes it and scrubs the context index. Used by
    /// spatial context refinement to weaken/delete context neurons that were missing from a match.
    pub fn weaken_neuron(&mut self, neuron_id: NeuronId) -> bool {
        match self.entries.get_mut(&neuron_id) {
            Some(strength) => { *strength -= 1.0; *strength <= 0.0 }
            None => false,
        }
    }

    /// Remove an entry explicitly. Used by the spatial death cascade to scrub a dying context
    /// neuron from a child pattern's stored context.
    pub fn remove(&mut self, neuron_id: NeuronId) {
        self.entries.remove(&neuron_id);
    }

    /// Direct access to the inner map.
    pub fn entries(&self) -> &FxHashMap<NeuronId, Strength> {
        &self.entries
    }

    /// Flatten all entries into a Vec for iteration / serialization.
    /// Each entry surfaces as a ContextEntry with `distance = 0` so the shared MatchResult /
    /// ContextEntry types stay reusable; readers should ignore the distance field.
    pub fn get_entries(&self) -> Vec<ContextEntry> {
        self.entries.iter()
            .map(|(&neuron_id, &strength)| ContextEntry { neuron_id, distance: 0, strength })
            .collect()
    }

    /// Match this known spatial context against an observed spatial context.
    /// Returns a match result with score, or None if below threshold.
    /// No offset — spatial is same-frame; there's no recency to align.
    ///
    /// Scores on the Jaccard union `common / (common + missing + novel)`: novel observed entries (co-actives
    /// the pattern does not contain) count against the match, so a small pattern can't over-fire on a large
    /// co-activation by matching a fraction of it. Same denominator as temporal — the grouping operation is
    /// identical across the two.
    pub fn match_observed(&self, observed: &SpatialContext, merge_threshold: f64, trace: bool) -> Option<MatchResult> {

        // Pass 1: walk the known context, classifying each entry into common / missing.
        let mut common = Vec::new();
        let mut missing = Vec::new();
        let mut score: f64 = 0.0;

        for (&neuron_id, &strength) in &self.entries {
            if strength <= 0.0 { continue; }
            let entry = ContextEntry { neuron_id, distance: 0, strength };
            if observed.entries.contains_key(&neuron_id) {
                common.push(entry);
                score += strength;
            } else {
                missing.push(entry);
                score -= strength;
            }
        }

        // if there are no known context entries, there cannot be a match
        if common.is_empty() && missing.is_empty() { return None; }

        // Pass 2: walk the observed context, finding entries not in the known context.
        let mut novel = Vec::new();
        for (&neuron_id, &strength) in &observed.entries {
            if !self.entries.contains_key(&neuron_id) {
                novel.push(ContextEntry { neuron_id, distance: 0, strength });
                score -= strength;
            }
        }

        // Jaccard union denominator: common / (common + missing + novel).
        let union_size = (common.len() + missing.len() + novel.len()) as f64;
        if union_size == 0.0 { return None; }
        if trace {
            let ratio = common.len() as f64 / union_size;
            eprintln!(
                "[match] known_ctx={} common={} missing={} novel={} ratio={:.3} thr={:.3} {}",
                common.len() + missing.len(), common.len(), missing.len(), novel.len(),
                ratio, merge_threshold, if ratio >= merge_threshold { "PASS" } else { "FAIL" }
            );
        }
        if (common.len() as f64 / union_size) < merge_threshold { return None; }

        // Round to 14 decimal places to avoid floating-point precision issues
        score = (score * 1e14).round() / 1e14;

        Some(MatchResult { score, common, missing, novel })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_match_exact() {
        let mut known = SpatialContext::new();
        known.add_neuron(1, 1.0);
        known.add_neuron(2, 1.0);

        let mut observed = SpatialContext::new();
        observed.add_neuron(1, 1.0);
        observed.add_neuron(2, 1.0);

        let result = known.match_observed(&observed, 0.5, false).unwrap();
        assert_eq!(result.common.len(), 2);
        assert_eq!(result.missing.len(), 0);
        assert_eq!(result.novel.len(), 0);
    }

    #[test]
    fn test_spatial_match_partial() {
        let mut known = SpatialContext::new();
        known.add_neuron(1, 1.0);
        known.add_neuron(2, 1.0);

        let mut observed = SpatialContext::new();
        observed.add_neuron(1, 1.0);

        // 1 common, 1 missing, 0 novel → 1/2 = 0.5
        let r05 = known.match_observed(&observed, 0.5, false).unwrap();
        assert_eq!(r05.common.len(), 1);
        assert_eq!(r05.missing.len(), 1);

        // below 0.9 threshold
        assert!(known.match_observed(&observed, 0.9, false).is_none());
    }

    #[test]
    fn test_spatial_match_novel() {
        let mut known = SpatialContext::new();
        known.add_neuron(1, 1.0);

        let mut observed = SpatialContext::new();
        observed.add_neuron(1, 1.0);
        observed.add_neuron(99, 1.0);

        let result = known.match_observed(&observed, 0.5, false).unwrap();
        assert_eq!(result.common.len(), 1);
        assert_eq!(result.novel.len(), 1);
        assert_eq!(result.novel[0].neuron_id, 99);
    }
}
