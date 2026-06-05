/// Context — a set of neurons at distances with strengths.
///
/// Used both for observed context (built by Brain each frame from Memory)
/// and known contexts (stored in neuron routing tables for pattern matching).
///
/// The inner storage mirrors the JS version:
///   entries: FxHashMap<NeuronId, FxHashMap<Distance, Strength>>
use rustc_hash::{FxHashMap, FxHashSet};

use crate::types::{ContextEntry, Distance, MatchResult, NeuronId, Strength};

#[derive(Debug, Clone)]
pub struct Context {
    entries: FxHashMap<NeuronId, FxHashMap<Distance, Strength>>,
}

impl Context {
    pub fn new() -> Self {
        Self { entries: FxHashMap::default() }
    }

    /// Total number of (neuronId, distance) entries across all neurons.
    pub fn size(&self) -> usize {
        self.entries.values().map(|dm| dm.len()).sum()
    }

    /// Flatten all entries into a Vec for iteration / serialization.
    pub fn get_entries(&self) -> Vec<ContextEntry> {
        let mut result = Vec::new();
        for (&neuron_id, distance_map) in &self.entries {
            for (&distance, &strength) in distance_map {
                result.push(ContextEntry { neuron_id, distance, strength });
            }
        }
        result
    }

    /// Direct access to the inner map (for pattern matching hot paths).
    pub fn entries(&self) -> &FxHashMap<NeuronId, FxHashMap<Distance, Strength>> {
        &self.entries
    }

    /// Add a neuron at a given distance with a strength (default 1.0).
    /// Panics if the entry already exists — caller must check first.
    pub fn add_neuron(&mut self, neuron_id: NeuronId, distance: Distance, strength: Strength) {
        let distance_map = self.entries.entry(neuron_id).or_insert_with(FxHashMap::default);
        if distance_map.contains_key(&distance) {
            panic!("Context entry already exists for neuron {} at distance {}", neuron_id, distance);
        }
        distance_map.insert(distance, strength);
    }

    /// Find an entry by neuron ID and distance. Test-only.
    #[cfg(test)]
    pub fn find(&self, neuron_id: NeuronId, distance: Distance) -> Option<ContextEntry> {
        self.entries.get(&neuron_id).and_then(|dm| dm.get(&distance)).map(|&strength| ContextEntry { neuron_id, distance, strength })
    }

    /// Increment the strength of an existing entry by 1. Panics if the
    /// entry does not exist — callers must check `has_key` first.
    pub fn strengthen_neuron(&mut self, neuron_id: NeuronId, distance: Distance) {
        let distance_map = self.entries.get_mut(&neuron_id).expect("Context entry not found for strengthening");
        let strength = distance_map.get_mut(&distance).expect("Context entry not found for strengthening");
        *strength += 1.0;
    }

    /// Remove an entry explicitly.
    pub fn remove(&mut self, neuron_id: NeuronId, distance: Distance) {
        let distance_map = self.entries.get_mut(&neuron_id).expect("Context entry not found for deletion");
        distance_map.remove(&distance);
        if distance_map.is_empty() {
            self.entries.remove(&neuron_id);
        }
    }

    /// Check if an entry exists at the given neuron ID and distance.
    pub fn has_key(&self, neuron_id: NeuronId, distance: Distance) -> bool {
        self.entries.get(&neuron_id).map_or(false, |dm| dm.contains_key(&distance))
    }

    /// Score a known context entry against the observed context.
    /// Returns full strength for exact match, partial credit for distance mismatch, negative for missing.
    fn get_match_score(strength: Strength, distance: Distance, observed_distances: Option<&FxHashMap<Distance, Strength>>) -> f64 {
        match observed_distances {
            Some(od) if od.contains_key(&distance) => strength,
            None => -strength,
            Some(od) => {
                let mut min_delta = f64::INFINITY;
                for &observed_distance in od.keys() {
                    let delta = (observed_distance as i64 - distance as i64).unsigned_abs() as f64;
                    if delta < min_delta { min_delta = delta; }
                }
                strength / (1.0 + min_delta)
            }
        }
    }

    /// Score a novel observed entry. Returns true if the neuron has a partial match
    /// in the known context (already accounted for), otherwise returns false.
    fn has_partial_match(distance: Distance, known_distances: Option<&FxHashMap<Distance, Strength>>) -> bool {
        if let Some(kd) = known_distances {
            for (&d, &strength) in kd {
                if d != distance && strength > 0.0 { return true; }
            }
        }
        false
    }

    /// Match this known context against an observed context.
    /// Returns match result with score, or None if below threshold.
    /// Uses effective strengths (with lazy decay applied) for scoring.
    /// * `observed` — the observed context to match against
    /// * `offset` — the parent's active age (shifts pattern distances to absolute)
    /// * `merge_threshold` — minimum required percentage for merge (0.0–1.0)
    /// * `exclude_ids` — optional set of observed neuron ids to mask out of
    ///        scoring (e.g. brand-new neurons that shouldn't count as unexplained novel entries)
    pub fn match_observed(&self, observed: &Context, offset: Distance, merge_threshold: f64, exclude_ids: Option<&FxHashSet<NeuronId>>) -> Option<MatchResult> {

        // Pass 1: walk the known context, classifying each entry into common/missing relative to observed.
        let mut common = Vec::new();
        let mut missing = Vec::new();
        let mut total_count: usize = 0;
        let mut score: f64 = 0.0;

        for (&neuron_id, distance_map) in &self.entries {
            let observed_distances = observed.entries.get(&neuron_id);
            for (&distance, &strength) in distance_map {
                if strength <= 0.0 { continue; }
                total_count += 1;

                let absolute_distance = distance + offset;
                let entry = ContextEntry { neuron_id, distance, strength };
                let is_common = observed_distances.map_or(false, |od| od.contains_key(&absolute_distance));
                if is_common { common.push(entry); }
                else { missing.push(entry); }

                score += Self::get_match_score(strength, absolute_distance, observed_distances);
            }
        }

        // if there are no known context entries, there cannot be a match
        if total_count == 0 { return None; }

        // Pass 2: walk the observed context, finding entries with no counterpart in the known context.
        // Done BEFORE the threshold gate so the gate can be symmetric — a pattern shouldn't fire on a
        // frame that contains many entries the pattern doesn't account for, no matter how much of the
        // pattern's own context is covered. The gate of common/total counts only pattern-side coverage,
        // which lets a small pattern over-fire on any frame containing a fraction of its entries.
        // Spatial entries sit at pattern_distance=0; temporal at >=1; we accept >=0 to cover both.
        let mut novel = Vec::new();
        for (&neuron_id, distance_map) in &observed.entries {
            if exclude_ids.map_or(false, |exc| exc.contains(&neuron_id)) { continue; }
            let known_distances = self.entries.get(&neuron_id);
            for (&absolute_distance, &strength) in distance_map {
                let pattern_distance = absolute_distance as i64 - offset as i64;
                if pattern_distance < 0 { continue; }
                let pattern_distance = pattern_distance as Distance;

                let is_known = known_distances.map_or(false, |kd| kd.get(&pattern_distance).map_or(false, |&s| s > 0.0));
                if !is_known && !Self::has_partial_match(pattern_distance, known_distances) {
                    novel.push(ContextEntry { neuron_id, distance: pattern_distance, strength });
                    score -= strength;
                }
            }
        }

        // Symmetric (Jaccard) match: common / (common + missing + novel) >= merge_threshold.
        // Treats the pattern AND the observed frame on equal footing — a pattern only fires when its
        // context entries dominate the union with what's observed. Asymmetric coverage (small pattern
        // matching a huge observed frame on a partial subset) is what drives runaway minting because
        // such a pattern fires on every overlapping frame and its predictions miss the rest.
        let union_size = (common.len() + missing.len() + novel.len()) as f64;
        if union_size == 0.0 { return None; }
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
    fn test_add_and_find() {
        let mut ctx = Context::new();
        ctx.add_neuron(1, 2, 1.0);
        assert_eq!(ctx.size(), 1);
        assert!(ctx.has_key(1, 2));
        assert!(!ctx.has_key(1, 3));

        let entry = ctx.find(1, 2).unwrap();
        assert_eq!(entry.neuron_id, 1);
        assert_eq!(entry.distance, 2);
        assert_eq!(entry.strength, 1.0);
    }

    #[test]
    #[should_panic(expected = "already exists")]
    fn test_add_duplicate_panics() {
        let mut ctx = Context::new();
        ctx.add_neuron(1, 2, 1.0);
        ctx.add_neuron(1, 2, 1.0);
    }

    #[test]
    fn test_remove() {
        let mut ctx = Context::new();
        ctx.add_neuron(1, 2, 5.0);
        ctx.add_neuron(1, 3, 5.0);
        ctx.remove(1, 2);
        assert!(!ctx.has_key(1, 2));
        assert!(ctx.has_key(1, 3));
    }

    #[test]
    fn test_get_entries() {
        let mut ctx = Context::new();
        ctx.add_neuron(10, 1, 2.0);
        ctx.add_neuron(10, 3, 4.0);
        ctx.add_neuron(20, 1, 1.0);

        let entries = ctx.get_entries();
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn test_match_exact() {
        // Known context: neuron 1 at distance 1, neuron 2 at distance 2
        let mut known = Context::new();
        known.add_neuron(1, 1, 1.0);
        known.add_neuron(2, 2, 1.0);

        // Observed context: same neurons at absolute distances (offset=0)
        let mut observed = Context::new();
        observed.add_neuron(1, 1, 1.0);
        observed.add_neuron(2, 2, 1.0);

        let result = known.match_observed(&observed, 0, 0.5, None).unwrap();
        assert_eq!(result.common.len(), 2);
        assert_eq!(result.missing.len(), 0);
        assert_eq!(result.novel.len(), 0);
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_match_below_threshold_returns_none() {
        let mut known = Context::new();
        known.add_neuron(1, 1, 1.0);
        known.add_neuron(2, 2, 1.0);

        // Observed only has neuron 1 — 50% match, below 0.9 threshold
        let mut observed = Context::new();
        observed.add_neuron(1, 1, 1.0);

        let result = known.match_observed(&observed, 0, 0.9, None);
        assert!(result.is_none());
    }

    #[test]
    fn test_match_with_offset() {
        // Known context: neuron 1 at pattern-relative distance 1
        let mut known = Context::new();
        known.add_neuron(1, 1, 1.0);

        // Observed: neuron 1 at absolute distance 4 (offset=3, so pattern-relative=1)
        let mut observed = Context::new();
        observed.add_neuron(1, 4, 1.0);

        let result = known.match_observed(&observed, 3, 0.5, None).unwrap();
        assert_eq!(result.common.len(), 1);
    }

    #[test]
    fn test_match_novel_detection() {
        let mut known = Context::new();
        known.add_neuron(1, 1, 1.0);

        // Observed has neuron 1 (common) + neuron 99 at absolute distance 3 (novel)
        let mut observed = Context::new();
        observed.add_neuron(1, 1, 1.0);
        observed.add_neuron(99, 3, 1.0);

        let result = known.match_observed(&observed, 0, 0.5, None).unwrap();
        assert_eq!(result.common.len(), 1);
        assert_eq!(result.novel.len(), 1);
        assert_eq!(result.novel[0].neuron_id, 99);
        // Novel distance should be pattern-relative (3 - 0 = 3)
        assert_eq!(result.novel[0].distance, 3);
    }

    #[test]
    fn test_match_excludes_masked_ids() {
        let mut known = Context::new();
        known.add_neuron(1, 1, 1.0);

        let mut observed = Context::new();
        observed.add_neuron(1, 1, 1.0);
        observed.add_neuron(99, 3, 1.0); // would be novel

        let mut exclude = FxHashSet::default();
        exclude.insert(99);

        let result = known.match_observed(&observed, 0, 0.5, Some(&exclude)).unwrap();
        assert_eq!(result.novel.len(), 0); // 99 was masked out
    }
}
