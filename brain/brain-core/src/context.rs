/// Context — observed co-activation / sequence structure that patterns match against.
///
/// Two distinct types because spatial and temporal have fundamentally different shapes:
///
/// * `SpatialContext` is a flat set of neurons at the current frame with strengths.
///   No distance dimension — spatial is same-frame co-activation.
/// * `TemporalContext` is a set of neurons at past distances with strengths.
///   Distance is the temporal recency: how many frames ago each context neuron fired.
///
/// Both are used in two roles:
/// 1. **Observed context** — built fresh each frame from the active set at that level.
/// 2. **Known context** — stored in neuron routing tables; what a pattern needs to see to match.
///
/// `match_observed` is a Jaccard-style score: `common / (common + missing + novel) >= threshold`.
/// Treats the pattern and the observed frame on equal footing — small patterns don't get to
/// over-fire on huge observed frames by matching a fraction of their own entries.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::types::{ContextEntry, Distance, MatchMode, MatchResult, NeuronId, Strength};

// ── Spatial ──────────────────────────────────────────────────────────────────

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
    pub fn match_observed(&self, observed: &SpatialContext, merge_threshold: f64) -> Option<MatchResult> {

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

        // Symmetric (Jaccard) match: common / (common + missing + novel) >= merge_threshold.
        let union_size = (common.len() + missing.len() + novel.len()) as f64;
        if union_size == 0.0 { return None; }
        if (common.len() as f64 / union_size) < merge_threshold { return None; }

        // Round to 14 decimal places to avoid floating-point precision issues
        score = (score * 1e14).round() / 1e14;

        Some(MatchResult { score, common, missing, novel })
    }
}

// ── Temporal ─────────────────────────────────────────────────────────────────

/// Temporal context: set of neurons at past distances with strengths.
/// Distance is the recency: how many frames ago each context neuron fired.
#[derive(Debug, Clone)]
pub struct TemporalContext {
    entries: FxHashMap<NeuronId, FxHashMap<Distance, Strength>>,
}

impl TemporalContext {
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

    /// Add a neuron at a given distance with a strength.
    /// Panics if the entry already exists — caller must check first.
    pub fn add_neuron(&mut self, neuron_id: NeuronId, distance: Distance, strength: Strength) {
        let distance_map = self.entries.entry(neuron_id).or_insert_with(FxHashMap::default);
        if distance_map.contains_key(&distance) {
            panic!("TemporalContext entry already exists for neuron {} at distance {}", neuron_id, distance);
        }
        distance_map.insert(distance, strength);
    }

    /// Find an entry by neuron ID and distance. Test-only.
    #[cfg(test)]
    pub fn find(&self, neuron_id: NeuronId, distance: Distance) -> Option<ContextEntry> {
        self.entries.get(&neuron_id).and_then(|dm| dm.get(&distance)).map(|&strength| ContextEntry { neuron_id, distance, strength })
    }

    /// Increment the strength of an existing entry by 1.
    /// Panics if the entry does not exist — callers must check `has_key` first.
    pub fn strengthen_neuron(&mut self, neuron_id: NeuronId, distance: Distance) {
        let distance_map = self.entries.get_mut(&neuron_id).expect("TemporalContext entry not found for strengthening");
        let strength = distance_map.get_mut(&distance).expect("TemporalContext entry not found for strengthening");
        *strength += 1.0;
    }

    /// Remove an entry explicitly.
    pub fn remove(&mut self, neuron_id: NeuronId, distance: Distance) {
        let distance_map = self.entries.get_mut(&neuron_id).expect("TemporalContext entry not found for deletion");
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

    /// Match this known temporal context against an observed temporal context.
    /// Returns match result with score, or None if below threshold.
    /// * `observed` — the observed context to match against
    /// * `offset` — the parent's active age (shifts pattern distances to absolute)
    /// * `merge_threshold` — minimum required percentage for merge (0.0–1.0)
    /// * `match_mode` — how the threshold denominator is formed (containment vs jaccard)
    /// * `exclude_ids` — optional set of observed neuron ids to mask out of scoring
    ///        (e.g. brand-new neurons that shouldn't count as unexplained novel entries)
    pub fn match_observed(&self, observed: &TemporalContext, offset: Distance, merge_threshold: f64, match_mode: MatchMode, exclude_ids: Option<&FxHashSet<NeuronId>>) -> Option<MatchResult> {

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
        let mut novel = Vec::new();
        for (&neuron_id, distance_map) in &observed.entries {
            if exclude_ids.map_or(false, |exc| exc.contains(&neuron_id)) { continue; }
            let known_distances = self.entries.get(&neuron_id);
            for (&absolute_distance, &strength) in distance_map {
                let pattern_distance = absolute_distance as i64 - offset as i64;
                // Temporal context neurons must be strictly older than the parent (pattern-relative distance >= 1).
                // Same-frame distance 0 observed entry is a spatial co-activation, not a temporal predecessor
                if pattern_distance < 1 { continue; }
                let pattern_distance = pattern_distance as Distance;

                let is_known = known_distances.map_or(false, |kd| kd.get(&pattern_distance).map_or(false, |&s| s > 0.0));
                if !is_known && !Self::has_partial_match(pattern_distance, known_distances) {
                    novel.push(ContextEntry { neuron_id, distance: pattern_distance, strength });
                    score -= strength;
                }
            }
        }

        // Threshold gate. Containment divides by the known-pattern size (common + missing =
        // total_count); Jaccard divides by the union (also counting novel observed entries). Both
        // apply the same novel score penalty above — only the gate denominator differs.
        let denom = match match_mode {
            MatchMode::Containment => total_count as f64,
            MatchMode::Jaccard => (common.len() + missing.len() + novel.len()) as f64,
        };
        if denom == 0.0 { return None; }
        if (common.len() as f64 / denom) < merge_threshold { return None; }

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
        let mut ctx = TemporalContext::new();
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
        let mut ctx = TemporalContext::new();
        ctx.add_neuron(1, 2, 1.0);
        ctx.add_neuron(1, 2, 1.0);
    }

    #[test]
    fn test_remove() {
        let mut ctx = TemporalContext::new();
        ctx.add_neuron(1, 2, 5.0);
        ctx.add_neuron(1, 3, 5.0);
        ctx.remove(1, 2);
        assert!(!ctx.has_key(1, 2));
        assert!(ctx.has_key(1, 3));
    }

    #[test]
    fn test_get_entries() {
        let mut ctx = TemporalContext::new();
        ctx.add_neuron(10, 1, 2.0);
        ctx.add_neuron(10, 3, 4.0);
        ctx.add_neuron(20, 1, 1.0);

        let entries = ctx.get_entries();
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn test_match_exact() {
        let mut known = TemporalContext::new();
        known.add_neuron(1, 1, 1.0);
        known.add_neuron(2, 2, 1.0);

        let mut observed = TemporalContext::new();
        observed.add_neuron(1, 1, 1.0);
        observed.add_neuron(2, 2, 1.0);

        let result = known.match_observed(&observed, 0, 0.5, MatchMode::Jaccard, None).unwrap();
        assert_eq!(result.common.len(), 2);
        assert_eq!(result.missing.len(), 0);
        assert_eq!(result.novel.len(), 0);
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_match_below_threshold_returns_none() {
        let mut known = TemporalContext::new();
        known.add_neuron(1, 1, 1.0);
        known.add_neuron(2, 2, 1.0);

        let mut observed = TemporalContext::new();
        observed.add_neuron(1, 1, 1.0);

        let result = known.match_observed(&observed, 0, 0.9, MatchMode::Jaccard, None);
        assert!(result.is_none());
    }

    #[test]
    fn test_match_with_offset() {
        let mut known = TemporalContext::new();
        known.add_neuron(1, 1, 1.0);

        let mut observed = TemporalContext::new();
        observed.add_neuron(1, 4, 1.0);

        let result = known.match_observed(&observed, 3, 0.5, MatchMode::Jaccard, None).unwrap();
        assert_eq!(result.common.len(), 1);
    }

    #[test]
    fn test_match_novel_detection() {
        let mut known = TemporalContext::new();
        known.add_neuron(1, 1, 1.0);

        let mut observed = TemporalContext::new();
        observed.add_neuron(1, 1, 1.0);
        observed.add_neuron(99, 3, 1.0);

        let result = known.match_observed(&observed, 0, 0.5, MatchMode::Jaccard, None).unwrap();
        assert_eq!(result.common.len(), 1);
        assert_eq!(result.novel.len(), 1);
        assert_eq!(result.novel[0].neuron_id, 99);
        assert_eq!(result.novel[0].distance, 3);
    }

    #[test]
    fn test_match_excludes_masked_ids() {
        let mut known = TemporalContext::new();
        known.add_neuron(1, 1, 1.0);

        let mut observed = TemporalContext::new();
        observed.add_neuron(1, 1, 1.0);
        observed.add_neuron(99, 3, 1.0);

        let mut exclude = FxHashSet::default();
        exclude.insert(99);

        let result = known.match_observed(&observed, 0, 0.5, MatchMode::Jaccard, Some(&exclude)).unwrap();
        assert_eq!(result.novel.len(), 0);
    }

    #[test]
    fn test_spatial_match_exact() {
        let mut known = SpatialContext::new();
        known.add_neuron(1, 1.0);
        known.add_neuron(2, 1.0);

        let mut observed = SpatialContext::new();
        observed.add_neuron(1, 1.0);
        observed.add_neuron(2, 1.0);

        let result = known.match_observed(&observed, 0.5).unwrap();
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
        let r05 = known.match_observed(&observed, 0.5).unwrap();
        assert_eq!(r05.common.len(), 1);
        assert_eq!(r05.missing.len(), 1);

        // below 0.9 threshold
        assert!(known.match_observed(&observed, 0.9).is_none());
    }

    #[test]
    fn test_spatial_match_novel() {
        let mut known = SpatialContext::new();
        known.add_neuron(1, 1.0);

        let mut observed = SpatialContext::new();
        observed.add_neuron(1, 1.0);
        observed.add_neuron(99, 1.0);

        let result = known.match_observed(&observed, 0.5).unwrap();
        assert_eq!(result.common.len(), 1);
        assert_eq!(result.novel.len(), 1);
        assert_eq!(result.novel[0].neuron_id, 99);
    }
}
