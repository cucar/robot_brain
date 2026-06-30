/// Footprints — the neighborhood primitive that lets corrections be coordinate-less.
///
/// A neuron's footprint is the set of base sensory/action neurons it ultimately covers.
/// A base neuron has `footprint = {itself}`; a correction (any distance) has
/// `footprint = ⋃ of its constituents' footprints`, computed once at mint.
/// Two neurons are neighbors iff their footprints touch in the base neighbor graph — there is a base
/// neuron in one footprint that is a spatial-neighbor of (or equal to) a base neuron in the other.
///
/// The set is stored as a dense bitset over base-neuron bit indices (assigned in allocation order by
/// the Thalamus), so union is a word-wise OR and the touch test is a word-wise AND.

use rustc_hash::FxHashMap;

/// A bitset over base-neuron bit indices.
///
/// Indices are the dense positions the Thalamus assigns to base neurons in allocation order, NOT raw
/// neuron ids — base neuron ids are interleaved with pattern ids, so a direct-id bitset would be sparse.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Footprint {
    /// 64-bit words; bit `i` lives at `words[i / 64]`, position `i % 64`. Trailing words are never all-zero
    /// unless the whole set is empty — `union_in_place` grows and `set_bit` grows, nothing shrinks.
    words: Vec<u64>,
}

impl Footprint {
    /// An empty footprint.
    pub fn new() -> Self {
        Self { words: Vec::new() }
    }

    /// A footprint holding exactly one base bit.
    pub fn single(bit: u32) -> Self {
        let mut fp = Self::new();
        fp.set_bit(bit);
        fp
    }

    /// Grow the word vector so index `bit` is addressable.
    fn ensure_word(&mut self, bit: u32) {
        let word = (bit / 64) as usize;
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
    }

    /// Set the base bit at `bit`.
    pub fn set_bit(&mut self, bit: u32) {
        self.ensure_word(bit);
        self.words[(bit / 64) as usize] |= 1u64 << (bit % 64);
    }

    /// Whether the base bit at `bit` is set. Inspection helper used by tests and diagnostics.
    #[allow(dead_code)]
    pub fn get_bit(&self, bit: u32) -> bool {
        let word = (bit / 64) as usize;
        word < self.words.len() && (self.words[word] & (1u64 << (bit % 64))) != 0
    }

    /// Whether no bits are set.
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Number of base bits set. Inspection helper used by tests and diagnostics.
    #[allow(dead_code)]
    pub fn count_ones(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    /// OR `other` into this footprint (set union in place).
    pub fn union_in_place(&mut self, other: &Footprint) {
        if other.words.len() > self.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        for (w, &o) in self.words.iter_mut().zip(other.words.iter()) {
            *w |= o;
        }
    }

    /// Whether the two footprints share at least one base bit (set intersection is non-empty).
    pub fn intersects(&self, other: &Footprint) -> bool {
        let n = self.words.len().min(other.words.len());
        for i in 0..n {
            if self.words[i] & other.words[i] != 0 {
                return true;
            }
        }
        false
    }

    /// Iterate the base bit indices that are set, ascending.
    pub fn iter_bits(&self) -> impl Iterator<Item = u32> + '_ {
        self.words.iter().enumerate().flat_map(|(wi, &word)| {
            (0..64).filter_map(move |b| {
                if word & (1u64 << b) != 0 {
                    Some(wi as u32 * 64 + b)
                } else {
                    None
                }
            })
        })
    }
}

/// Rebuild a correction's footprint by memoized recursion over its constituent graph.
///
/// `base_footprints` holds the already-known footprints of base neurons (`{self}`) — the recursion
/// grounds out there. `constituents` maps each correction to the neurons whose footprints union into
/// its own (its parent plus its context set). The recursion is dependency-order-free: it follows the
/// constituent edges and memoizes, so it does not need stored levels to sort by.
///
/// `in_progress` guards against a malformed cyclic constituent graph — a neuron already on the
/// recursion stack contributes nothing rather than looping forever.
pub fn rebuild_footprint(
    id: u64,
    constituents: &FxHashMap<u64, Vec<u64>>,
    base_footprints: &FxHashMap<u64, Footprint>,
    memo: &mut FxHashMap<u64, Footprint>,
    in_progress: &mut std::collections::HashSet<u64>,
) -> Footprint {
    if let Some(fp) = memo.get(&id) {
        return fp.clone();
    }
    if let Some(fp) = base_footprints.get(&id) {
        memo.insert(id, fp.clone());
        return fp.clone();
    }
    if in_progress.contains(&id) {
        return Footprint::new();
    }
    in_progress.insert(id);

    let mut fp = Footprint::new();
    if let Some(cons) = constituents.get(&id) {
        for &c in cons {
            let cfp = rebuild_footprint(c, constituents, base_footprints, memo, in_progress);
            fp.union_in_place(&cfp);
        }
    }

    in_progress.remove(&id);
    memo.insert(id, fp.clone());
    fp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_and_get() {
        let fp = Footprint::single(5);
        assert!(fp.get_bit(5));
        assert!(!fp.get_bit(4));
        assert_eq!(fp.count_ones(), 1);
        assert!(!fp.is_empty());
        assert!(Footprint::new().is_empty());
    }

    #[test]
    fn test_set_bit_grows_across_words() {
        let mut fp = Footprint::new();
        fp.set_bit(0);
        fp.set_bit(130); // forces a third word
        assert!(fp.get_bit(0));
        assert!(fp.get_bit(130));
        assert_eq!(fp.count_ones(), 2);
    }

    #[test]
    fn test_union_in_place() {
        let mut a = Footprint::single(1);
        let mut b = Footprint::single(200);
        b.set_bit(2);
        a.union_in_place(&b);
        assert!(a.get_bit(1));
        assert!(a.get_bit(2));
        assert!(a.get_bit(200));
        assert_eq!(a.count_ones(), 3);
    }

    #[test]
    fn test_intersects() {
        let a = {
            let mut f = Footprint::single(1);
            f.set_bit(3);
            f
        };
        let b = Footprint::single(3);
        let c = Footprint::single(2);
        assert!(a.intersects(&b));
        assert!(b.intersects(&a));
        assert!(!a.intersects(&c));
        assert!(!a.intersects(&Footprint::new()));
    }

    #[test]
    fn test_iter_bits_ascending() {
        let mut fp = Footprint::new();
        for &b in &[200u32, 1, 64, 5] {
            fp.set_bit(b);
        }
        let bits: Vec<u32> = fp.iter_bits().collect();
        assert_eq!(bits, vec![1, 5, 64, 200]);
    }

    #[test]
    fn test_rebuild_footprint_unions_constituents() {
        // bases 10, 11, 12 with bits 0, 1, 2
        let mut base = FxHashMap::default();
        base.insert(10u64, Footprint::single(0));
        base.insert(11u64, Footprint::single(1));
        base.insert(12u64, Footprint::single(2));

        // correction 100: parent 10 + context 11 → bits {0,1}
        // correction 101: parent 100 + context 12 → bits {0,1,2}
        let mut cons: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
        cons.insert(100, vec![10, 11]);
        cons.insert(101, vec![100, 12]);

        let mut memo = FxHashMap::default();
        let mut in_progress = std::collections::HashSet::new();
        let fp100 = rebuild_footprint(100, &cons, &base, &mut memo, &mut in_progress);
        assert_eq!(fp100.count_ones(), 2);
        assert!(fp100.get_bit(0) && fp100.get_bit(1));

        let fp101 = rebuild_footprint(101, &cons, &base, &mut memo, &mut in_progress);
        assert_eq!(fp101.count_ones(), 3);
        assert!(fp101.get_bit(0) && fp101.get_bit(1) && fp101.get_bit(2));
    }

    #[test]
    fn test_rebuild_footprint_tolerates_cycle() {
        let base = FxHashMap::default();
        let mut cons: FxHashMap<u64, Vec<u64>> = FxHashMap::default();
        cons.insert(1, vec![2]);
        cons.insert(2, vec![1]); // cycle — must terminate, not overflow the stack
        let mut memo = FxHashMap::default();
        let mut in_progress = std::collections::HashSet::new();
        let fp = rebuild_footprint(1, &cons, &base, &mut memo, &mut in_progress);
        assert!(fp.is_empty());
    }
}
