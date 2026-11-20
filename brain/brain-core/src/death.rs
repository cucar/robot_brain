//! Decommissioning (R38): an interneuron dies when its last function has left the dictionary and its last
//! activation, or its subtree's, has left the board; every function that names it leaves with it, and so on up.

use crate::machine::*;
use crate::types::*;

impl Machine {
    /// Decommission every interneuron on the ledger that is due: no function of it left, and no activation of it
    /// or of its subtree standing on the board (D9). Its subtree goes in the same pass (R38).
    pub(crate) fn reap(&mut self) {
        let fno = self.frame_no;
        let mut keep = Vec::new();
        let mut pending = std::mem::take(&mut self.ledger);
        while !pending.is_empty() {
            for c in pending.drain(..) {
                if !self.thalamus[c].alive || !self.dict.of_inter(c).is_empty() { self.ledger_due.remove(&c); continue; }
                // nothing of its subtree can have left the board before the frame it was found standing until
                if self.ledger_due.get(&c).map_or(false, |d| *d > fno) { keep.push(c); continue; }
                // it waits while an activation of it, or of any neuron in its subtree, still stands on the board
                let latest = self.latest_standing_over(c, fno);
                if let Some(t) = latest { self.ledger_due.insert(c, t + self.span_s as Frame + 1); keep.push(c); continue; }
                self.ledger_due.remove(&c);
                self.decommission(c);
            }
            pending = std::mem::take(&mut self.ledger);
        }
        self.ledger = keep;
        self.retired_funcs.retain(|_, (_, until)| *until >= fno);
    }

    /// The latest frame at which an activation of `c` or of any neuron in its subtree (every interneuron of a
    /// function that names `c`, and so on up) stands on the board. Read from what stands: an activation counts where
    /// its neuron reaches `c` going down, through the nodes of the functions it is the interneuron of (R38).
    fn latest_standing_over(&self, c: NeuronId, fno: Frame) -> Option<Frame> {
        let mut by_n: rustc_hash::FxHashMap<NeuronId, Frame> = rustc_hash::FxHashMap::default();
        for a in self.frames.iter().flat_map(|f| f.acts.iter()) {
            if !self.stands(a, fno) { continue; }
            let e = by_n.entry(a.neuron).or_insert(a.pos[0] as Frame);
            if (a.pos[0] as Frame) > *e { *e = a.pos[0] as Frame; }
        }
        // the neurons known to reach `c` going down, and those known not to
        let mut yes: rustc_hash::FxHashSet<NeuronId> = rustc_hash::FxHashSet::default();
        let mut no: rustc_hash::FxHashSet<NeuronId> = rustc_hash::FxHashSet::default();
        yes.insert(c);
        let mut best: Option<Frame> = None;
        for (&n, &t) in &by_n {
            if best.map_or(false, |b| b >= t) { continue; }
            if self.reaches_down(n, &mut yes, &mut no) { best = Some(t); }
        }
        best
    }

    /// Whether `n` reaches a neuron of `yes` going down: it is one, or the interneuron of a function one of whose
    /// nodes is one, and so on down. What it learns is kept in `yes` and `no`.
    fn reaches_down(&self, n: NeuronId, yes: &mut rustc_hash::FxHashSet<NeuronId>, no: &mut rustc_hash::FxHashSet<NeuronId>) -> bool {
        if yes.contains(&n) { return true; }
        if no.contains(&n) { return false; }
        // depth first, each neuron once per question
        let mut seen: rustc_hash::FxHashSet<NeuronId> = rustc_hash::FxHashSet::default();
        let mut stack = vec![n];
        seen.insert(n);
        let mut found = false;
        while let Some(m) = stack.pop() {
            if yes.contains(&m) { found = true; break; }
            if no.contains(&m) { continue; }
            for &id in self.dict.of_inter(m) {
                let Some(f) = self.dict.get(id) else { continue };
                if f.inter != Some(m) { continue; }
                for &s in &f.nodes {
                    let k = self.dict.neuron_of(s);
                    if seen.insert(k) { stack.push(k); }
                }
            }
        }
        if found { yes.insert(n); } else { for m in seen { no.insert(m); } }
        found
    }

    pub(crate) fn decommission(&mut self, c: NeuronId) {
        if !self.thalamus[c].alive { return; }
        self.thalamus[c].alive = false;
        self.alive -= 1;
        self.last_func.remove(&c);
        if self.trace { eprintln!("[decommission f{}] #{}", self.frame_no, c); }
        // every function that names it can no longer hold, and leaves (its subtree)
        for id in self.dict.naming(c) {
            if let Some(f) = self.dict.remove(id) {
                if self.trace_retire { eprintln!("[retire f{}] {} | it names #{}, decommissioned", self.frame_no, self.dict.describe(&f), c); }
                self.retire(f);
            }
        }
        let n = &mut self.thalamus[c];
        n.history.clear();
        n.index.clear();
        n.now_index.clear();
        n.connections.clear();
        let dead = self.dict.drop_syms(c);
        self.thalamus.forget(&dead);
        for f in self.frames.iter_mut() {
            for a in f.acts.iter_mut() { a.inferences.retain(|i| !dead.contains(&i.target)); }
        }
    }
}
