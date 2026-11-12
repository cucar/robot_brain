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
                let mut subtree = vec![c];
                let mut inside: rustc_hash::FxHashSet<NeuronId> = rustc_hash::FxHashSet::default();
                inside.insert(c);
                let mut k = 0;
                while k < subtree.len() {
                    for id in self.dict.naming(subtree[k]) {
                        if let Some(n) = self.dict.get(id).and_then(|f| f.inter) { if inside.insert(n) { subtree.push(n); } }
                    }
                    k += 1;
                }
                let latest = self.frames.iter().flat_map(|f| f.acts.iter()).filter(|a| inside.contains(&a.neuron) && self.stands(a, fno)).map(|a| a.pos[0] as Frame).max();
                if let Some(t) = latest { self.ledger_due.insert(c, t + self.span_s as Frame + 1); keep.push(c); continue; }
                self.ledger_due.remove(&c);
                self.decommission(c);
            }
            pending = std::mem::take(&mut self.ledger);
        }
        self.ledger = keep;
        self.retired_funcs.retain(|_, (_, until)| *until >= fno);
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
        n.connections.clear();
        let dead = self.dict.drop_syms(c);
        self.thalamus.forget(&dead);
        for f in self.frames.iter_mut() {
            for a in f.acts.iter_mut() { a.inferences.retain(|i| !dead.contains(&i.target)); }
        }
    }
}
