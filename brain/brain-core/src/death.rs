//! Decommissioning (R38): an interneuron dies when its last function has left the dictionary and the last
//! window over it has closed; every function that names it leaves with it, and so on up.

use crate::machine::*;
use crate::types::*;

impl Machine {
    /// Decommission every interneuron on the ledger that is due: no function of it left, and no open
    /// activation of it. Its subtree goes in the same pass (R38).
    pub(crate) fn reap(&mut self) {
        let fno = self.frame_no;
        let mut keep = Vec::new();
        let mut pending = std::mem::take(&mut self.ledger);
        while !pending.is_empty() {
            for c in pending.drain(..) {
                if !self.thalamus[c].alive || !self.dict.of_inter(c).is_empty() { continue; }
                let open = self.frames.iter().any(|f| f.acts.iter().any(|a| a.neuron == c && !a.gone && (fno - f.no) <= a.reach[0] as i64));
                if open { keep.push(c); continue; }
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
        let naming: Vec<_> = self.dict.naming.get(&c).cloned().unwrap_or_default();
        for id in naming { self.retire_func(id); }
        let anchored: Vec<_> = self.dict.anchored(c).to_vec();
        for id in anchored { self.retire_func(id); }
        let n = &mut self.thalamus[c];
        n.history.clear();
        n.index.clear();
        n.connections.clear();
        self.thalamus.forget(c);
        for f in self.frames.iter_mut() {
            for a in f.acts.iter_mut() { a.inferences.retain(|i| i.target != c); }
        }
    }
}
