//! Deaths (R38): the ledger and the subtree.


use crate::machine::*;
use crate::neuron::*;
use crate::types::*;

impl Machine {
    // ───────────────────────────── deaths (R38) ─────────────────────────────

    pub(crate) fn handle_retired(&mut self, owner: NeuronId, mut r: Retired) {
        // the ledger holds the pattern, not a handle to it: what it fired stays expandable until every
        // window over it has closed (R38)
        if let Some(line) = r.line.take() {
            let until = self.frame_no + self.max_reach_t as i64 + 1;
            self.retired_lines.insert((owner, r.pattern), (line, until));
        }
        if let Some(c) = r.child {
            if (c as usize) < self.thalamus.len() {
                self.thalamus[c].parents.retain(|p| *p != (owner, r.pattern));
                // merged: the function that took its place points at the child now (§5.5)
                if let Some(h) = r.heir { if !self.thalamus[c].parents.contains(&(owner, h)) { self.thalamus[c].parents.push((owner, h)); } }
                if self.thalamus[c].parents.is_empty() && !self.ledger.contains(&c) { self.ledger.push(c); }
            }
        }
    }

    /// Delete every child on the ledger whose last window has closed (R38); a deleted neuron's subtree
    /// goes on the ledger as it is deleted and is taken in the same pass.
    pub(crate) fn reap(&mut self) {
        let fno = self.frame_no;
        let mut keep = Vec::new();
        let mut pending = std::mem::take(&mut self.ledger);
        while !pending.is_empty() {
            for c in pending.drain(..) {
                if !self.thalamus[c].alive { continue; }
                if !self.thalamus[c].parents.is_empty() { continue; }
                let open = self.frames.iter().any(|f| f.acts.iter().any(|a| a.neuron == c && (fno - f.no) <= a.reach[0] as i64));
                if open { keep.push(c); continue; }
                self.delete_neuron(c);
            }
            pending = std::mem::take(&mut self.ledger);
        }
        self.ledger = keep;
        self.retired_lines.retain(|_, (_, until)| *until >= fno);
    }

    pub(crate) fn delete_neuron(&mut self, c: NeuronId) {
        if !self.thalamus[c].alive { return; }
        self.thalamus[c].alive = false;
        self.alive -= 1;
        self.definitions.remove(&c);
        self.def_classes.remove(&c);
        self.redefined.remove(&c);
        self.won.retain(|w| w.2 != c && w.0 != c);
        let key = self.thalamus[c].key.clone();
        if self.index.get(&key) == Some(&c) { self.index.remove(&key); }
        if self.trace { eprintln!("[death f{}] #{}", self.frame_no, c); }
        // its subtree: its own patterns' children with no parent outside
        let functions: Vec<Function> = std::mem::take(&mut self.thalamus[c].functions);
        for f in functions {
            self.handle_retired(c, Retired { pattern: f.id, child: f.child, line: Some(f), heir: None });
        }
        self.thalamus[c].classes.clear();
        self.thalamus[c].history.clear();
        self.thalamus[c].connections.clear();
        // scrub it from every pattern and saved activation that names it
        for (n, r) in self.thalamus.delete_neighbor(c) {
            if self.trace { eprintln!("[cascade f{} #{}] p{} child={:?} lost #{}", self.frame_no, n, r.pattern, r.child, c); }
            self.handle_retired(n, r);
        }
        for f in self.frames.iter_mut() {
            for a in f.acts.iter_mut() {
                a.inferences.retain(|i| i.target != c);
            }
        }
    }

}
