//! Deaths (R38): the ledger and the subtree.


use crate::machine::*;
use crate::neuron::*;
use crate::types::*;

impl Machine {
    // ───────────────────────────── deaths (R38) ─────────────────────────────

    pub(crate) fn handle_retired(&mut self, owner: NeuronId, r: Retired) {
        if let Some(c) = r.child {
            if (c as usize) < self.neurons.len() {
                self.neurons[c as usize].parents.retain(|p| *p != (owner, r.pattern));
                if self.neurons[c as usize].parents.is_empty() && !self.ledger.contains(&c) { self.ledger.push(c); }
            }
        }
        for v in r.value_neurons {
            if (v as usize) < self.neurons.len() {
                self.neurons[v as usize].parents.retain(|p| *p != (owner, r.pattern));
                if self.neurons[v as usize].parents.is_empty() && !self.ledger.contains(&v) { self.ledger.push(v); }
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
                if !self.neurons[c as usize].alive { continue; }
                if !self.neurons[c as usize].parents.is_empty() { continue; }
                let open = self.frames.iter().any(|f| f.acts.iter().any(|a| a.neuron == c && (fno - f.no) <= a.reach[0] as i64));
                if open { keep.push(c); continue; }
                self.delete_neuron(c);
            }
            pending = std::mem::take(&mut self.ledger);
        }
        self.ledger = keep;
    }

    pub(crate) fn delete_neuron(&mut self, c: NeuronId) {
        if !self.neurons[c as usize].alive { return; }
        self.neurons[c as usize].alive = false;
        self.alive -= 1;
        let key = self.neurons[c as usize].key.clone();
        if self.index.get(&key) == Some(&c) { self.index.remove(&key); }
        if self.trace { eprintln!("[death f{}] #{}", self.frame_no, c); }
        // its subtree: its own patterns' children with no parent outside
        let table: Vec<(PatternId, Option<NeuronId>, Vec<NeuronId>)> = self.neurons[c as usize].table.iter().map(|p| (p.id, p.child(), p.members().iter().filter_map(|m| m.value_neuron).collect())).collect();
        for (pid, child, values) in table {
            self.handle_retired(c, Retired { pattern: pid, child, value_neurons: values, member_only: false });
        }
        self.neurons[c as usize].table.clear();
        self.neurons[c as usize].history.clear();
        self.neurons[c as usize].connections.clear();
        // scrub it from every pattern and saved activation that names it
        let ids: Vec<NeuronId> = self.neurons.iter().filter(|n| n.alive).map(|n| n.id).collect();
        for n in ids {
            let retired = self.neurons[n as usize].delete_neighbor(c);
            for r in retired { if self.trace { eprintln!("[cascade f{} #{}] p{} child={:?} lost #{}", self.frame_no, n, r.pattern, r.child, c); } self.handle_retired(n, r); }
        }
        for f in self.frames.iter_mut() {
            for a in f.acts.iter_mut() {
                a.inferences.retain(|i| i.target != c);
            }
        }
    }

}
