//! The interneuron index (D36): what each stands for.

use rustc_hash::{FxHashMap};

use crate::machine::*;
use crate::neuron::*;
use crate::types::*;

impl Machine {
    // ───────────────────────────── the index (D36) ─────────────────────────────

    /// What a neuron stands for (D36): its expansion with variables left open, down to base symbols. No
    /// function names a neuron built over its own child (D27), so nothing is met twice; the path is a guard.
    pub(crate) fn key_of(&self, n: NeuronId, path: &mut Vec<NeuronId>) -> String {
        let nn = &self.thalamus[n];
        if nn.kind != Kind::Inter { return format!("b{}", n); }
        if path.contains(&n) || path.len() > 256 { self.cycle_guard.set(self.cycle_guard.get() + 1); return format!("~{}", n); }
        let Some(&(owner, pid)) = nn.parents.first() else { return format!("?{}", n) };
        path.push(n);
        let s = self.line_key(owner, pid, path);
        path.pop();
        s
    }

    /// A function's nodes and relations (D34) with the variables left open: a class place is open, and a
    /// parameter's places are open and tied.
    pub(crate) fn line_key(&self, owner: NeuronId, pid: PatternId, path: &mut Vec<NeuronId>) -> String {
        let Some(f) = self.line(owner, pid) else { return format!("?{}:{}", owner, pid) };
        let mut parts: Vec<String> = Vec::new();
        for (place, off) in &f.places {
            let k = match place {
                Place::Owner => format!("O{}", self.key_of(owner, path)),
                Place::Fixed(m) => self.key_of(*m, path),
                Place::Class(_) => "V".to_string(),
                Place::Param(k) => format!("P{}", k),
            };
            parts.push(format!("{}@{:?}", k, off));
        }
        format!("F[{}]", parts.join(","))
    }

    pub(crate) fn child_key(&self, owner: NeuronId, pid: PatternId) -> String { self.line_key(owner, pid, &mut Vec::new()) }

    /// Recompute what every interneuron stands for after the frame's functions moved (D36); where two stand
    /// for one thing the younger's functions are rewired to the older (D43, R43). Spans follow (D4).
    pub(crate) fn refresh_index(&mut self) {
        let ids: Vec<NeuronId> = self.thalamus.iter().filter(|n| n.alive && n.kind == Kind::Inter).map(|n| n.id).collect();
        // a child stands for one thing (D36): a parent whose function re-centered into something else than what
        // the child stands for is no longer its parent (D29), and is wired below to the neuron that stands for
        // what it now is, or gets a child of its own at its next accepted bid
        let mut detached: Vec<(NeuronId, PatternId)> = Vec::new();
        for &id in &ids {
            let parents = self.thalamus[id].parents.clone();
            if parents.len() < 2 { continue; }
            let first = self.child_key(parents[0].0, parents[0].1);
            for &(owner, pid) in &parents[1..] {
                if self.child_key(owner, pid) != first {
                    self.thalamus[id].parents.retain(|p| *p != (owner, pid));
                    if let Some(f) = self.thalamus[owner].function_mut(pid) { if f.child == Some(id) { f.child = None; } }
                    detached.push((owner, pid));
                }
            }
        }
        let mut index: FxHashMap<String, NeuronId> = FxHashMap::default();
        let mut rewire: Vec<(NeuronId, NeuronId)> = Vec::new();
        for &id in &ids {
            let key = self.key_of(id, &mut Vec::new());
            self.thalamus[id].key = key.clone();
            match index.get(&key) {
                Some(&older) => rewire.push((id, older)),
                None => { index.insert(key, id); }
            }
        }
        self.index = index;
        for (young, old) in rewire {
            let parents = std::mem::take(&mut self.thalamus[young].parents);
            for (owner, pid) in parents {
                self.thalamus[owner].wire_child(pid, old);
                if !self.thalamus[old].parents.contains(&(owner, pid)) { self.thalamus[old].parents.push((owner, pid)); }
            }
            if !self.ledger.contains(&young) { self.ledger.push(young); }
        }
        for (owner, pid) in detached {
            let key = self.child_key(owner, pid);
            if let Some(&n) = self.index.get(&key) {
                self.thalamus[owner].wire_child(pid, n);
                if !self.thalamus[n].parents.contains(&(owner, pid)) { self.thalamus[n].parents.push((owner, pid)); }
            }
        }
        // spans, from the patterns as they now stand (D4)
        let mut memo: FxHashMap<NeuronId, Pos> = FxHashMap::default();
        for &id in &ids {
            if !self.thalamus[id].alive { continue; }
            let s = self.neuron_span(id, &mut memo, &mut Vec::new());
            self.spans[id as usize] = s;
            let rt = s[0] + self.radii[0];
            if rt > self.max_reach_t { self.max_reach_t = rt; }
        }
    }

}
