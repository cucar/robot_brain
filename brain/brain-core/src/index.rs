//! The interneuron index (D36): what each stands for.

use rustc_hash::{FxHashMap};

use crate::machine::*;
use crate::neuron::*;
use crate::types::*;

impl Machine {
    // ───────────────────────────── the index (D36) ─────────────────────────────

    /// What a neuron stands for (D36): its expansion with variables left open, down to base symbols. No
    /// line names a neuron built over its own child (D27), so nothing is met twice; the path is a guard.
    pub(crate) fn key_of(&self, n: NeuronId, path: &mut Vec<NeuronId>) -> String {
        let nn = &self.thalamus[n];
        if nn.kind != Kind::Inter { return format!("b{}", n); }
        if path.contains(&n) || path.len() > 256 { self.cycle_guard.set(self.cycle_guard.get() + 1); return format!("~{}", n); }
        if let Some(v) = nn.value {
            path.push(n);
            let s = format!("V[{}={}]", self.var_key(v.owner, v.var, path), self.key_of(v.member, path));
            path.pop();
            return s;
        }
        let Some(&(owner, pid)) = nn.parents.first() else { return format!("?{}", n) };
        path.push(n);
        let s = self.line_key(owner, pid, path);
        path.pop();
        s
    }

    pub(crate) fn line_key(&self, owner: NeuronId, pid: PatternId, path: &mut Vec<NeuronId>) -> String {
        let Some(Body::Function { nodes, .. }) = self.line(owner, pid).map(|p| &p.body) else { return format!("?{}:{}", owner, pid) };
        let mut parts: Vec<String> = Vec::new();
        for (node, off) in nodes {
            let k = match node {
                Node::Owner => format!("O{}", self.key_of(owner, path)),
                Node::Neuron(m) => self.key_of(*m, path),
                Node::Var(v) => self.var_key(owner, *v, path),
            };
            parts.push(format!("{}@{:?}", k, off));
        }
        format!("F[{}]", parts.join(","))
    }

    pub(crate) fn var_key(&self, owner: NeuronId, var: PatternId, path: &mut Vec<NeuronId>) -> String {
        match self.line(owner, var).map(|p| &p.body) {
            // a call stands for its relations with the variables left open (D36): a class is named by its
            // members, not expanded through them
            Some(Body::Class { members, .. }) => {
                let mut ks: Vec<NeuronId> = members.iter().map(|m| m.neuron).collect();
                ks.sort();
                format!("C{{{}}}", ks.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(","))
            }
            Some(Body::Parameter { offsets, .. }) => format!("P{:?}", offsets),
            _ => format!("?v{}:{}", owner, var),
        }
    }

    pub(crate) fn child_key(&self, owner: NeuronId, pid: PatternId) -> String { self.line_key(owner, pid, &mut Vec::new()) }

    pub(crate) fn value_key(&self, owner: NeuronId, var: PatternId, member: NeuronId) -> String {
        let mut path = Vec::new();
        format!("V[{}={}]", self.var_key(owner, var, &mut path), self.key_of(member, &mut path))
    }

    /// Recompute what every interneuron stands for after the frame's lines moved (D36); where two stand
    /// for one thing the younger's patterns are rewired to the older (D43, R43).
    pub(crate) fn refresh_index(&mut self) {
        let ids: Vec<NeuronId> = self.thalamus.iter().filter(|n| n.alive && n.kind == Kind::Inter).map(|n| n.id).collect();
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
                let is_value = self.thalamus[young].value.is_some();
                if is_value {
                    if let Some(v) = self.thalamus[young].value { self.thalamus[owner].wire_value(pid, v.member, old); }
                } else {
                    self.thalamus[owner].wire_child(pid, old);
                }
                if !self.thalamus[old].parents.contains(&(owner, pid)) { self.thalamus[old].parents.push((owner, pid)); }
            }
            if !self.ledger.contains(&young) { self.ledger.push(young); }
        }
        // spans, for the owner's reach in prices
        for f in self.frames.iter() {
            for a in &f.acts {
                let s = &mut self.spans[a.neuron as usize];
                for d in 0..MAX_DIMS { if a.span[d] > s[d] { s[d] = a.span[d]; } }
                let rt = a.span[0] + self.radii[0];
                if rt > self.max_reach_t { self.max_reach_t = rt; }
            }
        }
    }

}
