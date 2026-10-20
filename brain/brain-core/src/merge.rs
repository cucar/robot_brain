//! Merging (D43, §5.5): identical lines, the oldest kept.


use crate::neuron::*;
use crate::types::*;

impl Neuron {
    // ───────────────────────────── merging (D43, §5.5) ─────────────────────────────

    /// An older pattern identical to this one, if any.
    pub(crate) fn identical(&self, pid: PatternId) -> Option<PatternId> {
        let p = self.pattern(pid)?;
        for q in &self.table {
            if q.id >= pid { continue; }
            let same = match (&p.body, &q.body) {
                (Body::Function { nodes: a, .. }, Body::Function { nodes: b, .. }) => a == b,
                (Body::Class { members: a, .. }, Body::Class { members: b, .. }) => a.iter().map(|m| m.neuron).collect::<Vec<_>>() == b.iter().map(|m| m.neuron).collect::<Vec<_>>(),
                (Body::Parameter { offsets: a, .. }, Body::Parameter { offsets: b, .. }) => a == b,
                _ => false,
            };
            if same { return Some(q.id); }
        }
        None
    }

    pub(crate) fn merge(&mut self) -> Vec<Retired> {
        let mut out = Vec::new();
        let ids: Vec<PatternId> = self.table.iter().map(|p| p.id).collect();
        for pid in ids {
            if self.pattern(pid).is_none() { continue; }
            let Some(keep) = self.identical(pid) else { continue };
            // the oldest takes the others' covers, and its child where it has none
            let moved_child = match (self.pattern(keep).and_then(|p| p.child()), self.pattern(pid).and_then(|p| p.child())) {
                (None, Some(c)) => { if let Some(k) = self.pattern_mut(keep) { if let Body::Function { child, .. } = &mut k.body { *child = Some(c); } } true }
                _ => false,
            };
            // members' value neurons: a member both held keeps the oldest's; one only the newer held moves
            let newer_members: Vec<Member> = self.pattern(pid).map(|p| p.members().to_vec()).unwrap_or_default();
            let mut retired_values = Vec::new();
            if let Some(k) = self.pattern_mut(keep) {
                if k.is_variable() {
                    for m in newer_members {
                        match k.members_mut().iter_mut().find(|x| x.neuron == m.neuron) {
                            Some(x) => { if x.value_neuron.is_none() { x.value_neuron = m.value_neuron; } else if let Some(v) = m.value_neuron { if Some(v) != x.value_neuron { retired_values.push(v); } } }
                            None => k.members_mut().push(m),
                        }
                    }
                }
            }
            for row in self.history.iter_mut() {
                for e in row.cover.iter_mut() {
                    if e.pattern == pid { e.pattern = keep; }
                    for nb in e.nodes.iter_mut() { if nb.node == Node::Var(pid) { nb.node = Node::Var(keep); } if let Some(v) = &mut nb.var { if v.var == pid { v.var = keep; } } }
                }
                for o in row.owner.iter_mut() { if *o == Some(pid) { *o = Some(keep); } }
                if row.owner_self == Some(pid) { row.owner_self = Some(keep); }
                for c in row.claim.iter_mut() { if let Some((v, _)) = c { if *v == pid { *v = keep; } } }
            }
            for q in self.table.iter_mut() {
                if let Body::Function { nodes, .. } = &mut q.body {
                    for (n, _) in nodes.iter_mut() { if *n == Node::Var(pid) { *n = Node::Var(keep); } }
                }
            }
            let idx = self.index_of(pid).unwrap();
            let p = self.table.remove(idx);
            let child = if moved_child { None } else { p.child() };
            out.push(Retired { pattern: pid, child, value_neurons: retired_values, member_only: false, line: Some(p) });
        }
        out
    }

}
