//! Expansion (R28): a neuron placed, recovered down to base symbols.


use crate::machine::*;
use crate::neuron::*;
use crate::types::*;

impl Machine {
    // ───────────────────────────── expansion (R28) ─────────────────────────────

    /// Expand a neuron placed at `at` to base symbols at their coordinates (R28).
    pub fn expand(&self, n: NeuronId, at: Pos, path: &mut Vec<NeuronId>, out: &mut Vec<(NeuronId, Pos)>, fired_by: Option<&FiredBy>) {
        self.expand_into(n, at, path, out, None, fired_by);
    }

    /// Expand a neuron placed at `at` (R28): the nodes of the function whose bid fired it, the latest at
    /// `at`'s frame and the rest by their relations, centered on its position in every other dimension,
    /// down to base symbols, which go to `base`; every neuron the descent passes through below the root,
    /// interneurons included, goes to `all` where it is asked for. No line names a neuron built over its
    /// own child (D27), so a descent never meets a neuron again; the path is a guard and is counted.
    pub(crate) fn expand_into(&self, n: NeuronId, at: Pos, path: &mut Vec<NeuronId>, base: &mut Vec<(NeuronId, Pos)>, mut all: Option<&mut Vec<(NeuronId, Pos)>>, fired_by: Option<&FiredBy>) {
        if base.len() > 4096 { return; }
        let Some(nn) = self.thalamus.get(n) else { return };
        if !nn.alive { return; }
        if !path.is_empty() { if let Some(a) = all.as_deref_mut() { a.push((n, at)); } }
        if nn.kind != Kind::Inter { base.push((n, at)); return; }
        if path.contains(&n) || path.len() > 256 { self.cycle_guard.set(self.cycle_guard.get() + 1); return; }
        // a value neuron: its variable with its value
        if let Some(v) = nn.value {
            path.push(n);
            self.expand_value(v.owner, v.var, v.member, at, path, base, all, None);
            path.pop();
            return;
        }
        let Some((owner, pid)) = fired_by.map(|f| (f.owner, f.pattern)).or(nn.parents.first().copied()) else { return };
        let Some(p) = self.line(owner, pid) else { return };
        let Body::Function { nodes, .. } = &p.body else { return };
        path.push(n);
        // the center of the box the nodes span in every other dimension
        let mut center = [0i32; MAX_DIMS];
        for d in 1..self.radii.len() {
            let lo = nodes.iter().map(|(_, o)| o[d]).min().unwrap_or(0);
            let hi = nodes.iter().map(|(_, o)| o[d]).max().unwrap_or(0);
            center[d] = lo + (hi - lo) / 2;
        }
        for (node, off) in nodes {
            let mut q = at;
            q[0] += off[0];
            for d in 1..MAX_DIMS { q[d] += off[d] - center[d]; }
            match node {
                Node::Owner => self.expand_into(owner, q, path, base, all.as_deref_mut(), None),
                Node::Neuron(m) => self.expand_into(*m, q, path, base, all.as_deref_mut(), None),
                Node::Var(v) => {
                    let value = fired_by.and_then(|f| f.values.iter().find(|(x, _, _)| x == v).map(|(_, m, _)| *m))
                        .or_else(|| self.bind_variable(owner, *v, &q));
                    if let Some(m) = value { self.expand_value(owner, *v, m, q, path, base, all.as_deref_mut(), None); }
                }
            }
        }
        path.pop();
    }

    /// A variable with its value, at its place: a class's member there, a parameter's at each position.
    pub(crate) fn expand_value(&self, owner: NeuronId, var: PatternId, member: NeuronId, at: Pos, path: &mut Vec<NeuronId>, base: &mut Vec<(NeuronId, Pos)>, mut all: Option<&mut Vec<(NeuronId, Pos)>>, _f: Option<&FiredBy>) {
        match self.line(owner, var).map(|p| &p.body) {
            Some(Body::Class { .. }) => self.expand_into(member, at, path, base, all, None),
            Some(Body::Parameter { offsets, .. }) => {
                for o in offsets { self.expand_into(member, add(&at, o), path, base, all.as_deref_mut(), None); }
            }
            _ => {}
        }
    }

    /// Bind a variable with no value neuron beside it (R28): a parameter from what stands at its positions
    /// that have happened, read through value neurons (D44); else, and for a class, its most frequent
    /// member, the oldest on a tie.
    pub(crate) fn bind_variable(&self, owner: NeuronId, var: PatternId, at: &Pos) -> Option<NeuronId> {
        let p = self.line(owner, var)?;
        if let Body::Parameter { offsets, .. } = &p.body {
            for o in offsets {
                let c = add(at, o);
                if c[0] as i64 > self.frame_no { continue; }
                if let Some(f) = self.frames.iter().find(|f| f.no == c[0] as i64) {
                    if let Some(a) = f.acts.iter().find(|a| !a.gone && a.strong && a.pos == c && a.kind != Kind::Inter) { return Some(a.neuron); }
                    if let Some(a) = f.acts.iter().find(|a| !a.gone && a.strong && a.pos == c) { return Some(self.follow(a.neuron)); }
                }
            }
        }
        p.frequent_member()
    }

}
