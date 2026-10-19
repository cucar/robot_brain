//! Expansion (R28): a neuron placed, recovered down to base symbols.


use crate::machine::*;
use crate::neuron::*;
use crate::types::*;

impl Machine {
    // ───────────────────────────── expansion (R28) ─────────────────────────────

    /// Expand a neuron placed at `at` to base symbols at their coordinates (R28): the nodes of the
    /// function whose bid fired it, the latest at `at` and the rest by their offsets, once through each
    /// neuron on the descent; a neuron met again is placed as its owner's symbol where its function
    /// names the owner, and the descent ends there.
    pub fn expand(&self, n: NeuronId, at: Pos, path: &mut Vec<NeuronId>, out: &mut Vec<(NeuronId, Pos)>, fired_by: Option<&FiredBy>) {
        if path.len() > 64 || out.len() > 4096 { return; }
        let nn = &self.thalamus[n];
        if !nn.alive { return; }
        if nn.kind != Kind::Inter { out.push((n, at)); return; }
        // a value neuron: its variable with its value
        if let Some(v) = nn.value {
            if path.contains(&n) { return; }
            path.push(n);
            self.expand_value(v.owner, v.var, v.member, at, path, out);
            path.pop();
            return;
        }
        let Some((owner, pid)) = fired_by.map(|f| (f.owner, f.pattern)).or(nn.parents.first().copied()) else { return };
        let Some(nodes) = self.thalamus[owner].function_nodes(pid) else { return };
        let names_owner = matches!(nodes.first(), Some((Node::Owner, _)));
        if path.contains(&n) {
            if names_owner { self.expand(owner, at, path, out, None); }
            return;
        }
        path.push(n);
        // the center of the box the nodes span in every other dimension
        let mut center = [0i32; MAX_DIMS];
        for d in 1..self.radii.len() {
            let lo = nodes.iter().map(|(_, o)| o[d]).min().unwrap_or(0);
            let hi = nodes.iter().map(|(_, o)| o[d]).max().unwrap_or(0);
            center[d] = lo + (hi - lo) / 2;
        }
        for (node, off) in nodes {
            let mut p = at;
            p[0] += off[0];
            for d in 1..MAX_DIMS { p[d] += off[d] - center[d]; }
            match node {
                Node::Owner => self.expand(owner, p, path, out, None),
                Node::Neuron(m) => self.expand(*m, p, path, out, None),
                Node::Var(v) => {
                    let value = fired_by.and_then(|f| f.values.iter().find(|(x, _, _)| x == v).map(|(_, m, _)| *m))
                        .or_else(|| self.bind_variable(owner, *v, &p));
                    if let Some(m) = value { self.expand_value(owner, *v, m, p, path, out); }
                }
            }
        }
        path.pop();
    }

    /// A variable with its value, at its place: a class's member there, a parameter's at each position.
    pub(crate) fn expand_value(&self, owner: NeuronId, var: PatternId, member: NeuronId, at: Pos, path: &mut Vec<NeuronId>, out: &mut Vec<(NeuronId, Pos)>) {
        match self.thalamus[owner].pattern(var).map(|p| &p.body) {
            Some(Body::Class { .. }) => self.expand(member, at, path, out, None),
            Some(Body::Parameter { offsets, .. }) => {
                for o in offsets { self.expand(member, add(&at, o), path, out, None); }
            }
            _ => {}
        }
    }

    /// Bind a variable with no value neuron beside it (R28): from what stands at its positions that
    /// have happened, read through value neurons, else its most frequent member.
    pub(crate) fn bind_variable(&self, owner: NeuronId, var: PatternId, at: &Pos) -> Option<NeuronId> {
        let p = self.thalamus[owner].pattern(var)?;
        if let Body::Parameter { offsets, .. } = &p.body {
            for o in offsets {
                let c = add(at, o);
                if c[0] as i64 > self.frame_no { continue; }
                if let Some(f) = self.frames.iter().find(|f| f.no == c[0] as i64) {
                    if let Some(a) = f.acts.iter().find(|a| a.strong && a.pos == c && a.kind != Kind::Inter) { return Some(a.neuron); }
                    if let Some(a) = f.acts.iter().find(|a| a.strong && a.pos == c) { return Some(self.follow(a.neuron)); }
                }
            }
        }
        p.frequent_member()
    }

}
