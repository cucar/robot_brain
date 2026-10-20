//! Neighborhoods (D5, D7): what an activation observes.


use crate::machine::*;
use crate::neuron::*;
use crate::types::*;

impl Machine {
    // ───────────────────────────── neighborhoods (D5, D7) ─────────────────────────────

    /// What one activation observes (D5, D7): every activation `b` in the same frame as `a` or an earlier
    /// one, within `a`'s reach in every activation dimension they share, standing on the apex as `a` does,
    /// whose kind the past table admits beside `a`'s (D50). What has not fired yet is not there.
    pub(crate) fn neighborhood(&self, me: ActRef) -> Vec<Nb> {
        let Some(a) = self.act(me) else { return vec![] };
        let mut out = Vec::new();
        for f in self.frames.iter() {
            if f.no > a.pos[0] as i64 || (a.pos[0] as i64 - f.no) > a.reach[0] as i64 { continue; }
            for (i, b) in f.acts.iter().enumerate() {
                let r = ActRef { frame: f.no, index: i as u32 };
                if r == me || b.gone { continue; }
                // both on the apex (R27)
                if b.covered_at.is_some() { continue; }
                if !self.cfg.policy.past[a.kind as usize][b.kind as usize] { continue; }
                let shared = a.axes & b.axes;
                let mut delta = [0i32; MAX_DIMS];
                delta[0] = b.pos[0] - a.pos[0];
                let mut ok = true;
                for d in 1..self.radii.len() {
                    if shared & (1 << d) == 0 { continue; }
                    delta[d] = b.pos[d] - a.pos[d];
                    if delta[d].abs() > a.reach[d] { ok = false; break; }
                }
                if !ok { continue; }
                let value = self.thalamus[b.neuron].value;
                // a child stands in for what it stands for (D22, D47): every node of its expansion (R28)
                let (covers, stands_for) = if b.kind == Kind::Inter {
                    let mut base = Vec::new();
                    let mut all = Vec::new();
                    self.expand_into(b.neuron, b.pos, &mut Vec::new(), &mut base, Some(&mut all), b.fired_by.as_ref());
                    // what it stands for counts inside the box the neighborhood spans, as every node does (D15)
                    let inside = |p: &Pos| {
                        let dt = a.pos[0] - p[0];
                        dt >= 0 && dt <= a.reach[0] && (1..self.radii.len()).all(|d| shared & (1 << d) == 0 || (p[d] - a.pos[d]).abs() <= a.reach[d])
                    };
                    (base.into_iter().filter(|(_, p)| inside(p)).map(|(n, p)| (n, sub(&p, &a.pos))).collect(),
                     all.into_iter().filter(|(_, p)| inside(p)).map(|(n, p)| (n, sub(&p, &a.pos))).collect())
                } else { (Vec::new(), Vec::new()) };
                out.push(Nb { act: r, neuron: b.neuron, delta, cost: b.cost, value, covers, stands_for });
            }
        }
        out
    }

    /// Every neuron what `n` stands for passes through (D36), over every line wired to it, down to base
    /// symbols: sorted, without `n` itself.
    pub(crate) fn through_of(&self, n: NeuronId) -> Vec<NeuronId> {
        let mut seen: rustc_hash::FxHashSet<NeuronId> = rustc_hash::FxHashSet::default();
        let mut stack = vec![n];
        while let Some(x) = stack.pop() {
            let Some(xn) = self.thalamus.get(x) else { continue };
            if xn.kind != Kind::Inter { continue; }
            let push = |m: NeuronId, seen: &mut rustc_hash::FxHashSet<NeuronId>, stack: &mut Vec<NeuronId>| { if m != n && seen.insert(m) { stack.push(m); } };
            if let Some(v) = xn.value {
                push(v.member, &mut seen, &mut stack);
                continue;
            }
            for &(owner, pid) in &xn.parents {
                let Some(p) = self.line(owner, pid) else { continue };
                let Body::Function { nodes, .. } = &p.body else { continue };
                for (node, _) in nodes {
                    match node {
                        Node::Owner => push(owner, &mut seen, &mut stack),
                        Node::Neuron(m) => push(*m, &mut seen, &mut stack),
                        // a call neuron stands for its relations with the variables left open (D36)
                        Node::Var(_) => {}
                    }
                }
            }
        }
        let mut v: Vec<NeuronId> = seen.into_iter().collect();
        v.sort_unstable();
        v
    }

}
