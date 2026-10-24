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
                // both on the apex (R27): what a bid bound stays there (D10)
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
                // a child stands for what is under it (D47, D54): every node of its expansion (R28)
                let (covers, stands_for) = if b.kind == Kind::Inter {
                    let mut base = Vec::new();
                    let mut all = Vec::new();
                    self.expand_into(b.neuron, b.pos, &mut Vec::new(), &mut base, Some(&mut all), b.fired_by.as_ref());
                    // one that stands for `a` itself is built over it, not beside it: a weak pattern placed at
                    // `a`'s coordinate (R30) is where the call `a`'s bid would buy fires, a level above (D53)
                    if base.iter().chain(all.iter()).any(|(n, p)| *n == a.neuron && *p == a.pos) { continue; }
                    // what it stands for counts inside the box the neighborhood spans, as every node does (D15)
                    let inside = |p: &Pos| {
                        let dt = a.pos[0] - p[0];
                        dt >= 0 && dt <= a.reach[0] && (1..self.radii.len()).all(|d| shared & (1 << d) == 0 || (p[d] - a.pos[d]).abs() <= a.reach[d])
                    };
                    (base.into_iter().filter(|(_, p)| inside(p)).map(|(n, p)| (n, sub(&p, &a.pos))).collect(),
                     all.into_iter().filter(|(_, p)| inside(p)).map(|(n, p)| (n, sub(&p, &a.pos))).collect())
                } else { (Vec::new(), Vec::new()) };
                // as it stands: a bound activation is its neuron, its position covered (D10, D24)
                let (cost, pos) = if b.bound_at.is_some() { (b.cost - b.pos_bits, 0.0) } else { (b.cost, b.pos_bits) };
                out.push(Nb { act: r, neuron: b.neuron, delta, cost, pos, covers, stands_for });
            }
        }
        out
    }

}
