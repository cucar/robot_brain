//! Expansion (R28) of an inferred call: read through its function in the dictionary, its nodes placed back from
//! its origin, each node that is an interneuron expanded in turn, down to base symbols. A cycle is read round
//! over the frames that have happened while the node it reaches has something to land on; beyond them, round up
//! to the nearest node the function places past the cycle in that direction, and once where there is none. A
//! descent stops where it comes back to an interneuron already on its path (D36).
//!
//! Read with no board, as the index search reads what an interneuron would stand for (D36), every lap is beyond
//! what has happened.

use crate::function::*;
use crate::machine::*;
use crate::types::*;

fn dot(a: &Pos, b: &Pos) -> i64 { (0..MAX_DIMS).map(|d| a[d] as i64 * b[d] as i64).sum() }

impl Machine {
    /// The function an interneuron is read through when nothing fired it there: the one that fired it last, or
    /// any of its functions (R28).
    pub(crate) fn func_of(&self, n: NeuronId) -> Option<&Func> {
        self.last_func.get(&n).and_then(|id| self.func(*id)).or_else(|| self.dict.of_inter(n).first().and_then(|id| self.dict.get(*id)))
    }

    /// The nodes of a function placed back from its origin at `at`, one expansion down (R28): each node at its
    /// place, and a cycle read round as R28 says. With `board`, a lap whose node falls in a frame that has happened
    /// is read only where something stands there to land on (D54); without, every lap is beyond what has happened.
    pub(crate) fn places(&self, f: &Func, at: Pos, board: bool) -> Vec<(NeuronId, Pos)> {
        let lay = f.layout();
        let shift = |p: &Pos, k: i32, l: &Pos| { let mut q = at; for d in 0..MAX_DIMS { q[d] += p[d] + k * l[d]; } q };
        let zero = [0i32; MAX_DIMS];
        let mut out: Vec<(NeuronId, Pos)> = f.nodes.iter().enumerate().map(|(i, m)| (*m, shift(&lay.place[i], 0, &zero))).collect();
        let Some((_, ua, ub, off)) = lay.closing else { return out };
        // one lap: from the node the closing relation reaches to the node it leaves, then the closing relation again
        let mut lap = [0i32; MAX_DIMS];
        for d in 0..MAX_DIMS { lap[d] = lay.place[ua][d] + off[d] - lay.place[ub][d]; }
        let path: Vec<usize> = Func::tree_path(&lay, ub, ua).iter().map(|s| s.2).collect();
        let mut on_cycle = vec![false; f.nodes.len()];
        on_cycle[ua] = true;
        on_cycle[ub] = true;
        for &v in &path { on_cycle[v] = true; }
        // the nearest node past the cycle in the direction the laps run, where the function has one
        let far_cycle = (0..f.nodes.len()).filter(|i| on_cycle[*i]).map(|i| dot(&lay.place[i], &lap)).max().unwrap_or(0);
        let past: Option<i64> = (0..f.nodes.len()).filter(|i| !on_cycle[*i]).map(|i| dot(&lay.place[i], &lap)).filter(|x| *x > far_cycle).min();
        for k in 1..=64 {
            let reached = shift(&lay.place[ub], k, &lap);
            let happened = board && (reached[0] as i64) <= self.frame_no;
            let go = if happened { self.stood(f.nodes[ub], &reached) } else {
                // beyond what has happened: round up to the nearest node past the cycle, and once where none
                k == 1 || past.map_or(false, |t| dot(&shift(&lay.place[ub], k - 1, &lap), &lap) - dot(&at, &lap) < t)
            };
            if !go { break; }
            // past the first round, the lap before it: from the node the last round reached, round to the node
            // the closing relation leaves
            if k > 1 { for &v in &path { if v != ub { out.push((f.nodes[v], shift(&lay.place[v], k - 1, &lap))); } } }
            out.push((f.nodes[ub], reached));
        }
        out
    }

    /// Expand a neuron completing at `at` to base symbols at their places (R28), reading nothing before frame
    /// `from`; every interneuron the descent passes through below the root goes to `all` where asked. With
    /// `board`, cycles are read round against what has happened; without, as the index reads them.
    pub fn expand_from(&self, n: NeuronId, at: Pos, from: Frame, board: bool, path: &mut Vec<NeuronId>, out: &mut Vec<(NeuronId, Pos)>, mut all: Option<&mut Vec<(NeuronId, Pos)>>) {
        if (at[0] as i64) < from { return; }
        let Some(nn) = self.thalamus.get(n) else { return };
        if !nn.alive { return; }
        if nn.kind != Kind::Inter { out.push((n, at)); return; }
        if path.contains(&n) || path.len() > 256 { return; }
        if !path.is_empty() { if let Some(v) = all.as_deref_mut() { v.push((n, at)); } }
        let Some(f) = self.func_of(n).cloned() else { return };
        path.push(n);
        for (m, q) in self.places(&f, at, board) {
            self.expand_from(m, q, from, board, path, out, all.as_deref_mut());
        }
        path.pop();
    }

    /// The nodes of an interneuron's function placed back from `at`, one expansion down, against what has
    /// happened (R49, R28).
    pub(crate) fn nodes_at(&self, n: NeuronId, at: Pos) -> Vec<(NeuronId, Pos)> {
        let Some(f) = self.func_of(n) else { return vec![] };
        self.places(f, at, true)
    }
}
