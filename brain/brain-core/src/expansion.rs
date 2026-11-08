//! Expansion (R28) of an inferred call: read back through its function in the dictionary to its nodes from
//! its origin, each node that is an interneuron expanded in turn, down to base symbols. A cycle is read round
//! over the frames that have happened while the node it reaches has something to land on; beyond them, round up
//! to the nearest node the function has past the cycle in that direction, and once where there is none. A
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

    /// What a function expands to from node `root` standing at `at`, one expansion down (R28): each node at its
    /// position, and a cycle read round as R28 says. With `board`, a lap whose node falls in a frame that has happened
    /// is read only where something stands there to land on (D54); without, every lap is beyond what has happened.
    pub(crate) fn expands_to(&self, f: &Func, root: usize, at: Pos, board: bool) -> Vec<(Sym, Pos)> {
        let lay = f.layout_from(root);
        let shift = |p: &Pos, k: i32, l: &Pos| { let mut q = at; for d in 0..MAX_DIMS { q[d] += p[d] + k * l[d]; } q };
        let zero = [0i32; MAX_DIMS];
        let mut out: Vec<(Sym, Pos)> = f.nodes.iter().enumerate().map(|(i, m)| (*m, shift(&lay.position[i], 0, &zero))).collect();
        let Some((_, ua, ub, off)) = lay.closing else { return out };
        // one lap: from the node the closing relation reaches to the node it leaves, then the closing relation again
        let mut lap = [0i32; MAX_DIMS];
        for d in 0..MAX_DIMS { lap[d] = lay.position[ua][d] + off[d] - lay.position[ub][d]; }
        let path: Vec<usize> = Func::tree_path(&lay, ub, ua).iter().map(|s| s.2).collect();
        let mut on_cycle = vec![false; f.nodes.len()];
        on_cycle[ua] = true;
        on_cycle[ub] = true;
        for &v in &path { on_cycle[v] = true; }
        // the nearest node past the cycle in the direction the laps run, where the function has one
        let far_cycle = (0..f.nodes.len()).filter(|i| on_cycle[*i]).map(|i| dot(&lay.position[i], &lap)).max().unwrap_or(0);
        let past: Option<i64> = (0..f.nodes.len()).filter(|i| !on_cycle[*i]).map(|i| dot(&lay.position[i], &lap)).filter(|x| *x > far_cycle).min();
        for k in 1..=64 {
            let reached = shift(&lay.position[ub], k, &lap);
            let happened = board && (reached[0] as i64) <= self.frame_no;
            let go = if happened { self.stood(f.nodes[ub], &reached) } else {
                // beyond what has happened: round up to the nearest node past the cycle, and once where none
                k == 1 || past.map_or(false, |t| dot(&shift(&lay.position[ub], k - 1, &lap), &lap) - dot(&at, &lap) < t)
            };
            if !go { break; }
            // past the first round, the lap before it: from the node the last round reached, round to the node
            // the closing relation leaves
            if k > 1 { for &v in &path { if v != ub { out.push((f.nodes[v], shift(&lay.position[v], k - 1, &lap))); } } }
            out.push((f.nodes[ub], reached));
        }
        out
    }

    /// Expand a neuron in a pose standing at `at` to base symbols at their positions (R28), from the node its pose
    /// starts at, or its origin where it names no pose, reading nothing before frame `from`; every interneuron the
    /// descent passes through below the root goes to `all` where asked. With `board`, cycles are read round
    /// against what has happened; without, as the index reads them.
    pub fn expand_from(&self, s: Sym, at: Pos, from: Frame, board: bool, path: &mut Vec<NeuronId>, out: &mut Vec<(NeuronId, Pos)>, mut all: Option<&mut Vec<(NeuronId, Pos)>>) {
        if (at[0] as i64) < from { return; }
        let n = self.dict.neuron_of(s);
        let Some(nn) = self.thalamus.get(n) else { return };
        if !nn.alive { return; }
        if nn.kind != Kind::Inter { out.push((n, at)); return; }
        if path.contains(&n) || path.len() > 256 { return; }
        if !path.is_empty() { if let Some(v) = all.as_deref_mut() { v.push((n, at)); } }
        let Some(f) = self.func_of(n).cloned() else { return };
        let root = self.dict.start_of(s).and_then(|x| self.dict.node_by_neuron(&f, x)).unwrap_or(0);
        path.push(n);
        for (m, q) in self.expands_to(&f, root, at, board) {
            self.expand_from(m, q, from, board, path, out, all.as_deref_mut());
        }
        path.pop();
    }

    /// What an interneuron's call standing at `at` expands to, one expansion down, against what has
    /// happened (R49, R28).
    pub(crate) fn nodes_at(&self, s: Sym, at: Pos) -> Vec<(Sym, Pos)> {
        let Some(f) = self.func_of(self.dict.neuron_of(s)) else { return vec![] };
        let root = self.dict.start_of(s).and_then(|x| self.dict.node_by_neuron(f, x)).unwrap_or(0);
        self.expands_to(f, root, at, true)
    }
}
