/// Expansion (R28), the one reading of a call through its function in the dictionary: from the node its phase
/// starts at, each node at its position, each node that is an interneuron expanded in turn in its own phase, down
/// to base symbols. A descent stops where it comes back to an interneuron already on its path (D36).
///
/// Inference, the match, the walk and the index search all read through it.

use crate::dictionary::*;
use crate::brain::*;
use crate::types::*;


impl Brain {
    /// The function an interneuron is read through when nothing fired it there: the one that fired it last, or
    /// any of its functions (R28).
    /// The function a call in its phase expands through (D8, D36, R28): of its interneuron's functions, the one with
    /// a node of each of the two nodes its phase names, since an interneuron never holds two functions over the
    /// same neurons (R43); else one with a node of its start neuron; else any of its functions.
    pub(crate) fn find_function_for(&self, s: Sym) -> Option<&Func> {
        let n = self.thalamus.dict.find_neuron_of(s);
        let Some(p) = self.thalamus.dict.find_pose(s) else { return self.find_function_of(n) };
        let fs: Vec<&Func> = self.thalamus.dict.list_of_inter(n).iter().filter_map(|id| self.thalamus.dict.get(*id)).collect();
        let has = |f: &Func, m: Sym| f.find_node(m).is_some();
        fs.iter().copied().find(|f| has(f, p.from) && has(f, p.to))
            .or_else(|| fs.iter().copied().find(|f| has(f, p.from)))
            .or_else(|| self.find_function_of(n))
    }

    pub(crate) fn find_function_of(&self, n: NeuronId) -> Option<&Func> {
        (if self.frozen { self.frozen_last_func.get(&n).or_else(|| self.last_func.get(&n)) } else { self.last_func.get(&n) }).and_then(|id| self.find_function(*id)).or_else(|| self.thalamus.dict.list_of_inter(n).first().and_then(|id| self.thalamus.dict.get(*id)))
    }

    /// Expand a call in its phase standing at `at`, one expansion down (R28): the function it is read through, and
    /// each node at its position. None for a base neuron, and for a call no function can be read for.
    pub(crate) fn expand_once(&self, s: Sym, at: Pos, open: bool) -> Option<(FuncId, Vec<(Sym, Pos)>)> {
        let n = self.thalamus.dict.find_neuron_of(s);
        if self.thalamus.registry.get(n).map_or(true, |nn| nn.kind != Kind::Inter) { return None; }
        let f = self.find_function_for(s)?;
        let phase = self.thalamus.dict.find_pose(s);
        // it is read from the node its phase starts at, at the call's position (R36, D37)
        let root = phase.and_then(|p| f.find_node(p.from)).unwrap_or(0);
        let lay = f.lay_out_from(root);
        let stand = |p: &Pos, k: i32, l: &Pos| Pos::make(at.len().max(p.len()).max(l.len()), |d| at[d] + p[d] + k * l[d]);
        let zero = Pos::zero();
        let mut out: Vec<(Sym, Pos)> = f.nodes.iter().enumerate().map(|(i, m)| (*m, stand(&lay.position[i], 0, &zero))).collect();
        let Some((_, ua, ub, off)) = lay.closing else { return Some((f.id, out)) };
        if !open {
            // the phase says its laps: the cycle is read back from the node the phase starts at, lap after lap, to
            // the node it ends at (D53)
            let order = f.place_back(root);
            let Some(l) = f.find_lap() else { return Some((f.id, out)) };
            if order.is_empty() { return Some((f.id, out)); }
            let to = phase.and_then(|p| f.find_node(p.to)).unwrap_or(order[order.len() - 1].0);
            let total = f.stretch(root, to, phase.map_or(0, |p| p.laps));
            let c = order.len();
            out.retain(|(m, _)| !order.iter().any(|(v, _)| f.nodes[*v] == *m));
            for i in 0..total {
                let ((v, back), k) = (&order[i % c], (i / c) as i32);
                out.push((f.nodes[*v], stand(back, k, &l)));
            }
            return Some((f.id, out));
        }
        // the phase leaves its laps open, as a call a connection names does (D25): the cycle is read round while the
        // node it reaches has something to land on in the frames that have happened, and beyond them once more
        // (D54, D15)
        // one lap: from the node the closing relation reaches to the node it leaves, then the closing relation again
        let lap = lay.position[ua].plus(&off).minus(&lay.position[ub]);
        let path: Vec<usize> = Func::find_tree_path(&lay, ub, ua).iter().map(|s| s.2).collect();
        // a closed shape comes back to the activation it started at: its lap advances nowhere, so it has no laps to read
        if lap.is_zero() { return Some((f.id, out)); }
        for k in 1i32.. {
            assert!(k <= MAX_LAPS as i32, "an inferred cycle was read back more than {} laps", MAX_LAPS);
            let reached = stand(&lay.position[ub], k, &lap);
            let happened = (reached[0] as i64) <= self.frame_no;
            let go = if happened { self.has_stood(f.nodes[ub], &reached) } else { k == 1 };
            if !go { break; }
            // past the first round, the lap before it: from the node the last round reached, round to the node
            // the closing relation leaves
            if k > 1 { for &v in &path { if v != ub { out.push((f.nodes[v], stand(&lay.position[v], k - 1, &lap))); } } }
            out.push((f.nodes[ub], reached));
        }
        Some((f.id, out))
    }

    /// Expand a call in its phase all the way down (R28).
    ///
    /// `open` says whether the call's own phase leaves its laps open; each call below it is read in its own phase.
    /// `visit` is handed each node with its position and the interneurons of the calls above it, the call's own
    /// first, and says whether the expansion goes on through that node.
    /// `deps` takes each function the expansion was read through.
    pub(crate) fn walk_expansion(&self, s: Sym, at: Pos, open: bool, path: &mut Vec<NeuronId>, deps: &mut Vec<FuncId>, visit: &mut dyn FnMut(Sym, Pos, &[NeuronId]) -> bool) {
        let n = self.thalamus.dict.find_neuron_of(s);
        if path.contains(&n) { return; }
        assert!(path.len() <= MAX_CALL_DEPTH, "an expansion went through more than {} calls, one under another", MAX_CALL_DEPTH);
        let Some((fid, nodes)) = self.expand_once(s, at, open) else { return };
        deps.push(fid);
        path.push(n);
        for (m, q) in nodes {
            if visit(m, q, path) { self.walk_expansion(m, q, false, path, deps, visit); }
        }
        path.pop();
    }

    /// Expand an inferred pose standing at `at` to base symbols at their positions (R28, R36), reading nothing
    /// before frame `from`.
    pub fn expand_from(&self, s: Sym, at: Pos, from: FrameNumber) -> Vec<(NeuronId, Pos)> {
        let mut out: Vec<(NeuronId, Pos)> = Vec::new();
        if (at[0] as i64) < from { return out; }
        let n = self.thalamus.dict.find_neuron_of(s);
        let Some(nn) = self.thalamus.registry.get(n) else { return out };
        if !nn.alive { return out; }
        if nn.kind != Kind::Inter { out.push((n, at)); return out; }
        // an inferred call is named by a connection, which leaves its laps open (D25)
        self.walk_expansion(s, at, true, &mut Vec::new(), &mut Vec::new(), &mut |m, q, _| {
            // an expansion reads back from where it stands, so nothing under a node before `from` is at or after it
            if (q[0] as i64) < from { return false; }
            let nm = self.thalamus.dict.find_neuron_of(m);
            let Some(mm) = self.thalamus.registry.get(nm) else { return false };
            if !mm.alive { return false; }
            if mm.kind != Kind::Inter { out.push((nm, q)); return false; }
            true
        });
        out
    }

    /// What an inferred call standing at `at` expands to, one expansion down (R49, R28).
    pub(crate) fn list_nodes_at(&self, s: Sym, at: Pos) -> Vec<(Sym, Pos)> {
        self.expand_once(s, at, true).map_or(Vec::new(), |(_, nodes)| nodes)
    }
}
