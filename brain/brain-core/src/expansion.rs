//! Expansion (R28): an inferred call is read through its function in the dictionary, its nodes placed back from
//! its origin, each node that is an interneuron expanded in turn, down to base symbols. A descent stops where it
//! comes back to an interneuron already on its path (D36).

use crate::function::*;
use crate::machine::*;
use crate::types::*;

impl Machine {
    /// The function an interneuron is read through when nothing fired it there: the one that fired it last, or
    /// any of its functions (R28).
    pub(crate) fn func_of(&self, n: NeuronId) -> Option<&Func> {
        self.last_func.get(&n).and_then(|id| self.func(*id)).or_else(|| self.dict.of_inter(n).first().and_then(|id| self.dict.get(*id)))
    }

    /// Expand a neuron completing at `at` to base symbols at their places (R28), reading nothing before frame
    /// `from`; every interneuron the descent passes through below the root goes to `all` where asked.
    pub fn expand_from(&self, n: NeuronId, at: Pos, from: Frame, path: &mut Vec<NeuronId>, out: &mut Vec<(NeuronId, Pos)>, mut all: Option<&mut Vec<(NeuronId, Pos)>>) {
        if (at[0] as i64) < from { return; }
        let Some(nn) = self.thalamus.get(n) else { return };
        if !nn.alive { return; }
        if nn.kind != Kind::Inter { out.push((n, at)); return; }
        if path.contains(&n) || path.len() > 256 { return; }
        if !path.is_empty() { if let Some(v) = all.as_deref_mut() { v.push((n, at)); } }
        let Some(f) = self.func_of(n).cloned() else { return };
        let lay = f.layout();
        path.push(n);
        for (i, m) in f.nodes.iter().enumerate() {
            let mut q = at;
            for d in 0..MAX_DIMS { q[d] += lay.place[i][d]; }
            self.expand_from(*m, q, from, path, out, all.as_deref_mut());
        }
        path.pop();
    }

    /// The nodes of an interneuron's function placed back from `at`, one expansion down (R49): each neuron at its
    /// place.
    pub(crate) fn nodes_at(&self, n: NeuronId, at: Pos) -> Vec<(NeuronId, Pos)> {
        let Some(f) = self.func_of(n) else { return vec![] };
        let lay = f.layout();
        f.nodes.iter().enumerate().map(|(i, m)| { let mut q = at; for d in 0..MAX_DIMS { q[d] += lay.place[i][d]; } (*m, q) }).collect()
    }
}
