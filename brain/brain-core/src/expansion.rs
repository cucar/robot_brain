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

    /// Expand a neuron placed at `at` (R28): the places of the function whose bid fired it, the owner's
    /// activation at `at` and every other place at its offset from it, a fixed node to its neuron and a
    /// variable to its argument, down to base symbols, which go to `base`; every neuron the descent passes
    /// through below the root, interneurons included, goes to `all` where it is asked for. A call with no
    /// arguments given is read off the activation of it standing at `at` where there is one, and bound as
    /// an inferred call otherwise. A descent passes through a neuron once: one met again on its own descent,
    /// as on a chain closed on itself, ends it there, so a closed chain expands to one turn (R28).
    pub(crate) fn expand_into(&self, n: NeuronId, at: Pos, path: &mut Vec<NeuronId>, base: &mut Vec<(NeuronId, Pos)>, mut all: Option<&mut Vec<(NeuronId, Pos)>>, fired_by: Option<&FiredBy>) {
        if base.len() > 4096 { return; }
        let Some(nn) = self.thalamus.get(n) else { return };
        if !nn.alive { return; }
        if !path.is_empty() { if let Some(a) = all.as_deref_mut() { a.push((n, at)); } }
        if nn.kind != Kind::Inter { base.push((n, at)); return; }
        if path.contains(&n) || path.len() > 256 { self.cycle_guard.set(self.cycle_guard.get() + 1); return; }
        // a recognized call's arguments stand beside it (D37): read them off its activation at `at`
        let found;
        let fired_by = match fired_by {
            Some(f) => Some(f),
            None => { found = self.fired_at(n, &at); found.as_ref() }
        };
        let Some((owner, pid)) = fired_by.map(|f| (f.owner, f.pattern)).or(nn.parents.first().copied()) else { return };
        let Some(f) = self.line(owner, pid) else { return };
        path.push(n);
        for (pi, (place, off)) in f.places.iter().enumerate() {
            // a recognized call recovers what its bid covered and bound, and a failed place took nothing
            if fired_by.map_or(false, |fb| fb.failed.contains(&pi)) { continue; }
            let q = add(&at, off);
            match place {
                Place::Owner => self.expand_into(owner, q, path, base, all.as_deref_mut(), None),
                Place::Fixed(m) => self.expand_into(*m, q, path, base, all.as_deref_mut(), None),
                Place::Class(_) | Place::Param(_) => {
                    let arg = fired_by.and_then(|fb| fb.args.iter().find(|(p, _, _)| *p == pi).map(|(_, m, _)| *m))
                        .or_else(|| self.bind_variable(owner, f, pi, &at));
                    if let Some(m) = arg { self.expand_into(m, q, path, base, all.as_deref_mut(), None); }
                }
            }
        }
        path.pop();
    }

    /// What fired the activation of `n` standing at `at`, if one does.
    pub(crate) fn fired_at(&self, n: NeuronId, at: &Pos) -> Option<FiredBy> {
        let f = self.frames.iter().find(|f| f.no == at[0] as i64)?;
        f.acts.iter().find(|a| !a.gone && a.neuron == n && a.pos == *at && a.fired_by.is_some()).and_then(|a| a.fired_by.clone())
    }

    /// Bind a variable of an inferred call (R28): a parameter from what stands at those of its places that
    /// have happened, else its most frequent member; a class its most frequent member, the oldest on a tie.
    pub(crate) fn bind_variable(&self, owner: NeuronId, f: &Function, pi: usize, at: &Pos) -> Option<NeuronId> {
        match f.places[pi].0 {
            Place::Class(k) => self.thalamus.get(owner).and_then(|o| o.class(k)).and_then(|c| c.frequent()),
            Place::Param(k) => {
                for qi in f.param_places(k) {
                    let c = add(at, &f.places[qi].1);
                    if c[0] as i64 > self.frame_no { continue; }
                    if let Some(fr) = self.frames.iter().find(|fr| fr.no == c[0] as i64) {
                        if let Some(a) = fr.acts.iter().find(|a| !a.gone && a.strong && a.pos == c && a.covered_at.is_none()) { return Some(a.neuron); }
                        if let Some(a) = fr.acts.iter().find(|a| !a.gone && a.strong && a.pos == c) { return Some(a.neuron); }
                    }
                }
                f.params.get(k as usize).and_then(|p| frequent(&p.members))
            }
            _ => None,
        }
    }

}
