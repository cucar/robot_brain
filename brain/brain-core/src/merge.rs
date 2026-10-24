//! Merging (D43, §5.5): identical functions and identical classes, the oldest kept.

use crate::neuron::*;
use crate::types::*;

impl Neuron {
    // ───────────────────────────── merging (D43, §5.5) ─────────────────────────────

    /// An older function with the same nodes, variables and relations, if any.
    pub(crate) fn identical(&self, fid: PatternId) -> Option<PatternId> {
        let f = self.function(fid)?;
        self.functions.iter().filter(|g| g.id < fid && g.places == f.places).map(|g| g.id).min()
    }

    pub(crate) fn merge(&mut self, ctx: Ctx) -> Vec<Retired> {
        let mut out = Vec::new();
        // classes: the newer's places go to the older
        let ids: Vec<PatternId> = self.classes.iter().map(|c| c.id).collect();
        for k in ids {
            let Some(c) = self.class(k) else { continue };
            let mut m: Vec<NeuronId> = c.members.iter().map(|x| x.neuron).collect();
            m.sort();
            let older = self.classes.iter().filter(|o| o.id < k).find(|o| { let mut n: Vec<NeuronId> = o.members.iter().map(|x| x.neuron).collect(); n.sort(); n == m }).map(|o| o.id);
            let Some(keep) = older else { continue };
            for f in self.functions.iter_mut() {
                let mut hit = false;
                for (p, _) in f.places.iter_mut() { if *p == Place::Class(k) { *p = Place::Class(keep); hit = true; } }
                if hit { f.canonicalize(); }
            }
            self.classes.retain(|c| c.id != k);
        }
        // functions: the oldest keeps its child, and takes the oldest child of the group where it has none
        let ids: Vec<PatternId> = self.functions.iter().map(|f| f.id).collect();
        for fid in ids {
            if self.function(fid).is_none() { continue; }
            let Some(keep) = self.identical(fid) else { continue };
            let newer_child = self.function(fid).and_then(|f| f.child);
            let mut moved_child = false;
            if let (Some(c), Some(k)) = (newer_child, self.function_mut(keep)) { if k.child.is_none() { k.child = Some(c); moved_child = true; } }
            let idx = self.functions.iter().position(|f| f.id == fid).unwrap();
            let f = self.functions.remove(idx);
            self.unsettled.remove(&fid);
            self.refit(keep, ctx);
            out.push(Retired { pattern: fid, child: f.child, line: Some(f), heir: if moved_child { Some(keep) } else { None } });
        }
        out
    }
}
