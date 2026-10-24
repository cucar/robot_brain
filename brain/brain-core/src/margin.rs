//! Margin and retirement (D30, R18, R41).

use crate::neuron::*;
use crate::types::*;

impl Neuron {
    // ───────────────────────────── margin and retirement (D30, R18, R41) ─────────────────────────────

    /// What a function is worth over the history less its entry in the table (D30).
    pub fn margin(&self, fid: PatternId, ctx: Ctx) -> f64 {
        let Some(f) = self.function(fid) else { return 0.0 };
        self.benefit(fid, ctx) - self.entry(f, ctx)
    }

    /// The benefit of a function (D30): over its uses, the row's cost without it in the table less its cost
    /// with it; zero in every other row, where it does not fit.
    pub(crate) fn benefit(&self, fid: PatternId, ctx: Ctx) -> f64 {
        let Some(f) = self.function(fid) else { return 0.0 };
        let with: Vec<&Function> = self.functions.iter().collect();
        let without: Vec<&Function> = self.functions.iter().filter(|g| g.id != fid).collect();
        let index = self.row_index();
        let mut total = 0.0;
        for a in &f.uses {
            let Some(&i) = index.get(a) else { continue };
            let row = &self.history[i];
            total += self.row_cost(row, &without, ctx) - self.row_cost(row, &with, ctx);
        }
        total
    }

    /// Price every function's margin against the table as it now stands and retire every one strictly
    /// negative, together, with every function that names nothing besides the owner, and every class no
    /// function uses or left with one member (R18, R41).
    pub(crate) fn retire(&mut self, ctx: Ctx) -> Vec<Retired> {
        let mut doomed: Vec<PatternId> = Vec::new();
        for f in &self.functions {
            if f.p_size() == 0 || self.margin(f.id, ctx) < 0.0 {
                if ctx.trace { eprintln!("[why f{} #{}] p{} {}: benefit {:.1} entry {:.1}", ctx.frame, self.id, f.id, describe_function(f), self.benefit(f.id, ctx), self.entry(f, ctx)); }
                doomed.push(f.id);
            }
        }
        let mut out = Vec::new();
        for fid in doomed { out.push(self.remove_function(fid)); }
        out.extend(self.drop_dead_classes());
        out
    }

    /// Delete the classes no function uses or that are left with one member (R41); the places that used one
    /// are re-decided at their function's next collapse, and a function left with nothing besides the owner
    /// goes.
    pub(crate) fn drop_dead_classes(&mut self) -> Vec<Retired> {
        let dead: Vec<PatternId> = self.classes.iter().filter(|c| c.members.len() < 2 || !self.functions.iter().any(|f| f.uses_class(c.id))).map(|c| c.id).collect();
        if dead.is_empty() { return Vec::new(); }
        self.classes.retain(|c| !dead.contains(&c.id));
        for f in self.functions.iter_mut() {
            if f.places.iter().any(|(p, _)| matches!(p, Place::Class(k) if dead.contains(k))) {
                f.places.retain(|(p, _)| !matches!(p, Place::Class(k) if dead.contains(k)));
                f.canonicalize();
                self.unsettled.insert(f.id);
            }
        }
        let empty: Vec<PatternId> = self.functions.iter().filter(|f| f.p_size() == 0).map(|f| f.id).collect();
        empty.into_iter().map(|fid| self.remove_function(fid)).collect()
    }

    /// Delete a function from the table.
    pub(crate) fn remove_function(&mut self, fid: PatternId) -> Retired {
        let idx = self.functions.iter().position(|f| f.id == fid).unwrap();
        let f = self.functions.remove(idx);
        self.unsettled.remove(&fid);
        Retired { pattern: fid, child: f.child, line: Some(f), heir: None }
    }
}
