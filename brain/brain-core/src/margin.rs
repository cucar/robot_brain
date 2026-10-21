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

    /// What one entry of a row's cover saves (D22), read with the row's owners as they stand: what it was
    /// credited, less the bidder, which function, its choices and its failures.
    pub(crate) fn entry_savings(&self, row: &Row, e: &CoverEntry) -> f64 {
        let wf = self.which_function_bits();
        let mut coverage = 0.0;
        let mut seen: Vec<usize> = Vec::new();
        for b in &e.binds { for &j in &b.nbs { if row.owner[j] == Some(e.function) && !seen.contains(&j) { seen.push(j); coverage += row.credit(j); } } }
        if e.credited_owner && row.owner_self == Some(e.function) { coverage += row.own_cost; }
        let p = self.function(e.function).map_or(1, |f| f.p_size().max(1));
        coverage - (row.own_cost + wf + e.choice + e.failed as f64 * log2(p as f64))
    }

    /// What a cover saves over a row (D22 summed), with its owners as that cover would set them.
    pub(crate) fn savings_of_cover(&self, i: usize, cover: &[CoverEntry]) -> f64 {
        let row = &self.history[i];
        let mut tmp = row.clone();
        let n = tmp.nbs.len();
        tmp.owner = vec![None; n];
        tmp.owner_self = None;
        for e in cover {
            if e.credited_owner && tmp.owner_self.is_none() { tmp.owner_self = Some(e.function); }
            for b in &e.binds { for &j in &b.nbs { if tmp.owner[j].is_none() { tmp.owner[j] = Some(e.function); } } }
        }
        cover.iter().map(|e| self.entry_savings(&tmp, e)).sum()
    }

    /// The benefit of a function (D30): over the rows it covers, what the cover saves with it less what the
    /// same cover saves without it.
    pub(crate) fn benefit(&self, fid: PatternId, _ctx: Ctx) -> f64 {
        let mut total = 0.0;
        for i in self.rows_of(fid) {
            // without it, what it held falls to the residual unless another function of the cover names it
            let cover = &self.history[i].cover;
            let without: Vec<CoverEntry> = cover.iter().filter(|e| e.function != fid).cloned().collect();
            total += self.savings_of_cover(i, cover) - self.savings_of_cover(i, &without);
        }
        total
    }

    /// Retire every function whose margin is strictly negative, measured on the table as it stands and
    /// retired together, and every class no function uses or left with one member (R18, R41).
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
        if !out.is_empty() { self.recognize(ctx); }
        out
    }

    /// Delete the classes no function uses or that are left with one member (R41); the places that used
    /// one are re-decided at their function's next collapse, and a function left with nothing besides the
    /// owner goes.
    pub(crate) fn drop_dead_classes(&mut self) -> Vec<Retired> {
        let dead: Vec<PatternId> = self.classes.iter().filter(|c| c.members.len() < 2 || !self.functions.iter().any(|f| f.uses_class(c.id))).map(|c| c.id).collect();
        if dead.is_empty() { return Vec::new(); }
        self.classes.retain(|c| !dead.contains(&c.id));
        let mut hit: Vec<PatternId> = Vec::new();
        for f in self.functions.iter_mut() {
            if f.places.iter().any(|(p, _)| matches!(p, Place::Class(k) if dead.contains(k))) {
                f.places.retain(|(p, _)| !matches!(p, Place::Class(k) if dead.contains(k)));
                f.canonicalize();
                hit.push(f.id);
            }
        }
        for row in self.history.iter_mut() { if row.cover.iter().any(|e| hit.contains(&e.function)) { row.dirty = true; } }
        let empty: Vec<PatternId> = self.functions.iter().filter(|f| f.p_size() == 0).map(|f| f.id).collect();
        empty.into_iter().map(|fid| self.remove_function(fid)).collect()
    }

    /// Delete a function from the table; what it held falls to the residual.
    pub(crate) fn remove_function(&mut self, fid: PatternId) -> Retired {
        let idx = self.functions.iter().position(|f| f.id == fid).unwrap();
        let f = self.functions.remove(idx);
        for row in self.history.iter_mut() { if row.cover.iter().any(|e| e.function == fid) { row.dirty = true; } }
        Retired { pattern: fid, child: f.child, line: Some(f) }
    }
}
