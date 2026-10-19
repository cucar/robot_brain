//! Margin and retirement (D30, R18, R41).

use rustc_hash::{FxHashSet};

use crate::neuron::*;
use crate::types::*;

impl Neuron {
    // ───────────────────────────── margin and retirement (D30, R18, R41) ─────────────────────────────

    /// What a pattern saves over the history less its line (D30).
    pub fn margin(&self, pid: PatternId, ctx: Ctx) -> f64 {
        let Some(p) = self.pattern(pid) else { return 0.0 };
        self.benefit(pid, ctx) - self.line_cost(p, ctx)
    }

    pub(crate) fn entry_savings(&self, row: &Row, e: &CoverEntry) -> f64 {
        let wp = self.which_pattern_bits();
        let mut coverage = 0.0;
        let mut choices = 0.0;
        let mut p_size = 0u32;
        for nb in &e.nodes {
            match &nb.var {
                Some(v) => {
                    let vp = self.pattern(v.var);
                    p_size += vp.map_or(1, |q| q.positions() as u32);
                    choices += v.choice;
                    for pj in v.positions.iter().flatten() { if row.owner[*pj] == Some(e.pattern) { coverage += row.credit(*pj); } }
                }
                None => {
                    p_size += 1;
                    for &j in &nb.nbs { if row.owner[j] == Some(e.pattern) { coverage += row.credit(j); } }
                }
            }
        }
        if e.credited_owner && row.owner_self == Some(e.pattern) { coverage += row.own_cost; }
        let price = row.own_cost + wp + choices + e.failed as f64 * log2(p_size.max(1) as f64);
        coverage - price
    }

    /// A row's cover savings as it stands (D22 summed), or without one pattern (D30).
    pub(crate) fn cover_savings(&self, i: usize, without: Option<PatternId>, ctx: Ctx) -> f64 {
        let row = &self.history[i];
        match without {
            None => row.cover.iter().map(|e| self.entry_savings(row, e)).sum(),
            Some(x) => {
                if !row.cover.iter().any(|e| e.pattern == x || e.nodes.iter().any(|nb| nb.var.as_ref().map_or(false, |v| v.var == x))) {
                    return row.cover.iter().map(|e| self.entry_savings(row, e)).sum();
                }
                let cover = self.greedy_cover_without(i, Some(x), ctx);
                self.savings_of_cover(i, &cover)
            }
        }
    }

    pub(crate) fn benefit(&self, pid: PatternId, ctx: Ctx) -> f64 {
        let Some(p) = self.pattern(pid) else { return 0.0 };
        let mut total = 0.0;
        if p.is_function() {
            // each use once, in its latest row
            let mut seen: FxHashSet<Pos> = FxHashSet::default();
            for i in (0..self.history.len()).rev() {
                let row = &self.history[i];
                let places: Vec<Pos> = row.cover.iter().filter(|e| e.pattern == pid).map(|e| e.place).collect();
                if places.is_empty() { continue; }
                if places.iter().all(|pl| seen.contains(pl)) { continue; }
                for pl in &places { seen.insert(*pl); }
                total += self.cover_savings(i, None, ctx) - self.cover_savings(i, Some(pid), ctx);
            }
        } else {
            let mut seen: FxHashSet<Pos> = FxHashSet::default();
            for i in (0..self.history.len()).rev() {
                let row = &self.history[i];
                for e in &row.cover {
                    for nb in &e.nodes {
                        let Some(v) = &nb.var else { continue };
                        if v.var != pid { continue; }
                        let Some(place) = nb_place(row, v) else { continue };
                        if !seen.insert(place) { continue; }
                        let mut c = 0.0;
                        for pj in v.positions.iter().flatten() { if row.owner[*pj] == Some(e.pattern) { c += row.credit(*pj); } }
                        total += c - v.choice - v.failed as f64 * log2(p.positions() as f64);
                    }
                }
            }
        }
        total
    }

    /// Retire every pattern whose margin is negative, every parameter left with fewer than two
    /// positions and every class left with one member (R18, R41).
    pub(crate) fn retire(&mut self, ctx: Ctx) -> Vec<Retired> {
        let mut out = Vec::new();
        // one at a time: the worst goes, and the rest are measured again without it, since two lines
        // covering the same rows each show no margin while the other stands (D30)
        loop {
            let mut worst: Option<(PatternId, f64, bool)> = None;
            for p in &self.table {
                let degenerate = match &p.body {
                    Body::Parameter { offsets, .. } => offsets.len() < 2,
                    Body::Class { members, .. } => members.len() < 2,
                    Body::Function { nodes, .. } => nodes.len() < 2,
                };
                let m = if degenerate { f64::NEG_INFINITY } else { self.margin(p.id, ctx) };
                if m < 0.0 && worst.map_or(true, |(_, w, _)| m < w) { worst = Some((p.id, m, degenerate)); }
            }
            let Some((pid, _, degenerate)) = worst else { break };
            if ctx.trace {
                let p = self.pattern(pid).unwrap();
                eprintln!("[why f{} #{}] p{} {} {}: benefit {:.1} line {:.1} uses {}", ctx.frame, self.id, p.id, describe_body(&p.body), if degenerate { "degenerate" } else { "margin<0" }, self.benefit(p.id, ctx), self.line_cost(p, ctx), self.history.iter().filter(|r| r.cover.iter().any(|e| e.pattern == p.id)).count());
            }
            out.push(self.remove_pattern(pid));
            // functions that named a retired variable re-center without it (R18); rows fall dirty
            self.recognize(ctx);
        }
        out
    }

    /// Delete a pattern from the table; its covers fall to the residual.
    pub(crate) fn remove_pattern(&mut self, pid: PatternId) -> Retired {
        let idx = self.index_of(pid).unwrap();
        let p = self.table.remove(idx);
        let value_neurons: Vec<NeuronId> = p.members().iter().filter_map(|m| m.value_neuron).collect();
        // functions naming the variable lose the node
        for q in self.table.iter_mut() {
            if let Body::Function { nodes, .. } = &mut q.body {
                if nodes.iter().any(|(n, _)| *n == Node::Var(pid)) {
                    let keep: Vec<(Node, Offset)> = nodes.iter().copied().filter(|(n, _)| *n != Node::Var(pid)).collect();
                    *nodes = canonical(keep);
                }
            }
        }
        for row in self.history.iter_mut() {
            let had = row.cover.iter().any(|e| e.pattern == pid || e.nodes.iter().any(|nb| nb.var.as_ref().map_or(false, |v| v.var == pid)));
            if had { row.dirty = true; }
        }
        Retired { pattern: pid, child: p.child(), value_neurons, member_only: false }
    }

}
