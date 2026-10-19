//! Uses as places and the collapse (D27, D29): what a pattern names, re-decided over its uses.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::neuron::*;
use crate::types::*;

#[derive(Clone, Copy)]
pub(crate) struct Around {
    pub(crate) node: Node,
    pub(crate) coord: Pos,
    pub(crate) credit: f64,
    /// The pattern of the latest row's cover holding it, if any (D19).
    pub(crate) owned_by: Option<PatternId>,
    /// For a variable: the position index this coordinate holds.
    pub(crate) position: usize,
    pub(crate) row: usize,
    pub(crate) nb: Option<usize>,
}

/// A line in one readable row: `F` nodes with offsets in time, `C` members, `P` offsets.

impl Neuron {
    // ───────────────────────────── uses and surroundings (D27) ─────────────────────────────


    /// The uses a pattern covers: places, each with the rows that saw it, the latest row last (D27).
    pub(crate) fn uses_of(&self, pid: PatternId) -> Vec<(Pos, Vec<usize>)> {
        let mut map: FxHashMap<Pos, Vec<usize>> = FxHashMap::default();
        for (i, row) in self.history.iter().enumerate() {
            for e in &row.cover {
                if e.pattern != pid { continue; }
                let v = map.entry(e.place).or_default();
                if !v.contains(&i) { v.push(i); }
            }
            // a variable's uses include the places functions name it at
            for e in &row.cover {
                for nb in &e.nodes {
                    if let Some(vb) = &nb.var {
                        if vb.var == pid {
                            let place = nb_place(row, vb).unwrap_or(add(&row.pos, &add(&sub(&e.place, &row.pos), &nb.offset)));
                            let v = map.entry(place).or_default();
                            if !v.contains(&i) { v.push(i); }
                        }
                    }
                }
            }
        }
        let mut out: Vec<(Pos, Vec<usize>)> = map.into_iter().collect();
        out.sort();
        out
    }

    /// What stands around a place: the union of what every row that saw it recorded, read in nodes
    /// (D33): a claimed coordinate as its variable at its first position, any other neighbor as the
    /// neuron there, and each row's own activation as the owner at its offset from the place.
    pub(crate) fn surroundings(&self, rows: &[usize]) -> Vec<Around> {
        let mut out: Vec<Around> = Vec::new();
        let mut seen: FxHashMap<(Node, Pos), usize> = FxHashMap::default();
        // the latest row first, so it is the one read for ownership and credit
        let mut order: Vec<usize> = rows.to_vec();
        order.sort_by(|a, b| self.history[*b].pos[0].cmp(&self.history[*a].pos[0]).then(b.cmp(a)));
        for &i in &order {
            let row = &self.history[i];
            let o = Around { node: Node::Owner, coord: row.pos, credit: row.own_cost, owned_by: row.owner_self, position: 0, row: i, nb: None };
            if !seen.contains_key(&(Node::Owner, row.pos)) { seen.insert((Node::Owner, row.pos), out.len()); out.push(o); }
            for (j, nb) in row.nbs.iter().enumerate() {
                let coord = row.coord(j);
                let (node, position, credit) = match row.claim[j] {
                    Some((v, k)) => (Node::Var(v), k, 0.0),
                    None => (Node::Neuron(nb.neuron), 0, row.credit(j)),
                };
                let key = (node, coord);
                if !seen.contains_key(&key) {
                    seen.insert(key, out.len());
                    out.push(Around { node, coord, credit, owned_by: row.owner[j], position, row: i, nb: Some(j) });
                }
                // the child read as what it stands for
                if row.claim[j].is_none() && !nb.covers.is_empty() {
                    let share = row.credit(j) / nb.covers.len() as f64;
                    for (m, d) in &nb.covers {
                        let k2 = (Node::Neuron(*m), add(&row.pos, d));
                        if seen.contains_key(&k2) { continue; }
                        seen.insert(k2, out.len());
                        out.push(Around { node: k2.0, coord: k2.1, credit: share, owned_by: row.owner[j], position: 0, row: i, nb: Some(j) });
                    }
                }
            }
        }
        out
    }

    /// A variable's part (D22) at a place in a row, over all its positions at once.
    pub(crate) fn var_part_at(&self, row: &Row, v: PatternId, first: &Pos) -> Option<(f64, u32)> {
        let vp = self.pattern(v)?;
        let anchor = sub(first, &row.pos);
        let taken = vec![false; row.nbs.len()];
        match &vp.body {
            Body::Parameter { offsets, .. } => {
                let vb = self.bind_parameter(row, vp, offsets, &anchor, &taken)?;
                if vb.failed > 0 { return None; }
                let mut c = 0.0;
                for pj in vb.positions.iter().flatten() { c += row.credit(*pj); }
                Some((c - vb.choice, 0))
            }
            Body::Class { members, .. } => {
                let found: Vec<usize> = self.nbs_at(row, &anchor, None).into_iter().filter(|&j| members.iter().any(|m| m.neuron == row.nbs[j].neuron)).collect();
                match found.first() { Some(&j) => Some((row.credit(j) - vp.choice(), 0)), None => None }
            }
            _ => None,
        }
    }

    // ───────────────────────────── collapse (D27) ─────────────────────────────

    /// Re-decide what a function names over its uses (D27). Returns true when the line changed.
    pub(crate) fn collapse_function(&mut self, pid: PatternId, uses: &[(Pos, Vec<usize>)], ctx: Ctx) -> bool {
        let Some(p) = self.pattern(pid) else { return false };
        let Body::Function { nodes, .. } = &p.body else { return false };
        let nodes = nodes.clone();
        let s = uses.len();
        if s == 0 { return false; }
        let wp = self.which_pattern_bits();
        let names_owner = matches!(nodes.first(), Some((Node::Owner, _)));
        // candidates keyed by (node, offset from the reference): per use, what the surroundings hold there
        // and what it is worth; a use where the coordinate is named by another function of the cover is
        // neither a hold nor a miss for that node
        let mut holds: FxHashMap<(Node, Offset), FxHashMap<usize, f64>> = FxHashMap::default();
        let mut skipped: FxHashMap<(Node, Offset), FxHashSet<usize>> = FxHashMap::default();
        let mut own: Vec<f64> = Vec::with_capacity(s);
        for (u, (place, rows)) in uses.iter().enumerate() {
            let latest = rows.iter().copied().max_by_key(|&i| (self.history[i].pos[0], i)).map(|i| self.history[i].own_cost).unwrap_or(0.0);
            own.push(latest);
            let around = self.surroundings(rows);
            let mut here: FxHashSet<(Node, Offset)> = FxHashSet::default();
            for a in &around {
                let off = self.round(&sub(&a.coord, place));
                if off[0] > 0 { continue; }
                let key = (a.node, off);
                if here.contains(&key) { continue; }
                here.insert(key);
                let is_skipped = a.owned_by.map_or(false, |o| o != pid && self.pattern(o).map_or(false, |q| q.is_function()));
                let credit = match a.node {
                    Node::Var(v) => self.var_part_at(&self.history[a.row], v, &a.coord).map(|(c, _)| c).unwrap_or(0.0).max(0.0),
                    _ => a.credit,
                };
                if is_skipped { skipped.entry(key).or_default().insert(u); } else { holds.entry(key).or_default().insert(u, credit); }
            }
        }
        // the reference node always holds: it is the use
        let reference = nodes[0];
        let mut kept: Vec<(Node, Offset)> = vec![reference];
        let mut live: Vec<usize> = (0..s).collect();
        let mut narrowed = false;
        // what the line saves in each use as it stands: the reference's own credit less the bid's price
        let mut sav: Vec<f64> = (0..s).map(|u| {
            let r = match reference.0 { Node::Owner => own[u], _ => holds.get(&reference).and_then(|h| h.get(&u)).copied().unwrap_or(0.0) };
            r - own[u] - wp
        }).collect();
        // the nodes the line has, in their order, then the rest by how often and how much they hold
        let mut order: Vec<(Node, Offset)> = nodes.iter().skip(1).map(|(n, o)| (*n, *o)).collect();
        let mut rest: Vec<(Node, Offset)> = holds.keys().copied().filter(|k| !order.contains(k) && *k != reference).collect();
        rest.sort_by(|a, b| {
            let (ha, hb) = (holds[a].len(), holds[b].len());
            let (ca, cb): (f64, f64) = (holds[a].values().sum(), holds[b].values().sum());
            hb.cmp(&ha).then(cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal)).then(a.cmp(b))
        });
        order.extend(rest);
        for key in order {
            if key == reference || kept.contains(&key) { continue; }
            if matches!(key.0, Node::Owner) && !names_owner && key.1 == [0; MAX_DIMS] { continue; }
            if matches!(key.0, Node::Owner) && names_owner { continue; }
            if let Node::Var(v) = key.0 { if self.pattern(v).is_none() { continue; } }
            let Some(h) = holds.get(&key) else { continue };
            let sk = skipped.get(&key);
            let entry = ctx.offset_bits + match key.0 { Node::Owner => 0.0, Node::Neuron(_) => ctx.neuron_bits, Node::Var(_) => wp };
            // with the node: the line holds where the node holds or is skipped, and is given up elsewhere
            let mut gain = 0.0;
            let mut loss = 0.0;
            let mut next: Vec<usize> = Vec::new();
            for &u in &live {
                if let Some(c) = h.get(&u) { gain += c; next.push(u); }
                else if sk.map_or(false, |x| x.contains(&u)) { next.push(u); }
                else { loss += sav[u].max(0.0); }
            }
            if next.is_empty() || gain <= entry + loss { continue; }
            kept.push(key);
            for &u in &next { if let Some(c) = h.get(&u) { sav[u] += c; } }
            if next.len() < live.len() { narrowed = true; }
            live = next;
        }
        if kept.len() < 2 {
            // nothing but the reference: no relation, no function (D15)
            kept.truncate(1);
        }
        let new_nodes = canonical(kept);
        let changed = new_nodes != nodes;
        if changed {
            // a line that comes to hold in fewer places stands for something narrower: its child keeps its
            // name, which the lines above name, and lets go what it learned under the places given up
            let mut child_of = None;
            if let Some(p) = self.pattern_mut(pid) { if let Body::Function { nodes, child } = &mut p.body { *nodes = new_nodes; if narrowed { child_of = *child; } } }
            if let Some(c) = child_of { self.narrowed_children.push((pid, c)); }
        }
        changed
    }

    /// Re-decide a parameter's positions and members over its uses (D27).
    pub(crate) fn collapse_parameter(&mut self, pid: PatternId, uses: &[(Pos, Vec<usize>)], ctx: Ctx) -> bool {
        let Some(p) = self.pattern(pid) else { return false };
        let Body::Parameter { offsets, .. } = &p.body else { return false };
        let offsets = offsets.clone();
        let s = uses.len();
        if s == 0 { return false; }
        // per use: the value at the first position, and every coordinate holding it
        struct Cand { count: usize, covers: f64 }
        let mut cands: FxHashMap<Offset, Cand> = FxHashMap::default();
        let mut member_counts: FxHashMap<NeuronId, usize> = FxHashMap::default();
        for (place, rows) in uses {
            let mut here: FxHashSet<Offset> = FxHashSet::default();
            let mut value: Option<NeuronId> = None;
            // the value at the first position, read from the latest row
            for &i in rows.iter().rev() {
                let row = &self.history[i];
                let anchor = sub(place, &row.pos);
                for j in self.nbs_at(row, &anchor, None) { value = Some(row.value_at(j)); break; }
                if value.is_some() { break; }
            }
            let Some(value) = value else { continue };
            *member_counts.entry(value).or_default() += 1;
            for &i in rows {
                let row = &self.history[i];
                for (j, nb) in row.nbs.iter().enumerate() {
                    if row.value_at(j) != value { continue; }
                    if row.claim[j].map_or(false, |(v, _)| v != pid) { continue; }
                    let off = self.round(&sub(&row.coord(j), place));
                    if off[0] > 0 { continue; }
                    if !here.insert(off) { continue; }
                    let c = cands.entry(off).or_insert(Cand { count: 0, covers: 0.0 });
                    c.count += 1;
                    c.covers += nb.cost;
                }
            }
        }
        let mut kept: Vec<Offset> = vec![[0; MAX_DIMS]];
        let mut keys: Vec<Offset> = cands.keys().copied().collect();
        keys.sort();
        for off in keys {
            if off == [0; MAX_DIMS] { continue; }
            let c = &cands[&off];
            // exact holding: a position is kept only where it holds in every use
            if c.count == s && c.covers > ctx.offset_bits { kept.push(off); }
        }
        let changed = kept != offsets;
        if let Some(p) = self.pattern_mut(pid) {
            if let Body::Parameter { offsets, members } = &mut p.body {
                *offsets = kept;
                // members: whatever has stood at all its positions within the history (a record)
                for m in members.iter_mut() { m.count = member_counts.get(&m.neuron).copied().unwrap_or(0); }
                for (n, c) in member_counts { if !members.iter().any(|m| m.neuron == n) { members.push(Member { neuron: n, count: c, value_neuron: None }); } }
                members.retain(|m| m.count > 0 || m.value_neuron.is_some());
            }
        }
        changed
    }

    /// Re-decide a class's members over its uses (D27): candidates in order of count, each while what it
    /// covers beats its entry and the widening it gives every use the class already has.
    pub(crate) fn collapse_class(&mut self, pid: PatternId, uses: &[(Pos, Vec<usize>)], ctx: Ctx) -> bool {
        let Some(p) = self.pattern(pid) else { return false };
        let Body::Class { members, anchor } = &p.body else { return false };
        let old: Vec<NeuronId> = members.iter().map(|m| m.neuron).collect();
        let anchor = *anchor;
        let s = uses.len();
        // the neurons standing at the class's places, and which stood together in one use
        let mut counts: FxHashMap<NeuronId, (usize, f64)> = FxHashMap::default();
        let mut together: FxHashSet<(NeuronId, NeuronId)> = FxHashSet::default();
        for (place, rows) in uses {
            let mut here: Vec<NeuronId> = Vec::new();
            for &i in rows {
                let row = &self.history[i];
                let target = sub(place, &row.pos);
                for j in self.nbs_at(row, &target, None) {
                    if row.claim[j].map_or(false, |(v, _)| v != pid) { continue; }
                    let n = row.nbs[j].neuron;
                    if here.contains(&n) { continue; }
                    here.push(n);
                    let e = counts.entry(n).or_insert((0, 0.0));
                    e.0 += 1;
                    e.1 += row.credit(j);
                }
            }
            for a in &here { for b in &here { if a < b { together.insert((*a, *b)); } } }
        }
        let _ = anchor;
        let mut order: Vec<(NeuronId, usize, f64)> = counts.iter().map(|(n, (c, cov))| (*n, *c, *cov)).collect();
        order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut kept: Vec<NeuronId> = Vec::new();
        for (n, _, cov) in order {
            if kept.iter().any(|k| together.contains(&(n.min(*k), n.max(*k)))) { continue; }
            let k = kept.len().max(1) as f64;
            let widening = if kept.is_empty() { 0.0 } else { s as f64 * log2((k + 1.0) / k) };
            if cov > ctx.neuron_bits + widening { kept.push(n); } else if !kept.is_empty() { break; }
        }
        kept.sort();
        let changed = kept != old;
        let mut dropped: Vec<(PatternId, NeuronId)> = Vec::new();
        if let Some(p) = self.pattern_mut(pid) {
            if let Body::Class { members, .. } = &mut p.body {
                let mut newm: Vec<Member> = Vec::new();
                for m in members.iter() { if !kept.contains(&m.neuron) { if let Some(v) = m.value_neuron { dropped.push((pid, v)); } } }
                for n in &kept {
                    let count = counts.get(n).map(|c| c.0).unwrap_or(0);
                    let value_neuron = members.iter().find(|m| m.neuron == *n).and_then(|m| m.value_neuron);
                    newm.push(Member { neuron: *n, count, value_neuron });
                }
                *members = newm;
            }
        }
        self.dropped_values.extend(dropped);
        changed
    }

    /// Re-center every pattern named (D29): collapse over the uses it now covers, and re-cover the rows
    /// it stands in where its line moved.
    pub(crate) fn recenter(&mut self, changed: &FxHashSet<PatternId>, ctx: Ctx) {
        let mut ids: Vec<PatternId> = changed.iter().copied().filter(|p| self.pattern(*p).is_some()).collect();
        ids.sort();
        // variables first: functions read the rows as the variables rewrite them (D33)
        ids.sort_by_key(|p| self.pattern(*p).map_or(2, |q| if q.is_variable() { 0 } else { 1 }));
        let mut moved: Vec<PatternId> = Vec::new();
        for pid in ids {
            let uses = self.uses_of(pid);
            let kind = match self.pattern(pid).map(|p| &p.body) { Some(Body::Function { .. }) => 0, Some(Body::Parameter { .. }) => 1, Some(Body::Class { .. }) => 2, None => continue };
            let changed = match kind { 0 => self.collapse_function(pid, &uses, ctx), 1 => self.collapse_parameter(pid, &uses, ctx), _ => self.collapse_class(pid, &uses, ctx) };
            if changed { moved.push(pid); }
        }
        if moved.is_empty() { return; }
        for i in 0..self.history.len() {
            let touches = self.history[i].cover.iter().any(|e| moved.contains(&e.pattern) || e.nodes.iter().any(|nb| nb.var.as_ref().map_or(false, |v| moved.contains(&v.var))));
            if touches { self.history[i].dirty = true; }
        }
        self.recognize(ctx);
    }

}
