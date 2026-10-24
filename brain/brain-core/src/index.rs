//! The interneuron index (D36): what each interneuron stands for, its definition and its parents, and the
//! update children step that keeps them (§6, §6.3).

use rustc_hash::FxHashMap;

use crate::machine::*;
use crate::neuron::*;
use crate::types::*;

impl Machine {
    // ───────────────────────────── what a neuron stands for (D36) ─────────────────────────────

    /// What a neuron stands for (D36): a base symbol, or its definition's nodes and relations with the
    /// variables left open, each node for what it stands for in turn. A neuron met again on its own descent, as
    /// on a chain closed on itself, is written as itself and not read again.
    pub(crate) fn key_of(&self, n: NeuronId, path: &mut Vec<NeuronId>) -> String {
        let Some(nn) = self.thalamus.get(n) else { return format!("?{}", n) };
        if nn.kind != Kind::Inter { return format!("b{}", n); }
        if path.contains(&n) || path.len() > 256 { self.cycle_guard.set(self.cycle_guard.get() + 1); return format!("~{}", n); }
        let Some((owner, f)) = self.definitions.get(&n).cloned() else { return format!("?{}", n) };
        path.push(n);
        let s = self.body_key(owner, &f, path);
        path.pop();
        s
    }

    /// A function's nodes and relations (D34) with the variables left open, written the same from whichever
    /// node it is anchored on: every node at its offset from the earliest, in one order.
    fn body_key(&self, owner: NeuronId, f: &Function, path: &mut Vec<NeuronId>) -> String {
        let mut nodes: Vec<(Offset, String, Option<u32>)> = Vec::new();
        for (place, off) in &f.places {
            match place {
                Place::Owner => nodes.push((*off, self.key_of(owner, path), None)),
                Place::Fixed(m) => nodes.push((*off, self.key_of(*m, path), None)),
                Place::Class(_) => nodes.push((*off, "V".to_string(), None)),
                Place::Param(k) => nodes.push((*off, "P".to_string(), Some(*k))),
            }
        }
        let base = nodes.iter().map(|(o, _, _)| *o).min().unwrap_or([0; MAX_DIMS]);
        for (o, _, _) in nodes.iter_mut() { for d in 0..MAX_DIMS { o[d] -= base[d]; } }
        nodes.sort();
        let mut order: Vec<u32> = Vec::new();
        let parts: Vec<String> = nodes.iter().map(|(o, k, p)| {
            let k = match p { Some(x) => { if !order.contains(x) { order.push(*x); } format!("P{}", order.iter().position(|y| y == x).unwrap()) } None => k.clone() };
            format!("{}@{:?}", k, o)
        }).collect();
        format!("F[{}]", parts.join(","))
    }

    /// What a function's child would stand for (D36, R43).
    pub(crate) fn function_key(&self, owner: NeuronId, f: &Function, _classes: &[(PatternId, Vec<NeuronId>)]) -> String {
        self.body_key(owner, f, &mut Vec::new())
    }

    /// What a function's child would stand for, read within `horizon` of the owner's activation (R43): every
    /// base symbol its definition reaches there, down through each node's own definition, and a mark at each
    /// place a variable leaves open. A chain closed on itself is read round again as far as the horizon goes.
    pub(crate) fn window_of(&self, owner: NeuronId, f: &Function, horizon: &Pos) -> Vec<(NeuronId, Pos)> {
        let mut out = Vec::new();
        let mut seen: rustc_hash::FxHashSet<(NeuronId, Pos)> = rustc_hash::FxHashSet::default();
        self.window_places(owner, f, [0; MAX_DIMS], horizon, &mut seen, &mut out);
        out.sort();
        out.dedup();
        out
    }

    fn window_places(&self, owner: NeuronId, f: &Function, at: Pos, horizon: &Pos, seen: &mut rustc_hash::FxHashSet<(NeuronId, Pos)>, out: &mut Vec<(NeuronId, Pos)>) {
        for (place, off) in &f.places {
            let q = add(&at, off);
            if (0..MAX_DIMS).any(|d| q[d].abs() > horizon[d]) { continue; }
            match place {
                Place::Owner => self.window_node(owner, q, horizon, seen, out),
                Place::Fixed(m) => self.window_node(*m, q, horizon, seen, out),
                Place::Class(_) | Place::Param(_) => out.push((NeuronId::MAX, q)),
            }
        }
    }

    fn window_node(&self, n: NeuronId, q: Pos, horizon: &Pos, seen: &mut rustc_hash::FxHashSet<(NeuronId, Pos)>, out: &mut Vec<(NeuronId, Pos)>) {
        if !seen.insert((n, q)) { return; }
        let is_inter = self.thalamus.get(n).map_or(false, |x| x.kind == Kind::Inter);
        match self.definitions.get(&n).filter(|_| is_inter) {
            Some((o, d)) => { let (o, d) = (*o, d.clone()); self.window_places(o, &d, q, horizon, seen, out); }
            None => out.push((n, q)),
        }
    }

    /// The interneuron a function's child comes round to (R43, D36): one the child would be built over, met
    /// on its descent within that neuron's own span, that stands within its span for exactly what the child
    /// would there. The stretch from the child back to it is then a turn of what it already stands for, so a
    /// chain on input that repeats fires the neuron it minted a turn ago, and closes on itself.
    pub(crate) fn reusable(&self, owner: NeuronId, f: &Function) -> Option<NeuronId> {
        let bound = self.thalamus.iter().filter(|n| n.alive && n.kind == Kind::Inter).fold([0; MAX_DIMS], |b, n| max_pos(&b, &self.spans[n.id as usize]));
        let inter = |m: NeuronId| self.thalamus.get(m).map_or(false, |x| x.alive && x.kind == Kind::Inter);
        let mut seen: rustc_hash::FxHashSet<(NeuronId, Pos)> = rustc_hash::FxHashSet::default();
        let mut stack: Vec<(NeuronId, Pos)> = Vec::new();
        let mut met: Vec<NeuronId> = Vec::new();
        let push = |o: NeuronId, d: &Function, at: Pos, stack: &mut Vec<(NeuronId, Pos)>| {
            for (place, off) in &d.places {
                let m = match place { Place::Owner => o, Place::Fixed(m) => *m, _ => continue };
                stack.push((m, add(&at, off)));
            }
        };
        push(owner, f, [0; MAX_DIMS], &mut stack);
        while let Some((n, q)) = stack.pop() {
            if !inter(n) || (0..MAX_DIMS).any(|d| q[d].abs() > bound[d]) || !seen.insert((n, q)) { continue; }
            if q != [0; MAX_DIMS] && (0..MAX_DIMS).all(|d| q[d].abs() <= self.spans[n as usize][d]) { met.push(n); }
            if let Some((o, d)) = self.definitions.get(&n) { push(*o, d, q, &mut stack); }
        }
        met.sort_unstable();
        met.dedup();
        met.into_iter().find(|n| {
            let Some((o, d)) = self.definitions.get(n) else { return false };
            let horizon = self.spans[*n as usize];
            let w = self.window_of(*o, d, &horizon);
            !w.is_empty() && self.window_of(owner, f, &horizon) == w
        })
    }

    // ───────────────────────────── update children (§6, §6.3) ─────────────────────────────

    /// Once the frame's last level has run (§6): update child for every function whose bid won a child, then,
    /// for every interneuron whose definition changed, its new definition into the index and update child for
    /// each of its parents. A parent is kept where the function points at the child, and dropped where it
    /// detached (D36). Then the index is recomputed, two interneurons that stand for one thing merged into the
    /// older (R43), and the spans follow (D4).
    pub(crate) fn update_children(&mut self) {
        let won = std::mem::take(&mut self.won);
        let redefined = std::mem::take(&mut self.redefined);
        let mut calls: Vec<(NeuronId, PatternId, NeuronId)> = won;
        for (child, _) in &redefined {
            for &(o, f) in self.thalamus[*child].parents.clone().iter() { if !calls.contains(&(o, f, *child)) { calls.push((o, f, *child)); } }
        }
        for (owner, fid, child) in calls {
            let Some((def_owner, def)) = self.definitions.get(&child).cloned() else { continue };
            if !self.thalamus.get(owner).map_or(false, |n| n.alive) { continue; }
            let ctx = self.ctx_for(owner);
            let before = self.thalamus[owner].function(fid).and_then(|f| f.child);
            let what = self.thalamus[owner].update_child(fid, child, def_owner, &def, ctx);
            match what {
                ChildUpdate::Pointed | ChildUpdate::Adapted => {
                    // a function points at one child: off the one it pointed at before
                    if let Some(old) = before.filter(|o| *o != child) {
                        if let Some(o) = self.thalamus.get_mut(old) { o.parents.retain(|p| *p != (owner, fid)); }
                    }
                    if !self.thalamus[child].parents.contains(&(owner, fid)) { self.thalamus[child].parents.push((owner, fid)); }
                }
                ChildUpdate::Detached | ChildUpdate::Gone => { self.thalamus[child].parents.retain(|p| *p != (owner, fid)); }
            }
            if self.trace { eprintln!("[update f{}] #{} p{} -> #{}: {:?}", self.frame_no, owner, fid, child, what); }
        }
        // children no function points at wait on the ledger (R38)
        let orphans: Vec<NeuronId> = self.thalamus.iter().filter(|n| n.alive && n.kind == Kind::Inter && n.parents.is_empty()).map(|n| n.id).collect();
        for c in orphans { if !self.ledger.contains(&c) { self.ledger.push(c); } }
        self.refresh_index();
    }

    /// Recompute what every interneuron stands for (D36); where two stand for one thing the younger's
    /// functions are rewired to the older (R43). Spans follow (D4).
    pub(crate) fn refresh_index(&mut self) {
        let ids: Vec<NeuronId> = self.thalamus.iter().filter(|n| n.alive && n.kind == Kind::Inter).map(|n| n.id).collect();
        let mut index: FxHashMap<String, NeuronId> = FxHashMap::default();
        let mut rewire: Vec<(NeuronId, NeuronId)> = Vec::new();
        for &id in &ids {
            let key = self.key_of(id, &mut Vec::new());
            self.thalamus[id].key = key.clone();
            if key.starts_with('?') { continue; }
            match index.get(&key) {
                Some(&older) => rewire.push((id, older)),
                None => { index.insert(key, id); }
            }
        }
        self.index = index;
        for (young, old) in rewire {
            let parents = std::mem::take(&mut self.thalamus[young].parents);
            for (owner, pid) in parents {
                self.thalamus[owner].wire_child(pid, old);
                if !self.thalamus[old].parents.contains(&(owner, pid)) { self.thalamus[old].parents.push((owner, pid)); }
            }
            if !self.ledger.contains(&young) { self.ledger.push(young); }
        }
        let mut memo: FxHashMap<NeuronId, Pos> = FxHashMap::default();
        for &id in &ids {
            if !self.thalamus[id].alive { continue; }
            let s = self.neuron_span(id, &mut memo, &mut Vec::new());
            self.spans[id as usize] = s;
            let rt = s[0] + self.radii[0];
            if rt > self.max_reach_t { self.max_reach_t = rt; }
        }
    }
}
