//! Functions and the dictionary (§3.6, D15, D34, D36, D56): a function is a graph over neurons, one node per
//! neuron, connected, with at most one cycle, listed from its origin; the dictionary holds every function, each
//! relation in one function only, and the indexes the machine reads them through.

use rustc_hash::FxHashMap;

use crate::types::*;

pub type FuncId = u32;

/// A relation in its one form (D34): two neurons and the offset from the first to the second, one component per
/// activation dimension. In time it runs from the later neuron to the earlier, at a negative offset; where the
/// offset is zero in time, the first nonzero component is positive, and where every component is zero, the older
/// neuron is first.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Rel {
    pub a: Sym,
    pub b: Sym,
    pub off: Offset,
}

fn neg(o: &Offset) -> Offset { let mut r = *o; for x in r.iter_mut() { *x = -*x; } r }

/// The one form of the relation from `n` to `m` standing at `off` from it (D34).
pub fn canon(n: Sym, m: Sym, off: Offset) -> Rel {
    let first = off.iter().copied().find(|x| *x != 0);
    let keep = if off[0] != 0 { off[0] < 0 } else {
        match first { Some(x) => x > 0, None => n <= m }
    };
    if keep { Rel { a: n, b: m, off } } else { Rel { a: m, b: n, off: neg(&off) } }
}

impl Rel {
    /// Each end of this relation as its own rows hold it, with the neighbor and offset there (D47, D34): both,
    /// since the earlier end's row takes what followed it (D18); a relation from a node to itself is one relation,
    /// read from the node toward the next it reaches.
    pub fn ends(&self) -> Vec<(Sym, Sym, Offset)> {
        if self.a == self.b { return vec![(self.a, self.b, self.off)]; }
        vec![(self.a, self.b, self.off), (self.b, self.a, neg(&self.off))]
    }
}

/// A function (D15, D56): its nodes, the origin first, one per neuron in its pose, and its relations.
#[derive(Clone, Debug)]
pub struct Func {
    pub id: FuncId,
    pub nodes: Vec<Sym>,
    pub rels: Vec<Rel>,
    /// Per relation, the node it is counted at: where it attached to the function (R50), none for a relation
    /// that has met no other yet.
    pub at: Vec<Option<Sym>>,
    /// Its interneuron once a fit of it has been accepted (R16, R43).
    pub inter: Option<NeuronId>,
    /// When it joined the dictionary: earlier is older (R24).
    pub joined: u64,
}

/// How a function reads from its origin (D15, D54): the relations of a spanning tree in the order they are
/// reached, each as (relation, from node, to node, offset from the one to the other); the relation that closes the
/// cycle, if there is one, the same way; and each node's position from the origin over one lap.
#[derive(Clone, Debug, Default)]
pub struct Layout {
    pub tree: Vec<(usize, usize, usize, Offset)>,
    pub closing: Option<(usize, usize, usize, Offset)>,
    pub position: Vec<Pos>,
    /// Per node, the tree relation that reached it and the node it came from.
    pub parent: Vec<Option<(usize, usize)>>,
}

impl Func {
    /// A function of these relations, listed from `origin` (D15).
    pub fn from_rels(origin: Sym, rels: &[Rel]) -> Func {
        let mut nodes = vec![origin];
        for r in rels { for x in [r.a, r.b] { if !nodes.contains(&x) { nodes.push(x); } } }
        Func { id: 0, nodes, rels: rels.to_vec(), at: vec![None; rels.len()], inter: None, joined: 0 }
    }
    /// Where each relation is counted once a relation attaches at `at`: a relation that had met none before and
    /// has `at` for an end meets it there (R50).
    pub fn meeting(rels: &[Rel], at: &[Option<Sym>], node: Sym) -> Vec<Option<Sym>> {
        rels.iter().zip(at).map(|(r, a)| a.or(if r.a == node || r.b == node { Some(node) } else { None })).collect()
    }
    /// Keep the relations that pass, with where each is counted.
    pub fn keep(&mut self, f: impl Fn(&Rel) -> bool) {
        let (rels, at): (Vec<Rel>, Vec<Option<Sym>>) = self.rels.iter().zip(&self.at).filter(|(r, _)| f(r)).map(|(r, a)| (*r, *a)).unzip();
        self.rels = rels;
        self.at = at;
    }
    pub fn origin(&self) -> Sym { self.nodes[0] }
    pub fn node_of(&self, n: Sym) -> Option<usize> { self.nodes.iter().position(|x| *x == n) }
    pub fn holds(&self, r: &Rel) -> bool { self.rels.contains(r) }
    pub fn has_cycle(&self) -> bool { self.rels.len() >= self.nodes.len() }

    pub fn layout(&self) -> Layout { self.layout_from(0) }

    /// The offset a lap of its cycle advances by (D15), where it has one.
    pub fn lap(&self) -> Option<Offset> {
        let lay = self.layout();
        let (_, ua, ub, off) = lay.closing?;
        let mut l = [0i32; MAX_DIMS];
        for d in 0..MAX_DIMS { l[d] = lay.position[ua][d] + off[d] - lay.position[ub][d]; }
        Some(l)
    }

    /// The nodes of its cycle in the order a lap reads them back from node `root` (D53, D54), the root first.
    pub fn back_order(&self, root: usize) -> Vec<usize> {
        let Some(l) = self.lap() else { return vec![] };
        let lay = self.layout_from(root);
        let mut v = self.cycle_nodes();
        v.sort_by_key(|&i| (lay.position[i].iter().zip(&l).map(|(a, b)| *a as i64 * *b as i64).sum::<i64>(), i != root, i));
        v
    }

    /// How many nodes of its cycle a call in a pose from node `from` to node `to` round `laps` stands for (D53).
    pub fn stretch(&self, from: usize, to: usize, laps: u32) -> usize {
        let order = self.back_order(from);
        let c = order.len();
        let rank = order.iter().position(|x| *x == to).unwrap_or(c.saturating_sub(1));
        laps as usize * c + rank + 1
    }

    /// The nodes of its cycle, where it has one.
    pub fn cycle_nodes(&self) -> Vec<usize> {
        let lay = self.layout();
        let Some((_, ua, ub, _)) = lay.closing else { return vec![] };
        let mut v = vec![ua, ub];
        for s in Func::tree_path(&lay, ub, ua) { v.push(s.1); v.push(s.2); }
        v.sort_unstable();
        v.dedup();
        v
    }

    /// How it reads entered at node `root` (D54): the same as `layout`, each position from that node. Where two
    /// relations would place a node, as in a cycle, the one that reads back from the node it is entered at places
    /// it, and the one leading forward is the one that closes the cycle.
    pub fn layout_from(&self, root: usize) -> Layout {
        let n = self.nodes.len();
        let mut position: Vec<Option<Pos>> = vec![None; n];
        let mut parent: Vec<Option<(usize, usize)>> = vec![None; n];
        position[root] = Some([0; MAX_DIMS]);
        let mut used = vec![false; self.rels.len()];
        let mut tree = Vec::new();
        let mut queue = vec![root];
        let mut qi = 0;
        while qi < queue.len() {
            let u = queue[qi];
            qi += 1;
            let mut steps: Vec<(bool, usize, usize, Offset)> = Vec::new();
            for (ri, r) in self.rels.iter().enumerate() {
                if used[ri] { continue; }
                let (ua, ub) = (self.node_of(r.a).unwrap(), self.node_of(r.b).unwrap());
                let (v, off) = if ua == u { (ub, r.off) } else if ub == u { (ua, neg(&r.off)) } else { continue };
                steps.push((later(&off, &[0; MAX_DIMS]), ri, v, off));
            }
            steps.sort_by_key(|s| s.0);
            for (_, ri, v, off) in steps {
                if used[ri] || position[v].is_some() { continue; }
                used[ri] = true;
                let mut p = position[u].unwrap();
                for d in 0..MAX_DIMS { p[d] += off[d]; }
                position[v] = Some(p);
                parent[v] = Some((ri, u));
                tree.push((ri, u, v, off));
                queue.push(v);
            }
        }
        let closing = self.rels.iter().enumerate().find(|(ri, _)| !used[*ri]).map(|(ri, r)| {
            let (ua, ub) = (self.node_of(r.a).unwrap(), self.node_of(r.b).unwrap());
            (ri, ua, ub, r.off)
        });
        Layout { tree, closing, position: position.into_iter().map(|p| p.unwrap_or([0; MAX_DIMS])).collect(), parent }
    }

    /// The tree relations from `from` to `to` along the spanning tree, in order, each with its offset in that
    /// direction (the cycle's path, read once round from the node the closing relation reaches).
    pub fn tree_path(lay: &Layout, from: usize, to: usize) -> Vec<(usize, usize, usize, Offset)> {
        let up = |mut x: usize| { let mut v = vec![x]; while let Some((_, p)) = lay.parent[x] { v.push(p); x = p; } v };
        let (pf, pt) = (up(from), up(to));
        let lca = *pf.iter().find(|x| pt.contains(x)).unwrap_or(pf.last().unwrap_or(&0));
        let mut steps = Vec::new();
        // from `from` up to the meeting node, each step against the tree's direction
        let mut x = from;
        while x != lca {
            let (ri, p) = lay.parent[x].unwrap();
            let off = lay.tree.iter().find(|t| t.0 == ri).unwrap().3;
            steps.push((ri, x, p, neg(&off)));
            x = p;
        }
        // then down to `to`, with it
        let mut down = Vec::new();
        let mut y = to;
        while y != lca {
            let (ri, p) = lay.parent[y].unwrap();
            let off = lay.tree.iter().find(|t| t.0 == ri).unwrap().3;
            down.push((ri, p, y, off));
            y = p;
        }
        down.reverse();
        steps.extend(down);
        steps
    }

    /// How far its activation sees (D4): one position beyond its farthest node from its origin, in each
    /// dimension of `axes`.
    pub fn reach(&self, axes: u8) -> Pos {
        let lay = self.layout();
        let mut r = [0; MAX_DIMS];
        for d in 0..MAX_DIMS {
            if d > 0 && axes & (1 << d) == 0 { continue; }
            r[d] = lay.position.iter().map(|p| p[d].abs()).max().unwrap_or(0) + 1;
        }
        r
    }

    /// Two functions with the same relations and the same origin are one (D36).
    pub fn key(&self) -> (Sym, Vec<Rel>) {
        let mut r = self.rels.clone();
        r.sort();
        (self.origin(), r)
    }
}

pub fn describe_rel(r: &Rel) -> String { format!("{}→{}{:?}", sym_name(r.a), sym_name(r.b), r.off) }

/// A symbol as the trace prints it: `#n` for a base neuron, `s<k>` for a call in a pose.
pub fn sym_name(s: Sym) -> String { if s < SYM_CALL { format!("#{}", s) } else { format!("s{}", s - SYM_CALL) } }

/// The dictionary (D36, D56).
#[derive(Default)]
pub struct Dictionary {
    pub funcs: FxHashMap<FuncId, Func>,
    /// Every relation to the one function that holds it (R50).
    pub by_rel: FxHashMap<Rel, FuncId>,
    /// Per interneuron, its functions (R43: it may hold more than one).
    pub by_inter: FxHashMap<NeuronId, Vec<FuncId>>,
    /// Per neuron, the functions with a node of it in any pose (R38): kept as the functions change.
    by_node: FxHashMap<NeuronId, Vec<FuncId>>,
    /// The names of the calls (D8): each interneuron in each pose it has fired in.
    pub syms: Vec<(NeuronId, Pose)>,
    sym_index: FxHashMap<(NeuronId, Pose), Sym>,
    next_id: FuncId,
    next_join: u64,
}

impl Dictionary {
    pub fn get(&self, id: FuncId) -> Option<&Func> { self.funcs.get(&id) }
    pub fn len(&self) -> usize { self.funcs.len() }
    pub fn holder(&self, r: &Rel) -> Option<FuncId> { self.by_rel.get(r).copied() }
    pub fn of_inter(&self, n: NeuronId) -> &[FuncId] { self.by_inter.get(&n).map_or(&[], |v| v.as_slice()) }
    /// The functions whose nodes name a neuron, in any pose (R38, the subtree).
    pub fn naming(&self, n: NeuronId) -> Vec<FuncId> {
        let mut v = self.by_node.get(&n).cloned().unwrap_or_default();
        v.sort_unstable();
        v
    }

    /// Keep the index of functions by neuron as a function's nodes go from `before` to `after`.
    fn reindex(&mut self, id: FuncId, before: &[Sym], after: &[Sym]) {
        let old: Vec<NeuronId> = before.iter().map(|s| self.neuron_of(*s)).collect();
        let new: Vec<NeuronId> = after.iter().map(|s| self.neuron_of(*s)).collect();
        for n in &old {
            if new.contains(n) { continue; }
            if let Some(v) = self.by_node.get_mut(n) { v.retain(|x| *x != id); if v.is_empty() { self.by_node.remove(n); } }
        }
        for n in &new {
            if old.contains(n) { continue; }
            let v = self.by_node.entry(*n).or_default();
            if !v.contains(&id) { v.push(id); }
        }
    }

    /// The neuron a symbol names.
    pub fn neuron_of(&self, s: Sym) -> NeuronId { if s < SYM_CALL { s } else { self.syms.get((s - SYM_CALL) as usize).map_or(NeuronId::MAX, |x| x.0) } }
    /// The pose a call's symbol names; none for a base neuron.
    pub fn pose_of(&self, s: Sym) -> Option<Pose> { if s < SYM_CALL { None } else { self.syms.get((s - SYM_CALL) as usize).map(|x| x.1) } }
    /// The node a call's pose starts at (D54): a fit lands that node on the call.
    pub fn start_of(&self, s: Sym) -> Option<NeuronId> { self.pose_of(s).map(|p| p.from) }
    /// Whether a call's pose starts at a node (D54): the pose's start neuron is the node's neuron.
    pub fn starts_at(&self, call: Sym, node: Sym) -> bool { self.start_of(call).map_or(false, |n| n == self.neuron_of(node)) }
    /// The node of a function that stands for a neuron: one node per neuron (D15).
    pub fn node_by_neuron(&self, f: &Func, n: NeuronId) -> Option<usize> { f.nodes.iter().position(|s| self.neuron_of(*s) == n) }
    /// The symbol of an interneuron in a pose, allotted the first time it is named.
    pub fn sym_of(&mut self, n: NeuronId, pose: Pose) -> Sym {
        if let Some(s) = self.sym_index.get(&(n, pose)) { return *s; }
        let s = SYM_CALL + self.syms.len() as Sym;
        self.syms.push((n, pose));
        self.sym_index.insert((n, pose), s);
        s
    }
    /// The symbol of an interneuron in a pose where one is allotted already.
    pub fn find_sym(&self, n: NeuronId, pose: Pose) -> Option<Sym> { self.sym_index.get(&(n, pose)).copied() }
    /// A call's symbol in its pose less its laps (D25): the node it started at and the node it stands at.
    pub fn sym_no_laps(&mut self, s: Sym) -> Sym {
        match self.pose_of(s) { Some(p) if p.laps > 0 => { let n = self.neuron_of(s); self.sym_of(n, Pose { laps: 0, ..p }) } _ => s }
    }
    /// The same, reading only symbols already allotted.
    pub fn sym_no_laps_read(&self, s: Sym) -> Sym {
        match self.pose_of(s) { Some(p) if p.laps > 0 => self.find_sym(self.neuron_of(s), Pose { laps: 0, ..p }).unwrap_or(s), _ => s }
    }
    /// Every symbol of a neuron, and its poses are forgotten: a neuron allocated again names its poses afresh.
    pub fn drop_syms(&mut self, n: NeuronId) -> Vec<Sym> {
        let mut v = vec![n];
        for (i, (m, _)) in self.syms.iter().enumerate() { if *m == n { v.push(SYM_CALL + i as Sym); } }
        self.sym_index.retain(|(m, _), _| *m != n);
        v
    }
    /// A symbol with its neuron and pose, for the trace.
    pub fn sym_text(&self, s: Sym) -> String {
        match self.pose_of(s) {
            None => format!("#{}", s),
            Some(p) => format!("#{}<{}..{}x{}>", self.neuron_of(s), format!("#{}", p.from), format!("#{}", p.to), p.laps),
        }
    }

    /// Functions with the same relations and origin (D36).
    pub fn find_key(&self, key: &(Sym, Vec<Rel>)) -> Option<FuncId> {
        self.funcs.values().find(|f| f.key() == *key).map(|f| f.id)
    }

    /// A new function of one relation, its origin the relation's first neuron (R50, D15).
    pub fn start(&mut self, r: Rel) -> FuncId {
        let nodes = if r.a == r.b { vec![r.a] } else { vec![r.a, r.b] };
        self.insert(Func { id: 0, nodes, rels: vec![r], at: vec![None], inter: None, joined: 0 })
    }

    fn insert(&mut self, mut f: Func) -> FuncId {
        let id = self.next_id;
        self.next_id += 1;
        f.id = id;
        f.joined = self.next_join;
        self.next_join += 1;
        for r in &f.rels { self.by_rel.insert(*r, id); }
        if let Some(n) = f.inter { self.by_inter.entry(n).or_default().push(id); }
        let nodes = f.nodes.clone();
        self.funcs.insert(id, f);
        self.reindex(id, &[], &nodes);
        id
    }

    /// A relation joins a function at node `at` (R54, R50): a new node where its other neuron is not one, a cycle
    /// where it is. It is counted at that node, and so is each relation of the function it meets there that had
    /// met none before.
    pub fn join(&mut self, id: FuncId, r: Rel, at: Sym) {
        let Some(f) = self.funcs.get_mut(&id) else { return };
        let before = f.nodes.clone();
        if !f.nodes.contains(&r.a) { f.nodes.push(r.a); }
        if !f.nodes.contains(&r.b) { f.nodes.push(r.b); }
        f.at = Func::meeting(&f.rels, &f.at, at);
        f.rels.push(r);
        f.at.push(Some(at));
        let after = f.nodes.clone();
        self.by_rel.insert(r, id);
        self.reindex(id, &before, &after);
    }

    /// A relation leaves its function (R18, R54). The function stays connected: what the origin no longer reaches
    /// leaves with it, and those relations are returned. A function left with none leaves the dictionary, and is
    /// returned.
    pub fn leave(&mut self, id: FuncId, r: &Rel) -> (Vec<Rel>, Option<Func>) {
        let Some(f) = self.funcs.get_mut(&id) else { return (vec![], None) };
        let before = f.nodes.clone();
        f.keep(|x| x != r);
        self.by_rel.remove(r);
        // what the origin still reaches
        let mut reach = vec![f.nodes[0]];
        loop {
            let before = reach.len();
            for x in &f.rels {
                if reach.contains(&x.a) && !reach.contains(&x.b) { reach.push(x.b); }
                if reach.contains(&x.b) && !reach.contains(&x.a) { reach.push(x.a); }
            }
            if reach.len() == before { break; }
        }
        let cut: Vec<Rel> = f.rels.iter().copied().filter(|x| !(reach.contains(&x.a) && reach.contains(&x.b))).collect();
        f.keep(|x| reach.contains(&x.a) && reach.contains(&x.b));
        f.nodes.retain(|n| reach.contains(n));
        let after = f.nodes.clone();
        let empty = f.rels.is_empty();
        for x in &cut { self.by_rel.remove(x); }
        self.reindex(id, &before, &after);
        if empty { return (cut, self.remove(id)); }
        (cut, None)
    }

    /// A function's nodes, relations and where each is counted, replaced in place (switch 19): its relations
    /// indexed afresh and its nodes reindexed.
    pub fn replace_body(&mut self, id: FuncId, g: &Func) {
        let Some(f) = self.funcs.get_mut(&id) else { return };
        let before = f.nodes.clone();
        let old = f.rels.clone();
        f.nodes = g.nodes.clone();
        f.rels = g.rels.clone();
        f.at = g.at.clone();
        let after = f.nodes.clone();
        for r in &old { if self.by_rel.get(r) == Some(&id) { self.by_rel.remove(r); } }
        for r in &g.rels { self.by_rel.insert(*r, id); }
        self.reindex(id, &before, &after);
    }

    /// A function's origin set to one of its nodes (switch 21), the others in their order.
    pub fn set_origin(&mut self, id: FuncId, o: Sym) {
        let Some(f) = self.funcs.get_mut(&id) else { return };
        let Some(i) = f.nodes.iter().position(|x| *x == o) else { return };
        let n = f.nodes.remove(i);
        f.nodes.insert(0, n);
    }

    /// Take a function out of the dictionary (R18, R38); returns it.
    pub fn remove(&mut self, id: FuncId) -> Option<Func> {
        let f = self.funcs.remove(&id)?;
        self.reindex(id, &f.nodes, &[]);
        for r in &f.rels { if self.by_rel.get(r) == Some(&id) { self.by_rel.remove(r); } }
        if let Some(n) = f.inter { if let Some(v) = self.by_inter.get_mut(&n) { v.retain(|x| *x != id); } }
        Some(f)
    }

    /// Give a function its interneuron (R16, R43).
    pub fn set_inter(&mut self, id: FuncId, n: NeuronId) {
        let Some(f) = self.funcs.get_mut(&id) else { return };
        if f.inter == Some(n) { return; }
        if let Some(old) = f.inter { if let Some(v) = self.by_inter.get_mut(&old) { v.retain(|x| *x != id); } }
        f.inter = Some(n);
        self.by_inter.entry(n).or_default().push(id);
    }

    pub fn describe(&self, f: &Func) -> String {
        let rels: Vec<String> = f.rels.iter().map(|r| format!("{}→{}{:?}", self.sym_text(r.a), self.sym_text(r.b), &r.off[..])).collect();
        let mut s = format!("f{} origin {} [{}]", f.id, self.sym_text(f.origin()), rels.join(" "));
        if f.has_cycle() { s.push_str(" cycle"); }
        if let Some(i) = f.inter { s.push_str(&format!(" -> #{}", i)); }
        s
    }
}
