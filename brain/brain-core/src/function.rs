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
    pub a: NeuronId,
    pub b: NeuronId,
    pub off: Offset,
}

fn neg(o: &Offset) -> Offset { let mut r = *o; for x in r.iter_mut() { *x = -*x; } r }

/// The one form of the relation from `n` to `m` standing at `off` from it (D34).
pub fn canon(n: NeuronId, m: NeuronId, off: Offset) -> Rel {
    let first = off.iter().copied().find(|x| *x != 0);
    let keep = if off[0] != 0 { off[0] < 0 } else {
        match first { Some(x) => x > 0, None => n <= m }
    };
    if keep { Rel { a: n, b: m, off } } else { Rel { a: m, b: n, off: neg(&off) } }
}

impl Rel {
    /// The neuron that sees this relation from its own activation, and the neighbor and offset it records (D47):
    /// in time the later end alone; in space either end, the first here.
    pub fn recorders(&self) -> Vec<(NeuronId, NeuronId, Offset)> {
        if self.off[0] != 0 { vec![(self.a, self.b, self.off)] } else { vec![(self.a, self.b, self.off), (self.b, self.a, neg(&self.off))] }
    }
}

/// A function (D15, D56): its nodes, the origin first, one per neuron, and its relations.
#[derive(Clone, Debug)]
pub struct Func {
    pub id: FuncId,
    pub nodes: Vec<NeuronId>,
    pub rels: Vec<Rel>,
    /// Its interneuron once a fit of it has been accepted (R16, R43).
    pub inter: Option<NeuronId>,
    /// When it joined the dictionary: earlier is older (R24).
    pub joined: u64,
}

/// How a function reads from its origin (D15, D54): the relations of a spanning tree in the order they are
/// reached, each as (relation, from node, to node, offset from the one to the other); the relation that closes the
/// cycle, if there is one, the same way; and each node's place from the origin over one lap.
#[derive(Clone, Debug, Default)]
pub struct Layout {
    pub tree: Vec<(usize, usize, usize, Offset)>,
    pub closing: Option<(usize, usize, usize, Offset)>,
    pub place: Vec<Pos>,
    /// Per node, the tree relation that reached it and the node it came from.
    pub parent: Vec<Option<(usize, usize)>>,
}

impl Func {
    pub fn origin(&self) -> NeuronId { self.nodes[0] }
    pub fn node_of(&self, n: NeuronId) -> Option<usize> { self.nodes.iter().position(|x| *x == n) }
    pub fn holds(&self, r: &Rel) -> bool { self.rels.contains(r) }
    pub fn has_cycle(&self) -> bool { self.rels.len() >= self.nodes.len() }

    pub fn layout(&self) -> Layout {
        let n = self.nodes.len();
        let mut place: Vec<Option<Pos>> = vec![None; n];
        let mut parent: Vec<Option<(usize, usize)>> = vec![None; n];
        place[0] = Some([0; MAX_DIMS]);
        let mut used = vec![false; self.rels.len()];
        let mut tree = Vec::new();
        let mut queue = vec![0usize];
        let mut qi = 0;
        while qi < queue.len() {
            let u = queue[qi];
            qi += 1;
            for (ri, r) in self.rels.iter().enumerate() {
                if used[ri] { continue; }
                let (ua, ub) = (self.node_of(r.a).unwrap(), self.node_of(r.b).unwrap());
                let (v, off) = if ua == u { (ub, r.off) } else if ub == u { (ua, neg(&r.off)) } else { continue };
                if place[v].is_some() { continue; }
                used[ri] = true;
                let mut p = place[u].unwrap();
                for d in 0..MAX_DIMS { p[d] += off[d]; }
                place[v] = Some(p);
                parent[v] = Some((ri, u));
                tree.push((ri, u, v, off));
                queue.push(v);
            }
        }
        let closing = self.rels.iter().enumerate().find(|(ri, _)| !used[*ri]).map(|(ri, r)| {
            let (ua, ub) = (self.node_of(r.a).unwrap(), self.node_of(r.b).unwrap());
            (ri, ua, ub, r.off)
        });
        Layout { tree, closing, place: place.into_iter().map(|p| p.unwrap_or([0; MAX_DIMS])).collect(), parent }
    }

    /// The tree relations from `from` to `to` along the spanning tree, in order, each with its offset in that
    /// direction (the cycle's path, read once round from the node the closing relation reaches).
    pub fn tree_path(lay: &Layout, from: usize, to: usize) -> Vec<(usize, usize, usize, Offset)> {
        let up = |mut x: usize| { let mut v = vec![x]; while let Some((_, p)) = lay.parent[x] { v.push(p); x = p; } v };
        let (pf, pt) = (up(from), up(to));
        let lca = *pf.iter().find(|x| pt.contains(x)).unwrap_or(&0);
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

    /// How far its activation sees (D4): one position beyond the farthest node it places from its origin, in each
    /// dimension of `axes`.
    pub fn reach(&self, axes: u8) -> Pos {
        let lay = self.layout();
        let mut r = [0; MAX_DIMS];
        for d in 0..MAX_DIMS {
            if d > 0 && axes & (1 << d) == 0 { continue; }
            r[d] = lay.place.iter().map(|p| p[d].abs()).max().unwrap_or(0) + 1;
        }
        r
    }

    /// Two functions with the same relations and the same origin are one (D36).
    pub fn key(&self) -> (NeuronId, Vec<Rel>) {
        let mut r = self.rels.clone();
        r.sort();
        (self.origin(), r)
    }
}

pub fn describe_rel(r: &Rel) -> String { format!("#{}→#{}{:?}", r.a, r.b, r.off) }

/// The dictionary (D36, D56).
#[derive(Default)]
pub struct Dictionary {
    pub funcs: FxHashMap<FuncId, Func>,
    /// Every relation to the one function that holds it (R50).
    pub by_rel: FxHashMap<Rel, FuncId>,
    /// Per interneuron, its functions (R43: it may hold more than one).
    pub by_inter: FxHashMap<NeuronId, Vec<FuncId>>,
    next_id: FuncId,
    next_join: u64,
}

impl Dictionary {
    pub fn get(&self, id: FuncId) -> Option<&Func> { self.funcs.get(&id) }
    pub fn len(&self) -> usize { self.funcs.len() }
    pub fn holder(&self, r: &Rel) -> Option<FuncId> { self.by_rel.get(r).copied() }
    pub fn of_inter(&self, n: NeuronId) -> &[FuncId] { self.by_inter.get(&n).map_or(&[], |v| v.as_slice()) }
    /// The functions whose nodes name a neuron (R38, the subtree).
    pub fn naming(&self, n: NeuronId) -> Vec<FuncId> {
        let mut v: Vec<FuncId> = self.funcs.values().filter(|f| f.nodes.contains(&n)).map(|f| f.id).collect();
        v.sort_unstable();
        v
    }
    /// Functions whose origin is a neuron (D54: a call stands for its origin's neuron).
    pub fn find_key(&self, key: &(NeuronId, Vec<Rel>)) -> Option<FuncId> {
        self.funcs.values().find(|f| f.key() == *key).map(|f| f.id)
    }

    /// A new function of one relation, its origin the relation's first neuron (R50, D15).
    pub fn start(&mut self, r: Rel) -> FuncId {
        let nodes = if r.a == r.b { vec![r.a] } else { vec![r.a, r.b] };
        self.insert(Func { id: 0, nodes, rels: vec![r], inter: None, joined: 0 })
    }

    fn insert(&mut self, mut f: Func) -> FuncId {
        let id = self.next_id;
        self.next_id += 1;
        f.id = id;
        f.joined = self.next_join;
        self.next_join += 1;
        for r in &f.rels { self.by_rel.insert(*r, id); }
        if let Some(n) = f.inter { self.by_inter.entry(n).or_default().push(id); }
        self.funcs.insert(id, f);
        id
    }

    /// A relation joins a function (R54): a new node where its other neuron is not one, a cycle where it is.
    pub fn join(&mut self, id: FuncId, r: Rel) {
        let Some(f) = self.funcs.get_mut(&id) else { return };
        if !f.nodes.contains(&r.a) { f.nodes.push(r.a); }
        if !f.nodes.contains(&r.b) { f.nodes.push(r.b); }
        f.rels.push(r);
        self.by_rel.insert(r, id);
    }

    /// A relation leaves its function (R18, R54). The function stays connected: what the origin no longer reaches
    /// leaves with it, and those relations are returned. A function left with none leaves the dictionary, and is
    /// returned.
    pub fn leave(&mut self, id: FuncId, r: &Rel) -> (Vec<Rel>, Option<Func>) {
        let Some(f) = self.funcs.get_mut(&id) else { return (vec![], None) };
        f.rels.retain(|x| x != r);
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
        f.rels.retain(|x| reach.contains(&x.a) && reach.contains(&x.b));
        f.nodes.retain(|n| reach.contains(n));
        for x in &cut { self.by_rel.remove(x); }
        if f.rels.is_empty() { return (cut, self.remove(id)); }
        (cut, None)
    }

    /// Take a function out of the dictionary (R18, R38); returns it.
    pub fn remove(&mut self, id: FuncId) -> Option<Func> {
        let f = self.funcs.remove(&id)?;
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
        let rels: Vec<String> = f.rels.iter().map(describe_rel).collect();
        let mut s = format!("f{} origin #{} [{}]", f.id, f.origin(), rels.join(" "));
        if f.has_cycle() { s.push_str(" cycle"); }
        if let Some(i) = f.inter { s.push_str(&format!(" -> #{}", i)); }
        s
    }
}
