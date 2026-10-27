//! Functions and the dictionary (§3.5, §3.6, D15, D16, D36, D38, D41, D44): a function is a set of relations,
//! each from its anchor, a fixed node at its latest frame, to a node at an offset in one dimension; the
//! machine's dictionary holds every function, the classes they use, and the indexes the machine reads them
//! through.

use rustc_hash::FxHashMap;

use crate::types::*;

pub type FuncId = u32;
pub type ClassId = u32;

/// What a relation names at its far end (D16): a particular neuron, a class of the dictionary, or a parameter
/// of the function.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Node {
    Fixed(NeuronId),
    Class(ClassId),
    Param(u8),
}

impl Node {
    pub fn is_fixed(&self) -> bool { matches!(self, Node::Fixed(_)) }
}

/// One relation of a function (D34): from the anchor to a node, at an offset in one dimension, written
/// rounded (D6).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Rel {
    pub dim: u8,
    pub off: i32,
    pub node: Node,
}

/// A function (D15): its anchor, the fixed node at its latest frame whose history is its evidence (D30), and
/// its relations from the anchor, in one canonical order.
#[derive(Clone, Debug)]
pub struct Func {
    pub id: FuncId,
    pub anchor: NeuronId,
    pub rels: Vec<Rel>,
    /// Its interneuron, once the election has accepted it (R16, R43).
    pub inter: Option<NeuronId>,
    /// When it joined the dictionary: earlier is older (R24, R18).
    pub joined: u64,
    /// The neurons that have stood at each parameter's nodes, a record (D44).
    pub param_members: Vec<Vec<NeuronId>>,
}

/// A function's relations as the dictionary keys them (D15, D36): each written from the older of its two
/// nodes, a neuron before a class before a parameter, the older neuron first; between two nodes alike, from the
/// smaller position. So the same relations are the same function whichever neuron returned them.
pub type FuncKey = Vec<(Node, Node, u8, i32)>;

pub fn key_of(anchor: NeuronId, rels: &[Rel]) -> FuncKey {
    let mut out: FuncKey = rels.iter().map(|r| {
        let (a, b, off) = (Node::Fixed(anchor), r.node, r.off);
        if b < a || (b == a && off < 0) { (b, a, r.dim, -off) } else { (a, b, r.dim, off) }
    }).collect();
    out.sort_unstable();
    out
}

impl Func {
    pub fn key(&self) -> FuncKey { key_of(self.anchor, &self.rels) }
    /// The nodes an activation of `n` can stand at to read the function (D31): the anchor, and every fixed
    /// node in the anchor's frame, the latest, so that the rest stands within its reach. Node 0 is the anchor,
    /// node `i + 1` the far node of relation `i`.
    pub fn nodes_for(&self, n: NeuronId) -> Vec<usize> {
        let mut v = Vec::new();
        if self.anchor == n { v.push(0); }
        for (i, r) in self.rels.iter().enumerate() {
            if r.node == Node::Fixed(n) && (r.dim != 0 || r.off == 0) { v.push(i + 1); }
        }
        v
    }
    pub fn params(&self) -> usize { self.param_members.len() }
    pub fn uses_class(&self, k: ClassId) -> bool { self.rels.iter().any(|r| r.node == Node::Class(k)) }
    /// Every neuron the function names as a fixed node, the anchor among them, each once (D15).
    pub fn fixed_neurons(&self) -> Vec<NeuronId> {
        let mut v = vec![self.anchor];
        for r in &self.rels { if let Node::Fixed(n) = r.node { if !v.contains(&n) { v.push(n); } } }
        v
    }
    pub fn names(&self, n: NeuronId) -> bool { self.anchor == n || self.rels.iter().any(|r| r.node == Node::Fixed(n)) }
}

/// Put relations in the one order the dictionary keys them by.
pub fn canonical(mut rels: Vec<Rel>) -> Vec<Rel> {
    rels.sort();
    rels.dedup();
    rels
}

pub fn describe_rels(anchor: NeuronId, rels: &[Rel]) -> String {
    let parts: Vec<String> = rels.iter().map(|r| {
        let n = match r.node { Node::Fixed(n) => format!("#{}", n), Node::Class(k) => format!("C{}", k), Node::Param(p) => format!("P{}", p) };
        let d = ["t", "x", "y"][r.dim as usize % 3];
        format!("{}{}{:+}", n, d, r.off)
    }).collect();
    format!("#{}[{}]", anchor, parts.join(" "))
}

/// The dictionary (D36): every function, each with its interneuron once it has one, and the classes they use.
#[derive(Default)]
pub struct Dictionary {
    pub funcs: FxHashMap<FuncId, Func>,
    pub classes: FxHashMap<ClassId, Vec<NeuronId>>,
    pub by_anchor: FxHashMap<NeuronId, Vec<FuncId>>,
    /// Per neuron, the functions it can stand in to read them (D31): at the anchor, or at a fixed node of the
    /// anchor's frame.
    pub by_label: FxHashMap<NeuronId, Vec<FuncId>>,
    pub by_key: FxHashMap<FuncKey, FuncId>,
    /// Per interneuron, its functions (R43: it may hold more than one).
    pub by_inter: FxHashMap<NeuronId, Vec<FuncId>>,
    /// Per neuron, the functions that name it at a fixed node other than the anchor, for the subtree (R38).
    pub naming: FxHashMap<NeuronId, Vec<FuncId>>,
    /// Bumped on every change to what a function or class says, so readings of the dictionary can be kept
    /// while it stands.
    pub version: u64,
    /// The latest changes to functions, each with the version it made and the function as it stood: added, or
    /// as it was when it left. A reading kept from an earlier version holds while none of them touches it.
    pub log: std::collections::VecDeque<(u64, Func)>,
    /// The version of the latest change the log let go of.
    log_floor: u64,
    next_id: FuncId,
    next_join: u64,
    next_class: ClassId,
}

impl Dictionary {
    pub fn get(&self, id: FuncId) -> Option<&Func> { self.funcs.get(&id) }
    pub fn len(&self) -> usize { self.funcs.len() }
    pub fn anchored(&self, n: NeuronId) -> &[FuncId] { self.by_anchor.get(&n).map_or(&[], |v| v.as_slice()) }
    /// The functions an activation of `n` reads (D31).
    pub fn standing(&self, n: NeuronId) -> &[FuncId] { self.by_label.get(&n).map_or(&[], |v| v.as_slice()) }
    pub fn of_inter(&self, n: NeuronId) -> &[FuncId] { self.by_inter.get(&n).map_or(&[], |v| v.as_slice()) }
    pub fn find_key(&self, anchor: NeuronId, rels: &[Rel]) -> Option<FuncId> { self.by_key.get(&key_of(anchor, rels)).copied() }
    pub fn class(&self, k: ClassId) -> &[NeuronId] { self.classes.get(&k).map_or(&[], |v| v.as_slice()) }

    /// A class with exactly these members, the existing one where there is one (D43).
    pub fn class_for(&mut self, mut members: Vec<NeuronId>) -> ClassId {
        members.sort_unstable();
        members.dedup();
        if let Some((k, _)) = self.classes.iter().find(|(_, m)| **m == members) { return *k; }
        let k = self.next_class;
        self.next_class += 1;
        self.classes.insert(k, members);
        self.version += 1;
        k
    }
    pub fn class_exists(&self, members: &[NeuronId]) -> bool {
        let mut m = members.to_vec();
        m.sort_unstable();
        m.dedup();
        self.classes.values().any(|c| *c == m)
    }

    /// A function as it would stand, not yet in the dictionary.
    pub fn draft(&self, anchor: NeuronId, rels: Vec<Rel>, params: usize) -> Func {
        Func { id: FuncId::MAX, anchor, rels: canonical(rels), inter: None, joined: u64::MAX, param_members: vec![Vec::new(); params] }
    }

    /// Add a function to the dictionary (R53, R52); returns its id.
    pub fn add(&mut self, mut f: Func) -> FuncId {
        let id = self.next_id;
        self.next_id += 1;
        self.version += 1;
        f.id = id;
        f.joined = self.next_join;
        self.next_join += 1;
        self.by_anchor.entry(f.anchor).or_default().push(id);
        let mut labels: Vec<NeuronId> = f.fixed_neurons().into_iter().filter(|n| !f.nodes_for(*n).is_empty()).collect();
        labels.sort_unstable();
        labels.dedup();
        for n in labels { self.by_label.entry(n).or_default().push(id); }
        self.by_key.insert(f.key(), id);
        for n in f.fixed_neurons().into_iter().filter(|n| *n != f.anchor) { self.naming.entry(n).or_default().push(id); }
        if let Some(i) = f.inter { self.by_inter.entry(i).or_default().push(id); }
        self.note(f.clone());
        self.funcs.insert(id, f);
        id
    }

    /// Keep a change in the log, the oldest leaving past a bound.
    fn note(&mut self, f: Func) {
        self.log.push_back((self.version, f));
        while self.log.len() > 4096 { if let Some((v, _)) = self.log.pop_front() { self.log_floor = v; } }
    }

    /// The changes since `version`; None where the log no longer reaches back that far.
    pub fn changes_since(&self, version: u64) -> Option<impl Iterator<Item = &Func> + '_> {
        if version < self.log_floor { return None; }
        let from = self.log.partition_point(|(v, _)| *v <= version);
        Some(self.log.range(from..).map(|(_, f)| f))
    }

    /// Take a function out of the dictionary (R18, R52, R38); returns it.
    pub fn remove(&mut self, id: FuncId) -> Option<Func> {
        let f = self.funcs.remove(&id)?;
        self.version += 1;
        if let Some(v) = self.by_anchor.get_mut(&f.anchor) { v.retain(|x| *x != id); }
        for n in f.fixed_neurons() { if let Some(v) = self.by_label.get_mut(&n) { v.retain(|x| *x != id); } }
        self.by_key.remove(&f.key());
        for n in f.fixed_neurons() { if let Some(v) = self.naming.get_mut(&n) { v.retain(|x| *x != id); } }
        if let Some(i) = f.inter { if let Some(v) = self.by_inter.get_mut(&i) { v.retain(|x| *x != id); } }
        self.note(f.clone());
        Some(f)
    }

    /// Move a function to new relations (the update of grouping): it keeps its id, its interneuron and when it
    /// joined, and every index reads it by its new relations.
    pub fn rewrite(&mut self, id: FuncId, rels: Vec<Rel>) {
        let Some(mut f) = self.remove(id) else { return };
        let inter = f.inter.take();
        f.rels = canonical(rels);
        let joined = f.joined;
        let (next_id, next_join) = (self.next_id, self.next_join);
        self.next_id = id;
        self.next_join = joined;
        self.add(f);
        self.next_id = next_id;
        self.next_join = next_join;
        if let Some(n) = inter { self.set_inter(id, n); }
    }

    /// Give a function its interneuron (R16, R43).
    pub fn set_inter(&mut self, id: FuncId, n: NeuronId) {
        let Some(f) = self.funcs.get_mut(&id) else { return };
        if f.inter == Some(n) { return; }
        if let Some(old) = f.inter { if let Some(v) = self.by_inter.get_mut(&old) { v.retain(|x| *x != id); } }
        f.inter = Some(n);
        self.by_inter.entry(n).or_default().push(id);
    }

    /// The classes no function uses any longer, or left with one member, are deleted (D41).
    pub fn drop_dead_classes(&mut self) {
        let used: Vec<ClassId> = self.classes.keys().copied().filter(|k| self.funcs.values().any(|f| f.uses_class(*k))).collect();
        let before = self.classes.len();
        self.classes.retain(|k, m| used.contains(k) && !m.is_empty());
        if self.classes.len() != before { self.version += 1; }
    }

    /// An entry in the dictionary (D15, D13): one number per relation, per fixed neuron named, and per class at a
    /// node; and, for each class it uses that is not yet in the dictionary, one per member (D41).
    pub fn entry(&self, f: &Func, new_classes: &[Vec<NeuronId>]) -> f64 {
        let rels = f.rels.len() as f64;
        let fixed = f.fixed_neurons().len() as f64;
        let mut classes: Vec<ClassId> = f.rels.iter().filter_map(|r| if let Node::Class(k) = r.node { Some(k) } else { None }).collect();
        classes.sort_unstable();
        classes.dedup();
        let members: f64 = new_classes.iter().map(|m| m.len() as f64).sum();
        rels + fixed + classes.len() as f64 + members
    }

    pub fn describe(&self, f: &Func) -> String {
        let mut s = describe_rels(f.anchor, &f.rels);
        for r in &f.rels { if let Node::Class(k) = r.node { s.push_str(&format!(" C{}={:?}", k, self.class(k))); } }
        if let Some(i) = f.inter { s.push_str(&format!(" -> #{}", i)); }
        s
    }
}
