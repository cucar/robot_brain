//! What a neuron names (D15, D16, D34, D38, D41, D44): functions, each the owner's activation with fixed
//! nodes and variables at offsets from it, and the classes its functions use.

use crate::types::*;

/// One place of a function (D16, D38): the owner's activation at offset zero, a fixed node, a class used
/// as a variable, or one place of a parameter of the function.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Place {
    Owner,
    Fixed(NeuronId),
    Class(PatternId),
    /// A parameter of this function, by its index in the function's parameters.
    Param(u32),
}

/// A member of a class or a parameter, with how many uses it stood in (D41, D44).
#[derive(Clone, Debug, PartialEq)]
pub struct Member {
    pub neuron: NeuronId,
    pub count: usize,
}

/// A parameter (D44): the places of its function that hold one and the same neuron, and as a record the
/// neurons that have stood at all of them.
#[derive(Clone, Debug, Default)]
pub struct Param {
    pub members: Vec<Member>,
}

/// A function (D15, D38): the owner's activation at offset zero and its other places at their offsets
/// from it, written rounded (D6); its parameters; its child, a call neuron (D48).
#[derive(Clone, Debug)]
pub struct Function {
    pub id: PatternId,
    /// `places[0]` is the owner at offset zero; the rest in canonical order.
    pub places: Vec<(Place, Offset)>,
    pub params: Vec<Param>,
    pub child: Option<NeuronId>,
}

/// A class (D41): a set of neurons the table holds, one of which stands where a function uses it.
#[derive(Clone, Debug)]
pub struct Class {
    pub id: PatternId,
    pub members: Vec<Member>,
}

impl Class {
    pub fn has(&self, n: NeuronId) -> bool { self.members.iter().any(|m| m.neuron == n) }
    /// Which member: `log₂ |K|` (D13).
    pub fn choice(&self) -> f64 { log2(self.members.len() as f64) }
    /// The most frequent member, the oldest on a tie (R28).
    pub fn frequent(&self) -> Option<NeuronId> { frequent(&self.members) }
}

impl Param {
    /// Which member: `log₂ |K|` over the record, the value counted where it is new (D13, D44).
    pub fn choice(&self) -> f64 { log2(self.members.len().max(1) as f64) }
}

pub fn frequent(members: &[Member]) -> Option<NeuronId> {
    let mut best: Option<&Member> = None;
    for m in members {
        if best.map_or(true, |b| m.count > b.count || (m.count == b.count && m.neuron < b.neuron)) { best = Some(m); }
    }
    best.map(|m| m.neuron)
}

impl Function {
    pub fn new(id: PatternId, rest: Vec<(Place, Offset)>, params: Vec<Param>) -> Self {
        let mut f = Function { id, places: vec![(Place::Owner, [0; MAX_DIMS])], params, child: None };
        f.places.extend(rest);
        f.canonicalize();
        f
    }

    /// `|p|` (D22): what the function names besides the owner, a class once per place and a parameter
    /// once per place it ties.
    pub fn p_size(&self) -> usize { self.places.len().saturating_sub(1) }

    pub fn fixed_count(&self) -> usize { self.places.iter().filter(|(p, _)| matches!(p, Place::Fixed(_))).count() }
    pub fn class_places(&self) -> usize { self.places.iter().filter(|(p, _)| matches!(p, Place::Class(_))).count() }
    pub fn uses_class(&self, k: PatternId) -> bool { self.places.iter().any(|(p, _)| *p == Place::Class(k)) }

    /// The places of parameter `k`, by index into `places`.
    pub fn param_places(&self, k: u32) -> Vec<usize> {
        self.places.iter().enumerate().filter(|(_, (p, _))| *p == Place::Param(k)).map(|(i, _)| i).collect()
    }

    /// One canonical form (D34, D43): the owner first, the rest sorted by offset then place, parameters
    /// renumbered in the order their first place appears, a parameter tying fewer than two places dropped
    /// with its place, and duplicate places merged.
    pub fn canonicalize(&mut self) {
        let mut rest: Vec<(Place, Offset)> = self.places.iter().copied().filter(|(p, _)| *p != Place::Owner).collect();
        // a parameter left with one place is nothing (R41)
        let mut counts: Vec<usize> = vec![0; self.params.len()];
        for (p, _) in &rest { if let Place::Param(k) = p { if (*k as usize) < counts.len() { counts[*k as usize] += 1; } } }
        rest.retain(|(p, _)| match p { Place::Param(k) => counts.get(*k as usize).map_or(false, |c| *c >= 2), _ => true });
        // one node per place and kind
        rest.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
        rest.dedup();
        // renumber parameters by first appearance
        let mut order: Vec<u32> = Vec::new();
        for (p, _) in &rest { if let Place::Param(k) = p { if !order.contains(k) { order.push(*k); } } }
        let old = std::mem::take(&mut self.params);
        let mut params: Vec<Param> = Vec::new();
        for k in &order { params.push(old.get(*k as usize).cloned().unwrap_or_default()); }
        for (p, _) in rest.iter_mut() {
            if let Place::Param(k) = p { *k = order.iter().position(|x| x == k).unwrap() as u32; }
        }
        // sort again with the new numbering, so identical functions compare equal
        rest.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
        self.params = params;
        self.places = vec![(Place::Owner, [0; MAX_DIMS])];
        self.places.extend(rest);
    }

    /// The body with no child and no records, for comparing two functions (D43).
    pub fn shape(&self) -> &[(Place, Offset)] { &self.places }
}

pub fn describe_function(f: &Function) -> String {
    let place = |p: &Place| match p {
        Place::Owner => "me".to_string(),
        Place::Fixed(n) => format!("#{}", n),
        Place::Class(k) => format!("C{}", k),
        Place::Param(k) => format!("P{}", k),
    };
    let params: Vec<String> = f.params.iter().enumerate().map(|(k, p)| format!("P{}{{{}}}", k, p.members.iter().map(|m| format!("#{}x{}", m.neuron, m.count)).collect::<Vec<_>>().join(","))).collect();
    format!("F[{}]{} child={}", f.places.iter().map(|(p, o)| format!("{}@{}", place(p), fmt_off(o))).collect::<Vec<_>>().join(" "),
        if params.is_empty() { String::new() } else { format!(" {}", params.join(" ")) },
        f.child.map_or("-".to_string(), |c| format!("#{}", c)))
}

pub fn describe_class(c: &Class) -> String {
    format!("C{}{{{}}}", c.id, c.members.iter().map(|m| format!("#{}x{}", m.neuron, m.count)).collect::<Vec<_>>().join(","))
}

fn fmt_off(o: &Offset) -> String {
    if o[1] == 0 && o[2] == 0 { o[0].to_string() } else { format!("{:?}", o) }
}
