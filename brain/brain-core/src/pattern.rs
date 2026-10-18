//! What a neuron names (D15, D16, D38): nodes, functions, classes and parameters, and their lines.


use crate::neuron::*;
use crate::types::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Node {
    Owner,
    Neuron(NeuronId),
    Var(PatternId),
}

/// A value neuron's value as a neighbor carries it (D7, D45): its variable and the member, followed
/// through value neurons to a neuron that is not one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Value {
    pub owner: NeuronId,
    pub var: PatternId,
    pub member: NeuronId,
}


#[derive(Clone, Debug)]
pub struct Member {
    pub neuron: NeuronId,
    pub count: usize,
    pub value_neuron: Option<NeuronId>,
}

#[derive(Clone, Debug)]
pub enum Body {
    /// A set of relations connecting its nodes (D15, D34), written as each node's offset from the
    /// reference node, `nodes[0]`, which stands at offset zero: the owner where the function names it,
    /// else the node latest in time.
    Function { nodes: Vec<(Node, Offset)>, child: Option<NeuronId> },
    /// Its members and nothing else (D41); `anchor` is the seed's offset from its node, the coordinates
    /// it stands at before any function names it (D27).
    Class { members: Vec<Member>, anchor: (Node, Offset) },
    /// Its offsets between positions, the first at zero (D44), and its members as a record.
    Parameter { offsets: Vec<Offset>, members: Vec<Member> },
}

#[derive(Clone, Debug)]
pub struct Pattern {
    pub id: PatternId,
    pub body: Body,
}

impl Pattern {
    pub fn is_function(&self) -> bool { matches!(self.body, Body::Function { .. }) }
    pub fn is_variable(&self) -> bool { !self.is_function() }
    pub fn child(&self) -> Option<NeuronId> { if let Body::Function { child, .. } = &self.body { *child } else { None } }
    pub fn members(&self) -> &[Member] {
        match &self.body { Body::Class { members, .. } | Body::Parameter { members, .. } => members, _ => &[] }
    }
    pub fn members_mut(&mut self) -> &mut Vec<Member> {
        match &mut self.body { Body::Class { members, .. } | Body::Parameter { members, .. } => members, _ => unreachable!() }
    }
    pub fn positions(&self) -> usize {
        match &self.body { Body::Parameter { offsets, .. } => offsets.len(), _ => 1 }
    }
    pub fn member(&self, n: NeuronId) -> Option<&Member> { self.members().iter().find(|m| m.neuron == n) }
    /// Which member: `log₂ |K|` (D13).
    pub fn choice(&self) -> f64 { log2(self.members().len() as f64) }
    /// The most frequent member, the oldest on a tie (R28).
    pub fn frequent_member(&self) -> Option<NeuronId> {
        let mut best: Option<&Member> = None;
        for m in self.members() {
            if best.map_or(true, |b| m.count > b.count || (m.count == b.count && m.neuron < b.neuron)) { best = Some(m); }
        }
        best.map(|m| m.neuron)
    }
}

/// A connection (D25): exposures and what they earned.

pub fn describe_body(b: &Body) -> String {
    let node = |n: &Node| match n { Node::Owner => "me".to_string(), Node::Neuron(x) => format!("#{}", x), Node::Var(v) => format!("p{}", v) };
    match b {
        Body::Function { nodes, child } => format!("F[{}] child={}", nodes.iter().map(|(n, o)| format!("{}@{}", node(n), o[0])).collect::<Vec<_>>().join(" "), child.map_or("-".to_string(), |c| format!("#{}", c))),
        Body::Class { members, anchor } => format!("C{{{}}} at {}@{}", members.iter().map(|m| format!("#{}x{}{}", m.neuron, m.count, m.value_neuron.map_or(String::new(), |v| format!("=#{}", v)))).collect::<Vec<_>>().join(","), node(&anchor.0), anchor.1[0]),
        Body::Parameter { offsets, members } => format!("P[{}] {{{}}}", offsets.iter().map(|o| o[0].to_string()).collect::<Vec<_>>().join(","), members.iter().map(|m| format!("#{}x{}{}", m.neuron, m.count, m.value_neuron.map_or(String::new(), |v| format!("=#{}", v)))).collect::<Vec<_>>().join(",")),
    }
}


/// The canonical form of a function's nodes (D34, D43): the owner first where named, else the node
/// latest in time, at offset zero; the rest sorted by offset then node.
pub fn canonical(nodes: Vec<(Node, Offset)>) -> Vec<(Node, Offset)> {
    if nodes.is_empty() { return nodes; }
    let ref_idx = match nodes.iter().position(|(n, _)| *n == Node::Owner) {
        Some(i) => i,
        None => {
            let mut best = 0;
            for i in 1..nodes.len() {
                let (n, o) = nodes[i];
                let (bn, bo) = nodes[best];
                if o > bo || (o == bo && n < bn) { best = i; }
            }
            best
        }
    };
    let base = nodes[ref_idx].1;
    let mut rest: Vec<(Node, Offset)> = nodes.iter().enumerate().filter(|(i, _)| *i != ref_idx).map(|(_, (n, o))| (*n, sub(o, &base))).collect();
    rest.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    let mut out = vec![(nodes[ref_idx].0, [0; MAX_DIMS])];
    out.extend(rest);
    out
}
