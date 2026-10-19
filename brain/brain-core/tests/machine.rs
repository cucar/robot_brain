//! Behavioral tests on the machine: what forms over a letter stream (algorithm-text.md), the six-bar
//! cycle (algorithm-stocks.md), and the weak activation rule (D40).
use brain_core::machine::Machine;
use brain_core::neuron::{Body, Node};
use brain_core::types::*;

fn stream(m: &mut Machine, dim: DimId, text: &str) -> Vec<Option<u32>> {
    let mut preds = Vec::new();
    for b in text.bytes() {
        let w = m.process_frame(&[Input { dim, bucket: b as u32, pos: [0; MAX_DIMS] }], vec![]);
        preds.push(w.iter().find(|w| w.dim == dim).map(|w| w.bucket));
    }
    preds
}

#[test]
fn a_repeated_pair_gets_a_function_and_a_child() {
    let mut m = Machine::new(64, Policy::all_on(), 1);
    let ch = m.declare_channel("text", Some(("letter", 256)), None, &[]);
    let dim = m.channels[ch as usize].event.unwrap();
    stream(&mut m, dim, "ab ab ab ab ");
    let b = m.base_neuron(dim, b'b' as u32);
    let a = m.base_neuron(dim, b'a' as u32);
    let nb = &m.thalamus[b];
    // b's table names the owner and a at -1 (D47, D27), and the election gave it a child (R16)
    let f = nb.table.iter().find(|p| matches!(&p.body, Body::Function { nodes, .. } if nodes.iter().any(|(n, o)| *n == Node::Neuron(a) && o[0] == -1) && nodes.iter().any(|(n, _)| *n == Node::Owner)));
    assert!(f.is_some(), "b should name a at -1: {}", m.describe_neuron(b));
    assert!(f.unwrap().child().is_some(), "the pattern should have been bought a child: {}", m.describe_neuron(b));
    assert!(m.height_counts().len() >= 2);
}

/// Different letters stand before h (algorithm-text.md §2). Under D27 and D33 as written each letter
/// that recurs is named outright by a function before a class over two of them pays, since a function
/// needs one pair twice and a class needs two members twice; so h holds a function per predecessor.
#[test]
fn varied_predecessors_are_each_named() {
    let mut m = Machine::new(64, Policy::all_on(), 1);
    let ch = m.declare_channel("text", Some(("letter", 256)), None, &[]);
    let dim = m.channels[ch as usize].event.unwrap();
    stream(&mut m, dim, "th ch sh wh th ch sh wh th ch sh wh th ch sh wh ");
    let h = m.base_neuron(dim, b'h' as u32);
    let nh = &m.thalamus[h];
    let named: Vec<NeuronId> = nh.table.iter().filter_map(|p| match &p.body {
        Body::Function { nodes, .. } => nodes.iter().find(|(n, o)| matches!(n, Node::Neuron(_)) && o[0] == -1).map(|(n, _)| if let Node::Neuron(x) = n { *x } else { 0 }),
        _ => None,
    }).collect();
    assert!(named.len() >= 4, "h should name its predecessors at -1: {}", m.describe_neuron(h));
    for p in &nh.table { assert!(p.child().is_some(), "every line should have been bought: {}", m.describe_neuron(h)); }
}

#[test]
fn prediction_improves_over_passes() {
    let mut m = Machine::new(256, Policy::all_on(), 1);
    let ch = m.declare_channel("text", Some(("letter", 256)), None, &[]);
    let dim = m.channels[ch as usize].event.unwrap();
    let text = "the cat sat on the mat. the dog sat on the log. ";
    let mut accs = Vec::new();
    for _ in 0..4 {
        m.reset_context();
        let preds = stream(&mut m, dim, text);
        let bytes: Vec<u8> = text.bytes().collect();
        let mut ok = 0;
        for i in 1..bytes.len() { if preds[i - 1] == Some(bytes[i] as u32) { ok += 1; } }
        accs.push(ok as f64 / (bytes.len() - 1) as f64);
    }
    assert!(accs[3] > accs[0] + 0.2, "accuracy should climb: {:?}", accs);
    assert!(accs[3] > 0.6, "accuracy should be well above a bigram: {:?}", accs);
}

#[test]
fn weak_activations_last_one_frame_and_write_nothing() {
    let mut m = Machine::new(64, Policy::all_on(), 1);
    let ch = m.declare_channel("text", Some(("letter", 256)), None, &[]);
    let dim = m.channels[ch as usize].event.unwrap();
    stream(&mut m, dim, "abababab");
    let a = m.base_neuron(dim, b'a' as u32);
    let b = m.base_neuron(dim, b'b' as u32);
    let before = m.thalamus[b].connections.get(&(a, [1, 0, 0])).map(|c| c.strength).unwrap_or(0.0);
    assert!(before > 0.0);
    let w = m.process_frame(&[], vec![]);
    assert_eq!(w.iter().find(|w| w.dim == dim).map(|w| w.bucket), Some(b'b' as u32), "the weak a infers b");
    let after = m.thalamus[b].connections.get(&(a, [1, 0, 0])).map(|c| c.strength).unwrap_or(0.0);
    assert_eq!(before, after, "nothing connects to a weak activation (D40)");
}

/// The six-bar cycle in the spec's configuration (R = 6, H = 256, one channel): the machine must own
/// before a rise and stand out before a fall, and settle. It does: every bar right from lap 11 and
/// nothing added or retired from lap 94.
#[test]
fn six_bar_cycle_settles() {
    let pattern = [0.009, -0.019, -0.029, 0.019, 0.029, -0.009];
    let mut m = Machine::new(256, Policy::events_to_actions(), 6);
    let ch = m.declare_channel("TEST", Some(("price_change", 2)), Some(("activity", 2)), &[]);
    let ev = m.channels[ch as usize].event.unwrap();
    let ac = m.channels[ch as usize].action.unwrap();
    let laps = 150;
    let mut pending: Option<u32> = None;
    let (mut right, mut wrong) = (0, 0);
    for k in 0..laps * 6 {
        let pct = pattern[k % 6] * 100.0;
        let up = (pct > 0.0) as u32;
        let (action, lesson) = match pending { Some(a) => (a, false), None => (up, true) };
        if !lesson && k >= (laps - 10) * 6 { if action == up { right += 1; } else { wrong += 1; } }
        let reward = if action == 1 { pct } else { -pct };
        let inputs = [Input { dim: ev, bucket: up, pos: [0; MAX_DIMS] }, Input { dim: ac, bucket: action, pos: [0; MAX_DIMS] }];
        let w = m.process_frame(&inputs, vec![Reward { reward, channels: vec![ch], frames: 1 }]);
        pending = w.iter().find(|w| w.dim == ac).map(|w| w.bucket);
    }
    assert_eq!((right, wrong), (60, 0), "last ten laps: {}/{} optimal", right, right + wrong);
    let s = m.last_stats.clone();
    assert_eq!((s.added, s.retired), (0, 0), "nothing turns over once settled");
}
