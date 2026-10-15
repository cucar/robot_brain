//! Run the summary text and dump a few neurons: usage: cargo run --release --example dump -- <ids...>
use ucar_core::machine::Machine;
use ucar_core::types::*;
fn main() {
    let ids: Vec<u32> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let bytes = std::fs::read("../../apps/text/data/abramov_summary.txt").unwrap();
    let mut m = Machine::new(1024, Policy::all_on(), 1);
    let ch = m.declare_channel("text", Some(("letter", 256)), None, &[]);
    let dim = m.channels[ch as usize].event.unwrap();
    for _ in 0..4 { m.reset_context(); for &b in &bytes { m.process_frame(&[Input { dim, bucket: b as u32, pos: [0; MAX_DIMS] }], vec![]); } }
    for id in ids {
        print!("{}", m.describe_neuron(id));
        let n = &m.neurons[id as usize];
        let mut cs: Vec<_> = n.connections.iter().collect();
        cs.sort_by(|a, b| b.1.strength.partial_cmp(&a.1.strength).unwrap());
        for ((t, o), c) in cs.iter().take(8) { println!("    -> #{} @{:?} s={} est={:.2}", t, o, c.strength, c.estimate()); }
    }
    // how many live interneurons are value neurons vs call neurons, and how many have any connection
    let (mut vals, mut calls, mut vconn, mut cconn) = (0, 0, 0, 0);
    for n in m.neurons.iter().filter(|n| n.alive && n.kind == Kind::Inter) {
        if n.value.is_some() { vals += 1; if !n.connections.is_empty() { vconn += 1; } } else { calls += 1; if !n.connections.is_empty() { cconn += 1; } }
    }
    println!("value neurons {} ({} with connections), call neurons {} ({} with connections)", vals, vconn, calls, cconn);
}
