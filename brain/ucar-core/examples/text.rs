//! Stream a text file through the machine and report next-letter accuracy per episode.
//! usage: cargo run --release --example text -- <file> [episodes] [H] [radius] [verbose]
use ucar_core::machine::Machine;
use ucar_core::types::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let file = args.get(1).cloned().unwrap_or("../../apps/text/data/test.txt".into());
    let episodes: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let h: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let radius: i32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let verbose = args.get(5).map_or(false, |s| s.starts_with('v'));
    let tail: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let bytes: Vec<u8> = std::fs::read(&file).expect("read file");

    let mut m = Machine::new(h, Policy::all_on(), radius);
    let ch = m.declare_channel("text", Some(("letter", 256)), None, &[]);
    let dim = m.channels[ch as usize].event.unwrap();

    for ep in 1..=episodes {
        m.reset_context();
        let mut correct = 0usize;
        let mut predicted = 0usize;
        let mut pending: Option<u32> = None;
        let mut apex_cost = 0.0;
        let mut ms = 0.0;
        let mut above = 0usize;
        let t0 = std::time::Instant::now();
        for (i, &b) in bytes.iter().enumerate() {
            if let Some(p) = pending { predicted += 1; if p == b as u32 { correct += 1; } }
            let winners = m.process_frame(&[Input { dim, bucket: b as u32, pos: [0; MAX_DIMS] }], vec![]);
            pending = winners.iter().find(|w| w.dim == dim).map(|w| w.bucket);
            let st = &m.last_stats;
            apex_cost += st.apex_cost;
            if st.voters_above_base > 0 { above += 1; }
            ms += st.elapsed_ms;
            if verbose && ep == episodes && i + tail >= bytes.len() {
                println!("f{} '{}' base={} apex={} cost={:.1} levels={} bids={} acc={} minted={} added={} retired={} voters={}/{} pred={:?} {:.2}ms",
                    st.frame, b as char, st.base_acts, st.apex_acts, st.apex_cost, st.levels, st.bids, st.accepted, st.minted, st.added, st.retired, st.voters_above_base, st.voters,
                    pending.map(|p| p as u8 as char), st.elapsed_ms);
            }
            let _ = i;
        }
        let st = &m.last_stats;
        println!("episode {} acc={:.1}% ({}/{}) frames_with_child_voter={:.1}% apex_cost/frame={:.1} neurons={} patterns={} conns={} levels={:?} {:.0}ms (brain {:.0}ms)",
            ep, 100.0 * correct as f64 / predicted.max(1) as f64, correct, predicted, 100.0 * above as f64 / bytes.len() as f64, apex_cost / bytes.len() as f64, st.neurons, st.patterns, st.connections, m.level_counts(), t0.elapsed().as_secs_f64() * 1000.0, ms);
    }
    let ph: Vec<f64> = ucar_core::neuron::PHASE_NS.iter().map(|a| a.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6).collect();
    println!("neuron phases ms: recognize={:.0} recenter={:.0} delete={:.0} create={:.0} merge+bids={:.0}", ph[0], ph[1], ph[2], ph[3], ph[4]);
    println!("timings ms: levels={:.0} connect={:.0} infer={:.0} reap={:.0} select={:.0}", m.timings[0], m.timings[1], m.timings[2], m.timings[3], m.timings[4]);
    if verbose {
        for n in m.neurons.iter().filter(|n| n.alive && (!n.table.is_empty() || n.kind == Kind::Inter)).take(40) {
            print!("{}", m.describe_neuron(n.id));
        }
    }
}
