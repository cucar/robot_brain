//! A tiny closed-loop world for the forward side (§7, §8): a cyclic event stream, an own/disown action
//! per channel, the right action run by the world where the machine is silent, and a reward scoped to
//! the channel and the frame (algorithm-stocks.md). Reports how often the machine's own action was right.
//! usage: cargo run --release --example world -- [frames] [H] [period]
use ucar_core::machine::Machine;
use ucar_core::types::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let frames: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let h: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let period: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    // moves: 0 down, 1 flat, 2 up, in a fixed cycle of `period` plus a slow drift
    let cycle: Vec<u32> = if period == 7 { vec![2, 2, 0, 1, 2, 1, 0] } else { (0..period).map(|i| ((i * 7 + 1) % 3) as u32).collect() };

    let mut m = Machine::new(h, Policy::events_to_actions(), 1);
    let ch = m.declare_channel("stock", Some(("move", 3)), Some(("own", 2)), &[]);
    let ev = m.channels[ch as usize].event.unwrap();
    let ac = m.channels[ch as usize].action.unwrap();

    let mut pending_action: Option<u32> = None; // what the machine said for the frame ahead
    let mut right = 0usize;
    let mut spoke = 0usize;
    let mut lessons = 0usize;
    let mut window_right = 0usize;
    let mut window_spoke = 0usize;
    for f in 0..frames {
        let mv = cycle[f % period];
        // the action in effect over this bar: the machine's, or the world's lesson
        let (action, lesson) = match pending_action { Some(a) => (a, false), None => ((mv == 2) as u32, true) };
        let reward = match (action, mv) { (1, 2) | (0, 0) => 1.0, (1, 0) | (0, 2) => -1.0, _ => 0.0 };
        if !lesson { spoke += 1; window_spoke += 1; if reward > 0.0 { right += 1; window_right += 1; } } else { lessons += 1; }
        let inputs = [Input { dim: ev, bucket: mv, pos: [0; MAX_DIMS] }, Input { dim: ac, bucket: action, pos: [0; MAX_DIMS] }];
        let winners = m.process_frame(&inputs, vec![Reward { reward, channels: vec![ch], frames: 1 }]);
        pending_action = winners.iter().find(|w| w.dim == ac).map(|w| w.bucket);
        if (f + 1) % 500 == 0 {
            let st = &m.last_stats;
            println!("frame {} right={}/{} ({:.0}%) last500 {:.0}% lessons={} neurons={} patterns={} conns={} levels={:?} {:.2}ms/frame",
                f + 1, right, spoke, 100.0 * right as f64 / spoke.max(1) as f64, 100.0 * window_right as f64 / window_spoke.max(1) as f64, lessons, st.neurons, st.patterns, st.connections, m.level_counts(), st.elapsed_ms);
            window_right = 0; window_spoke = 0;
        }
    }
    if std::env::var("UCAR_DUMP").is_ok() {
        for n in m.neurons.iter().filter(|n| n.alive && (!n.table.is_empty() || n.kind == Kind::Inter || !n.connections.is_empty())).take(30) {
            print!("{}", m.describe_neuron(n.id));
            let mut cs: Vec<_> = n.connections.iter().collect();
            cs.sort_by_key(|((t, o), _)| (*t, *o));
            for ((t, o), c) in cs { println!("    -> #{} @{:?} s={} est={:.2}", t, o, c.strength, c.estimate()); }
        }
    }
}
