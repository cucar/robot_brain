//! The simple synthetic cycle test: one channel, a binary price bucket and own/out, the six-bar pattern
//! +1% -2% -3% +2% +3% -1% repeated. The world runs the right action where the machine is silent; the
//! reward is the bar's return signed by the action, scoped to the channel and the frame. Spec
//! configuration, nothing switched off.
//! usage: cargo run --release --example simple -- [cycles] [H] [R] [v]
use brain_core::machine::Machine;
use brain_core::types::*;

const PATTERN: [f64; 6] = [0.009, -0.019, -0.029, 0.019, 0.029, -0.009];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cycles: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(50);
    let h: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let r: i32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(6);
    let verbose = args.get(4).map_or(false, |s| s.starts_with('v'));
    let n = PATTERN.len();

    let mut m = Machine::new(h, Policy::events_to_actions(), r);
    let ch = m.declare_channel("TEST", Some(("price_change", 2)), Some(("activity", 2)), &[]);
    let ev = m.channels[ch as usize].event.unwrap();
    let ac = m.channels[ch as usize].action.unwrap();

    let total = cycles * n;
    let mut pending: Option<u32> = None;
    let mut stats = vec![(0usize, 0usize); n];
    let mut last10 = vec![(0usize, 0usize); n];
    let (mut lessons, mut added, mut retired) = (0, 0, 0);
    // a trajectory line every ten laps: right answers per bar in the window, and churn
    let window: usize = 10;
    let mut win = vec![0usize; n];
    let (mut wmint, mut wadd, mut wret, mut wreuse) = (0, 0, 0, 0);
    let t0 = std::time::Instant::now();
    for k in 0..total {
        let pct = PATTERN[k % n] * 100.0;
        let up = (pct > 0.0) as u32;
        let (action, lesson) = match pending { Some(a) => (a, false), None => (up, true) };
        if lesson { lessons += 1; }
        let reward = if action == 1 { pct } else { -pct };
        let cf = k % n;
        if !lesson {
            if action == up { stats[cf].0 += 1; } else { stats[cf].1 += 1; }
            if k >= total - 10 * n { if action == up { last10[cf].0 += 1; } else { last10[cf].1 += 1; } }
        }
        let inputs = [Input { dim: ev, bucket: up, pos: [0; MAX_DIMS] }, Input { dim: ac, bucket: action, pos: [0; MAX_DIMS] }];
        let winners = m.process_frame(&inputs, vec![Reward { reward, channels: vec![ch], frames: 1 }]);
        let st = m.last_stats.clone();
        added += st.added; retired += st.retired;
        wmint += st.minted; wadd += st.added; wret += st.retired; wreuse += st.reused;
        let w = winners.iter().find(|w| w.dim == ac);
        pending = w.map(|w| w.bucket);
        if !lesson && action == up { win[cf] += 1; }
        if (k + 1) % (window * n) == 0 {
            let hc = m.height_counts();
            println!("laps {:4}-{:4} | right/bar {:?} of {} | minted {:3} reused {:3} added {:3} retired {:3} | alive neurons {:3} patterns {:3} | max height {:2} | {:.1}s",
                (k + 1) / n - window + 1, (k + 1) / n, win, window, wmint, wreuse, wadd, wret, st.neurons, st.patterns, hc.len().saturating_sub(1), t0.elapsed().as_secs_f64());
            win.iter_mut().for_each(|x| *x = 0); wmint = 0; wadd = 0; wret = 0; wreuse = 0;
        }
        if verbose && (k + 2 * n >= total || args.get(4).map_or(false, |s| s == "vv")) {
            let fr = m.frames.back().unwrap();
            let apex: Vec<String> = fr.acts.iter().filter(|a| a.covered_at.is_none()).map(|a| format!("#{}h{}c{}", a.neuron, a.height, m.thalamus[a.neuron].connections.len())).collect();
            let next_up = (PATTERN[(k + 1) % n] > 0.0) as u32;
            println!("f{:4} cf{} price={} ran={}{} r={:+.1} | apex {:?} levels={} bids={} acc={} minted={} added={} retired={} voters={}/{} | says {:?} est {:.2} -> {}",
                k + 1, cf + 1, up, action, if lesson { "L" } else { " " }, reward, apex, st.levels, st.bids, st.accepted, st.minted, st.added, st.retired, st.voters_above_base, st.voters,
                pending, w.map_or(0.0, |w| w.estimate), if pending == Some(next_up) { "ok" } else { "WRONG" });
        }
    }
    println!("cycle frame | optimal | suboptimal | last 10 cycles");
    let (mut to, mut ts, mut lo, mut ls) = (0, 0, 0, 0);
    for i in 0..n {
        println!("{:11} | {:7} | {:10} | {}/{}", i + 1, stats[i].0, stats[i].1, last10[i].0, last10[i].0 + last10[i].1);
        to += stats[i].0; ts += stats[i].1; lo += last10[i].0; ls += last10[i].1;
    }
    let st = &m.last_stats;
    println!("overall optimal {}/{} = {:.1}% | last 10 cycles {}/{} = {:.1}% | lessons {} | neurons {} alive of {} minted | patterns {} alive, {} added, {} retired | heights {:?}",
        to, to + ts, 100.0 * to as f64 / (to + ts).max(1) as f64, lo, lo + ls, 100.0 * lo as f64 / (lo + ls).max(1) as f64, lessons, st.neurons, m.thalamus.len(), st.patterns, added, retired, m.height_counts());
}
