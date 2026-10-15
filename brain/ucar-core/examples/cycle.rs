//! The synthetic extended cycle test (apps/stocks/jobs/synthetic-extended-test.js) on UCAR: the first
//! 12 rows of KGC repeated, binary price and volume buckets, own/out, the world running the right
//! action where the machine is silent, a reward scoped to the price channel and the frame.
//! usage: cargo run --release --example cycle -- [repeats] [H] [R] [v]
use ucar_core::machine::Machine;
use ucar_core::types::*;

const ROWS: [(f64, f64); 12] = [
    (8.10, 1447100.0), (8.11, 2112900.0), (8.35, 1411400.0), (8.29, 2091100.0), (8.20, 1247200.0), (8.15, 770000.0),
    (8.19, 1948400.0), (8.05, 2701100.0), (7.94, 1280800.0), (7.86, 3083500.0), (7.22, 2742800.0), (7.51, 1510600.0),
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let repeats: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let h: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let r: i32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let verbose = args.get(4).map_or(false, |s| s.starts_with('v'));
    let no_volume = std::env::var("UCAR_NO_VOLUME").is_ok();

    let mut policy = Policy::events_to_actions();
    if std::env::var("UCAR_NO_PATTERNS").is_ok() { policy.past = [[false; 3]; 3]; }
    let mut m = Machine::new(h, policy, r);
    let price = m.declare_channel("price", Some(("price_change", 2)), Some(("activity", 2)), &[]);
    let volume = if no_volume { None } else { Some(m.declare_channel("volume", Some(("volume_change", 2)), None, &[])) };
    let pev = m.channels[price as usize].event.unwrap();
    let pac = m.channels[price as usize].action.unwrap();
    let vev = volume.map(|v| m.channels[v as usize].event.unwrap());

    // rows cycle; frame k reads row k-1 → row k (mod 12); the optimal action at frame k is own iff
    // the price rises into frame k+1
    let n = 12;
    let total = repeats * n;
    let mut pending: Option<u32> = None;
    let mut stats = vec![(0usize, 0usize); n]; // (optimal, suboptimal) per cycle frame
    let mut last10 = vec![(0usize, 0usize); n]; // the same over the last ten cycles
    let mut lessons = 0;
    let mut pnl = 0.0;
    let (mut minted, mut retired, mut added) = (0usize, 0usize, 0usize);
    for k in 1..=total {
        let prev = ROWS[(k - 1) % n];
        let cur = ROWS[k % n];
        let pct = (cur.0 - prev.0) / prev.0 * 100.0;
        let vol = (cur.1 - prev.1) / prev.1 * 100.0;
        let up = (pct > 0.0) as u32;
        // the action in effect over this bar
        let (action, lesson) = match pending { Some(a) => (a, false), None => (up, true) };
        if lesson { lessons += 1; }
        let reward = if action == 1 { pct } else { -pct };
        if action == 1 { pnl += pct; }
        // score: did what we owned over this bar match the optimum (own iff the bar rose)
        let cf = (k - 1) % n;
        if !lesson { if action == up { stats[cf].0 += 1; } else { stats[cf].1 += 1; } }
        if !lesson && k > total.saturating_sub(10 * n) { if action == up { last10[cf].0 += 1; } else { last10[cf].1 += 1; } }
        let mut inputs = vec![Input { dim: pev, bucket: up, pos: [0; MAX_DIMS] }, Input { dim: pac, bucket: action, pos: [0; MAX_DIMS] }];
        if let Some(v) = vev { inputs.push(Input { dim: v, bucket: (vol > 0.0) as u32, pos: [0; MAX_DIMS] }); }
        let mut winners = m.process_frame(&inputs, vec![Reward { reward, channels: vec![price], frames: 1 }]);
        // UCAR_GAP: silent frames after each cycle, as the loop case's world "shows nothing for a while"
        let gap: usize = std::env::var("UCAR_GAP").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        if gap > 0 && k % n == 0 {
            for _ in 0..gap { winners = m.process_frame(&[], vec![]); }
        }
        minted += m.last_stats.minted; retired += m.last_stats.retired; added += m.last_stats.added;
        let w = winners.iter().find(|w| w.dim == pac);
        pending = w.map(|w| w.bucket);
        if verbose {
            let fr = m.frames.back().unwrap();
            let apex: Vec<String> = fr.acts.iter().filter(|a| a.covered_at.is_none()).map(|a| format!("#{}L{}c{}", a.neuron, a.level, m.neurons[a.neuron as usize].connections.len())).collect();
            let covered: Vec<String> = fr.acts.iter().filter(|a| a.covered_at.is_some()).map(|a| format!("#{}L{}", a.neuron, a.level)).collect();
            println!("      apex {:?} covered {:?}", apex, covered);
            let st = &m.last_stats;
            let next_up = ((ROWS[(k + 1) % n].0 - cur.0) > 0.0) as u32;
            println!("f{:3} cf{:2} price={} vol={} ran={}{} r={:+.2} | apex={} levels={} bids={} acc={} minted={} retired={} voters={}/{} | says {:?} (est {:.2}, share {:.2}, voters {}) next_up={} {}",
                k, cf + 1, up, (vol > 0.0) as u32, action, if lesson { "L" } else { " " }, reward,
                st.apex_acts, st.levels, st.bids, st.accepted, st.minted, st.retired, st.voters_above_base, st.voters,
                pending, w.map_or(0.0, |w| w.estimate), w.map_or(0.0, |w| w.share), w.map_or(0, |w| w.voters), next_up,
                if pending == Some(next_up) { "ok" } else { "WRONG" });
        }
    }
    println!("cycle frame | optimal | suboptimal | last 10 cycles");
    let (mut to, mut ts) = (0, 0);
    let (mut lo, mut ls) = (0, 0);
    for (i, (o, s)) in stats.iter().enumerate() {
        println!("{:11} | {:7} | {:10} | {}/{}", i + 1, o, s, last10[i].0, last10[i].0 + last10[i].1);
        to += o; ts += s; lo += last10[i].0; ls += last10[i].1;
    }
    println!("last 10 cycles optimal {}/{} = {:.1}%", lo, lo + ls, 100.0 * lo as f64 / (lo + ls).max(1) as f64);
    let st = &m.last_stats;
    println!("overall optimal {}/{} = {:.1}% | lessons {} | pnl {:+.1}% | neurons {} alive of {} minted | patterns {} alive, {} added, {} retired | conns {} levels {:?}",
        to, to + ts, 100.0 * to as f64 / (to + ts).max(1) as f64, lessons, pnl, st.neurons, m.neurons.len(), st.patterns, added, retired, st.connections, m.level_counts());
    if let Ok(ids) = std::env::var("UCAR_DUMP") {
        for id in ids.split(',').filter_map(|x| x.parse::<u32>().ok()) {
            print!("{}", m.describe_neuron(id));
            let nn = &m.neurons[id as usize];
            let mut cs: Vec<_> = nn.connections.iter().collect();
            cs.sort_by_key(|((t, o), _)| (o[0], *t));
            for ((t, o), c) in cs { println!("    -> #{} @{} s={} est={:+.2}", t, o[0], c.strength, c.estimate()); }
        }
    }
    // the last cycle alone
    let last: usize = stats.iter().map(|(o, _)| *o).sum();
    let _ = last;
}
