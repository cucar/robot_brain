//! Several channels in one machine, no reset (docs/algorithm.md D34, D24). One channel per stock: the price's move up or
//! down as the event, own or out as the action, the reward scoped to that stock's channel and the frame; every stock
//! fires in every frame. Neighbors by kind (Policy::events_to_actions): an event reads every event and call of any
//! channel in its frame and the one before; no action is anyone's neighbor.
//!
//! Modes:
//!   two    KGC and GLD, the first 12 daily opens of each (apps/stocks/data/1D), cycled
//!   three  KGC, GLD and SPY, the same way
//!   lap    channel A, the six-bar lap of the stocks example with its action, beside channel B, one event channel up
//!          or down from a fixed seeded sequence (seed 20261116), with no action
//!
//! usage: cargo run --release --example channels -- <two|three|lap> [cycles=10 for stocks, 30 for lap] [data-dir]
use brain_core::activation::Act;
use brain_core::machine::Machine;
use brain_core::types::*;
use std::collections::BTreeSet;

/// The first 12 daily opens of a symbol, as apps/stocks/jobs/multi-channel-test.js loads them.
fn opens(dir: &str, sym: &str) -> Vec<f64> {
    let txt = std::fs::read_to_string(format!("{}/{}.json", dir, sym)).unwrap_or_else(|e| panic!("{}/{}.json: {}", dir, sym, e));
    let bars: Vec<serde_json::Value> = serde_json::from_str(&txt).unwrap();
    bars.iter().take(12).map(|b| b["OpenPrice"].as_f64().unwrap()).collect()
}

/// The events a call stands for, at any depth: (event dimension, frame).
fn under(m: &Machine, a: &Act) -> BTreeSet<(DimId, i64)> {
    let mut out = BTreeSet::new();
    let mut stack = a.covers.clone();
    while let Some(r) = stack.pop() {
        let Some(c) = m.act(r) else { continue };
        if c.covers.is_empty() { if c.kind == Kind::Event { out.insert((m.thalamus[c.neuron].dim.unwrap(), r.frame)); } } else { stack.extend(c.covers.iter().copied()); }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).cloned().unwrap_or("two".into());
    let dir = args.get(3).cloned().unwrap_or(format!("{}/../../apps/stocks/data/1D", env!("CARGO_MANIFEST_DIR")));
    // each channel: its name, the percent move of each bar of its cycle, and whether it has an action
    let mut series: Vec<(String, Vec<f64>, bool)> = Vec::new();
    let (cycle, default_cycles) = match mode.as_str() {
        "two" | "three" => {
            let syms: &[&str] = if mode == "two" { &["KGC", "GLD"] } else { &["KGC", "GLD", "SPY"] };
            for s in syms {
                let p = opens(&dir, s);
                // bar i is the move from row i to row i+1; the last wraps to the first row
                series.push((s.to_string(), (0..12).map(|i| (p[(i + 1) % 12] - p[i]) / p[i] * 100.0).collect(), true));
            }
            (12usize, 10usize)
        }
        "lap" => {
            series.push(("A".into(), vec![1.0, -2.0, -3.0, 2.0, 3.0, -1.0], true));
            let mut seed: u64 = 20261116;
            let mut b = Vec::new();
            for _ in 0..180 { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); b.push(if (seed >> 33) & 1 == 1 { 1.0 } else { -1.0 }); }
            series.push(("B".into(), b, false));
            (6usize, 30usize)
        }
        _ => panic!("mode: two, three or lap"),
    };
    let cycles: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(default_cycles);
    let mut m = Machine::new(256, Policy::events_to_actions(), 1);
    let mut chans: Vec<(ChannelId, DimId, Option<DimId>)> = Vec::new();
    for (name, _, act) in &series {
        let ch = m.declare_channel(name, Some(("move", 2)), if *act { Some(("position", 2)) } else { None }, &[]);
        chans.push((ch, m.channels[ch as usize].event.unwrap(), m.channels[ch as usize].action));
    }
    println!("{}: {} cycles of {} bars", mode, cycles, cycle);
    for (name, pct, _) in &series { if pct.len() == cycle { println!("  {} moves per cycle: {}", name, pct.iter().map(|p| if *p > 0.0 { 'U' } else { 'D' }).collect::<String>()); } }

    let total = cycle * cycles;
    let mut pending: Vec<Option<u32>> = vec![None; series.len()];
    let mut grid: Vec<Vec<Vec<char>>> = vec![vec![vec!['-'; cycle]; cycles]; series.len()];
    let mut combos: Vec<Vec<usize>> = vec![Vec::new(); cycles];
    let mut six_laps: Vec<usize> = Vec::new();
    let (mut last_dict, mut last_change) = (String::new(), 0i64);
    let t0 = std::time::Instant::now();
    for k in 0..total {
        let mut inputs = Vec::new();
        let mut rewards = Vec::new();
        for (i, (_, pct, _)) in series.iter().enumerate() {
            let p = pct[k % pct.len()];
            let up: u32 = if p > 0.0 { 1 } else { 0 };
            inputs.push(Input { dim: chans[i].1, bucket: up, pos: [0; MAX_DIMS] });
            if let Some(ac) = chans[i].2 {
                // what the machine said for this bar runs; a lesson (the move itself) where it said nothing
                let action = pending[i].unwrap_or(up);
                inputs.push(Input { dim: ac, bucket: action, pos: [0; MAX_DIMS] });
                rewards.push(Reward { reward: if action == 1 { p } else { -p }, channels: vec![chans[i].0], frames: 1 });
            }
        }
        let w = m.process_frame(&inputs, rewards);
        let fno = m.frame_no;
        for (i, (_, pct, _)) in series.iter().enumerate() {
            let Some(ac) = chans[i].2 else { continue };
            pending[i] = w.iter().find(|x| x.dim == ac).map(|x| x.bucket);
            let next_up = if pct[(k + 1) % pct.len()] > 0.0 { 1 } else { 0 };
            if let Some(g) = pending[i] { grid[i][k / cycle][k % cycle] = if g == next_up { 'R' } else { 'W' }; }
        }
        // recognition: the dictionary's last change, and this frame's combination calls (a level-1 call over two of
        // this frame's bars)
        let mut d: Vec<String> = m.dict.funcs.values().map(|f| m.dict.describe(f)).collect();
        d.sort();
        let d = d.join("; ");
        if d != last_dict { last_change = fno; last_dict = d; }
        let fr = m.frames.back().unwrap();
        let combo = fr.acts.iter().filter(|a| a.level == 1 && a.fired_by.is_some() && a.covers.iter().filter(|c| c.frame == fno && m.act(**c).map_or(false, |x| x.kind == Kind::Event)).count() >= 2).count();
        combos[k / cycle].push(combo);
        // the lap: whether a call on the apex stands for channel A's six bars of this lap
        if mode == "lap" && k % cycle == cycle - 1 {
            let a_ev = chans[0].1;
            let lap: Vec<i64> = ((fno - 5)..=fno).collect();
            let stands = m.frames.iter().flat_map(|f| f.acts.iter()).any(|a| a.on_apex() && !a.covers.is_empty() && { let u = under(&m, a); lap.iter().all(|f| u.contains(&(a_ev, *f))) });
            if stands { six_laps.push(k / cycle + 1); }
        }
    }
    for (i, (name, _, act)) in series.iter().enumerate() {
        if !act { continue; }
        let (r, w) = grid[i].iter().flatten().fold((0, 0), |(r, w), c| (r + (*c == 'R') as usize, w + (*c == 'W') as usize));
        let from = (0..cycles).rev().take_while(|c| grid[i][*c].iter().all(|x| *x == 'R')).last();
        println!("inference {}: right {} of {} | every bar right from cycle {}", name, r, r + w, from.map_or("never".into(), |c| (c + 1).to_string()));
        for (c, row) in grid[i].iter().enumerate() { println!("  {} cycle {:2}: {}", name, c + 1, row.iter().collect::<String>()); }
    }
    println!("recognition: {} functions | the dictionary last changed at frame {}", m.dict.len(), last_change);
    println!("  combination calls per frame, per cycle: {}", combos.iter().enumerate().map(|(c, v)| format!("{}:{}", c + 1, v.iter().map(|x| x.to_string()).collect::<String>())).collect::<Vec<_>>().join(" "));
    if mode == "lap" { println!("  channel A's six-bar cycle stands at the end of laps: {}", if six_laps.is_empty() { "none".into() } else { format!("{:?}", six_laps) }); }
    println!("  {:.1} s", t0.elapsed().as_secs_f64());
}
