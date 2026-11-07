//! A single shape shown over and over: a 5x5 black image with a white 3x3 square in the middle, at (1..3, 1..3),
//! each pixel firing black or white every frame, as in the MNIST example. Two frames per showing: the image, then
//! the digit 0 with a reward of +1. Reports, per showing, the levels, what stands on the apex and how much of it
//! stood the time before, and the functions added; for the showings asked, every call each level made; at the
//! end, the dictionary.
//! With `hollow` as the fifth argument the middle is silent: only the black border reports, nothing at (1..3, 1..3).
//! usage: cargo run --release --example square -- [showings=30] [h=2048] [detail-every=10] [trace-showing=0] [hollow]
use brain_core::machine::Machine;
use brain_core::types::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let showings: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(30);
    let h: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let every: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let trace_at: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    let hollow = args.get(5).map_or(false, |s| s == "hollow");
    let size = 5;
    let mut m = Machine::new(h, Policy::events_to_actions(), 1);
    let ax = m.declare_axis("x", 1);
    let ay = m.declare_axis("y", 1);
    let image = m.declare_channel("image", Some(("pixel", 2)), None, &[ax, ay]);
    let digit = m.declare_channel("digit", None, Some(("digit", 10)), &[]);
    let pix = m.channels[image as usize].event.unwrap();
    let dig = m.channels[digit as usize].action.unwrap();
    let white = |x: usize, y: usize| (1..=3).contains(&x) && (1..=3).contains(&y);
    println!("a white 3x3 square at x=1..3, y=1..3, on a black {}x{} image, shown {} times, H={} (B = black #0, W = white #1)", size, size, showings, h);
    if hollow { println!("hollow: the middle reports nothing (shown as .)"); }
    for y in 0..size { println!("    {}", (0..size).map(|x| if white(x, y) { if hollow { '.' } else { 'W' } } else { 'B' }).collect::<String>()); }

    let mut last: Vec<String> = Vec::new();
    for k in 1..=showings {
        let mut inputs = Vec::new();
        for y in 0..size { for x in 0..size {
            let mut pos = [0i32; MAX_DIMS];
            pos[ax] = x as i32;
            pos[ay] = y as i32;
            if hollow && white(x, y) { continue; }
            inputs.push(Input { dim: pix, bucket: if white(x, y) { 1 } else { 0 }, pos });
        } }
        m.trace = k == trace_at;
        let w = m.process_frame(&inputs, vec![]);
        m.trace = false;
        let said = w.iter().find(|w| w.dim == dig).map(|w| w.bucket);
        let fr = m.frames.back().unwrap();
        let mut ap: Vec<String> = fr.acts.iter().filter(|a| a.on_apex() && a.kind != Kind::Action).map(|a| name(&m, a)).collect();
        ap.sort();
        let kept = ap.iter().filter(|x| last.contains(x)).count();
        let per_level: Vec<String> = m.last_stats.per_level.iter().map(|(acts, _, fits, acc, _, _)| format!("{}:{}>{}", acts, fits, acc)).collect();
        println!("  showing {:3}: said {} | levels {} [{}] | apex {} ({} kept) | functions added {} retired {} | functions {} neurons {}",
            k, said.map_or("-".into(), |s| s.to_string()), m.last_stats.levels, per_level.join(" "), ap.len(), kept, m.last_stats.added, m.last_stats.retired, m.dict.len(), m.alive);
        if k <= 3 || k == showings || (every > 0 && k % every == 0) || kept != ap.len() && k > 3 {
            println!("      apex [{}]", ap.join(" "));
            calls(&m);
        }
        last = ap;
        m.process_frame(&[Input { dim: dig, bucket: 0, pos: [0; MAX_DIMS] }], vec![Reward { reward: 1.0, channels: vec![digit], frames: 1 }]);
    }
    let mut d: Vec<String> = m.dict.funcs.values().map(|f| m.dict.describe(f)).collect();
    d.sort();
    println!("dictionary: {} functions", d.len());
    for s in d { println!("  {}", s); }
}

/// An activation as the report prints it: a pixel as B or W at its position, a call as its neuron and pose at its position.
fn name(m: &Machine, a: &brain_core::machine::Act) -> String {
    if a.kind == Kind::Inter { format!("{}@{},{}", m.dict.sym_text(a.sym), a.pos[1], a.pos[2]) }
    else { format!("{}@{},{}", if m.thalamus[a.neuron].bucket == 1 { "W" } else { "B" }, a.pos[1], a.pos[2]) }
}

/// Every call this showing made, level by level: its function, pose, position, and the pixels it stands for.
fn calls(m: &Machine) {
    let fno = m.frame_no;
    let Some(fr) = m.frames.iter().find(|f| f.no == fno) else { return };
    let mut by_level: std::collections::BTreeMap<u32, Vec<String>> = std::collections::BTreeMap::new();
    for a in fr.acts.iter().filter(|a| a.kind == Kind::Inter) {
        let mut px: Vec<(i32, i32)> = Vec::new();
        let mut stack = a.covers.clone();
        while let Some(r) = stack.pop() {
            let Some(c) = m.act(r) else { continue };
            if c.kind == Kind::Event { px.push((c.pos[1], c.pos[2])); }
            stack.extend(c.covers.iter().copied());
        }
        px.sort();
        px.dedup();
        let func = a.fired_by.as_ref().map_or("?".into(), |fb| format!("f{}", fb.func));
        let failed = a.fired_by.as_ref().map_or(0, |fb| fb.failed.len());
        by_level.entry(a.level).or_default().push(format!("{} {} covers {} [{}] pixels {}{}{}", func, name(m, a), a.covers.len(),
            a.covers.iter().filter_map(|r| m.act(*r)).map(|c| name(m, c)).collect::<Vec<_>>().join(" "), px.len(),
            if failed > 0 { format!(" failed {}", failed) } else { String::new() },
            if a.on_apex() { "" } else { " (covered)" }));
    }
    for (l, v) in by_level { for s in v { println!("      level {} call: {}", l, s); } }
}
