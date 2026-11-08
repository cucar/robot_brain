//! A straight line at an angle, shown over and over: a white image with a black line drawn across it, every pixel
//! firing black or white every frame, as in the MNIST example. At 45° or less the line has one pixel per column,
//! at y = round(x · tan angle); above 45°, one pixel per row, at x = round(y / tan angle). Two frames per showing:
//! the image, then the digit 0 with a reward of +1. Reports, per showing, the levels and what stands on the apex;
//! for the last showing, every call each level made.
//! usage: cargo run --release --example slope -- [angle=45] [showings=20] [size=9]
use brain_core::machine::Machine;
use brain_core::types::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let angle: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(45.0);
    let showings: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let size: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(9);
    let mut m = Machine::new(2048, Policy::events_to_actions(), 1);
    let ax = m.declare_axis("x", 1);
    let ay = m.declare_axis("y", 1);
    let image = m.declare_channel("image", Some(("pixel", 2)), None, &[ax, ay]);
    let digit = m.declare_channel("digit", None, Some(("digit", 10)), &[]);
    let pix = m.channels[image as usize].event.unwrap();
    let dig = m.channels[digit as usize].action.unwrap();
    let t = angle.to_radians().tan();
    let mut ink = vec![vec![false; size]; size];
    if angle <= 45.0 {
        for x in 0..size { let y = (x as f64 * t).round() as usize; if y < size { ink[y][x] = true; } }
    } else {
        for y in 0..size { let x = (y as f64 / t).round() as usize; if x < size { ink[y][x] = true; } }
    }
    println!("a black line at {}° on a white {}x{} image, shown {} times (# = black #1, . = white #0)", angle, size, size, showings);
    for y in 0..size { println!("    {}", (0..size).map(|x| if ink[y][x] { '#' } else { '.' }).collect::<String>()); }
    let mut last: Vec<String> = Vec::new();
    for k in 1..=showings {
        let mut inputs = Vec::new();
        for y in 0..size { for x in 0..size {
            let mut pos = [0i32; MAX_DIMS];
            pos[ax] = x as i32;
            pos[ay] = y as i32;
            inputs.push(Input { dim: pix, bucket: if ink[y][x] { 1 } else { 0 }, pos });
        } }
        m.process_frame(&inputs, vec![]);
        let fr = m.frames.back().unwrap();
        let mut ap: Vec<String> = fr.acts.iter().filter(|a| a.on_apex() && a.kind != Kind::Action).map(|a| name(&m, a)).collect();
        ap.sort();
        let kept = ap.iter().filter(|x| last.contains(x)).count();
        let per_level: Vec<String> = m.last_stats.per_level.iter().map(|(acts, _, fits, acc, _, _)| format!("{}:{}>{}", acts, fits, acc)).collect();
        println!("  showing {:3}: levels {} [{}] | apex {} ({} kept) | functions added {} retired {} | functions {}",
            k, m.last_stats.levels, per_level.join(" "), ap.len(), kept, m.last_stats.added, m.last_stats.retired, m.dict.len());
        if k == showings {
            println!("      apex [{}]", ap.join(" "));
            calls(&m);
        }
        last = ap;
        m.process_frame(&[Input { dim: dig, bucket: 0, pos: [0; MAX_DIMS] }], vec![Reward { reward: 1.0, channels: vec![digit], frames: 1 }]);
    }
}

/// An activation as the report prints it: a pixel as # or . at its position, a call as its neuron and pose there.
fn name(m: &Machine, a: &brain_core::machine::Act) -> String {
    if a.kind == Kind::Inter { format!("{}@{},{}", m.dict.sym_text(a.sym), a.pos[1], a.pos[2]) }
    else { format!("{}@{},{}", if m.thalamus[a.neuron].bucket == 1 { "#" } else { "." }, a.pos[1], a.pos[2]) }
}

/// Every call this showing made, level by level: its function, pose, position, and the pixels it stands for, the
/// black ones first.
fn calls(m: &Machine) {
    let fno = m.frame_no;
    let Some(fr) = m.frames.iter().find(|f| f.no == fno) else { return };
    let mut by_level: std::collections::BTreeMap<u32, Vec<String>> = std::collections::BTreeMap::new();
    for a in fr.acts.iter().filter(|a| a.kind == Kind::Inter) {
        let (mut black, mut white) = (Vec::new(), 0);
        let mut stack = a.covers.clone();
        let mut seen = std::collections::HashSet::new();
        while let Some(r) = stack.pop() {
            if !seen.insert(r) { continue; }
            let Some(c) = m.act(r) else { continue };
            if c.kind == Kind::Event { if m.thalamus[c.neuron].bucket == 1 { black.push((c.pos[1], c.pos[2])); } else { white += 1; } }
            stack.extend(c.covers.iter().copied());
        }
        black.sort();
        let failed = a.fired_by.as_ref().map_or(0, |fb| fb.failed.len());
        by_level.entry(a.level).or_default().push(format!("{} covers {} | black {:?} white {}{}{}", name(m, a), a.covers.len(), black, white,
            if failed > 0 { format!(" failed {}", failed) } else { String::new() }, if a.on_apex() { "" } else { " (covered)" }));
    }
    for (l, v) in by_level { for s in v { println!("      level {} call: {}", l, s); } }
}
