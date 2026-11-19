//! A straight line at an angle, shown over and over: a white image with a black line drawn across it, every pixel
//! firing black or white every frame, as in the MNIST example. At 45° or less the line has one pixel per column,
//! at y = round(x · tan angle); above 45°, one pixel per row, at x = round(y / tan angle). Two frames per showing:
//! the image, then the digit 0 with a reward of +1, then the context is reset. Reports, per showing, the levels and what stands on the apex;
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
    m.sw.closing = std::env::args().find_map(|a| a.strip_prefix("--closing=").map(|v| match v { "cycle" => 0, "round" => 1, _ => panic!("--closing=cycle|round") })).unwrap_or(0);
    m.trace_closing = std::env::args().any(|a| a == "--trace-closing");
    m.sw.join_at_a_loss = std::env::args().any(|a| a == "--join-at-a-loss");
    let trace_first = std::env::args().any(|a| a == "--trace-first");
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
        m.trace = trace_first && k == 1;
        let mut inputs = Vec::new();
        for y in 0..size { for x in 0..size {
            let mut pos = [0i32; MAX_DIMS];
            pos[ax] = x as i32;
            pos[ay] = y as i32;
            inputs.push(Input { dim: pix, bucket: if ink[y][x] { 1 } else { 0 }, pos });
        } }
        m.process_frame(&inputs, vec![]);
        if k == 1 || k == 2 || k == showings { let mut b = m.bare_after_l0.clone(); b.sort_by_key(|x| (x.1, x.0)); println!("BARE showing {}: {}", k, if b.is_empty() { "none".to_string() } else { b.iter().map(|(x, y, v)| format!("{}({},{})", if *v == 0 { "black" } else { "white" }, x, y)).collect::<Vec<_>>().join(" ") }); }
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
        // each showing stands alone: nothing of it is on the board when the next is shown
        m.reset_context();
    }
    println!("NEVERCALLED {} of {}", m.funcs_never_called(), m.dict.len());
    println!("CLOSING weighed {} joined {} (from a function of more relations {}) | cycles at the end {} of {} | build joins the calls count would refuse {} | largest such gain as a cycle {} | weighs beyond every offset the rows hold {} | weighs with neither end where the call stands {} | weighs where one lap on cannot be worked out {}", m.closing_weighed, m.closing_joined, m.closing_from_multi, m.dict.funcs.values().filter(|f| f.has_cycle()).count(), m.dict.len(), m.closing_refused[0], m.closing_max.map_or("-".to_string(), |(c, lo, _)| format!("{:.1} (on the calls {:.1})", c, lo)), m.closing_beyond, m.closing_no_end, m.closing_no_lap);
    println!("JOINREFUSED held by no function, gain not above zero {} | lowest gain refused {}", m.help_refused, m.help_refused_min.map_or("-".to_string(), |x| format!("{:.1}", x)));
    { let (t, s) = m.twice_counts(); println!("TWICE frames with an interneuron called more than once {} | of those, a row holds another call of the same interneuron {}", t, s); }
    let mut ids: Vec<_> = m.dict.funcs.keys().copied().collect();
    ids.sort();
    for id in ids { println!("DICT {}", m.dict.describe(m.dict.get(id).unwrap())); }
}

/// An activation as the report prints it: a pixel as # or . at its position, a call as its pose there.
fn name(m: &Machine, a: &brain_core::machine::Act) -> String {
    if a.kind == Kind::Inter { format!("{}@{},{}", m.dict.sym_text(a.sym), a.pos[1], a.pos[2]) }
    else { format!("{}@{},{}", if m.thalamus[a.neuron].bucket == 1 { "#" } else { "." }, a.pos[1], a.pos[2]) }
}

/// Every call this showing made, level by level: its function, phase, position, and the pixels it stands for, the
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
