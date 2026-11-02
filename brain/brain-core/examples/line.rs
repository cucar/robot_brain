//! A single shape shown over and over: a 7x7 image, white everywhere but a vertical black line from (3, 1) to (3, 5),
//! each pixel firing black or white every frame, as in the MNIST example. Two frames per showing: the image, then
//! the digit 1 with a reward of +1. Reports, per showing, the levels, what stands on the apex and how much of it
//! stood the time before, and the functions added; at the end, the dictionary.
//! usage: cargo run --release --example line -- [showings=100] [h=2048] [trace-showing=0]
use brain_core::machine::Machine;
use brain_core::types::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let showings: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let h: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let trace_at: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let size = 7;
    let mut m = Machine::new(h, Policy::events_to_actions(), 1);
    let ax = m.declare_axis("x", 1);
    let ay = m.declare_axis("y", 1);
    let image = m.declare_channel("image", Some(("pixel", 2)), None, &[ax, ay]);
    let digit = m.declare_channel("digit", None, Some(("digit", 10)), &[]);
    let pix = m.channels[image as usize].event.unwrap();
    let dig = m.channels[digit as usize].action.unwrap();
    let ink = |x: usize, y: usize| x == 3 && (1..=5).contains(&y);
    println!("a vertical line at x=3, y=1..5, on a {}x{} image, shown {} times, H={}", size, size, showings, h);
    for y in 0..size { println!("    {}", (0..size).map(|x| if ink(x, y) { '#' } else { '.' }).collect::<String>()); }

    let mut last: Vec<String> = Vec::new();
    for k in 1..=showings {
        let mut inputs = Vec::new();
        for y in 0..size { for x in 0..size {
            let mut pos = [0i32; MAX_DIMS];
            pos[ax] = x as i32;
            pos[ay] = y as i32;
            inputs.push(Input { dim: pix, bucket: if ink(x, y) { 1 } else { 0 }, pos });
        } }
        m.trace = k == trace_at;
        let w = m.process_frame(&inputs, vec![]);
        m.trace = false;
        let said = w.iter().find(|w| w.dim == dig).map(|w| w.bucket);
        let fr = m.frames.back().unwrap();
        let mut ap: Vec<String> = fr.acts.iter().filter(|a| a.on_apex() && a.kind != Kind::Action)
            .map(|a| if a.kind == Kind::Inter { format!("#{}@{},{}", a.neuron, a.pos[1], a.pos[2]) } else { format!("{}@{},{}", if m.thalamus[a.neuron].bucket == 1 { "B" } else { "W" }, a.pos[1], a.pos[2]) }).collect();
        ap.sort();
        let kept = ap.iter().filter(|x| last.contains(x)).count();
        let levels = m.last_stats.levels;
        let added = m.last_stats.added;
        let retired = m.last_stats.retired;
        let per_level: Vec<String> = m.last_stats.per_level.iter().map(|(acts, _, _, acc, _, _)| format!("{}>{}", acts, acc)).collect();
        println!("  showing {:3}: said {} | levels {} [{}] | apex {} ({} kept) | functions added {} retired {} | functions {} neurons {}{}",
            k, said.map_or("-".into(), |s| s.to_string()), levels, per_level.join(" "), ap.len(), kept, added, retired, m.dict.len(), m.alive,
            if k <= 3 || k == showings || kept != ap.len() && k % 10 == 0 { format!("\n      apex [{}]", ap.join(" ")) } else { String::new() });
        if k == trace_at { level1_report(&m); }
        last = ap;
        m.process_frame(&[Input { dim: dig, bucket: 1, pos: [0; MAX_DIMS] }], vec![Reward { reward: 1.0, channels: vec![digit], frames: 1 }]);
    }
    let mut d: Vec<String> = m.dict.funcs.values().map(|f| m.dict.describe(f)).collect();
    d.sort();
    println!("dictionary: {} functions", d.len());
    for s in d { println!("  {}", s); }
}

/// Every relation a level-1 call returned on this showing, with whether the pixels the two ends stand for touch:
/// share a side or a corner (diagnostic).
fn level1_report(m: &Machine) {
    use brain_core::machine::ActRef;
    let fno = m.frame_no;
    let Some(fr) = m.frames.iter().find(|f| f.no == fno) else { return };
    let pixels = |r: ActRef| -> Vec<(i32, i32)> {
        let mut out = Vec::new();
        let mut stack = vec![r];
        while let Some(x) = stack.pop() {
            let Some(a) = m.act(x) else { continue };
            if a.kind == Kind::Event { out.push((a.pos[1], a.pos[2])); }
            stack.extend(a.covers.iter().copied());
        }
        out
    };
    let name = |r: ActRef| -> String {
        let a = m.act(r).unwrap();
        let n = match a.neuron { 0 => "W".to_string(), 1 => "B".to_string(), 12 => "Wrow".into(), 13 => "Wcol".into(), 14 => "Bcol".into(), x => format!("#{}", x) };
        format!("{}@({},{})", n, a.pos[1], a.pos[2])
    };
    let span = |px: &Vec<(i32, i32)>| -> String {
        let (x0, x1) = (px.iter().map(|p| p.0).min().unwrap(), px.iter().map(|p| p.0).max().unwrap());
        let (y0, y1) = (px.iter().map(|p| p.1).min().unwrap(), px.iter().map(|p| p.1).max().unwrap());
        format!("x{}-{} y{}-{}", x0, x1, y0, y1)
    };
    let (mut total, mut touching) = (0, 0);
    let mut seen = std::collections::HashSet::new();
    for (i, a) in fr.acts.iter().enumerate() {
        if a.kind != Kind::Inter || a.level != 1 { continue; }
        let me = ActRef { frame: fno, index: i as u32 };
        let mine = pixels(me);
        eprintln!("[l1] {} stands for {} pixels, {} | reach x{} y{} | returned {}:", name(me), mine.len(), span(&mine), a.reach[1], a.reach[2], a.rels.len());
        for br in &a.rels {
            let theirs = pixels(br.other);
            let touch = mine.iter().any(|p| theirs.iter().any(|q| p != q && (p.0 - q.0).abs() <= 1 && (p.1 - q.1).abs() <= 1));
            let pair = if me <= br.other { (me, br.other) } else { (br.other, me) };
            if seen.insert(pair) { total += 1; if touch { touching += 1; } }
            eprintln!("[l1]     to {:14} at ({:2},{:2}), stands for {} | strength {:2} | {}", name(br.other), br.off[1], br.off[2], span(&theirs), br.strength, if touch { "touching" } else { "not touching" });
        }
    }
    eprintln!("[l1] distinct pairs related at level 1: {} | touching: {} | not touching: {}", total, touching, total - touching);
}
