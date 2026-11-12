//! Simple images, each shown again and again to a fresh machine, as in the square example: the image, then the
//! digit 0 with a reward of +1, then the context is reset. Modes: hline, vline, square, hollow, two, three; move_a
//! (the 5x5 square image learned, then the same 5x5 image at another place on a larger canvas where nothing else
//! reports); move_b (learned, then a 10x10 black image with the white square at another place); teach23 (the 2 and the
//! 3 in turn, each taught with its digit, then each shown frozen without it).
//! usage: shapes <mode> [showings=30]
use brain_core::function::{Func, Rel};
use brain_core::machine::Machine;
use brain_core::types::*;

type Img = Vec<Vec<char>>; // 'W' white (bucket 1), 'B' black (bucket 0), '.' reports nothing

fn img(rows: &[&str]) -> Img { rows.iter().map(|r| r.chars().collect()).collect() }

fn shape(mode: &str) -> Img {
    match mode {
        "hline" => img(&["BBBBBBB", "BBBBBBB", "BWWWWWB", "BBBBBBB", "BBBBBBB"]),
        "vline" => img(&["BBBBB", "BBBBB", "BBBBB", "BBBBB", "BBBBB", "BBBBB", "BBBBB"]).iter().enumerate().map(|(y, r)| r.iter().enumerate().map(|(x, c)| if x == 2 && (1..=5).contains(&y) { 'W' } else { *c }).collect()).collect(),
        "square" => img(&["BBBBB", "BWWWB", "BWWWB", "BWWWB", "BBBBB"]),
        "hollow" => img(&["BBBBB", "B...B", "B...B", "B...B", "BBBBB"]),
        "two" => img(&["BBBBBBB", "BWWWWWB", "BBBBBWB", "BBBBBWB", "BWWWWWB", "BWBBBBB", "BWBBBBB", "BWWWWWB", "BBBBBBB"]),
        "three" => img(&["BBBBBBB", "BWWWWWB", "BBBBBWB", "BBBBBWB", "BWWWWWB", "BBBBBWB", "BBBBBWB", "BWWWWWB", "BBBBBBB"]),
        _ => panic!("unknown shape"),
    }
}

/// An image placed at (ox, oy) on a canvas of w x h filled with `fill`.
fn place(i: &Img, w: usize, h: usize, ox: usize, oy: usize, fill: char) -> Img {
    let mut c = vec![vec![fill; w]; h];
    for (y, r) in i.iter().enumerate() { for (x, ch) in r.iter().enumerate() { c[oy + y][ox + x] = *ch; } }
    c
}

struct Env { ax: usize, ay: usize, pix: DimId, dig: DimId, digit: ChannelId }

fn inputs(e: &Env, i: &Img) -> Vec<Input> {
    let mut v = Vec::new();
    for (y, r) in i.iter().enumerate() { for (x, ch) in r.iter().enumerate() {
        if *ch == '.' { continue; }
        let mut pos = [0i32; MAX_DIMS];
        pos[e.ax] = x as i32;
        pos[e.ay] = y as i32;
        v.push(Input { dim: e.pix, bucket: if *ch == 'W' { 1 } else { 0 }, pos });
    } }
    v
}

fn print_img(i: &Img) { for r in i { println!("    {}", r.iter().collect::<String>()); } }

/// A symbol in words: a pixel's colour, or a call with its function, pose and laps.
fn sname(m: &Machine, s: u32) -> String {
    match m.dict.pose_of(s) {
        None => match m.thalamus[s].bucket { 1 => "white".into(), _ => "black".into() },
        Some(p) => {
            let n = m.dict.neuron_of(s);
            let fs: Vec<String> = m.dict.of_inter(n).iter().map(|f| format!("f{}", f)).collect();
            let end = |x: u32| if m.thalamus[x].kind == Kind::Inter { format!("call #{}", x) } else { sname(m, x) };
            format!("the call #{} of {} (from {} to {}, {} laps)", n, if fs.is_empty() { "no function now".into() } else { fs.join(",") }, end(p.from), end(p.to), p.laps)
        }
    }
}

fn off_words(o: &[i32]) -> String { if o[0] == 0 { format!("({:+}, {:+})", o[1], o[2]) } else { format!("(t {:+}, {:+}, {:+})", o[0], o[1], o[2]) } }

fn rel_words(m: &Machine, r: &Rel) -> String { format!("{} sees {} at {}", sname(m, r.a), sname(m, r.b), off_words(&r.off)) }

fn rel_text(m: &Machine, r: &Rel) -> String { format!("{}→{}{:?}", m.dict.sym_text(r.a), m.dict.sym_text(r.b), &r.off[..]) }

fn fingerprint(m: &Machine) -> Vec<String> { let mut d: Vec<String> = m.dict.funcs.values().map(|f| m.dict.describe(f)).collect(); d.sort(); d }

fn act_name(m: &Machine, a: &brain_core::machine::Act) -> String {
    if a.kind == Kind::Inter { format!("{}@{},{}", m.dict.sym_text(a.sym), a.pos[1], a.pos[2]) }
    else { format!("{}@{},{}", if m.thalamus[a.neuron].bucket == 1 { "W" } else { "B" }, a.pos[1], a.pos[2]) }
}

/// The pixels an activation stands for, followed down.
fn pixels(m: &Machine, a: &brain_core::machine::Act) -> Vec<(i32, i32)> {
    let mut px = Vec::new();
    let mut stack = a.covers.clone();
    if a.kind == Kind::Event { px.push((a.pos[1], a.pos[2])); }
    while let Some(r) = stack.pop() {
        let Some(c) = m.act(r) else { continue };
        if c.kind == Kind::Event { px.push((c.pos[1], c.pos[2])); }
        stack.extend(c.covers.iter().copied());
    }
    px.sort();
    px.dedup();
    px
}

/// One showing: the image, then the digit `d` with its reward (or, frozen, the machine's own answer unrewarded); the
/// context reset after. Returns what the machine said.
fn show(m: &mut Machine, e: &Env, i: &Img, d: Option<u32>, trace: bool) -> Option<u32> {
    m.trace = trace;
    let w = m.process_frame(&inputs(e, i), vec![]);
    m.trace = false;
    let said = w.iter().find(|w| w.dim == e.dig).map(|w| w.bucket);
    let ran = d.or(said);
    let rewards = if d.is_some() { vec![Reward { reward: 1.0, channels: vec![e.digit], frames: 1 }] } else { vec![] };
    m.process_frame(&ran.map(|b| Input { dim: e.dig, bucket: b, pos: [0; MAX_DIMS] }).into_iter().collect::<Vec<_>>(), rewards);
    said
}

/// What stands on the apex at the end of a showing, level by level, the relations among what stands, and the bare
/// pixels. Called right after the image frame, before the digit frame.
fn report_apex(m: &Machine) {
    let fno = m.frame_no;
    let Some(fr) = m.frames.iter().find(|f| f.no == fno) else { return };
    let mut by_level: std::collections::BTreeMap<u32, Vec<String>> = Default::default();
    let mut bare = Vec::new();
    let mut apex_refs = Vec::new();
    for (i, a) in fr.acts.iter().enumerate() {
        if !a.on_apex() || a.kind == Kind::Action { continue; }
        apex_refs.push(i);
        if a.kind == Kind::Event { bare.push(act_name(m, a)); continue; }
        let px = pixels(m, a);
        by_level.entry(a.level).or_default().push(format!("{} at ({}, {}) | {} | f{} | covers [{}] | stands for {} pixels",
            m.dict.sym_text(a.sym), a.pos[1], a.pos[2], sname(m, a.sym), a.fired_by.as_ref().map_or(0, |f| f.func),
            a.covers.iter().filter_map(|r| m.act(*r)).map(|c| act_name(m, c)).collect::<Vec<_>>().join(" "), px.len()));
    }
    let n_calls: usize = by_level.values().map(|v| v.len()).sum();
    println!("  on the apex: {} calls, {} bare pixels", n_calls, bare.len());
    for (l, v) in &by_level { println!("    level {}: {} calls", l, v.len()); for s in v { println!("      {}", s); } }
    if !bare.is_empty() { println!("    bare pixels: {}", bare.join(" ")); }
    // the relations the board holds between what stands on the apex
    let mut seen = std::collections::BTreeSet::new();
    let mut lines = Vec::new();
    for &i in &apex_refs {
        let a = &fr.acts[i];
        for br in &a.rels {
            let Some(b) = m.act(br.other) else { continue };
            if !b.on_apex() || b.kind == Kind::Action { continue; }
            let r = brain_core::function::canon(a.sym, b.sym, br.off);
            let key = (act_name(m, a), act_name(m, b));
            if !seen.insert(key.clone()) { continue; }
            lines.push(format!("      {} – {} : {} | as held: {} | returned at strength {} | held by {}", key.0, key.1, rel_words(m, &r), rel_text(m, &r), br.strength,
                m.dict.holder(&r).map_or("no function".into(), |f| format!("f{}", f))));
        }
    }
    println!("    relations between what stands on the apex: {}", lines.len());
    for l in lines { println!("{}", l); }
}

/// The top call on the apex: the one standing for the most pixels.
fn top_call(m: &Machine) -> Option<(String, usize, usize)> {
    let fr = m.frames.back()?;
    fr.acts.iter().filter(|a| a.on_apex() && a.kind == Kind::Inter)
        .map(|a| (act_name(m, a), a.fired_by.as_ref().map_or(usize::MAX, |f| f.func as usize), pixels(m, a).len()))
        .max_by_key(|x| x.2)
}

fn func_words(m: &Machine, fid: usize) -> String {
    match m.dict.funcs.get(&(fid as _)) {
        None => format!("f{} is no longer in the dictionary", fid),
        Some(f) => format!("f{} ({} relations{}): {}", fid, f.rels.len(), if f.has_cycle() { ", a cycle" } else { "" },
            f.rels.iter().map(|r| format!("{} [{}]", rel_words(m, r), rel_text(m, r))).collect::<Vec<_>>().join("; ")),
    }
}

/// One round with every frame traced: the 2's top function after each frame of the round.
fn follow_round(m: &mut Machine, e: &Env, two: &Img, three: &Img, k: usize) -> (Option<u32>, Option<u32>) {
    m.trace_retire = true;
    let mut said = Vec::new();
    let mut fid = None;
    for (name, i, d) in [("2", two, 2u32), ("3", three, 3u32)] {
        eprintln!("=== round {} the {}: image frame {}", k, name, m.frame_no + 1);
        m.trace = true;
        let w = m.process_frame(&inputs(e, i), vec![]);
        let s = w.iter().find(|w| w.dim == e.dig).map(|w| w.bucket);
        said.push(s);
        let top = top_call(m);
        if name == "2" { fid = top.as_ref().map(|t| t.1); }
        println!("  follow, round {}: after the {} is shown (frame {}), its top call {} | the 2's top function: {}", k, name, m.frame_no,
            top.map_or("none".into(), |t| format!("{} of f{}, {} pixels", t.0, t.1, t.2)), fid.map_or("none".into(), |f| func_words(m, f)));
        eprintln!("=== round {} the {}: digit frame {}", k, name, m.frame_no + 1);
        m.process_frame(&[Input { dim: e.dig, bucket: d, pos: [0; MAX_DIMS] }], vec![Reward { reward: 1.0, channels: vec![e.digit], frames: 1 }]);
        m.trace = false;
        m.reset_context();
        println!("  follow, round {}: after the {}'s digit frame and reset | the 2's top function: {}", k, name, fid.map_or("none".into(), |f| func_words(m, f)));
    }
    m.trace_retire = false;
    (said[0], said[1])
}

fn report_dict(m: &Machine) {
    let mut fs: Vec<&Func> = m.dict.funcs.values().collect();
    fs.sort_by_key(|f| f.id);
    println!("  dictionary: {} functions", fs.len());
    for f in fs {
        println!("    f{}: origin {}{}{}", f.id, sname(m, f.origin()), if f.has_cycle() { ", a cycle" } else { "" }, f.inter.map_or(String::new(), |n| format!(", interneuron #{}", n)));
        for r in &f.rels { println!("      {}   [{}]", rel_words(m, r), rel_text(m, r)); }
    }
}

/// Show an image until the dictionary stops changing, or `n` times; the last showing traced to stderr. Returns the
/// last showing at which the dictionary changed.
fn train(m: &mut Machine, e: &Env, i: &Img, n: usize, label: &str) -> usize {
    let mut last_change = 0;
    let mut prev = fingerprint(m);
    for k in 1..=n {
        let trace = k == n || std::env::args().nth(3).map_or(false, |a| a == "trace-all");
        if trace { eprintln!("=== {} showing {}", label, k); }
        m.trace = trace;
        let w = m.process_frame(&inputs(e, i), vec![]);
        m.trace = false;
        let said = w.iter().find(|w| w.dim == e.dig).map(|w| w.bucket);
        let fp = fingerprint(m);
        let changed = fp != prev;
        if changed { last_change = k; }
        prev = fp;
        let apex = m.frames.back().unwrap().acts.iter().filter(|a| a.on_apex() && a.kind != Kind::Action).count();
        println!("  {} showing {:2}: said {} | levels {} | on the apex {} | functions {} | dictionary {}", label, k, said.map_or("-".into(), |s| s.to_string()), m.last_stats.levels, apex, m.dict.len(), if changed { "changed" } else { "same" });
        if k == n { report_apex(m); }
        m.process_frame(&[Input { dim: e.dig, bucket: 0, pos: [0; MAX_DIMS] }], vec![Reward { reward: 1.0, channels: vec![e.digit], frames: 1 }]);
        m.reset_context();
    }
    println!("  {}: the dictionary last changed at showing {}{}", label, last_change, if last_change < n { format!(" (unchanged over the last {} showings)", n - last_change) } else { " (still changing at the last showing)".into() });
    last_change
}

fn main() {
    // `--s15-off` runs the build with switch 15 off, the one choice it leaves open
    let s15_off = std::env::args().any(|a| a == "--s15-off");
    let args: Vec<String> = std::env::args().filter(|a| a != "--s15-off").collect();
    let mode = args.get(1).cloned().unwrap_or("square".into());
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    let mut m = Machine::new(2048, Policy::events_to_actions(), 1);
    if s15_off { m.sw.s15_uncovered_relations = false; }
    let ax = m.declare_axis("x", 1);
    let ay = m.declare_axis("y", 1);
    let image = m.declare_channel("image", Some(("pixel", 2)), None, &[ax, ay]);
    let digit = m.declare_channel("digit", None, Some(("digit", 10)), &[]);
    let e = Env { ax, ay, pix: m.channels[image as usize].event.unwrap(), dig: m.channels[digit as usize].action.unwrap(), digit };
    match mode.as_str() {
        "move_a" | "move_b" => {
            let sq = shape("square");
            println!("first position: the 5x5 square image");
            print_img(&sq);
            train(&mut m, &e, &sq, n, "first");
            report_dict(&m);
            let before = fingerprint(&m);
            let second = if mode == "move_a" { place(&sq, 10, 10, 4, 3, '.') } else { place(&img(&["WWW", "WWW", "WWW"]), 10, 10, 5, 4, 'B') };
            println!("second position ({}):", if mode == "move_a" { "the same 5x5 image at (4, 3) on a 10x10 canvas where nothing else reports" } else { "a 10x10 black image with the white square at (5..7, 4..6)" });
            print_img(&second);
            // the first showing there, in full
            eprintln!("=== second showing 1");
            m.trace = true;
            m.process_frame(&inputs(&e, &second), vec![]);
            m.trace = false;
            println!("  second showing 1: functions {} | dictionary {}", m.dict.len(), if fingerprint(&m) == before { "same as at the first position" } else { "changed" });
            report_apex(&m);
            m.process_frame(&[Input { dim: e.dig, bucket: 0, pos: [0; MAX_DIMS] }], vec![Reward { reward: 1.0, channels: vec![e.digit], frames: 1 }]);
            m.reset_context();
            train(&mut m, &e, &second, n, "second");
            let after = fingerprint(&m);
            let added: Vec<&String> = after.iter().filter(|x| !before.contains(x)).collect();
            let gone: Vec<&String> = before.iter().filter(|x| !after.contains(x)).collect();
            println!("  against the dictionary learned at the first position: {} functions added or changed, {} gone or changed", added.len(), gone.len());
            for x in added { println!("    now: {}", x); }
            for x in gone { println!("    was: {}", x); }
            report_dict(&m);
        }
        "teach23" => {
            let (two, three) = (shape("two"), shape("three"));
            println!("the 2:"); print_img(&two);
            println!("the 3:"); print_img(&three);
            let mut prev = fingerprint(&m);
            let mut last_change = 0;
            // the round whose showings are traced, and the 2's top function followed through it (argument 3)
            let follow: usize = std::env::args().nth(3).and_then(|a| a.parse().ok()).unwrap_or(0);
            for k in 1..=n {
                let (s2, s3) = if k == follow { follow_round(&mut m, &e, &two, &three, k) } else {
                    let all = std::env::args().nth(4).map_or(false, |a| a == "all");
                    if all { eprintln!("=== round {} the 2: image frame {}", k, m.frame_no + 1); }
                    let s2 = show(&mut m, &e, &two, Some(2), all);
                    m.reset_context();
                    if all { eprintln!("=== round {} the 3: image frame {}", k, m.frame_no + 1); }
                    let s3 = show(&mut m, &e, &three, Some(3), all);
                    m.reset_context();
                    (s2, s3)
                };
                let fp = fingerprint(&m);
                let changed = fp != prev;
                if changed { last_change = k; }
                prev = fp;
                let edges: usize = m.dict.funcs.values().map(|f| f.rels.len()).sum();
                println!("  round {:2}: the 2 said {} | the 3 said {} | functions {} edges {} | dictionary {}", k, s2.map_or("-".into(), |s| s.to_string()), s3.map_or("-".into(), |s| s.to_string()), m.dict.len(), edges, if changed { "changed" } else { "same" });
            }
            println!("  the dictionary last changed at round {}", last_change);
            report_dict(&m);
            m.frozen = true;
            for (name, i) in [("2", &two), ("3", &three)] {
                eprintln!("=== test {}", name);
                m.trace = true;
                let w = m.process_frame(&inputs(&e, i), vec![]);
                m.trace = false;
                let said = w.iter().find(|w| w.dim == e.dig).map(|w| w.bucket);
                println!("  test, the {} without its digit: says {} | voters {}", name, said.map_or("nothing".into(), |s| s.to_string()), m.last_stats.voters);
                report_apex(&m);
                m.process_frame(&said.map(|b| Input { dim: e.dig, bucket: b, pos: [0; MAX_DIMS] }).into_iter().collect::<Vec<_>>(), vec![]);
                m.reset_context();
            }
        }
        s => {
            let i = shape(s);
            println!("{}:", s);
            print_img(&i);
            train(&mut m, &e, &i, n, s);
            report_dict(&m);
        }
    }
    println!("{}", m.middle_summary());
}
