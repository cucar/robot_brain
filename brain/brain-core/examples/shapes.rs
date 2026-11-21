/// Simple images, each shown again and again to a fresh machine, as in the square example: the image, then the
/// digit 0 with a reward of +1, then the context is reset. Modes: hline, vline, square, hollow; teach23 (the 2 and the
/// 3 in turn, each taught with its digit, then each shown without it to a copy loaded with learning off).
/// usage: shapes <mode> [showings=30] [--regions=R] [--columns=C]
use brain_core::board::Act;
use brain_core::brain::Brain;
use brain_core::dictionary::{canon, Func, Rel};
use brain_core::types::*;

type Img = Vec<Vec<char>>; // 'W' white (bucket 1), 'B' black (bucket 0), '.' reports nothing

fn make_img(rows: &[&str]) -> Img { rows.iter().map(|r| r.chars().collect()).collect() }

fn make_shape(mode: &str) -> Img {
    match mode {
        "hline" => make_img(&["BBBBBBB", "BBBBBBB", "BWWWWWB", "BBBBBBB", "BBBBBBB"]),
        "vline" => make_img(&["BBBBB", "BBBBB", "BBBBB", "BBBBB", "BBBBB", "BBBBB", "BBBBB"]).iter().enumerate().map(|(y, r)| r.iter().enumerate().map(|(x, c)| if x == 2 && (1..=5).contains(&y) { 'W' } else { *c }).collect()).collect(),
        "square" => make_img(&["BBBBB", "BWWWB", "BWWWB", "BWWWB", "BBBBB"]),
        "hollow" => make_img(&["BBBBB", "B...B", "B...B", "B...B", "BBBBB"]),
        "two" => make_img(&["BBBBBBB", "BWWWWWB", "BBBBBWB", "BBBBBWB", "BWWWWWB", "BWBBBBB", "BWBBBBB", "BWWWWWB", "BBBBBBB"]),
        "three" => make_img(&["BBBBBBB", "BWWWWWB", "BBBBBWB", "BBBBBWB", "BWWWWWB", "BBBBBWB", "BBBBBWB", "BWWWWWB", "BBBBBBB"]),
        _ => panic!("unknown shape"),
    }
}

struct Env { ax: usize, ay: usize, pix: DimensionId, dig: DimensionId, digit: ChannelId }

fn make_inputs(e: &Env, i: &Img) -> Vec<Input> {
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

/// A symbol in words: a pixel's colour, or a call with its function and phase.
fn name_sym(m: &Brain, s: u32) -> String {
    let dict = m.get_dictionary();
    match dict.find_pose(s) {
        None => match m.get_registry()[s].bucket { 1 => "white".into(), _ => "black".into() },
        Some(p) => {
            let n = dict.find_neuron_of(s);
            let fs: Vec<String> = dict.list_of_inter(n).iter().map(|f| format!("f{}", f)).collect();
            let end = |x: u32| if dict.find_pose(x).is_some() || m.get_registry()[dict.find_neuron_of(x)].kind == Kind::Inter { dict.describe_sym(x) } else { name_sym(m, x) };
            format!("the call #{} of {} (from {} to {}, {} laps)", n, if fs.is_empty() { "no function now".into() } else { fs.join(",") }, end(p.from), end(p.to), p.laps)
        }
    }
}

fn describe_offset(o: &[i32]) -> String { if o[0] == 0 { format!("({:+}, {:+})", o[1], o[2]) } else { format!("(t {:+}, {:+}, {:+})", o[0], o[1], o[2]) } }

fn describe_rel_words(m: &Brain, r: &Rel) -> String { format!("{} sees {} at {}", name_sym(m, r.a), name_sym(m, r.b), describe_offset(&r.off)) }

fn describe_rel(m: &Brain, r: &Rel) -> String { let d = m.get_dictionary(); format!("{}→{}{:?}", d.describe_sym(r.a), d.describe_sym(r.b), &r.off[..]) }

fn read_fingerprint(m: &Brain) -> Vec<String> { let d = m.get_dictionary(); let mut v: Vec<String> = d.funcs.values().map(|f| d.describe(f)).collect(); v.sort(); v }

fn name_act(m: &Brain, a: &Act) -> String {
    if a.kind == Kind::Inter { format!("{}@{},{}", m.get_dictionary().describe_sym(a.sym), a.pos[1], a.pos[2]) }
    else { format!("{}@{},{}", if m.get_registry()[a.neuron].bucket == 1 { "W" } else { "B" }, a.pos[1], a.pos[2]) }
}

/// The pixels an activation stands for, followed down.
fn list_pixels(m: &Brain, a: &Act) -> Vec<(i32, i32)> {
    let mut px = Vec::new();
    let mut stack = a.covers.clone();
    if a.kind == Kind::Event { px.push((a.pos[1], a.pos[2])); }
    while let Some(r) = stack.pop() {
        let Some(c) = m.find_act(r) else { continue };
        if c.kind == Kind::Event { px.push((c.pos[1], c.pos[2])); }
        stack.extend(c.covers.iter().copied());
    }
    px.sort();
    px.dedup();
    px
}

/// One showing: the image, then the digit `d` with its reward; the context reset after. Returns what the machine
/// said.
fn show(m: &mut Brain, e: &Env, i: &Img, d: Option<u32>) -> Option<u32> {
    let w = m.process_frame(&make_inputs(e, i), vec![]);
    let said = w.iter().find(|w| w.dim == e.dig).map(|w| w.bucket);
    let ran = d.or(said);
    let rewards = if d.is_some() { vec![Reward { reward: 1.0, channels: vec![e.digit], frames: 1 }] } else { vec![] };
    m.process_frame(&ran.map(|b| Input { dim: e.dig, bucket: b, pos: [0; MAX_DIMS] }).into_iter().collect::<Vec<_>>(), rewards);
    said
}

/// What stands on the apex at the end of a showing, level by level, the relations among what stands, and the bare
/// pixels. Called right after the image frame, before the digit frame.
fn report_apex(m: &Brain) {
    let fno = m.frame_no;
    let dict = m.get_dictionary();
    let Some(fr) = m.get_board().iter().find(|f| f.no == fno) else { return };
    let mut by_level: std::collections::BTreeMap<u32, Vec<String>> = Default::default();
    let mut bare = Vec::new();
    let mut apex_refs = Vec::new();
    for (i, a) in fr.acts.iter().enumerate() {
        if !a.on_apex() || a.kind == Kind::Action { continue; }
        apex_refs.push(i);
        if a.kind == Kind::Event { bare.push(name_act(m, a)); continue; }
        let px = list_pixels(m, a);
        by_level.entry(a.level).or_default().push(format!("{} at ({}, {}) | {} | f{} | covers [{}] | stands for {} pixels",
            dict.describe_sym(a.sym), a.pos[1], a.pos[2], name_sym(m, a.sym), a.fired_by.as_ref().map_or(0, |f| f.func),
            a.covers.iter().filter_map(|r| m.find_act(*r)).map(|c| name_act(m, c)).collect::<Vec<_>>().join(" "), px.len()));
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
            let Some(b) = m.find_act(br.other) else { continue };
            if !b.on_apex() || b.kind == Kind::Action { continue; }
            let r = canon(a.sym, b.sym, br.off);
            let key = (name_act(m, a), name_act(m, b));
            if !seen.insert(key.clone()) { continue; }
            lines.push(format!("      {} – {} : {} | as held: {} | returned at strength {} | held by {}", key.0, key.1, describe_rel_words(m, &r), describe_rel(m, &r), br.strength,
                dict.find_holder(&r).map_or("no function".into(), |f| format!("f{}", f))));
        }
    }
    println!("    relations between what stands on the apex: {}", lines.len());
    for l in lines { println!("{}", l); }
}

fn report_dict(m: &Brain) {
    let mut fs: Vec<&Func> = m.get_dictionary().funcs.values().collect();
    fs.sort_by_key(|f| f.id);
    println!("  dictionary: {} functions", fs.len());
    for f in fs {
        println!("    f{}: {}{}{}", f.id, f.find_origin().map_or("a cycle, no origin".to_string(), |o| format!("origin {}", name_sym(m, o))), "", f.inter.map_or(String::new(), |n| format!(", interneuron #{}", n)));
        for r in &f.rels { println!("      {}   [{}]", describe_rel_words(m, r), describe_rel(m, r)); }
    }
}

/// Show an image until the dictionary stops changing, or `n` times. Returns the last showing at which the dictionary
/// changed.
fn train(m: &mut Brain, e: &Env, i: &Img, n: usize, label: &str) -> usize {
    let mut last_change = 0;
    let mut prev = read_fingerprint(m);
    for k in 1..=n {
        let w = m.process_frame(&make_inputs(e, i), vec![]);
        let said = w.iter().find(|w| w.dim == e.dig).map(|w| w.bucket);
        let fp = read_fingerprint(m);
        let changed = fp != prev;
        if changed { last_change = k; }
        prev = fp;
        let apex = m.get_board().back().unwrap().acts.iter().filter(|a| a.on_apex() && a.kind != Kind::Action).count();
        println!("  {} showing {:2}: said {} | levels {} | on the apex {} | functions {} | dictionary {}", label, k, said.map_or("-".into(), |s| s.to_string()), m.last_stats.levels, apex, m.get_dictionary().len(), if changed { "changed" } else { "same" });
        if k == n { report_apex(m); }
        m.process_frame(&[Input { dim: e.dig, bucket: 0, pos: [0; MAX_DIMS] }], vec![Reward { reward: 1.0, channels: vec![e.digit], frames: 1 }]);
        m.reset_context();
    }
    println!("NEVERCALLED {} {} of {}", label, m.count_functions_never_called(), m.get_dictionary().len());
    println!("  {}: the dictionary last changed at showing {}{}", label, last_change, if last_change < n { format!(" (unchanged over the last {} showings)", n - last_change) } else { " (still changing at the last showing)".into() });
    last_change
}

fn main() {
    // `--regions=R --columns=C` lay out the dispatch, one region of one column by default
    let flag = |name: &str| std::env::args().find_map(|a| a.strip_prefix(name).and_then(|v| v.parse::<usize>().ok())).unwrap_or(1);
    let (regions, columns) = (flag("--regions="), flag("--columns="));
    let args: Vec<String> = std::env::args().filter(|a| !a.starts_with("--")).collect();
    let mode = args.get(1).cloned().unwrap_or("square".into());
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    let mut m = Brain::new(2048, Policy::events_to_actions(), regions, columns, true);
    let ax = m.declare_axis("x", 1);
    let ay = m.declare_axis("y", 1);
    let image = m.declare_channel("image", Some(("pixel", 2)), None, &[ax, ay]);
    let digit = m.declare_channel("digit", None, Some(("digit", 10)), &[]);
    let e = Env { ax, ay, pix: m.channels[image as usize].event.unwrap(), dig: m.channels[digit as usize].action.unwrap(), digit };
    match mode.as_str() {
        "teach23" => {
            let (two, three) = (make_shape("two"), make_shape("three"));
            println!("the 2:"); print_img(&two);
            println!("the 3:"); print_img(&three);
            let mut prev = read_fingerprint(&m);
            let mut last_change = 0;
            for k in 1..=n {
                let s2 = show(&mut m, &e, &two, Some(2));
                m.reset_context();
                let s3 = show(&mut m, &e, &three, Some(3));
                m.reset_context();
                let fp = read_fingerprint(&m);
                let changed = fp != prev;
                if changed { last_change = k; }
                prev = fp;
                let edges: usize = m.get_dictionary().funcs.values().map(|f| f.rels.len()).sum();
                println!("  round {:2}: the 2 said {} | the 3 said {} | functions {} edges {} | dictionary {}", k, s2.map_or("-".into(), |s| s.to_string()), s3.map_or("-".into(), |s| s.to_string()), m.get_dictionary().len(), edges, if changed { "changed" } else { "same" });
            }
            println!("  the dictionary last changed at round {}", last_change);
            report_dict(&m);
            // the test runs on a copy with learning off: the brain and its context saved, and loaded into a new brain
            let job = std::path::PathBuf::from(format!("{}/../target/shapes-teach23-{}", env!("CARGO_MANIFEST_DIR"), std::process::id()));
            m.save(&job, "trained").unwrap();
            m.save_context(&job, "trained").unwrap();
            let mut t = Brain::new(2048, Policy::events_to_actions(), regions, columns, false);
            t.load(&job, "trained").unwrap();
            t.load_context(&job, "trained").unwrap();
            std::fs::remove_dir_all(&job).unwrap();
            for (name, i) in [("2", &two), ("3", &three)] {
                let w = t.process_frame(&make_inputs(&e, i), vec![]);
                let said = w.iter().find(|w| w.dim == e.dig).map(|w| w.bucket);
                println!("  test, the {} without its digit: says {} | voters {}", name, said.map_or("nothing".into(), |s| s.to_string()), t.last_stats.voters);
                report_apex(&t);
                t.process_frame(&said.map(|b| Input { dim: e.dig, bucket: b, pos: [0; MAX_DIMS] }).into_iter().collect::<Vec<_>>(), vec![]);
                t.reset_context();
            }
        }
        s => {
            let i = make_shape(s);
            println!("{}:", s);
            print_img(&i);
            train(&mut m, &e, &i, n, s);
            report_dict(&m);
        }
    }
}
