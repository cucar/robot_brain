//! MNIST (docs/algorithm-mnist.md): an image channel over x, y and time with one event dimension of two buckets,
//! white and black, every pixel firing one of them in every frame; a digit channel over time alone with ten digit
//! actions. Two frames per image: the image, then the digit that ran and its reward, +1 where it named the image's
//! digit and −1 where not, scoped to the digit channel and that one frame (R29, D35).
//! The world names the digit itself for the first `teach` training images (a lesson), and wherever the machine is
//! silent; after that what the machine says runs. Then `test` held-out images with the machine frozen: its answer
//! runs, with no reward and no lesson, and nothing of the machine changes.
//! The data files are the raw (gunzipped) MNIST idx files in `data-dir`, taken in the file's order: the first
//! `train` images, or with `per-class` the first that many of each digit among them, shown `episodes` times over.
//! usage: cargo run --release --example mnist -- <data-dir> [train=all] [test=500] [h=2048] [size=14] [teach=all]
//!        [threshold=128] [trace-image=0] [per-class=all] [episodes=1]
use brain_core::machine::Machine;
use brain_core::types::*;
use std::time::Instant;

fn read(path: &str) -> Vec<u8> { std::fs::read(path).unwrap_or_else(|e| panic!("{}: {}", path, e)) }

fn images(path: &str) -> Vec<Vec<u8>> {
    let b = read(path);
    let n = u32::from_be_bytes([b[4], b[5], b[6], b[7]]) as usize;
    (0..n).map(|i| b[16 + i * 784..16 + (i + 1) * 784].to_vec()).collect()
}

fn labels(path: &str) -> Vec<u32> { read(path)[8..].iter().map(|&x| x as u32).collect() }

fn main() {
    // `--s15-off` runs the build with switch 15 off, the one choice it leaves open
    let s15_off = std::env::args().any(|a| a == "--s15-off");
    let args: Vec<String> = std::env::args().filter(|a| a != "--s15-off" && !a.starts_with("--closing=") && a != "--trace-closing" && a != "--list-cycles" && !a.starts_with("--stop-settled=") && a != "--landed" && a != "--cycles" && !a.starts_with("--only=") && !a.starts_with("--changes-from=") && a != "--apex-all" && !a.starts_with("--test-per-class=") && !a.starts_with("--watch=") && !a.starts_with("--neurons=")).collect();
    // --only=i: train on image i of the file alone
    let only: Option<usize> = std::env::args().find_map(|a| a.strip_prefix("--only=").map(|s| s.parse().unwrap()));
    // --changes-from=k: report every change to the dictionary from episode k on (default: the last 5 episodes)
    // --watch=TEXT: from the episode --changes-from names, every relation whose text holds TEXT, after every image
    let watch: Option<String> = std::env::args().find_map(|a| a.strip_prefix("--watch=").map(|s| s.to_string()));
    let changes_from: Option<usize> = std::env::args().find_map(|a| a.strip_prefix("--changes-from=").map(|s| s.parse().unwrap()));
    // test only, --stop-settled=k: stop once k episodes in a row leave the dictionary as it was, and report it
    let stop_settled: Option<usize> = std::env::args().find_map(|a| a.strip_prefix("--stop-settled=").map(|s| s.parse().unwrap()));
    let mut same_run = 0usize;
    // --apex-all: the apex of every showing of a single image; --neurons=a,b: those neurons after every showing
    let apex_all = std::env::args().any(|a| a == "--apex-all");
    let show_neurons: Vec<usize> = std::env::args().find_map(|a| a.strip_prefix("--neurons=").map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())).unwrap_or_default();
    let dir = args.get(1).expect("usage: mnist <data-dir> [train] [test] [h] [size] [teach] [threshold]").clone();
    let num = |i: usize| -> Option<usize> { args.get(i).and_then(|s| if s == "all" { None } else { s.parse().ok() }) };
    let train_imgs = images(&format!("{}/train-images-idx3-ubyte", dir));
    let train_lbls = labels(&format!("{}/train-labels-idx1-ubyte", dir));
    let test_imgs = images(&format!("{}/t10k-images-idx3-ubyte", dir));
    let test_lbls = labels(&format!("{}/t10k-labels-idx1-ubyte", dir));
    let train_n = num(2).unwrap_or(train_imgs.len()).min(train_imgs.len());
    // --test-per-class=k: the test is the first k test images of each digit, in the file's order, in place of the
    // first [test] images
    let test_per_class: Option<usize> = std::env::args().find_map(|a| a.strip_prefix("--test-per-class=").map(|s| s.parse().unwrap()));
    let test_order: Vec<usize> = match test_per_class {
        Some(k) => { let mut got = [0usize; 10]; (0..test_imgs.len()).filter(|&i| { let d = test_lbls[i] as usize; if got[d] < k { got[d] += 1; true } else { false } }).collect() }
        None => (0..num(3).unwrap_or(500).min(test_imgs.len())).collect(),
    };
    let test_n = test_order.len();
    let h = num(4).unwrap_or(2048);
    let size = num(5).unwrap_or(14);
    let teach_arg = num(6);
    let threshold = num(7).unwrap_or(128) as u32;
    // the one training presentation (counted from 1, over every episode) to trace, or none
    let trace_n = num(8).unwrap_or(0);
    let per_class = num(9);
    let episodes = num(10).unwrap_or(1).max(1);
    // the images shown each episode, in the file's order
    let mut order: Vec<usize> = Vec::new();
    let mut picked = [0usize; 10];
    for i in 0..train_n {
        let d = train_lbls[i] as usize;
        if per_class.map_or(true, |k| picked[d] < k) { picked[d] += 1; order.push(i); }
    }
    if let Some(i) = only { order = vec![i]; }
    // what the dictionary holds, function by function, for seeing what changes
    let dict_map = |m: &Machine| -> std::collections::BTreeMap<u64, String> { m.dict.funcs.iter().map(|(id, f)| (*id as u64, m.dict.describe(f))).collect() };
    let mut ep_dict = dict_map(&Machine::new(1, Policy::events_to_actions(), 1));
    let shown = order.len() * episodes;
    assert!(28 % size == 0, "size must divide 28");
    let f = 28 / size;

    let mut m = Machine::new(h, Policy::events_to_actions(), 1);
    m.print_cycles = std::env::args().any(|a| a == "--cycles");
    m.sw.closing = std::env::args().find_map(|a| a.strip_prefix("--closing=").map(|v| match v { "cycle" => 0, "round" => 1, _ => panic!("--closing=cycle|round") })).unwrap_or(0);
    m.trace_closing = std::env::args().any(|a| a == "--trace-closing");
    // test only, --list-cycles: record each function's calls (the join log only records) and list the cycles at the end
    let list_cycles = std::env::args().any(|a| a == "--list-cycles");
    if list_cycles { m.join_log = true; }

    if s15_off { m.sw.s15_uncovered_relations = false; }
    m.trace_retire = true;
    let ax = m.declare_axis("x", 1);
    let ay = m.declare_axis("y", 1);
    let image = m.declare_channel("image", Some(("pixel", 2)), None, &[ax, ay]);
    let digit = m.declare_channel("digit", None, Some(("digit", 10)), &[]);
    let pix = m.channels[image as usize].event.unwrap();
    let dig = m.channels[digit as usize].action.unwrap();
    let teach = teach_arg.unwrap_or(usize::MAX);
    println!("mnist: {}x{}, black and white pixels, {} training images x {} episodes ({} taught), test {}, H={}", size, size, order.len(), episodes,
        if teach >= shown { "all".to_string() } else { format!("first {}", teach) }, test_n, h);

    // the pixels of an image: 1 (black, ink) where the block's mean reaches the threshold, 0 (white) elsewhere
    let frame_of = |px: &[u8]| -> Vec<Input> {
        let mut v = Vec::with_capacity(size * size);
        for y in 0..size {
            for x in 0..size {
                let mut s = 0u32;
                for dy in 0..f { for dx in 0..f { s += px[(y * f + dy) * 28 + x * f + dx] as u32; } }
                let ink = if s / (f * f) as u32 >= threshold { 1 } else { 0 };
                let mut pos = [0i32; MAX_DIMS];
                pos[ax] = x as i32;
                pos[ay] = y as i32;
                v.push(Input { dim: pix, bucket: ink, pos });
            }
        }
        v
    };

    // two frames per image: the image, then the digit that ran with what it earned (R29); the context is reset
    // after each, so no image sees another
    let run = |m: &mut Machine, px: &[u8]| -> (Option<u32>, f64) {
        let t = Instant::now();
        let w = m.process_frame(&frame_of(px), vec![]);
        (w.iter().find(|w| w.dim == dig).map(|w| w.bucket), t.elapsed().as_secs_f64() * 1000.0)
    };
    let answer = |m: &mut Machine, d: Option<u32>, reward: Option<f64>| -> f64 {
        let t = Instant::now();
        let inputs: Vec<Input> = d.map(|d| Input { dim: dig, bucket: d, pos: [0; MAX_DIMS] }).into_iter().collect();
        let rewards = reward.map(|r| Reward { reward: r, channels: vec![digit], frames: 1 }).into_iter().collect();
        m.process_frame(&inputs, rewards);
        t.elapsed().as_secs_f64() * 1000.0
    };

    let t0 = Instant::now();
    let (mut spoke, mut right, mut lessons, mut ms, mut apex) = (0usize, 0usize, 0usize, 0.0f64, 0usize);
    let (mut w_spoke, mut w_right, mut w_ms) = (0usize, 0usize, 0.0f64);
    let (mut w_added, mut w_retired) = (0usize, 0usize);
    let mut k = 0usize;
    let mut last_apex: Vec<String> = Vec::new();
    for ep in 0..episodes {
        let (mut e_right, mut e_spoke, mut e_levels, mut e_ms) = (0usize, 0usize, 0usize, 0.0f64);
        let mut said_list = Vec::new();
        let (mut e_added, mut e_retired) = (0usize, 0usize);
        let mut e_apex = 0usize;
        let mut apex_note = String::new();
        for &i in &order {
            k += 1;
            let label = train_lbls[i];
            let tracing = trace_n > 0 && k == trace_n;
            m.trace_levels = tracing;
            if tracing { m.stop_after_level = None; }
            m.trace = tracing;
            m.trace_cover = tracing;
            let before = if changes_from.map_or(ep + 5 >= episodes, |c| ep + 1 >= c) { Some((dict_map(&m), m.joins_by, m.moves_by)) } else { None };
            let (said, t) = run(&mut m, &train_imgs[i]);
            if let Some(fr) = m.frames.back() { e_apex += fr.acts.iter().filter(|a| a.on_apex() && a.kind == Kind::Inter).count(); }
            if ep + 1 == episodes {
                let fr = m.frames.back().unwrap();
                let calls = fr.acts.iter().filter(|a| a.on_apex() && a.kind == Kind::Inter).count();
                let bare = fr.acts.iter().filter(|a| a.on_apex() && a.kind == Kind::Event).count();
                println!("LASTEP image {} (digit {}) | calls on the apex {} | bare pixels on the apex {} | levels {}", i, train_lbls[i], calls, bare, m.last_stats.levels);
            }
            w_added += m.last_stats.added;
            e_added += m.last_stats.added;
            e_retired += m.last_stats.retired;
            w_retired += m.last_stats.retired;
            ms += t;
            e_ms += t;
            e_levels = e_levels.max(m.last_stats.levels);
            if tracing { level0_view(&m, &train_imgs[i], size); }
            if tracing { eprintln!("[image {}] {:.0} ms, levels {}, apex {}, neurons {}, functions {}", k, t, m.last_stats.levels, m.last_stats.apex_acts, m.alive, m.dict.len()); }
            apex += m.last_stats.apex_acts;
            if order.len() == 1 {
                // what stands on the apex after this image: each neuron with its place, and whether it is what stood last time
                let fr = m.frames.back().unwrap();
                let mut ap: Vec<String> = fr.acts.iter().filter(|a| a.on_apex() && a.kind != Kind::Action)
                    .map(|a| if a.kind == Kind::Inter { format!("#{}@{},{}", a.neuron, a.pos[1], a.pos[2]) } else { format!("{}@{},{}", if m.thalamus[a.neuron].bucket == 1 { "B" } else { "W" }, a.pos[1], a.pos[2]) }).collect();
                ap.sort();
                let kept = ap.iter().filter(|x| last_apex.contains(x)).count();
                apex_note = format!(" | apex {} ({} kept, {} new, {} gone) levels {} added {}", ap.len(), kept, ap.len() - kept, last_apex.len() - kept, m.last_stats.levels, m.last_stats.added);
                if ep + 1 == episodes || ep < 3 || apex_all { apex_note.push_str(&format!(" [{}]", ap.join(" "))); }
                last_apex = ap;
            }
            said_list.push(format!("{}:{}", label, said.map_or("-".to_string(), |s| s.to_string())));
            if let Some(s) = said { spoke += 1; w_spoke += 1; e_spoke += 1; if s == label { right += 1; w_right += 1; e_right += 1; } }
            let lesson = k <= teach || said.is_none();
            if lesson { lessons += 1; }
            let ran = if lesson { label } else { said.unwrap() };
            ms += answer(&mut m, Some(ran), Some(if ran == label { 1.0 } else { -1.0 }));
            if let (Some(pat), Some(c)) = (watch.as_ref(), changes_from) { if ep + 1 >= c { for l in m.watch_lines(pat) { println!("WATCH episode {} image {} | {}", ep + 1, i, l); } } }
            if let Some((b, j0, v0)) = before {
                let a = dict_map(&m);
                let started: Vec<String> = a.iter().filter(|(k, _)| !b.contains_key(k)).map(|(k, d)| format!("f{} {}", k, d)).collect();
                let left: Vec<String> = b.iter().filter(|(k, _)| !a.contains_key(k)).map(|(k, d)| format!("f{} {}", k, d)).collect();
                let changed: Vec<String> = a.iter().filter(|(k, d)| b.get(k).map_or(false, |x| x != *d)).map(|(k, d)| format!("f{} {}", k, d)).collect();
                let (jn, mv) = ((m.joins_by[0] + m.joins_by[1]) - (j0[0] + j0[1]), (m.moves_by[0] + m.moves_by[1]) - (v0[0] + v0[1]));
                if !started.is_empty() || !left.is_empty() || !changed.is_empty() || jn > 0 || mv > 0 {
                    println!("CHANGE episode {} image {} | started {} | left {} | changed {} | joins {} | moves {}", ep + 1, i, started.len(), left.len(), changed.len(), jn, mv);
                    for s in &started { println!("    + {}", s); }
                    for s in &left { println!("    - {}", s); }
                    for s in &changed { println!("    ~ {}", s); }
                }
            }
            for &n in &show_neurons { if n < m.thalamus.len() { let nn = &m.thalamus[n as u32]; let (s, e): (f64, f64) = nn.connections.values().fold((0.0, 0.0), |(s, e), c| (s + c.strength, e + c.earnings)); println!("NEURON showing {} #{} alive {} | history rows {} | connections {} (exposures {:.0}, earnings {:.2}) | functions {:?}", k, n, nn.alive, nn.history.len(), nn.connections.len(), s, e, m.dict.of_inter(n as u32)); } }
            // each image stands alone: nothing of it is on the board when the next is shown
            m.reset_context();
            if k % 100 == 0 {
                eprintln!("  image {}/{} | last 100: spoke {} right {} ({:.1}% of spoken) {:.0} ms/image, functions added {} retired {} | neurons {} functions {} | apex/image {:.1} | {:.1} ms/image so far",
                    k, shown, w_spoke, w_right, 100.0 * w_right as f64 / w_spoke.max(1) as f64, (ms - w_ms) / 100.0, w_added, w_retired, m.alive, m.dict.len(), apex as f64 / k as f64, ms / k as f64);
                w_added = 0;
                w_retired = 0;
                w_spoke = 0;
                w_right = 0;
                w_ms = ms;
            }
        }
        let now = dict_map(&m);
        let dict_changed = now != ep_dict;
        ep_dict = now;
        println!("EPISODE {} | dictionary {} | functions {} | right {}/{} | calls on the apex per image {:.1}", ep + 1, if dict_changed { "changed" } else { "same" }, m.dict.len(), e_right, order.len(), e_apex as f64 / order.len().max(1) as f64);
        if dict_changed { same_run = 0; } else { same_run += 1; }
        if let Some(kk) = stop_settled { if same_run >= kk { println!("SETTLED after episode {} | the last change at episode {}", ep + 1, ep + 1 - kk); break; } }
        if episodes > 1 {
            println!("  episode {:2}: added {} retired {} | right {}/{} (spoke {}) | max levels {} | neurons {} functions {} | {:.0} ms/image | digit:said {}{}",
                ep + 1, e_added, e_retired, e_right, order.len(), e_spoke, e_levels, m.alive, m.dict.len(), e_ms / order.len().max(1) as f64, said_list.join(" "), apex_note);
        }
    }
    println!("NEVERCALLED {} of {}", m.funcs_never_called(), m.dict.len());
    println!("CLOSING weighed {} joined {} (from a function of more relations {}) | cycles at the end {} of {} | build joins the calls count would refuse {} | largest such gain as a cycle {} | weighs beyond every offset the rows hold {} | weighs with neither end where the call stands {} | weighs where one lap on cannot be worked out {}", m.closing_weighed, m.closing_joined, m.closing_from_multi, m.dict.funcs.values().filter(|f| f.has_cycle()).count(), m.dict.len(), m.closing_refused[0], m.closing_max.map_or("-".to_string(), |(c, lo, _)| format!("{:.1} (on the calls {:.1})", c, lo)), m.closing_beyond, m.closing_no_end, m.closing_no_lap);
    println!("UNPAID {} relations left a function because they did not pay there", m.left_unpaid);
    println!("LIFTED TO THE STANDING CALL {} relations went to other calls than their direct coverers", m.lift_redirected);
    println!("LIFT PASSED OVER {} dropped weak calls, {} calls aged off the board", m.lift_passed_gone, m.lift_passed_aged);
    { let (t, s) = m.twice_counts(); println!("TWICE frames with an interneuron called more than once {} | of those, a row holds another call of the same interneuron {}", t, s); }
    for l in m.cycle_lap_lines() { println!("{}", l); }
    if list_cycles {
        let mut ids: Vec<_> = m.dict.funcs.keys().copied().collect(); ids.sort();
        for id in ids {
            let f = m.dict.get(id).unwrap();
            if !f.has_cycle() { continue; }
            let calls = m.func_calls.get(&id).map_or(0, |v| v.len());
            let frames = m.func_calls.get(&id).map_or(0, |v| { let mut w = v.clone(); w.dedup(); w.len() });
            println!("CYCLEFN f{} | relations {} | nodes {} | calls {} in {} frames | {}", id, f.rels.iter().map(|r| format!("{}→{}{:?}", m.dict.sym_text(r.a), m.dict.sym_text(r.b), &r.off[..])).collect::<Vec<_>>().join("; "), f.nodes.len(), calls, frames, m.dict.describe(f));
        }
    }
    println!("JOINSMOVES joins frame {} time {} | moves frame {} time {}", m.joins_by[0], m.joins_by[1], m.moves_by[0], m.moves_by[1]);
    println!("  train: {} images | spoke {} | right {} ({:.1}% of all, {:.1}% of spoken) | lessons {} | apex/image {:.1} | neurons {} functions {} | {:.0}s",
        shown, spoke, right, 100.0 * right as f64 / shown.max(1) as f64, 100.0 * right as f64 / spoke.max(1) as f64, lessons, apex as f64 / shown.max(1) as f64, m.alive, m.dict.len(), t0.elapsed().as_secs_f64());

    // what the dictionary holds: functions by relations and nodes, and by whether their interneuron is allocated
    {
        let mut by: std::collections::BTreeMap<(usize, usize, bool, bool), usize> = Default::default();
        for f in m.dict.funcs.values() { *by.entry((f.rels.len(), f.nodes.len(), f.has_cycle(), f.inter.is_some())).or_default() += 1; }
        println!("  dictionary by (relations, nodes, cycle, has interneuron): {:?}", by);
        let mut big: Vec<&brain_core::function::Func> = m.dict.funcs.values().filter(|f| f.rels.len() > 1).collect();
        big.sort_by_key(|f| std::cmp::Reverse(f.rels.len()));
        for f in big.iter().take(12) { println!("    {}", m.dict.describe(f)); }
    }
    // test: frozen, the machine's own answer runs, unrewarded, and nothing is taught
    m.frozen = true;
    let t1 = Instant::now();
    let (mut spoke, mut right, mut apex) = (0usize, 0usize, 0usize);
    let mut confusion = [[0usize; 10]; 10];
    for (t, &i) in test_order.iter().enumerate() {
        let label = test_lbls[i];
        let (said, _) = run(&mut m, &test_imgs[i]);
        apex += m.last_stats.apex_acts;
        if let Some(s) = said { spoke += 1; confusion[label as usize][s as usize] += 1; if s == label { right += 1; } }
        answer(&mut m, said, None);
        m.reset_context();
        if (t + 1) % 100 == 0 { eprintln!("  test {}/{} | spoke {} right {} ({:.1}%)", t + 1, test_n, spoke, right, 100.0 * right as f64 / (t + 1) as f64); }
    }
    println!("  test: {} images | spoke {} | right {} ({:.1}%) | apex/image {:.1} | neurons {} functions {} | {:.0}s",
        test_n, spoke, right, 100.0 * right as f64 / test_n.max(1) as f64, apex as f64 / test_n.max(1) as f64, m.alive, m.dict.len(), t1.elapsed().as_secs_f64());
    println!("  confusion (row = digit, column = said):");
    for (d, row) in confusion.iter().enumerate() { println!("    {} {:?}", d, row); }
}

/// What stands active once the election of `level` has run: every activation of this frame fired at `level + 1` or
/// below that no call fired at `level + 1` or below covers. One letter per activation, on the pixels it stands for
/// (what it covers, followed down), lower case at its own place; each listed with its neuron, place and function.
fn active_after(m: &Machine, px: &[u8], size: usize, label: u32, level: u32) {
    use brain_core::machine::ActRef;
    let fno = m.frame_no;
    let Some(fr) = m.frames.iter().find(|f| f.no == fno) else { return };
    let f = 28 / size;
    let ink = |x: usize, y: usize| { let mut s = 0u32; for dy in 0..f { for dx in 0..f { s += px[(y * f + dy) * 28 + x * f + dx] as u32; } } s / (f * f) as u32 >= 128 };
    let top = level + 1;
    let mut grid = vec![vec!['?'; size]; size];
    let letters: Vec<char> = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@$%&".chars().collect();
    let mut rows = Vec::new();
    let mut k = 0;
    for (i, a) in fr.acts.iter().enumerate() {
        if a.kind == Kind::Action || a.gone || a.level > top { continue; }
        let hidden = a.coverers.iter().filter_map(|c| m.act(*c)).any(|c| c.pos[0] as i64 == fno && c.level <= top);
        if hidden { continue; }
        let mut stack = vec![ActRef { frame: fno, index: i as u32 }];
        let mut pix = Vec::new();
        while let Some(r) = stack.pop() {
            let Some(c) = m.act(r) else { continue };
            if c.kind == Kind::Event && c.pos[0] as i64 == fno { pix.push((c.pos[1] as usize, c.pos[2] as usize)); }
            stack.extend(c.covers.iter().copied());
        }
        let (x, y) = (a.pos[1] as usize, a.pos[2] as usize);
        if a.kind == Kind::Event {
            grid[y][x] = if m.thalamus[a.neuron].bucket == 1 { '*' } else { '-' };
            continue;
        }
        let ch = letters[k % letters.len()];
        k += 1;
        for &(px_, py_) in &pix { if px_ < size && py_ < size { grid[py_][px_] = ch; } }
        if x < size && y < size { grid[y][x] = ch.to_ascii_lowercase(); }
        let what = a.fired_by.as_ref().and_then(|fb| m.dict.get(fb.func)).map(|f| m.dict.describe(f)).unwrap_or_else(|| "(retired)".into());
        rows.push(format!("  {} = #{} at ({}, {}), level {}, {} pixels | {}", ch, a.neuron, x, y, a.level, pix.len(), what));
    }
    let bare = grid.iter().flatten().filter(|c| **c == '-' || **c == '*').count();
    eprintln!("[active] image of a {}, active after the election of level {}: {} calls, {} bare pixels (- white, * black)", label, level, rows.len(), bare);
    for y in 0..size {
        let img: String = (0..size).map(|x| if ink(x, y) { '#' } else { '.' }).collect();
        let g: String = grid[y].iter().collect();
        eprintln!("[active]   {}      {}", img, g);
    }
    for r in rows { eprintln!("[active]{}", r); }
}

/// What each call that fired at level 1 returned at level 1: where it stands, its reach, and its relations, by
/// what stands at the other end and how far.
fn level1_view(m: &Machine, size: usize) {
    let fno = m.frame_no;
    let Some(fr) = m.frames.iter().find(|f| f.no == fno) else { return };
    let name = |a: &brain_core::machine::Act| -> String {
        match a.kind {
            Kind::Inter => a.fired_by.as_ref().and_then(|fb| m.dict.get(fb.func)).map(|f| {
                let r = f.rels[0];
                let col = |n: u32| if m.thalamus[n].bucket == 1 { "B" } else { "W" };
                let dir = match (r.off[1], r.off[2]) { (1, 0) => "-", (0, 1) => "|", (1, 1) => "\\", (1, -1) => "/", _ => "?" };
                format!("{}{}{}", col(r.a), col(r.b), dir)
            }).unwrap_or("?".into()),
            _ => if m.thalamus[a.neuron].bucket == 1 { "black px".into() } else { "white px".into() },
        }
    };
    let mut total = 0;
    let mut rows = Vec::new();
    for a in fr.acts.iter().filter(|a| a.kind == Kind::Inter && a.level == 1) {
        let n = a.rels.len();
        total += n;
        let far = a.rels.iter().filter(|r| r.off[1].abs().max(r.off[2].abs()) > 1).count();
        let to_calls = a.rels.iter().filter(|r| m.act(r.other).map_or(false, |o| o.kind == Kind::Inter)).count();
        let same_place = a.rels.iter().filter(|r| m.act(r.other).map_or(false, |o| o.pos == a.pos)).count();
        rows.push((n, format!("  {:5} at ({:2},{:2}) reach x{:2} y{:2} | box x {}..{} y {}..{} | returned {:3}: to calls {:3}, to pixels {:3}, farther than 1: {:3}",
            name(a), a.pos[1], a.pos[2], a.reach[1], a.reach[2], a.lo[1], a.hi[1], a.lo[2], a.hi[2], n, to_calls, n - to_calls, far)));
        let _ = same_place;
    }
    rows.sort_by(|x, y| y.0.cmp(&x.0));
    eprintln!("[level1] {} calls at level 1 returned {} relations ({} x {} image)", rows.len(), total, size, size);
    for (_, r) in &rows { eprintln!("[level1]{}", r); }
}

/// What level 0 left on the image: the board edges between pixels, covered or bare, by direction; and how many
/// calls landed on each pixel.
fn level0_view(m: &Machine, px: &[u8], size: usize) {
    use brain_core::machine::ActRef;
    let fno = m.frame_no;
    let Some(fr) = m.frames.iter().find(|f| f.no == fno) else { return };
    let f = 28 / size;
    let ink = |x: usize, y: usize| { let mut s = 0u32; for dy in 0..f { for dx in 0..f { s += px[(y * f + dy) * 28 + x * f + dx] as u32; } } s / (f * f) as u32 >= 128 };
    let mut at = vec![vec![None; size]; size];
    for (i, a) in fr.acts.iter().enumerate() { if a.kind == Kind::Event { at[a.pos[2] as usize][a.pos[1] as usize] = Some(ActRef { frame: fno, index: i as u32 }); } }
    let covered: std::collections::HashSet<(ActRef, ActRef)> = m.covered_edges.iter().copied().collect();
    let dirs = [("along x", 1i32, 0i32), ("down y", 0, 1), ("diagonal \\", 1, 1), ("diagonal /", 1, -1)];
    let (mut total, mut bare) = (0, 0);
    for (name, dx, dy) in dirs {
        let (mut t, mut b, mut bw, mut bb, mut mixed) = (0, 0, 0, 0, 0);
        for y in 0..size as i32 { for x in 0..size as i32 {
            let (x2, y2) = (x + dx, y + dy);
            if x2 < 0 || y2 < 0 || x2 >= size as i32 || y2 >= size as i32 { continue; }
            let (p, q) = (at[y as usize][x as usize].unwrap(), at[y2 as usize][x2 as usize].unwrap());
            t += 1;
            let e = if p <= q { (p, q) } else { (q, p) };
            if !covered.contains(&e) {
                b += 1;
                match (ink(x as usize, y as usize), ink(x2 as usize, y2 as usize)) { (false, false) => bw += 1, (true, true) => bb += 1, _ => mixed += 1 }
            }
        } }
        eprintln!("[level0] {:11} edges {:3} | covered {:3} | bare {:3} (white-white {}, black-black {}, black-white {})", name, t, t - b, b, bw, bb, mixed);
        total += t; bare += b;
    }
    eprintln!("[level0] all          edges {:3} | covered {:3} | bare {:3}", total, total - bare, bare);
    // per pixel: how many calls landed on it
    let mut kinds = std::collections::BTreeMap::new();
    for y in 0..size { for x in 0..size {
        let a = m.act(at[y][x].unwrap()).unwrap();
        *kinds.entry(a.coverers.len()).or_insert(0) += 1;
    }}
    for (n, k) in &kinds { eprintln!("[level0] pixels landed on by {} calls: {}", n, k); }
}
