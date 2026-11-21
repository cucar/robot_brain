/// MNIST (docs/algorithm-mnist.md): an image channel over x, y and time with one event dimension of two buckets,
/// white and black, every pixel firing one of them in every frame; a digit channel over time alone with ten digit
/// actions. Two frames per image: the image, then the digit that ran and its reward, +1 where it named the image's
/// digit and −1 where not, scoped to the digit channel and that one frame (R29, D35).
/// The world names the digit itself for the first `teach` training images (a lesson), and wherever the machine is
/// silent; after that what the machine says runs. Then `test` held-out images on a copy of the brain loaded with
/// learning off: its answer runs, with no reward and no lesson, and nothing of it changes.
/// The data files are the raw (gunzipped) MNIST idx files in `data-dir`, taken in the file's order: the first
/// `train` images, or with `per-class` the first that many of each digit among them, shown `episodes` times over.
/// usage: cargo run --release --example mnist -- <data-dir> [train=all] [test=500] [h=2048] [size=14] [teach=all]
///        [threshold=128] [trace-image=0] [per-class=all] [episodes=1] [--regions=R] [--columns=C]
/// The trace-image argument is read by nothing.
use brain_core::brain::Brain;
use brain_core::dictionary::Rel;
use brain_core::types::*;
use std::collections::BTreeMap;
use std::time::Instant;

fn read_file(path: &str) -> Vec<u8> { std::fs::read(path).unwrap_or_else(|e| panic!("{}: {}", path, e)) }

fn read_images(path: &str) -> Vec<Vec<u8>> {
    let b = read_file(path);
    let n = u32::from_be_bytes([b[4], b[5], b[6], b[7]]) as usize;
    (0..n).map(|i| b[16 + i * 784..16 + (i + 1) * 784].to_vec()).collect()
}

fn read_labels(path: &str) -> Vec<u32> { read_file(path)[8..].iter().map(|&x| x as u32).collect() }

/// What each function was made of and how it reads, by id.
type Desc = BTreeMap<u64, (Vec<u32>, Vec<Rel>, Option<u32>, String)>;

/// Bring `desc` up to the dictionary as it stands.
fn sync_desc(m: &Brain, desc: &mut Desc) {
    let dict = m.get_dictionary();
    desc.retain(|id, _| dict.funcs.contains_key(&(*id as u32)));
    for (id, f) in &dict.funcs {
        let same = desc.get(&(*id as u64)).map_or(false, |(n, r, i, _)| *n == f.nodes && *r == f.rels && *i == f.inter);
        if !same { desc.insert(*id as u64, (f.nodes.clone(), f.rels.clone(), f.inter, dict.describe(f))); }
    }
}

/// What started, left and changed in the dictionary since `desc` was brought up to it, each as `f<id> <text>`, in
/// the order of the functions' ids; `desc` is brought up to the dictionary as it stands.
fn list_changes(m: &Brain, desc: &mut Desc) -> (Vec<String>, Vec<String>, Vec<String>) {
    let dict = m.get_dictionary();
    let mut ids: Vec<u64> = dict.funcs.keys().map(|k| *k as u64).collect();
    ids.sort_unstable();
    let (mut started, mut changed) = (Vec::new(), Vec::new());
    for id in ids {
        let f = &dict.funcs[&(id as u32)];
        match desc.get(&id) {
            None => { let d = dict.describe(f); started.push(format!("f{} {}", id, d)); desc.insert(id, (f.nodes.clone(), f.rels.clone(), f.inter, d)); }
            Some((n, r, i, old)) => {
                if *n == f.nodes && *r == f.rels && *i == f.inter { continue; }
                let d = dict.describe(f);
                if d != *old { changed.push(format!("f{} {}", id, d)); }
                desc.insert(id, (f.nodes.clone(), f.rels.clone(), f.inter, d));
            }
        }
    }
    let gone: Vec<u64> = desc.keys().copied().filter(|id| !dict.funcs.contains_key(&(*id as u32))).collect();
    let left: Vec<String> = gone.iter().map(|id| format!("f{} {}", id, desc.remove(id).unwrap().3)).collect();
    (started, left, changed)
}

/// The dictionary, function by function, for seeing whether an episode changed it.
fn map_dict(m: &Brain) -> BTreeMap<u64, String> { let d = m.get_dictionary(); d.funcs.iter().map(|(id, f)| (*id as u64, d.describe(f))).collect() }

fn main() {
    // `--regions=R --columns=C` lay out the dispatch, one region of one column by default
    let flag = |name: &str| std::env::args().find_map(|a| a.strip_prefix(name).and_then(|v| v.parse::<usize>().ok())).unwrap_or(1);
    let (regions, columns) = (flag("--regions="), flag("--columns="));
    let args: Vec<String> = std::env::args().filter(|a| !a.starts_with("--")).collect();
    let dir = args.get(1).expect("usage: mnist <data-dir> [train] [test] [h] [size] [teach] [threshold]").clone();
    let num = |i: usize| -> Option<usize> { args.get(i).and_then(|s| if s == "all" { None } else { s.parse().ok() }) };
    let train_imgs = read_images(&format!("{}/train-images-idx3-ubyte", dir));
    let train_lbls = read_labels(&format!("{}/train-labels-idx1-ubyte", dir));
    let test_imgs = read_images(&format!("{}/t10k-images-idx3-ubyte", dir));
    let test_lbls = read_labels(&format!("{}/t10k-labels-idx1-ubyte", dir));
    let train_n = num(2).unwrap_or(train_imgs.len()).min(train_imgs.len());
    let test_order: Vec<usize> = (0..num(3).unwrap_or(500).min(test_imgs.len())).collect();
    let test_n = test_order.len();
    let h = num(4).unwrap_or(2048);
    let size = num(5).unwrap_or(14);
    let teach_arg = num(6);
    let threshold = num(7).unwrap_or(128) as u32;
    let per_class = num(9);
    let episodes = num(10).unwrap_or(1).max(1);
    // the images shown each episode, in the file's order
    let mut order: Vec<usize> = Vec::new();
    let mut picked = [0usize; 10];
    for i in 0..train_n {
        let d = train_lbls[i] as usize;
        if per_class.map_or(true, |k| picked[d] < k) { picked[d] += 1; order.push(i); }
    }
    let mut ep_dict: BTreeMap<u64, String> = BTreeMap::new();
    // the dictionary as last read, function by function: what each was made of and how it reads; a function is read
    // again only where what it is made of changed
    let mut desc: Desc = Default::default();
    let shown = order.len() * episodes;
    assert!(28 % size == 0, "size must divide 28");
    let f = 28 / size;

    let mut m = Brain::new(h, Policy::events_to_actions(), regions, columns, true);
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
    let run = |m: &mut Brain, px: &[u8]| -> (Option<u32>, f64) {
        let t = Instant::now();
        let w = m.process_frame(&frame_of(px), vec![]);
        (w.iter().find(|w| w.dim == dig).map(|w| w.bucket), t.elapsed().as_secs_f64() * 1000.0)
    };
    let answer = |m: &mut Brain, d: Option<u32>, reward: Option<f64>| -> f64 {
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
            // every change to the dictionary over the last 5 episodes
            let watching = ep + 5 >= episodes;
            if watching { sync_desc(&m, &mut desc); }
            let (said, t) = run(&mut m, &train_imgs[i]);
            if let Some(fr) = m.get_board().back() { e_apex += fr.acts.iter().filter(|a| a.on_apex() && a.kind == Kind::Inter).count(); }
            if ep + 1 == episodes {
                let fr = m.get_board().back().unwrap();
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
            apex += m.last_stats.apex_acts;
            if order.len() == 1 {
                // what stands on the apex after this image: each neuron with its place, and whether it is what stood last time
                let fr = m.get_board().back().unwrap();
                let reg = m.get_registry();
                let mut ap: Vec<String> = fr.acts.iter().filter(|a| a.on_apex() && a.kind != Kind::Action)
                    .map(|a| if a.kind == Kind::Inter { format!("#{}@{},{}", a.neuron, a.pos[1], a.pos[2]) } else { format!("{}@{},{}", if reg[a.neuron].bucket == 1 { "B" } else { "W" }, a.pos[1], a.pos[2]) }).collect();
                ap.sort();
                let kept = ap.iter().filter(|x| last_apex.contains(x)).count();
                apex_note = format!(" | apex {} ({} kept, {} new, {} gone) levels {} added {}", ap.len(), kept, ap.len() - kept, last_apex.len() - kept, m.last_stats.levels, m.last_stats.added);
                if ep + 1 == episodes || ep < 3 { apex_note.push_str(&format!(" [{}]", ap.join(" "))); }
                last_apex = ap;
            }
            said_list.push(format!("{}:{}", label, said.map_or("-".to_string(), |s| s.to_string())));
            if let Some(s) = said { spoke += 1; w_spoke += 1; e_spoke += 1; if s == label { right += 1; w_right += 1; e_right += 1; } }
            let lesson = k <= teach || said.is_none();
            if lesson { lessons += 1; }
            let ran = if lesson { label } else { said.unwrap() };
            ms += answer(&mut m, Some(ran), Some(if ran == label { 1.0 } else { -1.0 }));
            if watching {
                let (started, left, changed) = list_changes(&m, &mut desc);
                if !started.is_empty() || !left.is_empty() || !changed.is_empty() {
                    println!("CHANGE episode {} image {} | started {} | left {} | changed {}", ep + 1, i, started.len(), left.len(), changed.len());
                    for s in &started { println!("    + {}", s); }
                    for s in &left { println!("    - {}", s); }
                    for s in &changed { println!("    ~ {}", s); }
                }
            }
            // each image stands alone: nothing of it is on the board when the next is shown
            m.reset_context();
            if k % 100 == 0 {
                eprintln!("  image {}/{} | last 100: spoke {} right {} ({:.1}% of spoken) {:.0} ms/image, functions added {} retired {} | neurons {} functions {} | apex/image {:.1} | {:.1} ms/image so far",
                    k, shown, w_spoke, w_right, 100.0 * w_right as f64 / w_spoke.max(1) as f64, (ms - w_ms) / 100.0, w_added, w_retired, m.alive, m.get_dictionary().len(), apex as f64 / k as f64, ms / k as f64);
                w_added = 0;
                w_retired = 0;
                w_spoke = 0;
                w_right = 0;
                w_ms = ms;
            }
        }
        let now = map_dict(&m);
        let dict_changed = now != ep_dict;
        ep_dict = now;
        println!("EPISODE {} | dictionary {} | functions {} | right {}/{} | calls on the apex per image {:.1}", ep + 1, if dict_changed { "changed" } else { "same" }, m.get_dictionary().len(), e_right, order.len(), e_apex as f64 / order.len().max(1) as f64);
        if episodes > 1 {
            println!("  episode {:2}: added {} retired {} | right {}/{} (spoke {}) | max levels {} | neurons {} functions {} | {:.0} ms/image | digit:said {}{}",
                ep + 1, e_added, e_retired, e_right, order.len(), e_spoke, e_levels, m.alive, m.get_dictionary().len(), e_ms / order.len().max(1) as f64, said_list.join(" "), apex_note);
        }
    }
    println!("NEVERCALLED {} of {}", m.count_functions_never_called(), m.get_dictionary().len());
    println!("  train: {} images | spoke {} | right {} ({:.1}% of all, {:.1}% of spoken) | lessons {} | apex/image {:.1} | neurons {} functions {} | {:.0}s",
        shown, spoke, right, 100.0 * right as f64 / shown.max(1) as f64, 100.0 * right as f64 / spoke.max(1) as f64, lessons, apex as f64 / shown.max(1) as f64, m.alive, m.get_dictionary().len(), t0.elapsed().as_secs_f64());

    // what the dictionary holds: functions by relations and nodes, and by whether their interneuron is allocated
    {
        let dict = m.get_dictionary();
        let mut by: BTreeMap<(usize, usize, bool, bool), usize> = Default::default();
        for f in dict.funcs.values() { *by.entry((f.rels.len(), f.nodes.len(), f.has_cycle(), f.inter.is_some())).or_default() += 1; }
        println!("  dictionary by (relations, nodes, cycle, has interneuron): {:?}", by);
        let mut big: Vec<&brain_core::dictionary::Func> = dict.funcs.values().filter(|f| f.rels.len() > 1).collect();
        big.sort_by_key(|f| std::cmp::Reverse(f.rels.len()));
        for f in big.iter().take(12) { println!("    {}", dict.describe(f)); }
    }
    // test: on a copy loaded with learning off, the machine's own answer runs, unrewarded, and nothing is taught
    if test_n > 0 {
        let job = std::path::PathBuf::from(format!("{}/../target/mnist-test-{}", env!("CARGO_MANIFEST_DIR"), std::process::id()));
        m.save(&job, "trained").unwrap();
        m.save_context(&job, "trained").unwrap();
        let mut t = Brain::new(h, Policy::events_to_actions(), regions, columns, false);
        t.load(&job, "trained").unwrap();
        t.load_context(&job, "trained").unwrap();
        std::fs::remove_dir_all(&job).unwrap();
        m = t;
    }
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
        test_n, spoke, right, 100.0 * right as f64 / test_n.max(1) as f64, apex as f64 / test_n.max(1) as f64, m.alive, m.get_dictionary().len(), t1.elapsed().as_secs_f64());
    println!("  confusion (row = digit, column = said):");
    for (d, row) in confusion.iter().enumerate() { println!("    {} {:?}", d, row); }
}
