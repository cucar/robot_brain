/// The alternation case (docs/algorithm-xy.md): a letter channel over time alone, a pair of letters
/// alternating for a stretch. Each frame prints the letter, what the machine had placed for it, what stands on
/// the apex, and the dictionary as it changes.
/// usage: cargo run --release --example xy -- [frames] [trace-to-frame] [stream] [--regions=R] [--columns=C]
/// The stream is stretches separated by commas: two letters and a count of frames, or `_` and a count of
/// silent frames, as in `xy:12,_:4,ab:12,_:4,xy:8`. Without one, `x y` alternates for every frame.
use brain_core::brain::Brain;
use brain_core::types::*;

fn main() {
    // `--regions=R --columns=C` lay out the dispatch, one region of one column by default
    let flag = |name: &str| std::env::args().find_map(|a| a.strip_prefix(name).and_then(|v| v.parse::<usize>().ok())).unwrap_or(1);
    let (regions, columns) = (flag("--regions="), flag("--columns="));
    let args: Vec<String> = std::env::args().filter(|a| !a.starts_with("--")).collect();
    let frames: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(30);
    let _trace_to: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut m = Brain::new(64, Policy::all_on(), regions, columns, true);
    let (mut wrong_at, mut last_change): (Vec<usize>, usize) = (Vec::new(), 0);
    let ch = m.declare_channel("letters", Some(("letter", 26)), None, &[]);
    let dim = m.channels[ch as usize].event.unwrap();
    let reg = |m: &Brain, n: u32| m.get_registry()[n].bucket;
    let letter = |n: u32| (b'a' + n as u8) as char;
    let spec = args.get(3).cloned().unwrap_or_else(|| format!("xy:{}", frames));
    let mut stream: Vec<Option<u32>> = Vec::new();
    for part in spec.split(',') {
        let (what, n) = part.split_once(':').unwrap();
        let n: usize = n.parse().unwrap();
        let b = what.as_bytes();
        for i in 0..n { stream.push(if b[0] == b'_' { None } else { Some((b[i % 2] - b'a') as u32) }); }
    }
    let mut said: Option<u32> = None;
    let (mut right, mut asked) = (0, 0);
    let mut last_dict = String::new();
    for (k, &b) in stream.iter().enumerate().take(frames) {
        if let (Some(s), Some(b)) = (said, b) { asked += 1; if s == b { right += 1; } else { wrong_at.push(k + 1); } }
        let inputs: Vec<Input> = b.map(|b| Input { dim, bucket: b, pos: [0; MAX_DIMS] }).into_iter().collect();
        let w = m.process_frame(&inputs, vec![]);
        let fr = m.get_board().back().unwrap();
        let apex: Vec<String> = fr.acts.iter().filter(|a| a.on_apex()).map(|a| if a.kind == Kind::Inter { format!("#{}", a.neuron) } else { letter(reg(&m, a.neuron)).to_string() }).collect();
        let open: Vec<String> = m.get_board().iter().filter(|f| f.no < fr.no).flat_map(|f| f.acts.iter().map(move |a| (f.no, a))).filter(|(_, a)| a.on_apex() && a.strong && fr.no - a.pos[0] as i64 <= a.reach[0] as i64)
            .map(|(no, a)| format!("{}@{}", if a.kind == Kind::Inter { format!("#{}", a.neuron) } else { letter(reg(&m, a.neuron)).to_string() }, no)).collect();
        let next = w.iter().find(|w| w.dim == dim).map(|w| w.bucket);
        let mark = match (said, b) { (None, _) | (_, None) => "  ", (Some(s), Some(b)) if s == b => "ok", _ => "NO" };
        println!("f{:3} {} had {} {} | apex now [{}] open before [{}] | says next {}", k + 1, b.map_or('_', letter), said.map_or("-".to_string(), |s| letter(s).to_string()), mark, apex.join(" "), open.join(" "), next.map_or("-".to_string(), |s| letter(s).to_string()));
        let mut d: Vec<String> = m.get_dictionary().funcs.values().map(|f| m.get_dictionary().describe(f)).collect();
        d.sort();
        let d = d.join("; ");
        if d != last_dict { println!("      dictionary: {}", if d.is_empty() { "(empty)".into() } else { d.clone() }); last_dict = d; last_change = k + 1; }
        said = next;
    }
    println!("right {}/{} | neurons {} | functions {}", right, asked, m.alive, m.get_dictionary().len());
    println!("alternation inference {}/{} | wrong at frames {:?}", right, asked, wrong_at);
    println!("recognition: {} function(s): {} | the dictionary last changed at frame {}", m.get_dictionary().len(), last_dict, last_change);
}
