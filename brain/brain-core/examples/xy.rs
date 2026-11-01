//! The alternation case (docs/algorithm-xy.md): a letter channel over time alone, a pair of letters
//! alternating for a stretch. Each frame prints the letter, what the machine had placed for it, what stands on
//! the apex, and the dictionary as it changes.
//! usage: cargo run --release --example xy -- [frames] [trace-to-frame] [stream]
//! The stream is stretches separated by commas: two letters and a count of frames, or `_` and a count of
//! silent frames, as in `xy:12,_:4,ab:12,_:4,xy:8`. Without one, `x y` alternates for every frame.
use brain_core::machine::Machine;
use brain_core::types::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let frames: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(30);
    let trace_to: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut m = Machine::new(64, Policy::all_on(), 1);
    let ch = m.declare_channel("letters", Some(("letter", 26)), None, &[]);
    let dim = m.channels[ch as usize].event.unwrap();
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
        m.trace = k < trace_to;
        if let (Some(s), Some(b)) = (said, b) { asked += 1; if s == b { right += 1; } }
        let inputs: Vec<Input> = b.map(|b| Input { dim, bucket: b, pos: [0; MAX_DIMS] }).into_iter().collect();
        let w = m.process_frame(&inputs, vec![]);
        let fr = m.frames.back().unwrap();
        let apex: Vec<String> = fr.acts.iter().filter(|a| a.on_apex()).map(|a| if a.kind == Kind::Inter { format!("#{}", a.neuron) } else { letter(m.thalamus[a.neuron].bucket).to_string() }).collect();
        let open: Vec<String> = m.frames.iter().filter(|f| f.no < fr.no).flat_map(|f| f.acts.iter().map(move |a| (f.no, a))).filter(|(_, a)| a.on_apex() && a.strong && fr.no - a.pos[0] as i64 <= a.reach[0] as i64)
            .map(|(no, a)| format!("{}@{}", if a.kind == Kind::Inter { format!("#{}", a.neuron) } else { letter(m.thalamus[a.neuron].bucket).to_string() }, no)).collect();
        let next = w.iter().find(|w| w.dim == dim).map(|w| w.bucket);
        let mark = match (said, b) { (None, _) | (_, None) => "  ", (Some(s), Some(b)) if s == b => "ok", _ => "NO" };
        println!("f{:3} {} had {} {} | apex now [{}] open before [{}] | says next {}", k + 1, b.map_or('_', letter), said.map_or("-".to_string(), |s| letter(s).to_string()), mark, apex.join(" "), open.join(" "), next.map_or("-".to_string(), |s| letter(s).to_string()));
        let mut d: Vec<String> = m.dict.funcs.values().map(|f| m.dict.describe(f)).collect();
        d.sort();
        let d = d.join("; ");
        if d != last_dict { println!("      dictionary: {}", if d.is_empty() { "(empty)".into() } else { d.clone() }); last_dict = d; }
        said = next;
    }
    println!("right {}/{} | neurons {} | functions {}", right, asked, m.alive, m.dict.len());
}
