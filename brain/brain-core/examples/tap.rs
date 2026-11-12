//! Tap while go, then done (docs/algorithm-loop.md): one channel over time alone, an event dimension `cue` (go,
//! stop) and an action dimension `act` (tap, done). A stretch: go in frames 0, 1 and 2, stop in frame 3, no cue
//! after. Teaching: the world runs tap itself in frames 1, 2 and 3 and done in frame 4, each rewarded, then shows
//! nothing and runs nothing for longer than any reach; six stretches. Test: the world shows the cues, runs nothing
//! itself and executes what the machine outputs: a tap in the frame after a go and done in the frame after a stop
//! are rewarded 1, any other action −1. The neighborhood policy is the stocks one: no action is anyone's
//! neighbor, and only events and interneurons connect, to actions alone. Nothing runs where every candidate's
//! estimate is negative. Each frame prints the cue, what ran, what stands on the apex (a guessed activation
//! marked ?), and the machine's output; with a trace, the machine's own trace goes to stderr.
//! usage: cargo run --release --example tap -- [teaching stretches=6] [silence=8] [trace=1]
use brain_core::machine::Machine;
use brain_core::types::*;

fn main() {
    // `--s15-off` runs the build with switch 15 off, the one choice it leaves open
    let s15_off = std::env::args().any(|a| a == "--s15-off");
    let args: Vec<String> = std::env::args().filter(|a| a != "--s15-off").collect();
    let stretches: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(6);
    let silence: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let trace: bool = args.get(3).map_or(true, |s| s != "0");
    let mut policy = Policy::events_to_actions();
    policy.action = ActionPolicy::Nothing;
    let mut m = Machine::new(256, policy, 1);
    if s15_off { m.sw.s15_uncovered_relations = false; }
    let ch = m.declare_channel("tapper", Some(("cue", 2)), Some(("act", 2)), &[]);
    let cue = m.channels[ch as usize].event.unwrap();
    let act = m.channels[ch as usize].action.unwrap();
    let (go, stop, tap, done) = ((cue, 0u32), (cue, 1u32), (act, 0u32), (act, 1u32));
    let name = |m: &Machine, n: u32| -> String {
        let nn = &m.thalamus[n];
        match (nn.dim, nn.bucket) {
            (Some(d), 0) if d == cue => "go".into(),
            (Some(d), 1) if d == cue => "stop".into(),
            (Some(d), 0) if d == act => "tap".into(),
            (Some(d), 1) if d == act => "done".into(),
            _ => format!("#{}", n),
        }
    };
    let word = |b: Option<u32>| -> &'static str { match b { Some(0) => "tap", Some(_) => "done", None => "nothing" } };
    let mut guessed: Vec<(usize, String)> = Vec::new();
    let mut longest_reach = 0;
    // one frame: the cue shown (if any), the action run (if any) and its reward; returns the machine's output
    let mut step = |m: &mut Machine, c: Option<(DimId, u32)>, ran: Option<u32>, reward: f64, label: &str| -> Option<u32> {
        let frame = m.frame_no as usize + 1;
        m.trace = trace;
        if trace { eprintln!("=== frame {} {}", frame, label); }
        let mut inputs = Vec::new();
        if let Some(c) = c { inputs.push(Input { dim: c.0, bucket: c.1, pos: [0; MAX_DIMS] }); }
        if let Some(b) = ran { inputs.push(Input { dim: act, bucket: b, pos: [0; MAX_DIMS] }); }
        let rewards = if ran.is_some() { vec![Reward { reward, channels: vec![ch], frames: 1 }] } else { vec![] };
        let w = m.process_frame(&inputs, rewards);
        let out = w.iter().find(|w| w.dim == act).map(|w| w.bucket);
        longest_reach = longest_reach.max(m.max_reach_t);
        let fr = m.frames.back().unwrap();
        for a in fr.acts.iter().filter(|a| !a.strong && a.kind != Kind::Inter) { guessed.push((frame, name(m, a.neuron))); }
        let apex: Vec<String> = fr.acts.iter().filter(|a| a.on_apex()).map(|a| {
            let s = if a.kind == Kind::Inter { m.dict.sym_text(a.sym) } else { name(m, a.neuron) };
            if a.strong { s } else { format!("{}?", s) }
        }).collect();
        let shown = match c { Some(x) if x == go => "go", Some(x) if x == stop => "stop", _ => "-" };
        println!("f{:3} {:8} cue {:4} ran {:4} | apex [{}] | outputs {}", frame, label, shown, word(ran).replace("nothing", "-"), apex.join(" "), word(out));
        out
    };
    // the stretch's cues, frame by frame: go for `n` frames, then stop, then no cue
    let cues = |n: usize| -> Vec<Option<(DimId, u32)>> { let mut v = vec![Some(go); n]; v.push(Some(stop)); for _ in 0..silence { v.push(None); } v };
    for r in 1..=stretches {
        let l = format!("teach {}", r);
        let cs = cues(3);
        for (k, c) in cs.iter().enumerate() {
            let ran = match k { 1 | 2 | 3 => Some(tap.1), 4 => Some(done.1), _ => None };
            step(&mut m, *c, ran, 1.0, &l);
        }
    }
    let mut table: Vec<(usize, &str, Option<u32>, Option<u32>)> = Vec::new();
    let mut said: Option<u32> = None;
    let mut prev: Option<(DimId, u32)> = None;
    for (i, n) in [3usize, 5, 2].iter().enumerate() {
        let l = format!("test {}", i + 1);
        for c in cues(*n) {
            let ran = said;
            let reward = match (ran, prev) { (Some(0), Some(p)) if p == go => 1.0, (Some(1), Some(p)) if p == stop => 1.0, _ => -1.0 };
            said = step(&mut m, c, ran, reward, &l);
            let expected = match c { Some(x) if x == go => Some(tap.1), Some(x) if x == stop => Some(done.1), _ => None };
            let shown = match c { Some(x) if x == go => "go", Some(x) if x == stop => "stop", _ => "-" };
            table.push((m.frame_no as usize, shown, said, expected));
            prev = c;
        }
    }
    println!("\ntest: frame | cue shown | machine outputs (for the frame after) | expected");
    let mut right = 0;
    for (f, c, o, e) in &table {
        if o == e { right += 1; }
        println!("  f{:3} | {:4} | {:7} | {:7}{}", f, c, word(*o), word(*e), if o == e { "" } else { "   <- differs" });
    }
    println!("right {}/{}", right, table.len());
    println!("longest reach in time seen: {} (silence {})", longest_reach, silence);
    println!("guessed activations that stood: {:?}", guessed);
    println!("dictionary: {} functions", m.dict.len());
    let mut d: Vec<String> = m.dict.funcs.values().map(|f| m.dict.describe(f)).collect();
    d.sort();
    for s in d { println!("  {}", s); }
    println!("neurons and their connections:");
    for id in 0..m.thalamus.len() as u32 { if m.thalamus[id].alive { print!("{}", m.describe_neuron(id)); } }
}
