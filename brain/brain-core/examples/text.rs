//! Stream a text through one letter channel and dump the tables of the letters named.
//! usage: cargo run --release --example text -- "<text>" [R] [H] [letters to dump]
use brain_core::machine::Machine;
use brain_core::types::*;

fn main() {
    // `--s15-off` runs the build with switch 15 off, the one choice it leaves open
    let s15_off = std::env::args().any(|a| a == "--s15-off");
    let args: Vec<String> = std::env::args().filter(|a| a != "--s15-off" && !a.starts_with("--closing=") && a != "--trace-closing" && a != "--join-at-a-loss").collect();
    let text = args.get(1).cloned().unwrap_or_else(|| "th ch sh wh th ch sh wh th ch sh wh th ch sh wh ".into());
    let r: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let h: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let dump = args.get(4).cloned().unwrap_or_else(|| "h".into());
    let mut m = Machine::new(h, Policy::all_on(), r);
    m.sw.closing = std::env::args().find_map(|a| a.strip_prefix("--closing=").map(|v| match v { "cycle" => 0, "round" => 1, _ => panic!("--closing=cycle|round") })).unwrap_or(0);
    m.trace_closing = std::env::args().any(|a| a == "--trace-closing");
    m.sw.join_at_a_loss = std::env::args().any(|a| a == "--join-at-a-loss");
    if s15_off { m.sw.s15_uncovered_relations = false; }
    let ch = m.declare_channel("text", Some(("letter", 256)), None, &[]);
    let dim = m.channels[ch as usize].event.unwrap();
    let mut right = 0;
    let bytes: Vec<u8> = text.bytes().collect();
    for (i, b) in bytes.iter().enumerate() {
        let w = m.process_frame(&[Input { dim, bucket: *b as u32, pos: [0; MAX_DIMS] }], vec![]);
        let pred = w.iter().find(|w| w.dim == dim).map(|w| w.bucket);
        if i + 1 < bytes.len() && pred == Some(bytes[i + 1] as u32) { right += 1; }
    }
    println!("right {}/{} | neurons {} alive of {} | patterns {} | heights {:?}", right, bytes.len() - 1, m.last_stats.neurons, m.thalamus.len(), m.last_stats.patterns, m.height_counts());
    println!("CLOSING weighed {} joined {} (from a function of more relations {}) | cycles at the end {} of {} | build joins the calls count would refuse {} | largest such gain as a cycle {} | weighs beyond every offset the rows hold {} | weighs with neither end where the call stands {} | weighs where one lap on cannot be worked out {}", m.closing_weighed, m.closing_joined, m.closing_from_multi, m.dict.funcs.values().filter(|f| f.has_cycle()).count(), m.dict.len(), m.closing_refused[0], m.closing_max.map_or("-".to_string(), |(c, lo, _)| format!("{:.1} (on the calls {:.1})", c, lo)), m.closing_beyond, m.closing_no_end, m.closing_no_lap);
    println!("JOINREFUSED held by no function, gain not above zero {} | lowest gain refused {}", m.help_refused, m.help_refused_min.map_or("-".to_string(), |x| format!("{:.1}", x)));
    let mut ids: Vec<u32> = m.dict.funcs.keys().map(|x| *x as u32).collect();
    ids.sort();
    for id in ids { let f = m.dict.get(id as _).unwrap(); println!("DICTF f{} | {}", id, m.dict.describe(f)); }
    for c in dump.bytes() {
        let id = m.base_neuron(dim, c as u32);
        print!("{}", m.describe_neuron(id));
    }
}
