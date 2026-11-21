/// Stream a text through one letter channel and dump the tables of the letters named.
/// usage: cargo run --release --example text -- "<text>" [R] [H] [letters to dump] [--regions=R] [--columns=C]
/// The second argument is read by nothing: a base neuron sees one position in every dimension (D4, R4).
use brain_core::brain::Brain;
use brain_core::types::*;

fn main() {
    // `--regions=R --columns=C` lay out the dispatch, one region of one column by default
    let flag = |name: &str| std::env::args().find_map(|a| a.strip_prefix(name).and_then(|v| v.parse::<usize>().ok())).unwrap_or(1);
    let (regions, columns) = (flag("--regions="), flag("--columns="));
    let args: Vec<String> = std::env::args().filter(|a| !a.starts_with("--")).collect();
    let text = args.get(1).cloned().unwrap_or_else(|| "th ch sh wh th ch sh wh th ch sh wh th ch sh wh ".into());
    let h: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let dump = args.get(4).cloned().unwrap_or_else(|| "h".into());
    let mut m = Brain::new(h, Policy::all_on(), regions, columns, true);
    let ch = m.declare_channel("text", Some(("letter", 256)), None, &[]);
    let dim = m.channels[ch as usize].event.unwrap();
    let mut right = 0;
    let bytes: Vec<u8> = text.bytes().collect();
    for (i, b) in bytes.iter().enumerate() {
        let w = m.process_frame(&[Input { dim, bucket: *b as u32, pos: [0; MAX_DIMS] }], vec![]);
        let pred = w.iter().find(|w| w.dim == dim).map(|w| w.bucket);
        if i + 1 < bytes.len() && pred == Some(bytes[i + 1] as u32) { right += 1; }
    }
    println!("right {}/{} | neurons {} alive of {} | patterns {} | heights {:?}", right, bytes.len() - 1, m.last_stats.neurons, m.get_registry().len(), m.last_stats.patterns, m.count_heights());
    let mut ids: Vec<u32> = m.get_dictionary().funcs.keys().map(|x| *x as u32).collect();
    ids.sort();
    for id in ids { let f = m.get_dictionary().get(id as _).unwrap(); println!("DICTF f{} | {}", id, m.get_dictionary().describe(f)); }
    let letters: Vec<u32> = dump.bytes().map(|c| m.find_base_neuron(dim, c as u32)).collect();
    for s in m.describe_neurons(&letters) { print!("{}", s); }
}
