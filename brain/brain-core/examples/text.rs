//! Stream a text through one letter channel and dump the tables of the letters named.
//! usage: cargo run --release --example text -- "<text>" [R] [H] [letters to dump]
use brain_core::machine::Machine;
use brain_core::types::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = args.get(1).cloned().unwrap_or_else(|| "th ch sh wh th ch sh wh th ch sh wh th ch sh wh ".into());
    let r: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let h: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let dump = args.get(4).cloned().unwrap_or_else(|| "h".into());
    let mut m = Machine::new(h, Policy::all_on(), r);
    let ch = m.declare_channel("text", Some(("letter", 256)), None, &[]);
    let dim = m.channels[ch as usize].event.unwrap();
    let mut right = 0;
    let bytes: Vec<u8> = text.bytes().collect();
    for (i, b) in bytes.iter().enumerate() {
        let w = m.process_frame(&[Input { dim, bucket: *b as u32, pos: [0; MAX_DIMS] }], vec![]);
        let pred = w.iter().find(|w| w.dim == dim).map(|w| w.bucket);
        if i + 1 < bytes.len() && pred == Some(bytes[i + 1] as u32) { right += 1; }
    }
    println!("right {}/{} | neurons {} alive of {} | patterns {} | heights {:?} | expansions that met a neuron again {}", right, bytes.len() - 1, m.last_stats.neurons, m.thalamus.len(), m.last_stats.patterns, m.height_counts(), m.cycle_guard.get());
    for c in dump.bytes() {
        let id = m.base_neuron(dim, c as u32);
        print!("{}", m.describe_neuron(id));
    }
}
