/// The six-bar cycle (docs/algorithm-stocks.md §4): one instrument, `rise` and `fall`, the bars +1% −2% −3% +2%
/// +3% −1% repeated, and the action `own` or `disown` for the bar ahead. Where the machine says nothing the world
/// runs the right action; each frame's reward is the bar's change signed by the action that stood over it, scoped
/// to the channel and that one frame. Reports how often what the machine said was right, per bar of the lap, over
/// the run and over the last ten laps, and the ring as it stands at the end.
/// usage: cargo run --release --example stocks -- [laps] [h] [--regions=R] [--columns=C]
use brain_core::brain::Brain;
use brain_core::types::*;

const PATTERN: [f64; 6] = [1.0, -2.0, -3.0, 2.0, 3.0, -1.0];

fn main() {
    // `--regions=R --columns=C` lay out the dispatch, one region of one column by default
    let flag = |name: &str| std::env::args().find_map(|a| a.strip_prefix(name).and_then(|v| v.parse::<usize>().ok())).unwrap_or(1);
    let (regions, columns) = (flag("--regions="), flag("--columns="));
    let args: Vec<String> = std::env::args().filter(|a| !a.starts_with("--")).collect();
    let laps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(150);
    let h: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let n = PATTERN.len();
    let mut m = Brain::new(h, Policy::events_to_actions(), regions, columns, true);
    // every inference the machine makes, the last frame's included, read against the bar it was made for; and the
    // frame the dictionary's functions and relations last changed
    let (mut inf_right, mut inf_all) = (0usize, 0usize);
    let (mut last_dict, mut last_change) = (String::new(), 0u64);
    let ch = m.declare_channel("price", Some(("move", 2)), Some(("position", 2)), &[]);
    let (ev, ac) = (m.channels[ch as usize].event.unwrap(), m.channels[ch as usize].action.unwrap());
    let total = laps * n;
    let mut pending: Option<u32> = None;
    let (mut right, mut wrong, mut lessons) = (vec![0usize; n], vec![0usize; n], 0usize);
    let (mut right10, mut wrong10) = (vec![0usize; n], vec![0usize; n]);
    let mut first_all_right: Option<usize> = None;
    let mut lap_ok = 0;
    for k in 0..total {
        let pct = PATTERN[k % n];
        let up: u32 = if pct > 0.0 { 1 } else { 0 };
        let lesson = pending.is_none();
        let action = pending.unwrap_or(up);
        let reward = if action == 1 { pct } else { -pct };
        let cf = k % n;
        if lesson { lessons += 1; } else {
            if action == up { right[cf] += 1; } else { wrong[cf] += 1; }
            if k + 10 * n >= total { if action == up { right10[cf] += 1; } else { wrong10[cf] += 1; } }
        }
        if cf == 0 { lap_ok = 0; }
        if !lesson && action == up { lap_ok += 1; }
        if cf == n - 1 && lap_ok == n && first_all_right.is_none() { first_all_right = Some(k / n + 1); }
        let inputs = [Input { dim: ev, bucket: up, pos: [0; MAX_DIMS] }, Input { dim: ac, bucket: action, pos: [0; MAX_DIMS] }];
        let w = m.process_frame(&inputs, vec![Reward { reward, channels: vec![ch], frames: 1 }]);
        let said = w.iter().find(|w| w.dim == ac).cloned();
        pending = said.as_ref().map(|w| w.bucket);
        if let Some(p) = pending { inf_all += 1; if p == (if PATTERN[(k + 1) % n] > 0.0 { 1 } else { 0 }) { inf_right += 1; } }
        let dict = m.get_dictionary();
        let mut dd: Vec<String> = dict.funcs.values().map(|f| dict.describe(f)).collect();
        dd.sort();
        let dd = dd.join("; ");
        if dd != last_dict { last_change = k as u64 + 1; last_dict = dd; }
    }
    println!("six-bar cycle: {} laps, H={}", laps, h);
    println!("bar | right | wrong | last 10 laps");
    let (mut tr, mut tw, mut lr, mut lw) = (0, 0, 0, 0);
    for i in 0..n {
        println!("{:3} | {:5} | {:5} | {}/{}", i + 1, right[i], wrong[i], right10[i], right10[i] + wrong10[i]);
        tr += right[i]; tw += wrong[i]; lr += right10[i]; lw += wrong10[i];
    }
    println!("overall right {}/{} = {:.1}% | last 10 laps {}/{} = {:.1}% | lessons {} | first lap all right: {}", tr, tr + tw, 100.0 * tr as f64 / (tr + tw).max(1) as f64, lr, lr + lw, 100.0 * lr as f64 / (lr + lw).max(1) as f64, lessons, first_all_right.map_or("none".to_string(), |l| l.to_string()));
    let dict = m.get_dictionary();
    println!("neurons {} | functions {}", m.alive, dict.len());
    let mut d: Vec<String> = dict.funcs.values().map(|f| dict.describe(f)).collect();
    d.sort();
    if d.len() <= 40 { for s in d { println!("  {}", s); } }
    println!("stocks inference {}/{}", inf_right, inf_all);
    println!("recognition: {} functions at the end; the dictionary's functions and relations last changed at frame {}", dict.len(), last_change);
}
