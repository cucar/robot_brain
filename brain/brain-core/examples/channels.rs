//! Several channels in one machine, no reset (docs/algorithm.md D34, D24). One channel per stock: the price's move up or
//! down as the event, own or out as the action, the reward scoped to that stock's channel and the frame; every stock
//! fires in every frame. Neighbors by kind (Policy::events_to_actions): an event reads every event and call of any
//! channel in its frame and the one before; no action is anyone's neighbor.
//!
//! Modes:
//!   two    KGC and GLD, the first 12 daily opens of each (apps/stocks/data/1D), cycled
//!   three  KGC, GLD and SPY, the same way
//!   lap    two stocks on six-bar cycles, each with its action: A the stocks example's lap, +1 −2 −3 +2 +3 −1, and
//!          B +2 +1 −1 −3 +3 −2
//!   lapa   A alone; lapb  B alone
//!   three1 one channel whose event has one value per combination of KGC, GLD and SPY's moves (named by the three
//!          moves, K G S), the same 12-bar sequence as `three`, the reward KGC's move: the control for `three`
//!   four   one channel whose event has four values, the cycle a b c d a c (A and B's moves together: UU DU DD UD
//!          UU DD), the action own or out, the reward A's move
//!
//! usage: cargo run --release --example channels -- <two|three|lap|lapa|lapb> [cycles=10 for stocks, 30 for laps]
//!        [data-dir]
use brain_core::activation::Act;
use brain_core::machine::Machine;
use brain_core::types::*;
use std::collections::BTreeSet;

/// The first 12 daily opens of a symbol, as apps/stocks/jobs/multi-channel-test.js loads them.
fn opens(dir: &str, sym: &str) -> Vec<f64> {
    let txt = std::fs::read_to_string(format!("{}/{}.json", dir, sym)).unwrap_or_else(|e| panic!("{}/{}.json: {}", dir, sym, e));
    let bars: Vec<serde_json::Value> = serde_json::from_str(&txt).unwrap();
    bars.iter().take(12).map(|b| b["OpenPrice"].as_f64().unwrap()).collect()
}

/// The events a call stands for, at any depth: (event dimension, frame).
/// The function a call's symbol is a call of: of its interneuron's functions, the one with a node its phase starts at.
fn call_func(m: &Machine, s: Sym) -> Option<u32> {
    let p = m.dict.pose_of(s)?;
    let fs = m.dict.of_inter(m.dict.neuron_of(s));
    fs.iter().find(|f| m.dict.get(**f).map_or(false, |g| g.node_of(p.from).is_some())).or(fs.first()).map(|f| *f as u32)
}

/// A symbol in words: a bar by its channel and move, a call by its function.
fn sym_words(m: &Machine, names: &std::collections::BTreeMap<DimId, String>, four: bool, s: Sym) -> String {
    if m.dict.pose_of(s).is_some() { return format!("call(f{})", call_func(m, s).map_or(-1, |g| g as i64)); }
    let n = &m.thalamus[s];
    let nm = n.dim.and_then(|d| names.get(&d)).cloned().unwrap_or_default();
    match n.kind { Kind::Event => if four && nm == "D" { format!("D={}{}{}{}", if n.bucket & 8 != 0 { "↑" } else { "↓" }, if n.bucket & 4 != 0 { "↑" } else { "↓" }, if n.bucket & 2 != 0 { "↑" } else { "↓" }, if n.bucket & 1 != 0 { "↑" } else { "↓" }) } else if four && nm == "C" { format!("C={}{}{}", if n.bucket & 4 != 0 { "↑" } else { "↓" }, if n.bucket & 2 != 0 { "↑" } else { "↓" }, if n.bucket & 1 != 0 { "↑" } else { "↓" }) } else if four { format!("{}={}", nm, ["a", "b", "c", "d"][n.bucket as usize]) } else { format!("{}{}", nm, if n.bucket == 1 { "↑" } else { "↓" }) }, _ => format!("{}:{}", nm, if n.bucket == 1 { "own" } else { "out" }) }
}

/// An activation in words: what it is, its level and the frame it stands at.
fn act_words(m: &Machine, names: &std::collections::BTreeMap<DimId, String>, four: bool, r: brain_core::machine::ActRef) -> String {
    let Some(a) = m.act(r) else { return "?".into() };
    if a.covers.is_empty() { format!("{}@{}", sym_words(m, names, four, a.sym), r.frame) } else { format!("call(f{}) L{}@{}", a.fired_by.as_ref().map_or(0, |f| f.func), a.level, r.frame) }
}

fn under(m: &Machine, a: &Act) -> BTreeSet<(DimId, i64)> {
    let mut out = BTreeSet::new();
    let mut stack = a.covers.clone();
    while let Some(r) = stack.pop() {
        let Some(c) = m.act(r) else { continue };
        if c.covers.is_empty() { if c.kind == Kind::Event { out.insert((m.thalamus[c.neuron].dim.unwrap(), r.frame)); } } else { stack.extend(c.covers.iter().copied()); }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).cloned().unwrap_or("two".into());
    let dir = args.get(3).filter(|a| !a.starts_with("--")).cloned().unwrap_or(format!("{}/../../apps/stocks/data/1D", env!("CARGO_MANIFEST_DIR")));
    // each channel: its name, the percent move of each bar of its cycle, and whether it has an action
    let mut series: Vec<(String, Vec<f64>, bool)> = Vec::new();
    let mut combos8: Vec<u32> = Vec::new();
    // seq mode: each stock's prices, for the trader
    let mut prices: Vec<Vec<f64>> = Vec::new();
    let (cycle, default_cycles) = match mode.as_str() {
        "two" | "three" => {
            let syms: &[&str] = if mode == "two" { &["KGC", "GLD"] } else { &["KGC", "GLD", "SPY"] };
            for s in syms {
                let p = opens(&dir, s);
                // bar i is the move from row i to row i+1; the last wraps to the first row
                series.push((s.to_string(), (0..12).map(|i| (p[(i + 1) % 12] - p[i]) / p[i] * 100.0).collect(), true));
            }
            (12usize, 10usize)
        }
        "lap" | "lapa" | "lapb" => {
            if mode != "lapb" { series.push(("A".into(), vec![1.0, -2.0, -3.0, 2.0, 3.0, -1.0], true)); }
            if mode != "lapa" { series.push(("B".into(), vec![2.0, 1.0, -1.0, -3.0, 3.0, -2.0], true)); }
            (6usize, 30usize)
        }
        // A on its six-bar lap, B on a five-bar lap UUDUD: together they repeat every 30 frames
        "lap65" => {
            series.push(("A".into(), vec![1.0, -2.0, -3.0, 2.0, 3.0, -1.0], true));
            series.push(("B".into(), vec![2.0, 1.0, -1.0, 3.0, -2.0], true));
            (30usize, 12usize)
        }
        // B alone on its five-bar lap
        "lapb5" => {
            series.push(("B".into(), vec![2.0, 1.0, -1.0, 3.0, -2.0], true));
            (5usize, 72usize)
        }
        // four stocks at one position: KGC, GLD and SPY on the 12-bar lap, and Q, made up, moving one percent a bar
        "three4" => {
            for s in ["KGC", "GLD", "SPY"] {
                let p = opens(&dir, s);
                series.push((s.to_string(), (0..12).map(|i| (p[(i + 1) % 12] - p[i]) / p[i] * 100.0).collect(), true));
            }
            series.push(("Q".into(), "UDDUUDUDDUDD".chars().map(|c| if c == 'U' { 1.0 } else { -1.0 }).collect(), true));
            (12usize, 10usize)
        }
        // its control: one channel whose event has one value per combination of the four moves, graded on KGC's
        "three4c" => {
            let mut pcts: Vec<Vec<f64>> = Vec::new();
            for s in ["KGC", "GLD", "SPY"] { let p = opens(&dir, s); pcts.push((0..12).map(|i| (p[(i + 1) % 12] - p[i]) / p[i] * 100.0).collect()); }
            let q: Vec<bool> = "UDDUUDUDDUDD".chars().map(|c| c == 'U').collect();
            combos8 = (0..12).map(|i| (pcts[0][i] > 0.0) as u32 * 8 + (pcts[1][i] > 0.0) as u32 * 4 + (pcts[2][i] > 0.0) as u32 * 2 + q[i] as u32).collect();
            series.push(("D".into(), pcts[0].clone(), true));
            (12usize, 10usize)
        }
        "three1" => {
            // one value per combination of the three moves: KGC up 4, GLD up 2, SPY up 1
            let mut pcts: Vec<Vec<f64>> = Vec::new();
            for s in ["KGC", "GLD", "SPY"] { let p = opens(&dir, s); pcts.push((0..12).map(|i| (p[(i + 1) % 12] - p[i]) / p[i] * 100.0).collect()); }
            combos8 = (0..12).map(|i| (pcts[0][i] > 0.0) as u32 * 4 + (pcts[1][i] > 0.0) as u32 * 2 + (pcts[2][i] > 0.0) as u32).collect();
            series.push(("C".into(), pcts[0].clone(), true));
            (12usize, 10usize)
        }
        // the stock sequence memorization of the main branch's demos: KGC, GOLD and SPY over their whole daily
        // history as the CSVs hold it, one lap the whole sequence, one episode a lap
        "seq" => {
            let bars: Option<usize> = args.iter().find(|a| a.starts_with("--bars=")).map(|a| a[7..].parse().unwrap());
            // the stocks (KGC, GOLD and SPY unless given), and only the last so many rows of each where asked
            let syms: Vec<String> = args.iter().find(|a| a.starts_with("--symbols=")).map_or(vec!["KGC".into(), "GOLD".into(), "SPY".into()], |a| a[10..].split(',').map(|x| x.to_string()).collect());
            let last: Option<usize> = args.iter().find(|a| a.starts_with("--last=")).map(|a| a[7..].parse().unwrap());
            for s in &syms {
                let s = s.as_str();
                let txt = std::fs::read_to_string(format!("{}/{}.csv", dir, s)).unwrap();
                let p: Vec<f64> = txt.lines().filter(|l| !l.trim().is_empty()).map(|l| l.split(',').next().unwrap().parse().unwrap()).collect();
                let p: Vec<f64> = match last { Some(l) if l < p.len() => p[p.len() - l..].to_vec(), _ => p };
                let n = bars.map_or(p.len(), |b| b.min(p.len()));
                series.push((s.to_string(), (0..n - 1).map(|i| (p[i + 1] - p[i]) / p[i] * 100.0).collect(), true));
                prices.push(p[..n].to_vec());
            }
            let n = series[0].1.len();
            (n, 5usize)
        }
        "four" => {
            // a = UU, b = DU, c = DD, d = UD; the reward is A's move
            series.push(("S".into(), vec![1.0, -2.0, -3.0, 2.0, 3.0, -1.0], true));
            (6usize, 30usize)
        }
        _ => panic!("mode: two, three, three1, lap, lapa, lapb or four"),
    };
    let cycles: usize = args.get(2).filter(|a| !a.starts_with("--")).and_then(|s| s.parse().ok()).unwrap_or(default_cycles);
    let mut m = Machine::new(256, Policy::events_to_actions(), 1);
    m.print_cycles = std::env::args().any(|a| a == "--cycles");
    if args.iter().any(|a| a == "--levels") { m.trace_levels = true; }
    let trace_frame: Option<i64> = args.iter().find(|a| a.starts_with("--trace-frame=")).map(|a| a[14..].parse().unwrap());
    let mut chans: Vec<(ChannelId, DimId, Option<DimId>)> = Vec::new();
    for (name, _, act) in &series {
        let ch = m.declare_channel(name, Some(("move", if mode == "four" { 4 } else if mode == "three1" { 8 } else if mode == "three4c" { 16 } else { 2 })), if *act { Some(("position", 2)) } else { None }, &[]);
        chans.push((ch, m.channels[ch as usize].event.unwrap(), m.channels[ch as usize].action));
    }
    println!("{}: {} cycles of {} bars", mode, cycles, cycle);
    for (name, pct, _) in &series { if pct.len() == cycle { println!("  {} moves per cycle: {}", name, pct.iter().map(|p| if *p > 0.0 { 'U' } else { 'D' }).collect::<String>()); } }

    let total = cycle * cycles;
    let mut pending: Vec<Option<u32>> = vec![None; series.len()];
    let mut grid: Vec<Vec<Vec<char>>> = vec![vec![vec!['-'; cycle]; cycles]; series.len()];
    let mut combos: Vec<Vec<usize>> = vec![Vec::new(); cycles];
    // per channel set: the laps at whose end one call on the apex stands for those channels' six bars of the lap
    let mut six_laps: Vec<Vec<usize>> = vec![Vec::new(); series.len() + 1];
    let mut called_last: BTreeSet<u32> = BTreeSet::new();
    let mut under_l3: Vec<(DimId, Option<i64>)> = chans.iter().map(|c| (c.1, None)).collect();
    let (mut last_dict, mut last_change) = (String::new(), 0i64);
    let t0 = std::time::Instant::now();
    // the log for comparing runs: every function after every frame, the build's own trace of what moved, the board at
    // one frame, and for the first two cycles what each bar read in the frame before
    let log = args.iter().any(|a| a == "--log");
    let board_at: Option<i64> = args.iter().find(|a| a.starts_with("--board=")).map(|a| a[8..].parse().unwrap());
    // the calls to follow frame by frame: --watch=9,11
    let watch: Vec<u32> = args.iter().find(|a| a.starts_with("--watch=")).map_or(vec![], |a| a[8..].split(',').map(|x| x.parse().unwrap()).collect());
    let (mut last_change_c, mut last_dict_c) = (0i64, String::new());
    let four = mode == "four" || mode == "three1" || mode == "three4c";
    let dictf = args.iter().any(|a| a == "--dictf");
    let assoc = args.iter().any(|a| a == "--assoc");
    let kinds = args.iter().any(|a| a == "--kinds");
    let heard = args.iter().any(|a| a == "--heard");
    let bars = args.iter().any(|a| a == "--bars");
    let all4 = args.iter().any(|a| a == "--all4");
    let trace_frames: Vec<i64> = args.iter().find(|a| a.starts_with("--trace-frames=")).map(|a| a[15..].split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or_default();
    let names: std::collections::BTreeMap<DimId, String> = chans.iter().zip(series.iter()).flat_map(|(c, s)| [(c.1, s.0.clone())].into_iter().chain(c.2.map(|a| (a, s.0.clone())))).collect();
    if log { m.trace = true; m.trace_retire = true; m.trace_cover = true; }
    // the main branch's trader (apps/stocks/trader.js) in seq mode: $15,000 a episode, each stock own (action 1) or
    // out, at most one position, the own with the highest estimate (then the higher price, then the symbol), every
    // dollar in it in whole shares, no transaction cost; profit is cash and holdings less the start
    let nch = series.len();
    let (mut cash, mut shares, mut trades) = (15000.0f64, vec![0i64; nch], 0usize);
    let mut ep_profit: Vec<(f64, usize)> = Vec::new();
    let mut est = vec![0.0f64; nch];
    let mut last_act: Vec<Option<u32>> = vec![None; nch];
    // seq mode: the trader's positions and cost (trader.js maxPositions, transactionCost in percent), and the report
    let max_positions: usize = args.iter().find(|a| a.starts_with("--max-positions=")).map_or(1, |a| a[16..].parse().unwrap());
    let tcost: f64 = args.iter().find(|a| a.starts_with("--cost=")).map_or(0.0, |a| a[7..].parse().unwrap());
    let report_every: usize = args.iter().find(|a| a.starts_with("--every=")).map_or(0, |a| a[8..].parse().unwrap());
    for k in 0..total {
        if k % cycle == 0 { cash = 15000.0; shares = vec![0; nch]; trades = 0; }
        let mut inputs = Vec::new();
        let mut rewards = Vec::new();
        for (i, (_, pct, _)) in series.iter().enumerate() {
            let p = pct[k % pct.len()];
            let up: u32 = if p > 0.0 { 1 } else { 0 };
            let ev = if mode == "four" { [0u32, 1, 2, 3, 0, 2][k % 6] } else if mode == "three1" || mode == "three4c" { combos8[k % 12] } else { up };
            inputs.push(Input { dim: chans[i].1, bucket: ev, pos: [0; MAX_DIMS] });
            if let Some(ac) = chans[i].2 {
                // what the machine said for this bar runs; a lesson (the move itself) where it said nothing
                let action = pending[i].unwrap_or(up);
                inputs.push(Input { dim: ac, bucket: action, pos: [0; MAX_DIMS] });
                rewards.push(Reward { reward: if action == 1 { p } else { -p }, channels: vec![chans[i].0], frames: 1 });
            }
        }
        if log { eprintln!("=== frame {}", m.frame_no + 1); }
        if !trace_frames.is_empty() { let n = m.frame_no + 1; let on = trace_frames.contains(&n); m.trace_vote = on; if on { eprintln!("=== frame {}", n); } }
        if let Some(tf) = trace_frame { let on = m.frame_no + 1 == tf; m.trace = on; m.trace_cover = on; if on { eprintln!("=== frame {}", tf); } }
        let w = m.process_frame(&inputs, rewards);
        let fno = m.frame_no;
        for (i, (_, pct, _)) in series.iter().enumerate() {
            let Some(ac) = chans[i].2 else { continue };
            pending[i] = w.iter().find(|x| x.dim == ac).map(|x| x.bucket);
            est[i] = w.iter().find(|x| x.dim == ac).map_or(0.0, |x| x.estimate);
            let next_up = if pct[(k + 1) % pct.len()] > 0.0 { 1 } else { 0 };
            if let Some(g) = pending[i] { grid[i][k / cycle][k % cycle] = if g == next_up { 'R' } else { 'W' }; }
        }
        // the trader, at the price this frame's bar ended at: a stock keeps its last action where the machine says
        // nothing new; where none has an action yet, nothing trades
        if !prices.is_empty() {
            for i in 0..nch { if pending[i].is_some() { last_act[i] = pending[i]; } }
            if last_act.iter().any(|a| a.is_some()) {
                let at = (k % cycle) + 1;
                let price = |i: usize| prices[i][at];
                let total_value = cash + (0..nch).map(|i| shares[i] as f64 * price(i)).sum::<f64>();
                let mut own: Vec<usize> = (0..nch).filter(|&i| last_act[i] == Some(1) && price(i) > 0.0).collect();
                // more owns than positions: the best by estimate, then the higher price, then the symbol, priced under
                // $5,000 (trader.js distributeAllocations)
                if own.len() > max_positions {
                    own.sort_by(|&a, &b| est[b].exp().partial_cmp(&est[a].exp()).unwrap().then(price(b).partial_cmp(&price(a)).unwrap()).then(series[a].0.cmp(&series[b].0)));
                    own.retain(|&i| price(i) < 5000.0);
                    own.truncate(max_positions);
                }
                // the plan (trader.js getActionPlan): sell what is out, bring each own to its equal share of the value,
                // sweep what cash is left into the cheapest own; then sells, then buys, each at its cost
                let mul = tcost / 100.0;
                let mut remaining = cash;
                let mut sells: Vec<(usize, i64)> = Vec::new();
                let mut buys: Vec<(usize, i64)> = Vec::new();
                let mut cheapest: Option<usize> = None;
                for i in 0..nch {
                    if !own.contains(&i) {
                        if shares[i] > 0 { sells.push((i, shares[i])); remaining += shares[i] as f64 * price(i) * (1.0 - mul); }
                        continue;
                    }
                    if cheapest.map_or(true, |c| price(i) < price(c)) { cheapest = Some(i); }
                    let buy_price = price(i) * (1.0 + mul);
                    let target = ((total_value / own.len() as f64) / buy_price).floor() as i64;
                    let diff = target - shares[i];
                    if diff < 0 { sells.push((i, -diff)); remaining += (-diff) as f64 * price(i) * (1.0 - mul); }
                    else if diff > 0 { buys.push((i, diff)); remaining -= diff as f64 * buy_price; }
                }
                if let Some(c) = cheapest {
                    if remaining > 0.0 {
                        let extra = (remaining / (price(c) * (1.0 + mul))).floor() as i64;
                        if extra > 0 { if let Some(b) = buys.iter_mut().find(|b| b.0 == c) { b.1 += extra; } else { buys.push((c, extra)); } }
                    }
                }
                for (i, n) in sells { cash += n as f64 * price(i) * (1.0 - mul); shares[i] -= n; trades += 1; }
                for (i, n) in buys {
                    let eff = price(i) * (1.0 + mul);
                    let n = n.min((cash / eff).floor() as i64);
                    if n <= 0 { continue; }
                    cash -= n as f64 * eff; shares[i] += n; trades += 1;
                }
            }
            // every 100 frames: neurons, functions, the share right over the last 100 frames and so far, and the profit
            if report_every > 0 && (k % cycle + 1) % report_every == 0 {
                let lap = k / cycle;
                let lo = (k % cycle + 1) - report_every;
                let (mut r100, mut n100, mut rall, mut nall) = (0usize, 0usize, 0usize, 0usize);
                for i in 0..nch {
                    for (c, x) in grid[i][lap].iter().enumerate().take(k % cycle + 1) {
                        if *x == '-' { continue; }
                        nall += 1; if *x == 'R' { rall += 1; }
                        if c >= lo { n100 += 1; if *x == 'R' { r100 += 1; } }
                    }
                }
                let at = (k % cycle) + 1;
                let profit = cash + (0..nch).map(|i| shares[i] as f64 * prices[i][at]).sum::<f64>() - 15000.0;
                println!("EVERY100 frame {} | neurons {} | functions {} | right last 100 frames {:.2}% | right so far {:.2}% | net profit ${:.2} | trades {} | {:.0}s",
                    k + 1, m.alive, m.dict.len(), 100.0 * r100 as f64 / n100.max(1) as f64, 100.0 * rall as f64 / nall.max(1) as f64, profit, trades, t0.elapsed().as_secs_f64());
            }
        }
        // recognition: the dictionary's last change, and this frame's combination calls (a level-1 call over two of
        // this frame's bars)
        let mut d: Vec<String> = m.dict.funcs.values().map(|f| m.dict.describe(f)).collect();
        d.sort();
        let d = d.join("; ");
        if d != last_dict { last_change = fno; last_dict = d; }
        let _ = (&mut last_change_c, &mut last_dict_c);
        if k % cycle == cycle - 1 {
            let top = m.frames.iter().filter(|f| f.no > fno - cycle as i64).flat_map(|f| f.acts.iter()).map(|a| a.level).max().unwrap_or(0);
            let accs: Vec<String> = series.iter().enumerate().filter(|(_, x)| x.2).map(|(i, x)| { let row = &grid[i][k / cycle]; let (r, w) = row.iter().fold((0, 0), |(r, w), c| (r + (*c == 'R') as usize, w + (*c == 'W') as usize)); format!("{} {:.2}%", x.0, 100.0 * r as f64 / (r + w).max(1) as f64) }).collect();
            let profit = if prices.is_empty() { 0.0 } else { cash + (0..nch).map(|i| shares[i] as f64 * prices[i][(k % cycle) + 1]).sum::<f64>() - 15000.0 };
            if !prices.is_empty() { ep_profit.push((profit, trades)); }
            println!("CYCLE {} | neurons {} | functions {} | highest call level in the cycle {} | the dictionary last changed at frame {} | right {}{}", k / cycle + 1, m.alive, m.dict.len(), top, last_change, accs.join(", "),
                if prices.is_empty() { String::new() } else { format!(" | net profit ${:.2}, ROI {:.2}% ({} trades)", profit, profit / 15000.0 * 100.0, trades) });
        }
        let fr = m.frames.back().unwrap();
        if dictf && !log {
            let mut ids: Vec<u32> = m.dict.funcs.keys().map(|x| *x as u32).collect();
            ids.sort();
            for id in ids { let f = m.dict.get(id as _).unwrap(); println!("DICTF {} | f{} | {}", fno, id, f.rels.iter().map(|r| format!("{} → {} at t{}", sym_words(&m, &names, four, r.a), sym_words(&m, &names, four, r.b), r.off[0])).collect::<Vec<_>>().join("; ")); }
        }
        if bars {
            // each bar of the frame before: the kind its first covered relation decided, whether it is covered and at
            // what age, and the calls over it with their level and function
            if let Some(pf) = m.frames.iter().find(|f| f.no == fno - 1) {
                let parts: Vec<String> = pf.acts.iter().enumerate().filter(|(_, a)| a.kind == Kind::Event).map(|(i, a)| {
                    let r = brain_core::machine::ActRef { frame: pf.no, index: i as u32 };
                    let by: Vec<String> = a.coverers.iter().filter_map(|c| m.act(*c).map(|ca| format!("L{} f{}@{}", ca.level, ca.fired_by.as_ref().map_or(0, |fb| fb.func), c.frame))).collect();
                    format!("{}={} cov:{} by:[{}]", act_words(&m, &names, four, r), match m.kind_decided(r) { Some(true) => "inside", Some(false) => "time", None => "none" }, a.covered_at.map_or("-".to_string(), |x| x.to_string()), by.join(" "))
                }).collect();
                println!("BARS {} | {}", pf.no, parts.join(", "));
            }
        }
        if all4 {
            // the calls standing in this frame over every bar of this frame
            if let Some(fr) = m.frames.iter().find(|f| f.no == fno) {
                let nb = fr.acts.iter().filter(|a| a.kind == Kind::Event).count();
                let parts: Vec<String> = fr.acts.iter().filter(|a| !a.covers.is_empty() && !a.gone).filter(|a| under(&m, a).iter().filter(|x| x.1 == fno).count() == nb).map(|a| format!("L{} f{}", a.level, a.fired_by.as_ref().map_or(0, |fb| fb.func))).collect();
                println!("ALL4 {} | {}", fno, parts.join("; "));
            }
        }
        if heard {
            // what was heard in this frame's selection: each voter, its level and the function it is a call of
            let parts: Vec<String> = m.last_heard.iter().map(|r| { let a = m.act(*r); format!("{}@{} L{} {}", a.map_or("?".to_string(), |a| m.dict.sym_text(a.sym).split('<').next().unwrap_or("").to_string()), r.frame, a.map_or(0, |a| a.level), a.and_then(|a| a.fired_by.as_ref()).map_or("base".to_string(), |fb| format!("f{}", fb.func))) }).collect();
            println!("HEARD {} | {}", fno, parts.join("; "));
        }
        if kinds {
            // the kind each bar of the frame before was taken by (its first covered relation): inside, time, or none
            if let Some(pf) = m.frames.iter().find(|f| f.no == fno - 1) {
                let parts: Vec<String> = pf.acts.iter().enumerate().filter(|(_, a)| a.kind == Kind::Event).map(|(i, _)| {
                    let r = brain_core::machine::ActRef { frame: pf.no, index: i as u32 };
                    format!("{}={}", act_words(&m, &names, four, r), match m.kind_decided(r) { Some(true) => "inside", Some(false) => "time", None => "none" })
                }).collect();
                println!("KIND {} | {}", pf.no, parts.join(", "));
            }
        }
        if assoc {
            // the calls over this frame's bars: each call that landed on a bar of this frame, what it covers, and the
            // bars under no call
            let mut parts: Vec<String> = Vec::new();
            for f in m.frames.iter() { for (i, a) in f.acts.iter().enumerate() {
                let Some(fb) = &a.fired_by else { continue };
                if !a.covers.iter().any(|c| c.frame == fno && m.act(*c).map_or(false, |x| x.kind == Kind::Event)) && !a.covers.iter().any(|c| m.act(*c).map_or(false, |x| x.fired_by.is_some() && x.covers.iter().any(|cc| cc.frame == fno && m.act(*cc).map_or(false, |y| y.kind == Kind::Event)))) { continue; }
                let _ = i;
                parts.push(format!("f{} L{} over {}", fb.func, a.level, a.covers.iter().map(|c| act_words(&m, &names, four, *c)).collect::<Vec<_>>().join(" + ")));
            } }
            let bare: Vec<String> = fr.acts.iter().enumerate().filter(|(_, a)| a.kind == Kind::Event && a.coverers.is_empty()).map(|(i, _)| act_words(&m, &names, four, brain_core::machine::ActRef { frame: fno, index: i as u32 })).collect();
            println!("ASSOC {} | {}.{} | {} | bars under no call: {}", fno, k / cycle + 1, k % cycle + 1, parts.join("; "), if bare.is_empty() { "none".into() } else { bare.join(", ") });
        }
        if log {
            let mut ids: Vec<u32> = m.dict.funcs.keys().map(|x| *x as u32).collect();
            ids.sort();
            for id in ids { let f = m.dict.get(id as _).unwrap(); println!("DICTF {} | f{} | {}", fno, id, f.rels.iter().map(|r| format!("{} → {} at t{}", sym_words(&m, &names, four, r.a), sym_words(&m, &names, four, r.b), r.off[0])).collect::<Vec<_>>().join("; ")); }
            // every call made in this frame (a call that took a lap is made again, its record grown), with the frames
            // it stands for, and how each stands at the frame's end
            for (i, a) in fr.acts.iter().enumerate() {
                if a.fired_by.is_none() || a.opened_at != fno { continue; }
                let r = brain_core::machine::ActRef { frame: fno, index: i as u32 };
                let fs: BTreeSet<i64> = under(&m, a).iter().map(|x| x.1).collect();
                let combo = a.level == 1 && a.covers.iter().filter(|c| c.frame == fno && m.act(**c).map_or(false, |x| x.kind == Kind::Event)).count() >= 2;
                println!("CALL {} | {} | frames {:?} | {} | calls on it: {}", fno, act_words(&m, &names, four, r), fs, if combo { "the combination call" } else { "" },
                    if a.covered_at.is_some() { format!("covered by {}", a.coverers.iter().map(|c| act_words(&m, &names, four, *c)).collect::<Vec<_>>().join(", ")) } else if a.coverers.is_empty() { "none".into() } else { format!("not covered, landed on by {}", a.coverers.iter().map(|c| act_words(&m, &names, four, *c)).collect::<Vec<_>>().join(", ")) });
            }
            // each followed call standing in the last six frames: what it stands for, whether it is covered and by
            // what, what its row read (and whether the call read shares an activation with it), and the relations it
            // returned with their strengths
            for f in m.frames.iter().filter(|f| f.no > fno - 6) { for (i, a) in f.acts.iter().enumerate() {
                let Some(fb) = &a.fired_by else { continue };
                if !watch.contains(&(fb.func as u32)) { continue; }
                let r = brain_core::machine::ActRef { frame: f.no, index: i as u32 };
                let fs: BTreeSet<i64> = under(&m, a).iter().map(|x| x.1).collect();
                let cov = if a.covered_at.is_some() { format!("covered, calls on it: {}", a.coverers.iter().map(|c| act_words(&m, &names, four, *c)).collect::<Vec<_>>().join(", ")) } else { format!("not covered{}", if a.coverers.is_empty() { String::new() } else { format!(", landed on by {}", a.coverers.iter().map(|c| act_words(&m, &names, four, *c)).collect::<Vec<_>>().join(", ")) }) };
                let mine: BTreeSet<brain_core::machine::ActRef> = a.covers.iter().copied().collect();
                let row = m.thalamus[a.neuron].history.iter().find(|row| row.act == r);
                let read: Vec<String> = row.map_or(vec![], |row| row.nbs.iter().map(|nb| {
                    let shared: Vec<String> = m.act(nb.act).map_or(vec![], |x| x.covers.iter().filter(|c| mine.contains(c)).map(|c| act_words(&m, &names, four, *c)).collect());
                    format!("{} at t{}{}", act_words(&m, &names, four, nb.act), nb.off[0], if shared.is_empty() { if m.act(nb.act).map_or(false, |x| x.covers.is_empty()) { " (a bar)".to_string() } else { " (shares no activation: read through a lifted relation)".to_string() } } else { format!(" (shares {})", shared.join(", ")) })
                }).collect());
                let rels: Vec<String> = a.rels.iter().map(|br| format!("{} at t{} strength {}", act_words(&m, &names, four, br.other), br.off[0], br.strength)).collect();
                println!("WATCH {} | {} | frames {:?} | {} | read: {} | relations: {}", fno, act_words(&m, &names, four, r), fs, cov, if read.is_empty() { "-".into() } else { read.join("; ") }, if rels.is_empty() { "-".into() } else { rels.join("; ") });
            } }
            if k < 2 * cycle {
                for (i, a) in fr.acts.iter().enumerate() {
                    let r = brain_core::machine::ActRef { frame: fno, index: i as u32 };
                    if a.level == 1 && a.fired_by.is_some() { println!("LEVEL1 {} | {} | covers {}", fno, act_words(&m, &names, four, r), a.covers.iter().map(|c| act_words(&m, &names, four, *c)).collect::<Vec<_>>().join(", ")); }
                    if a.covers.is_empty() && a.kind == Kind::Event {
                        let row = m.thalamus[a.neuron].history.iter().find(|row| row.act == r);
                        let back: Vec<String> = row.map_or(vec![], |row| row.nbs.iter().filter(|nb| nb.off[0] == -1).map(|nb| act_words(&m, &names, four, nb.act)).collect());
                        let here: Vec<String> = row.map_or(vec![], |row| row.nbs.iter().filter(|nb| nb.off[0] == 0).map(|nb| act_words(&m, &names, four, nb.act)).collect());
                        println!("READ {} | {} | the frame before read as: {} | this frame: {}", fno, act_words(&m, &names, four, r), if back.is_empty() { "nothing".into() } else { back.join(", ") }, if here.is_empty() { "nothing".into() } else { here.join(", ") });
                    }
                }
            }
            if board_at == Some(fno) {
                for f in m.frames.iter() { for (i, a) in f.acts.iter().enumerate() {
                    if !a.on_apex() || a.kind == Kind::Action { continue; }
                    let r = brain_core::machine::ActRef { frame: f.no, index: i as u32 };
                    let fr_under: BTreeSet<i64> = if a.covers.is_empty() { [f.no].into_iter().collect() } else { under(&m, a).iter().map(|x| x.1).collect() };
                    println!("BOARD {} | {} | level {} | frames {:?}", fno, act_words(&m, &names, four, r), a.level, fr_under);
                } }
            }
        }
        let combo = fr.acts.iter().filter(|a| a.level == 1 && a.fired_by.is_some() && a.covers.iter().filter(|c| c.frame == fno && m.act(**c).map_or(false, |x| x.kind == Kind::Event)).count() >= 2).count();
        combos[k / cycle].push(combo);
        // the laps: whether one call on the apex stands for each channel's six bars of this lap, and for all of them
        if (mode.starts_with("lap") || mode == "four") && k % cycle == cycle - 1 {
            let lap: Vec<i64> = ((fno - 5)..=fno).collect();
            let sets: Vec<Vec<DimId>> = chans.iter().map(|c| vec![c.1]).chain(std::iter::once(chans.iter().map(|c| c.1).collect())).collect();
            for (j, set) in sets.iter().enumerate() {
                let stands = m.frames.iter().flat_map(|f| f.acts.iter()).any(|a| a.on_apex() && !a.covers.is_empty() && { let u = under(&m, a); set.iter().all(|e| lap.iter().all(|f| u.contains(&(*e, *f)))) });
                if stands { six_laps[j].push(k / cycle + 1); }
            }
        }
        if k / cycle == cycles - 1 { for a in fr.acts.iter() { if let Some(fb) = &a.fired_by { called_last.insert(fb.func as u32); } } }
        // the frame at which bar 6 of the second-last cycle first comes under a level-3 call
        if mode.starts_with("lap") && cycles >= 2 {
            let target = (cycle * (cycles - 1)) as i64;
            if fno >= target {
                for u in under_l3.iter_mut() {
                    if u.1.is_none() && m.frames.iter().flat_map(|f| f.acts.iter()).any(|a| a.level == 3 && under(&m, a).contains(&(u.0, target))) { u.1 = Some(fno); }
                }
            }
        }
    }
    for (i, (name, _, act)) in series.iter().enumerate() {
        if !act { continue; }
        let (r, w) = grid[i].iter().flatten().fold((0, 0), |(r, w), c| (r + (*c == 'R') as usize, w + (*c == 'W') as usize));
        let from = (0..cycles).rev().take_while(|c| grid[i][*c].iter().all(|x| *x == 'R')).last();
        println!("inference {}: right {} of {} | every bar right from cycle {}", name, r, r + w, from.map_or("never".into(), |c| (c + 1).to_string()));
        if cycle > 60 {
            for (c, row) in grid[i].iter().enumerate() { let (r, w) = row.iter().fold((0, 0), |(r, w), x| (r + (*x == 'R') as usize, w + (*x == 'W') as usize)); println!("  {} lap {:2}: right {} of {} ({:.2}%)", name, c + 1, r, r + w, 100.0 * r as f64 / (r + w).max(1) as f64); }
        } else {
            for (c, row) in grid[i].iter().enumerate() { println!("  {} cycle {:2}: {}", name, c + 1, row.iter().collect::<String>()); }
        }
    }
    println!("recognition: {} functions | the dictionary last changed at frame {}", m.dict.len(), last_change);
    println!("NEVERCALLED {} of {}", m.funcs_never_called(), m.dict.len());
    println!("JOINSMOVES joins frame {} time {} | moves frame {} time {}", m.joins_by[0], m.joins_by[1], m.moves_by[0], m.moves_by[1]);
    println!("  combination calls per frame, per cycle: {}", combos.iter().enumerate().map(|(c, v)| format!("{}:{}", c + 1, v.iter().map(|x| x.to_string()).collect::<String>())).collect::<Vec<_>>().join(" "));
    if mode.starts_with("lap") || mode == "four" {
        let show = |v: &Vec<usize>| if v.is_empty() { "never".to_string() } else { format!("from lap {} ({} of {} laps)", v[0], v.len(), cycles) };
        for (j, (name, _, _)) in series.iter().enumerate() { println!("  one call on the apex stands for {}'s six bars of a lap: {}", name, show(&six_laps[j])); }
        if series.len() > 1 { println!("  one call on the apex stands for every channel's six bars of a lap: {}", show(&six_laps[series.len()])); }
    }
    if mode.starts_with("lap") { for (j, u) in under_l3.iter().enumerate() { println!("  {}: bar 6 of cycle {} (frame {}) first comes under a level-3 call at frame {}", series[j].0, cycles - 1, cycle * (cycles - 1), u.1.map_or("never".into(), |f| f.to_string())); } }
    // the dictionary at the end, by level and kind, with how many had a call in the last cycle
    {
        let names: std::collections::BTreeMap<DimId, String> = chans.iter().zip(series.iter()).flat_map(|(c, s)| [(c.1, s.0.clone())].into_iter().chain(c.2.map(|a| (a, s.0.clone())))).collect();
        let func_of = |s: Sym| -> Option<u32> { let p = m.dict.pose_of(s)?; let fs = m.dict.of_inter(m.dict.neuron_of(s)); fs.iter().find(|f| m.dict.get(**f).map_or(false, |g| g.node_of(p.from).is_some())).or(fs.first()).map(|f| *f as u32) };
        let assoc_bars = |id: u32| m.dict.get(id as _).map_or(false, |f| f.rels.iter().any(|r| r.off[0] == 0 && m.dict.pose_of(r.a).is_none() && m.dict.pose_of(r.b).is_none()));
        // each function's level and channels, read once per function (a function's nodes may name itself)
        fn info(m: &Machine, id: u32, f_of: &dyn Fn(Sym) -> Option<u32>, memo: &mut std::collections::BTreeMap<u32, (usize, BTreeSet<DimId>)>, open: &mut BTreeSet<u32>) -> (usize, BTreeSet<DimId>) {
            if let Some(x) = memo.get(&id) { return x.clone(); }
            if !open.insert(id) { return (0, BTreeSet::new()); }
            let mut lv = 0;
            let mut ch = BTreeSet::new();
            if let Some(f) = m.dict.get(id as _) {
                for n in &f.nodes {
                    if m.dict.pose_of(*n).is_some() { if let Some(g) = f_of(*n) { let (l, c) = info(m, g, f_of, memo, open); lv = lv.max(l); ch.extend(c); } }
                    else if let Some(d) = m.thalamus[*n].dim { ch.insert(d); }
                }
            }
            open.remove(&id);
            memo.insert(id, (lv + 1, ch.clone()));
            (lv + 1, ch)
        }
        let mut memo = Default::default();
        let mut table: std::collections::BTreeMap<(usize, String), (usize, usize)> = Default::default();
        for (id, f) in m.dict.funcs.iter() {
            let id = *id as u32;
            let (lv, ch) = info(&m, id, &func_of, &mut memo, &mut BTreeSet::new());
            let kind = if assoc_bars(id) { "association between bars".to_string() }
                else if f.nodes.iter().any(|n| m.dict.pose_of(*n).is_some() && func_of(*n).map_or(false, |g| assoc_bars(g))) { "over the combination calls".to_string() }
                else if ch.len() == 1 { format!("{} only", names.get(ch.iter().next().unwrap()).cloned().unwrap_or_default()) }
                else { "across the channels in time".to_string() };
            let e = table.entry((lv, kind)).or_default();
            e.0 += 1;
            if called_last.contains(&id) { e.1 += 1; }
        }
        for ((lv, kind), (n, c)) in &table { println!("  dictionary: level {} | {} | {} | with a call in the last cycle {}", lv, kind, n, c); }
        let short = |x: Sym| -> String {
            if m.dict.pose_of(x).is_some() { return format!("call(f{})", func_of(x).map_or(-1, |g| g as i64)); }
            let n = &m.thalamus[x];
            let nm = n.dim.and_then(|d| names.get(&d)).cloned().unwrap_or_default();
            match n.kind { Kind::Event => if mode == "four" { format!("{}={}", nm, ["a", "b", "c", "d"][n.bucket as usize]) } else { format!("{}{}", nm, if n.bucket == 1 { "↑" } else { "↓" }) }, _ => format!("{}:{}", nm, if n.bucket == 1 { "own" } else { "out" }) }
        };
        let mut ids: Vec<u32> = m.dict.funcs.keys().map(|x| *x as u32).collect();
        ids.sort_by_key(|id| (memo.get(id).map_or(0, |x: &(usize, BTreeSet<DimId>)| x.0), *id));
        for id in ids {
            let f = m.dict.get(id as _).unwrap();
            println!("  function f{} | level {} | nodes {} | relations {}{}{}", id, memo.get(&id).map_or(0, |x| x.0), f.nodes.iter().map(|n| short(*n)).collect::<Vec<_>>().join(", "),
                f.rels.iter().map(|r| format!("{} → {} at t{}", short(r.a), short(r.b), r.off[0])).collect::<Vec<_>>().join("; "), if f.has_cycle() { " | a cycle" } else { "" }, if called_last.contains(&id) { " | a call in the last cycle" } else { "" });
        }
    }
    // the apex at the last frame: each call, its level and the frames it stands for
    let last = m.frame_no;
    println!("  calls on the apex at the last frame: {} | reaching frames {}-{}: {}", m.frames.iter().flat_map(|f| f.acts.iter()).filter(|a| a.on_apex() && !a.covers.is_empty()).count(), last - 5, last,
        m.frames.iter().flat_map(|f| f.acts.iter()).filter(|a| a.on_apex() && !a.covers.is_empty() && under(&m, a).iter().any(|x| x.1 >= last - 5)).count());
    for a in m.frames.iter().flat_map(|f| f.acts.iter()).filter(|a| a.on_apex() && !a.covers.is_empty()) {
        let u = under(&m, a);
        let frames: BTreeSet<i64> = u.iter().map(|x| x.1).collect();
        let chs: BTreeSet<DimId> = u.iter().map(|x| x.0).collect();
        println!("  apex: call of f{} | level {} | frames {:?} | channels {}", a.fired_by.as_ref().map_or(0, |f| f.func), a.level, frames, chs.len());
    }
    println!("  {:.1} s", t0.elapsed().as_secs_f64());
}
