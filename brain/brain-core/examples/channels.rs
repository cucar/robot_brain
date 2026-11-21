/// Several channels in one machine, no reset (docs/algorithm.md D34, D24). One channel per stock: the price's move up or
/// down as the event, own or out as the action, the reward scoped to that stock's channel and the frame; every stock
/// fires in every frame. Neighbors by kind (Policy::events_to_actions): an event reads every event and call of any
/// channel in its frame and the one before; no action is anyone's neighbor.
///
/// Modes:
///   two    KGC and GLD, the first 12 daily opens of each (apps/stocks/data/1D), cycled
///   three  KGC, GLD and SPY, the same way
///   three4 KGC, GLD and SPY, the same way, and Q, made up, moving one percent a bar
///   lap    two stocks on six-bar cycles, each with its action: A the stocks example's lap, +1 −2 −3 +2 +3 −1, and
///          B +2 +1 −1 −3 +3 −2
///   lap65  A on its six-bar lap and B on a five-bar lap +2 +1 −1 +3 −2: together they repeat every 30 frames
///   seq    stocks over their whole daily history as the CSVs hold it, one lap the whole sequence, with a trader
///
/// usage: cargo run --release --example channels -- <two|three|three4|lap|lap65|seq> [cycles] [data-dir]
///        [--all4] [--symbols=A,B] [--max-positions=N] [--cost=PCT] [--every=N] [--regions=R] [--columns=C]
use brain_core::board::Act;
use brain_core::brain::Brain;
use brain_core::types::*;
use std::collections::BTreeSet;

/// The first 12 daily opens of a symbol, as apps/stocks/jobs/multi-channel-test.js loads them.
fn read_opens(dir: &str, sym: &str) -> Vec<f64> {
    let txt = std::fs::read_to_string(format!("{}/{}.json", dir, sym)).unwrap_or_else(|e| panic!("{}/{}.json: {}", dir, sym, e));
    let bars: Vec<serde_json::Value> = serde_json::from_str(&txt).unwrap();
    bars.iter().take(12).map(|b| b["OpenPrice"].as_f64().unwrap()).collect()
}

/// The events a call stands for, at any depth: (event dimension, frame).
fn list_under(m: &Brain, a: &Act) -> BTreeSet<(DimensionId, i64)> {
    let mut out = BTreeSet::new();
    let mut stack = a.covers.clone();
    while let Some(r) = stack.pop() {
        let Some(c) = m.find_act(r) else { continue };
        if c.covers.is_empty() { if c.kind == Kind::Event { out.insert((m.get_registry()[c.neuron].dim.unwrap(), r.frame)); } } else { stack.extend(c.covers.iter().copied()); }
    }
    out
}

fn main() {
    // `--regions=R --columns=C` lay out the dispatch, one region of one column by default
    let flag = |name: &str| std::env::args().find_map(|a| a.strip_prefix(name).and_then(|v| v.parse::<usize>().ok())).unwrap_or(1);
    let (regions, columns) = (flag("--regions="), flag("--columns="));
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).cloned().unwrap_or("two".into());
    let dir = args.get(3).filter(|a| !a.starts_with("--")).cloned().unwrap_or(format!("{}/../../apps/stocks/data/1D", env!("CARGO_MANIFEST_DIR")));
    // each channel: its name, the percent move of each bar of its cycle, and whether it has an action
    let mut series: Vec<(String, Vec<f64>, bool)> = Vec::new();
    // seq mode: each stock's prices, for the trader
    let mut prices: Vec<Vec<f64>> = Vec::new();
    let (cycle, default_cycles) = match mode.as_str() {
        "two" | "three" => {
            let syms: &[&str] = if mode == "two" { &["KGC", "GLD"] } else { &["KGC", "GLD", "SPY"] };
            for s in syms {
                let p = read_opens(&dir, s);
                // bar i is the move from row i to row i+1; the last wraps to the first row
                series.push((s.to_string(), (0..12).map(|i| (p[(i + 1) % 12] - p[i]) / p[i] * 100.0).collect(), true));
            }
            (12usize, 10usize)
        }
        "lap" => {
            series.push(("A".into(), vec![1.0, -2.0, -3.0, 2.0, 3.0, -1.0], true));
            series.push(("B".into(), vec![2.0, 1.0, -1.0, -3.0, 3.0, -2.0], true));
            (6usize, 30usize)
        }
        // A on its six-bar lap, B on a five-bar lap UUDUD: together they repeat every 30 frames
        "lap65" => {
            series.push(("A".into(), vec![1.0, -2.0, -3.0, 2.0, 3.0, -1.0], true));
            series.push(("B".into(), vec![2.0, 1.0, -1.0, 3.0, -2.0], true));
            (30usize, 12usize)
        }
        // four stocks at one position: KGC, GLD and SPY on the 12-bar lap, and Q, made up, moving one percent a bar
        "three4" => {
            for s in ["KGC", "GLD", "SPY"] {
                let p = read_opens(&dir, s);
                series.push((s.to_string(), (0..12).map(|i| (p[(i + 1) % 12] - p[i]) / p[i] * 100.0).collect(), true));
            }
            series.push(("Q".into(), "UDDUUDUDDUDD".chars().map(|c| if c == 'U' { 1.0 } else { -1.0 }).collect(), true));
            (12usize, 10usize)
        }
        // the stock sequence memorization of the main branch's demos: KGC, GOLD and SPY over their whole daily
        // history as the CSVs hold it, one lap the whole sequence, one episode a lap
        "seq" => {
            let syms: Vec<String> = args.iter().find(|a| a.starts_with("--symbols=")).map_or(vec!["KGC".into(), "GOLD".into(), "SPY".into()], |a| a[10..].split(',').map(|x| x.to_string()).collect());
            for s in &syms {
                let s = s.as_str();
                let txt = std::fs::read_to_string(format!("{}/{}.csv", dir, s)).unwrap();
                let p: Vec<f64> = txt.lines().filter(|l| !l.trim().is_empty()).map(|l| l.split(',').next().unwrap().parse().unwrap()).collect();
                let n = p.len();
                series.push((s.to_string(), (0..n - 1).map(|i| (p[i + 1] - p[i]) / p[i] * 100.0).collect(), true));
                prices.push(p[..n].to_vec());
            }
            let n = series[0].1.len();
            (n, 5usize)
        }
        _ => panic!("mode: two, three, three4, lap, lap65 or seq"),
    };
    let cycles: usize = args.get(2).filter(|a| !a.starts_with("--")).and_then(|s| s.parse().ok()).unwrap_or(default_cycles);
    let mut m = Brain::new(256, Policy::events_to_actions(), regions, columns, true);
    let mut chans: Vec<(ChannelId, DimensionId, Option<DimensionId>)> = Vec::new();
    for (name, _, act) in &series {
        let ch = m.declare_channel(name, Some(("move", 2)), if *act { Some(("position", 2)) } else { None }, &[]);
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
    let mut under_l3: Vec<(DimensionId, Option<i64>)> = chans.iter().map(|c| (c.1, None)).collect();
    let (mut last_dict, mut last_change) = (String::new(), 0i64);
    let t0 = std::time::Instant::now();
    let all4 = args.iter().any(|a| a == "--all4");
    // the main branch's trader (apps/stocks/trader.js) in seq mode: $15,000 a episode, each stock own (action 1) or
    // out, at most one position, the own with the highest estimate (then the higher price, then the symbol), every
    // dollar in it in whole shares, no transaction cost; profit is cash and holdings less the start
    let nch = series.len();
    let (mut cash, mut shares, mut trades) = (15000.0f64, vec![0i64; nch], 0usize);
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
            inputs.push(Input { dim: chans[i].1, bucket: up, pos: [0; MAX_DIMS] });
            if let Some(ac) = chans[i].2 {
                // what the machine said for this bar runs; a lesson (the move itself) where it said nothing
                let action = pending[i].unwrap_or(up);
                inputs.push(Input { dim: ac, bucket: action, pos: [0; MAX_DIMS] });
                rewards.push(Reward { reward: if action == 1 { p } else { -p }, channels: vec![chans[i].0], frames: 1 });
            }
        }
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
                    k + 1, m.alive, m.get_dictionary().len(), 100.0 * r100 as f64 / n100.max(1) as f64, 100.0 * rall as f64 / nall.max(1) as f64, profit, trades, t0.elapsed().as_secs_f64());
            }
        }
        // recognition: the dictionary's last change, and this frame's combination calls (a level-1 call over two of
        // this frame's bars)
        let dict = m.get_dictionary();
        let mut d: Vec<String> = dict.funcs.values().map(|f| dict.describe(f)).collect();
        d.sort();
        let d = d.join("; ");
        if d != last_dict { last_change = fno; last_dict = d; }
        let board = m.get_board();
        if k % cycle == cycle - 1 {
            let top = board.iter().filter(|f| f.no > fno - cycle as i64).flat_map(|f| f.acts.iter()).map(|a| a.level).max().unwrap_or(0);
            let accs: Vec<String> = series.iter().enumerate().filter(|(_, x)| x.2).map(|(i, x)| { let row = &grid[i][k / cycle]; let (r, w) = row.iter().fold((0, 0), |(r, w), c| (r + (*c == 'R') as usize, w + (*c == 'W') as usize)); format!("{} {:.2}%", x.0, 100.0 * r as f64 / (r + w).max(1) as f64) }).collect();
            let profit = if prices.is_empty() { 0.0 } else { cash + (0..nch).map(|i| shares[i] as f64 * prices[i][(k % cycle) + 1]).sum::<f64>() - 15000.0 };
            println!("CYCLE {} | neurons {} | functions {} | highest call level in the cycle {} | the dictionary last changed at frame {} | right {}{}", k / cycle + 1, m.alive, dict.len(), top, last_change, accs.join(", "),
                if prices.is_empty() { String::new() } else { format!(" | net profit ${:.2}, ROI {:.2}% ({} trades)", profit, profit / 15000.0 * 100.0, trades) });
        }
        let fr = board.back().unwrap();
        if all4 {
            // the calls standing in this frame over every bar of this frame
            if let Some(fr) = board.iter().find(|f| f.no == fno) {
                let nb = fr.acts.iter().filter(|a| a.kind == Kind::Event).count();
                let parts: Vec<String> = fr.acts.iter().filter(|a| !a.covers.is_empty() && !a.gone).filter(|a| list_under(&m, a).iter().filter(|x| x.1 == fno).count() == nb).map(|a| format!("L{} f{}", a.level, a.fired_by.as_ref().map_or(0, |fb| fb.func))).collect();
                println!("ALL4 {} | {}", fno, parts.join("; "));
            }
        }
        let combo = fr.acts.iter().filter(|a| a.level == 1 && a.fired_by.is_some() && a.covers.iter().filter(|c| c.frame == fno && m.find_act(**c).map_or(false, |x| x.kind == Kind::Event)).count() >= 2).count();
        combos[k / cycle].push(combo);
        // the laps: whether one call on the apex stands for each channel's six bars of this lap, and for all of them
        if mode.starts_with("lap") && k % cycle == cycle - 1 {
            let lap: Vec<i64> = ((fno - 5)..=fno).collect();
            let sets: Vec<Vec<DimensionId>> = chans.iter().map(|c| vec![c.1]).chain(std::iter::once(chans.iter().map(|c| c.1).collect())).collect();
            for (j, set) in sets.iter().enumerate() {
                let stands = board.iter().flat_map(|f| f.acts.iter()).any(|a| a.on_apex() && !a.covers.is_empty() && { let u = list_under(&m, a); set.iter().all(|e| lap.iter().all(|f| u.contains(&(*e, *f)))) });
                if stands { six_laps[j].push(k / cycle + 1); }
            }
        }
        if k / cycle == cycles - 1 { for a in fr.acts.iter() { if let Some(fb) = &a.fired_by { called_last.insert(fb.func as u32); } } }
        // the frame at which bar 6 of the second-last cycle first comes under a level-3 call
        if mode.starts_with("lap") && cycles >= 2 {
            let target = (cycle * (cycles - 1)) as i64;
            if fno >= target {
                for u in under_l3.iter_mut() {
                    if u.1.is_none() && board.iter().flat_map(|f| f.acts.iter()).any(|a| a.level == 3 && list_under(&m, a).contains(&(u.0, target))) { u.1 = Some(fno); }
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
    let dict = m.get_dictionary();
    println!("recognition: {} functions | the dictionary last changed at frame {}", dict.len(), last_change);
    println!("NEVERCALLED {} of {}", m.count_functions_never_called(), dict.len());
    println!("  combination calls per frame, per cycle: {}", combos.iter().enumerate().map(|(c, v)| format!("{}:{}", c + 1, v.iter().map(|x| x.to_string()).collect::<String>())).collect::<Vec<_>>().join(" "));
    if mode.starts_with("lap") {
        let show = |v: &Vec<usize>| if v.is_empty() { "never".to_string() } else { format!("from lap {} ({} of {} laps)", v[0], v.len(), cycles) };
        for (j, (name, _, _)) in series.iter().enumerate() { println!("  one call on the apex stands for {}'s six bars of a lap: {}", name, show(&six_laps[j])); }
        if series.len() > 1 { println!("  one call on the apex stands for every channel's six bars of a lap: {}", show(&six_laps[series.len()])); }
        for (j, u) in under_l3.iter().enumerate() { println!("  {}: bar 6 of cycle {} (frame {}) first comes under a level-3 call at frame {}", series[j].0, cycles - 1, cycle * (cycles - 1), u.1.map_or("never".into(), |f| f.to_string())); }
    }
    // the dictionary at the end, by level and kind, with how many had a call in the last cycle
    {
        let reg = m.get_registry();
        let names: std::collections::BTreeMap<DimensionId, String> = chans.iter().zip(series.iter()).flat_map(|(c, s)| [(c.1, s.0.clone())].into_iter().chain(c.2.map(|a| (a, s.0.clone())))).collect();
        let func_of = |s: Sym| -> Option<u32> { let p = dict.find_pose(s)?; let fs = dict.list_of_inter(dict.find_neuron_of(s)); fs.iter().find(|f| dict.get(**f).map_or(false, |g| g.find_node(p.from).is_some())).or(fs.first()).map(|f| *f as u32) };
        let assoc_bars = |id: u32| dict.get(id as _).map_or(false, |f| f.rels.iter().any(|r| r.off[0] == 0 && dict.find_pose(r.a).is_none() && dict.find_pose(r.b).is_none()));
        // each function's level and channels, read once per function (a function's nodes may name itself)
        fn read_info(m: &Brain, id: u32, f_of: &dyn Fn(Sym) -> Option<u32>, memo: &mut std::collections::BTreeMap<u32, (usize, BTreeSet<DimensionId>)>, open: &mut BTreeSet<u32>) -> (usize, BTreeSet<DimensionId>) {
            if let Some(x) = memo.get(&id) { return x.clone(); }
            if !open.insert(id) { return (0, BTreeSet::new()); }
            let mut lv = 0;
            let mut ch = BTreeSet::new();
            let dict = m.get_dictionary();
            if let Some(f) = dict.get(id as _) {
                for n in &f.nodes {
                    if dict.find_pose(*n).is_some() { if let Some(g) = f_of(*n) { let (l, c) = read_info(m, g, f_of, memo, open); lv = lv.max(l); ch.extend(c); } }
                    else if let Some(d) = m.get_registry()[*n].dim { ch.insert(d); }
                }
            }
            open.remove(&id);
            memo.insert(id, (lv + 1, ch.clone()));
            (lv + 1, ch)
        }
        let mut memo = Default::default();
        let mut table: std::collections::BTreeMap<(usize, String), (usize, usize)> = Default::default();
        for (id, f) in dict.funcs.iter() {
            let id = *id as u32;
            let (lv, ch) = read_info(&m, id, &func_of, &mut memo, &mut BTreeSet::new());
            let kind = if assoc_bars(id) { "association between bars".to_string() }
                else if f.nodes.iter().any(|n| dict.find_pose(*n).is_some() && func_of(*n).map_or(false, |g| assoc_bars(g))) { "over the combination calls".to_string() }
                else if ch.len() == 1 { format!("{} only", names.get(ch.iter().next().unwrap()).cloned().unwrap_or_default()) }
                else { "across the channels in time".to_string() };
            let e = table.entry((lv, kind)).or_default();
            e.0 += 1;
            if called_last.contains(&id) { e.1 += 1; }
        }
        for ((lv, kind), (n, c)) in &table { println!("  dictionary: level {} | {} | {} | with a call in the last cycle {}", lv, kind, n, c); }
        let short = |x: Sym| -> String {
            if dict.find_pose(x).is_some() { return format!("call(f{})", func_of(x).map_or(-1, |g| g as i64)); }
            let n = &reg[x];
            let nm = n.dim.and_then(|d| names.get(&d)).cloned().unwrap_or_default();
            match n.kind { Kind::Event => format!("{}{}", nm, if n.bucket == 1 { "↑" } else { "↓" }), _ => format!("{}:{}", nm, if n.bucket == 1 { "own" } else { "out" }) }
        };
        let mut ids: Vec<u32> = dict.funcs.keys().map(|x| *x as u32).collect();
        ids.sort_by_key(|id| (memo.get(id).map_or(0, |x: &(usize, BTreeSet<DimensionId>)| x.0), *id));
        for id in ids {
            let f = dict.get(id as _).unwrap();
            println!("  function f{} | level {} | nodes {} | relations {}{}{}", id, memo.get(&id).map_or(0, |x| x.0), f.nodes.iter().map(|n| short(*n)).collect::<Vec<_>>().join(", "),
                f.rels.iter().map(|r| format!("{} → {} at t{}", short(r.a), short(r.b), r.off[0])).collect::<Vec<_>>().join("; "), if f.has_cycle() { " | a cycle" } else { "" }, if called_last.contains(&id) { " | a call in the last cycle" } else { "" });
        }
    }
    // the apex at the last frame: each call, its level and the frames it stands for
    let last = m.frame_no;
    let board = m.get_board();
    println!("  calls on the apex at the last frame: {} | reaching frames {}-{}: {}", board.iter().flat_map(|f| f.acts.iter()).filter(|a| a.on_apex() && !a.covers.is_empty()).count(), last - 5, last,
        board.iter().flat_map(|f| f.acts.iter()).filter(|a| a.on_apex() && !a.covers.is_empty() && list_under(&m, a).iter().any(|x| x.1 >= last - 5)).count());
    for a in board.iter().flat_map(|f| f.acts.iter()).filter(|a| a.on_apex() && !a.covers.is_empty()) {
        let u = list_under(&m, a);
        let frames: BTreeSet<i64> = u.iter().map(|x| x.1).collect();
        let chs: BTreeSet<DimensionId> = u.iter().map(|x| x.0).collect();
        println!("  apex: call of f{} | level {} | frames {:?} | channels {}", a.fired_by.as_ref().map_or(0, |f| f.func), a.level, frames, chs.len());
    }
    println!("  {:.1} s", t0.elapsed().as_secs_f64());
}
