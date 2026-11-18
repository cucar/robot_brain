//! Train and test on split data: stocks' daily open prices, an episode of training on one period, then a test on a
//! later period, frozen (the machine runs as trained and must change nothing of itself) or with learning left on.
//!
//! usage: split [episodes=1] [--symbols=KGC,GOLD,SPY] [--train=2015,2020] [--test=2020,2025]
//!              [--test-mode=frozen|learn] [--max-positions=1] [--cost=0] [--every=0] [--csv --dates=SO] [--coins=50]
//! (years: from the first up to, not including, the second)
//!
//! Prices come from apps/stocks/data/1D/<symbol>.json, on the dates every symbol has; with `--csv`, from the
//! <symbol>.csv files (which pad a stock's history before it starts), row for row on the dates of `--dates`'s JSON.
//! The trader is the main branch's (apps/stocks/trader.js), as the channels runner's seq mode has it: $15,000 at the
//! start of each period, each stock own (action 1) or out, at most `--max-positions` positions, the best by estimate,
//! then the higher price, then the symbol, priced under $5,000; an equal share each, what cash is left swept into the
//! cheapest; sells then buys, whole shares, `--cost` percent each side.
use brain_core::machine::Machine;
use brain_core::types::*;
use std::collections::{BTreeMap, BTreeSet};

fn opens_by_date(dir: &str, sym: &str) -> BTreeMap<String, f64> {
    let txt = std::fs::read_to_string(format!("{}/{}.json", dir, sym)).unwrap_or_else(|e| panic!("{}/{}.json: {}", dir, sym, e));
    let bars: Vec<serde_json::Value> = serde_json::from_str(&txt).unwrap();
    bars.iter().map(|b| (b["Timestamp"].as_str().unwrap()[..10].to_string(), b["OpenPrice"].as_f64().unwrap())).collect()
}

/// Everything a neuron or the dictionary holds, summed: if a test changes any of it, this changes.
fn fingerprint(m: &Machine) -> (usize, usize, usize, usize, usize, u64, u64, usize) {
    let mut d: Vec<String> = m.dict.funcs.values().map(|f| m.dict.describe(f)).collect();
    d.sort();
    let h = d.join(";").bytes().fold(1469598103934665603u64, |h, b| (h ^ b as u64).wrapping_mul(1099511628211));
    let (mut rows, mut conns, mut index) = (0usize, 0usize, 0usize);
    let mut cs: u64 = 0;
    for n in m.thalamus.iter() {
        rows += n.history.len();
        index += n.index.values().map(|v| *v as usize).sum::<usize>();
        conns += n.connections.len();
        for (k, c) in &n.connections { cs = cs.wrapping_add((c.strength.to_bits() ^ c.earnings.to_bits()).wrapping_mul(k.0 as u64 + 31 * k.1 as u64 + 1)); }
    }
    (m.alive, m.dict.len(), rows, conns, index, cs, h, m.dict.syms.len())
}

/// The trader (trader.js): one bar's plan and its trades at `price`, from each stock's action and estimate.
struct Trader { cash: f64, shares: Vec<i64>, trades: usize, max_positions: usize, mul: f64 }

impl Trader {
    fn new(n: usize, max_positions: usize, cost: f64) -> Self { Trader { cash: 15000.0, shares: vec![0; n], trades: 0, max_positions, mul: cost / 100.0 } }
    fn value(&self, price: &dyn Fn(usize) -> f64) -> f64 { self.cash + (0..self.shares.len()).map(|i| self.shares[i] as f64 * price(i)).sum::<f64>() }
    fn step(&mut self, acts: &[Option<u32>], est: &[f64], names: &[String], price: &dyn Fn(usize) -> f64) {
        let nch = acts.len();
        if !acts.iter().any(|a| a.is_some()) { return; }
        let total_value = self.value(price);
        let mut own: Vec<usize> = (0..nch).filter(|&i| acts[i] == Some(1) && price(i) > 0.0).collect();
        if own.len() > self.max_positions {
            own.sort_by(|&a, &b| est[b].exp().partial_cmp(&est[a].exp()).unwrap().then(price(b).partial_cmp(&price(a)).unwrap()).then(names[a].cmp(&names[b])));
            own.retain(|&i| price(i) < 5000.0);
            own.truncate(self.max_positions);
        }
        let mul = self.mul;
        let mut remaining = self.cash;
        let mut sells: Vec<(usize, i64)> = Vec::new();
        let mut buys: Vec<(usize, i64)> = Vec::new();
        let mut cheapest: Option<usize> = None;
        for i in 0..nch {
            if !own.contains(&i) {
                if self.shares[i] > 0 { sells.push((i, self.shares[i])); remaining += self.shares[i] as f64 * price(i) * (1.0 - mul); }
                continue;
            }
            if cheapest.map_or(true, |c| price(i) < price(c)) { cheapest = Some(i); }
            let buy_price = price(i) * (1.0 + mul);
            let target = ((total_value / own.len() as f64) / buy_price).floor() as i64;
            let diff = target - self.shares[i];
            if diff < 0 { sells.push((i, -diff)); remaining += (-diff) as f64 * price(i) * (1.0 - mul); }
            else if diff > 0 { buys.push((i, diff)); remaining -= diff as f64 * buy_price; }
        }
        if let Some(c) = cheapest {
            if remaining > 0.0 {
                let extra = (remaining / (price(c) * (1.0 + mul))).floor() as i64;
                if extra > 0 { if let Some(b) = buys.iter_mut().find(|b| b.0 == c) { b.1 += extra; } else { buys.push((c, extra)); } }
            }
        }
        for (i, n) in sells { self.cash += n as f64 * price(i) * (1.0 - mul); self.shares[i] -= n; self.trades += 1; }
        for (i, n) in buys {
            let eff = price(i) * (1.0 + mul);
            let n = n.min((self.cash / eff).floor() as i64);
            if n <= 0 { continue; }
            self.cash -= n as f64 * eff; self.shares[i] += n; self.trades += 1;
        }
    }
}

struct Result { right: Vec<(usize, usize)>, profit: f64, trades: usize, top: u32, marks: Vec<(usize, f64, usize, usize)> }

/// One pass over a period: every bar into the machine, its guess for the next bar graded, and the trader run on its
/// positions at the price each bar ended at.
fn run(m: &mut Machine, names: &[String], chans: &[(ChannelId, DimId, DimId)], prices: &[Vec<f64>], max_positions: usize, cost: f64, every: usize) -> Result {
    let n = prices[0].len() - 1;
    let nch = names.len();
    let mut pending: Vec<Option<u32>> = vec![None; nch];
    let mut last_act: Vec<Option<u32>> = vec![None; nch];
    let mut est = vec![0.0f64; nch];
    let mut right = vec![(0usize, 0usize); nch];
    let mut tr = Trader::new(nch, max_positions, cost);
    let mut top = 0u32;
    let mut marks = Vec::new();
    for k in 0..n {
        let mut inputs = Vec::new();
        let mut rewards = Vec::new();
        for i in 0..nch {
            let p = (prices[i][k + 1] - prices[i][k]) / prices[i][k] * 100.0;
            let up: u32 = if p > 0.0 { 1 } else { 0 };
            inputs.push(Input { dim: chans[i].1, bucket: up, pos: [0; MAX_DIMS] });
            // what the machine said for this bar runs; a lesson (the move itself) where it said nothing
            let action = pending[i].unwrap_or(up);
            inputs.push(Input { dim: chans[i].2, bucket: action, pos: [0; MAX_DIMS] });
            rewards.push(Reward { reward: if action == 1 { p } else { -p }, channels: vec![chans[i].0], frames: 1 });
        }
        let w = m.process_frame(&inputs, rewards);
        if let Some(f) = m.frames.back() { top = top.max(f.acts.iter().map(|a| a.level).max().unwrap_or(0)); }
        for i in 0..nch {
            let x = w.iter().find(|x| x.dim == chans[i].2);
            pending[i] = x.map(|x| x.bucket);
            est[i] = x.map_or(0.0, |x| x.estimate);
            if k + 1 < n {
                let next_up = if prices[i][k + 2] > prices[i][k + 1] { 1 } else { 0 };
                if let Some(g) = pending[i] { right[i].1 += 1; if g == next_up { right[i].0 += 1; } }
            }
        }
        for i in 0..nch { if pending[i].is_some() { last_act[i] = pending[i]; } }
        let price = |i: usize| prices[i][k + 1];
        tr.step(&last_act, &est, names, &price);
        if every > 0 && (k + 1) % every == 0 { marks.push((k + 1, tr.value(&price) - 15000.0, m.alive, m.dict.len())); }
    }
    let price = |i: usize| prices[i][n];
    Result { right, profit: tr.value(&price) - 15000.0, trades: tr.trades, top, marks }
}

/// The same trader with coin flips for the machine: each bar each stock own or out at even odds, and an estimate
/// drawn at random, so the choice among more owns than positions is random too.
fn coin(prices: &[Vec<f64>], names: &[String], max_positions: usize, cost: f64, seed: u64) -> f64 {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut next = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 33) as f64 / (1u64 << 31) as f64 };
    let n = prices[0].len() - 1;
    let nch = names.len();
    let mut tr = Trader::new(nch, max_positions, cost);
    for k in 0..n {
        let acts: Vec<Option<u32>> = (0..nch).map(|_| Some(if next() < 0.5 { 1 } else { 0 })).collect();
        let est: Vec<f64> = (0..nch).map(|_| next()).collect();
        let price = |i: usize| prices[i][k + 1];
        tr.step(&acts, &est, names, &price);
    }
    let price = |i: usize| prices[i][n];
    tr.value(&price) - 15000.0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let episodes: usize = args.get(1).filter(|a| !a.starts_with("--")).and_then(|s| s.parse().ok()).unwrap_or(1);
    let opt = |k: &str| args.iter().find(|a| a.starts_with(k)).map(|a| a[k.len()..].to_string());
    let syms: Vec<String> = opt("--symbols=").map_or(vec!["KGC".into(), "GOLD".into(), "SPY".into()], |s| s.split(',').map(|x| x.to_string()).collect());
    let years = |k: &str, d: (u32, u32)| opt(k).map_or(d, |s| { let v: Vec<u32> = s.split(',').map(|x| x.parse().unwrap()).collect(); (v[0], v[1]) });
    let (tr, te) = (years("--train=", (2015, 2020)), years("--test=", (2020, 2025)));
    let learn = opt("--test-mode=").map_or(false, |s| s == "learn");
    let max_positions: usize = opt("--max-positions=").map_or(1, |s| s.parse().unwrap());
    let cost: f64 = opt("--cost=").map_or(0.0, |s| s.parse().unwrap());
    let every: usize = opt("--every=").map_or(0, |s| s.parse().unwrap());
    let coins: u64 = opt("--coins=").map_or(50, |s| s.parse().unwrap());
    let dir = format!("{}/../../apps/stocks/data/1D", env!("CARGO_MANIFEST_DIR"));
    // the prices on each date
    let (dates, data): (Vec<String>, Vec<Vec<f64>>) = if args.iter().any(|a| a == "--csv") {
        let refsym = opt("--dates=").unwrap_or_else(|| syms[0].clone());
        let dates: Vec<String> = opens_by_date(&dir, &refsym).into_keys().collect();
        let data: Vec<Vec<f64>> = syms.iter().map(|s| {
            let txt = std::fs::read_to_string(format!("{}/{}.csv", dir, s)).unwrap();
            let p: Vec<f64> = txt.lines().filter(|l| !l.trim().is_empty()).map(|l| l.split(',').next().unwrap().parse().unwrap()).collect();
            assert_eq!(p.len(), dates.len(), "{}.csv has {} rows, {}.json {} dates", s, p.len(), refsym, dates.len());
            p
        }).collect();
        (dates, data)
    } else {
        let by: Vec<BTreeMap<String, f64>> = syms.iter().map(|s| opens_by_date(&dir, s)).collect();
        let mut ds: BTreeSet<String> = by[0].keys().cloned().collect();
        for d in &by[1..] { ds.retain(|x| d.contains_key(x)); }
        let dates: Vec<String> = ds.into_iter().collect();
        let data = by.iter().map(|d| dates.iter().map(|x| d[x]).collect()).collect();
        (dates, data)
    };
    let period = |(a, b): (u32, u32)| -> (Vec<Vec<f64>>, String, String) {
        let ix: Vec<usize> = (0..dates.len()).filter(|&i| { let y: u32 = dates[i][..4].parse().unwrap(); y >= a && y < b }).collect();
        (data.iter().map(|d| ix.iter().map(|&i| d[i]).collect()).collect(), dates[ix[0]].clone(), dates[ix[ix.len() - 1]].clone())
    };
    let (train, t0, t1) = period(tr);
    let (test, s0, s1) = period(te);
    println!("split: {} | train {} to {} ({} bars) | test {} to {} ({} bars) | {} episodes | test {} | trader: at most {} positions, cost {}% each side",
        syms.join(","), t0, t1, train[0].len() - 1, s0, s1, test[0].len() - 1, episodes, if learn { "with learning on" } else { "frozen" }, max_positions, cost);

    let mut m = Machine::new(256, Policy::events_to_actions(), 1);
    let mut chans: Vec<(ChannelId, DimId, DimId)> = Vec::new();
    for s in &syms {
        let ch = m.declare_channel(s, Some(("move", 2)), Some(("position", 2)), &[]);
        chans.push((ch, m.channels[ch as usize].event.unwrap(), m.channels[ch as usize].action.unwrap()));
    }
    let show = |label: &str, ep: usize, r: &Result, m: &Machine, secs: f64| {
        let accs: Vec<String> = syms.iter().zip(&r.right).map(|(s, (a, b))| format!("{} {:.2}%", s, 100.0 * *a as f64 / (*b).max(1) as f64)).collect();
        let (a, b) = r.right.iter().fold((0, 0), |(a, b), x| (a + x.0, b + x.1));
        println!("{} {} | neurons {} | functions {} | never called {} | highest level {} | right {} | average {:.2}% | net profit ${:.2}, ROI {:.2}% ({} trades) | {:.0}s",
            label, ep, m.alive, m.dict.len(), m.funcs_never_called(), r.top, accs.join(", "), 100.0 * a as f64 / b.max(1) as f64, r.profit, r.profit / 150.0, r.trades, secs);
    };
    for ep in 1..=episodes {
        // training: the machine learns
        m.frozen = false;
        m.reset_context();
        let t = std::time::Instant::now();
        let r = run(&mut m, &syms, &chans, &train, max_positions, cost, 0);
        show("TRAIN", ep, &r, &m, t.elapsed().as_secs_f64());
        let before = fingerprint(&m);
        println!("      trained machine {:?} | poses {}", before, m.dict.syms.len());
        // the test, from the trained machine as it stands
        m.frozen = !learn;
        m.reset_context();
        let t = std::time::Instant::now();
        let r = run(&mut m, &syms, &chans, &test, max_positions, cost, every);
        let after = fingerprint(&m);
        for (k, p, n, f) in &r.marks { println!("      test frame {} | net profit ${:.2} | neurons {} | functions {}", k, p, n, f); }
        show(if learn { "LEARN" } else { "TEST " }, ep, &r, &m, t.elapsed().as_secs_f64());
        if !learn {
            println!("      read-only check: {} (neurons, functions, history rows, connections, index, connection sum, dictionary, pose table) {:?}{} | poses named while frozen, kept apart {}",
                if before == after { "unchanged" } else { "CHANGED" }, before, if before == after { String::new() } else { format!(" -> {:?}", after) }, m.dict.frozen_syms.len());
        }
    }
    // beside it, over the test period: equal-weight buy-and-hold from the first price the trader trades at, and the
    // coin-flip trader under the same rules
    let n = test[0].len() - 1;
    let bh: f64 = test.iter().map(|p| 15000.0 / syms.len() as f64 * p[n] / p[1]).sum::<f64>() - 15000.0;
    let mut c: Vec<f64> = (1..=coins).map(|s| coin(&test, &syms, max_positions, cost, s)).collect();
    c.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| c[((c.len() - 1) as f64 * f).round() as usize];
    println!("BASELINE buy-and-hold ${:.2} ({:.2}%) | coin flip, {} seeds: median ${:.2}, middle half ${:.2} to ${:.2}, best ${:.2}, worst ${:.2}",
        bh, bh / 150.0, coins, q(0.5), q(0.25), q(0.75), c[c.len() - 1], c[0]);
}
