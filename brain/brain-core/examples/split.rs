//! Train and test on split data: stocks' daily open prices, an episode of training on one period, then a test on a
//! later period with the machine frozen, which must change nothing of itself; then another training episode and test.
//!
//! usage: split [episodes=2] [--symbols=KGC,GOLD,SPY] [--train=2015,2020] [--test=2020,2025]
//! (years: from the first up to, not including, the second)
use brain_core::machine::Machine;
use brain_core::types::*;
use std::collections::{BTreeMap, BTreeSet};

fn opens_by_date(dir: &str, sym: &str) -> BTreeMap<String, f64> {
    let txt = std::fs::read_to_string(format!("{}/{}.json", dir, sym)).unwrap_or_else(|e| panic!("{}/{}.json: {}", dir, sym, e));
    let bars: Vec<serde_json::Value> = serde_json::from_str(&txt).unwrap();
    bars.iter().map(|b| (b["Timestamp"].as_str().unwrap()[..10].to_string(), b["OpenPrice"].as_f64().unwrap())).collect()
}

/// Everything a neuron or the dictionary holds, summed: if a test changes any of it, this changes.
fn fingerprint(m: &Machine) -> (usize, usize, usize, usize, usize, u64, u64) {
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
    (m.alive, m.dict.len(), rows, conns, index, cs, h)
}

struct Result { right: Vec<(usize, usize)>, profit: f64, trades: usize }

/// One pass over a period: every bar into the machine, its guess for the next bar graded, and the main branch's
/// trader (apps/stocks/trader.js) run on its positions: $15,000, own (1) or out, at most one position, the own with
/// the highest estimate, then the higher price, then the symbol, whole shares, no transaction cost.
fn run(m: &mut Machine, names: &[String], chans: &[(ChannelId, DimId, DimId)], prices: &[Vec<f64>]) -> Result {
    let n = prices[0].len() - 1;
    let nch = names.len();
    let mut pending: Vec<Option<u32>> = vec![None; nch];
    let mut last_act: Vec<Option<u32>> = vec![None; nch];
    let mut est = vec![0.0f64; nch];
    let mut right = vec![(0usize, 0usize); nch];
    let (mut cash, mut shares, mut trades) = (15000.0f64, vec![0i64; nch], 0usize);
    for k in 0..n {
        let mut inputs = Vec::new();
        let mut rewards = Vec::new();
        for i in 0..nch {
            let p = (prices[i][k + 1] - prices[i][k]) / prices[i][k] * 100.0;
            let up: u32 = if p > 0.0 { 1 } else { 0 };
            inputs.push(Input { dim: chans[i].1, bucket: up, pos: [0; MAX_DIMS] });
            let action = pending[i].unwrap_or(up);
            inputs.push(Input { dim: chans[i].2, bucket: action, pos: [0; MAX_DIMS] });
            rewards.push(Reward { reward: if action == 1 { p } else { -p }, channels: vec![chans[i].0], frames: 1 });
        }
        let w = m.process_frame(&inputs, rewards);
        for i in 0..nch {
            let x = w.iter().find(|x| x.dim == chans[i].2);
            pending[i] = x.map(|x| x.bucket);
            est[i] = x.map_or(0.0, |x| x.estimate);
            if k + 1 < n {
                let next_up = if prices[i][k + 2] > prices[i][k + 1] { 1 } else { 0 };
                if let Some(g) = pending[i] { right[i].1 += 1; if g == next_up { right[i].0 += 1; } }
            }
        }
        // the trader, at the price this bar ended at
        for i in 0..nch { if pending[i].is_some() { last_act[i] = pending[i]; } }
        if last_act.iter().any(|a| a.is_some()) {
            let price = |i: usize| prices[i][k + 1];
            let total = cash + (0..nch).map(|i| shares[i] as f64 * price(i)).sum::<f64>();
            let mut own: Vec<usize> = (0..nch).filter(|&i| last_act[i] == Some(1) && price(i) > 0.0).collect();
            own.sort_by(|&a, &b| est[b].exp().partial_cmp(&est[a].exp()).unwrap().then(price(b).partial_cmp(&price(a)).unwrap()).then(names[a].cmp(&names[b])));
            own.truncate(1);
            for i in 0..nch {
                if own.contains(&i) || shares[i] == 0 { continue; }
                cash += shares[i] as f64 * price(i);
                shares[i] = 0;
                trades += 1;
            }
            if let Some(&c) = own.first() {
                let target = (total / price(c)).floor() as i64;
                let diff = target - shares[c];
                if diff != 0 { cash -= diff as f64 * price(c); shares[c] = target; trades += 1; }
            }
        }
    }
    let profit = cash + (0..nch).map(|i| shares[i] as f64 * prices[i][n]).sum::<f64>() - 15000.0;
    Result { right, profit, trades }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let episodes: usize = args.get(1).filter(|a| !a.starts_with("--")).and_then(|s| s.parse().ok()).unwrap_or(2);
    let opt = |k: &str| args.iter().find(|a| a.starts_with(k)).map(|a| a[k.len()..].to_string());
    let syms: Vec<String> = opt("--symbols=").map_or(vec!["KGC".into(), "GOLD".into(), "SPY".into()], |s| s.split(',').map(|x| x.to_string()).collect());
    let years = |k: &str, d: (u32, u32)| opt(k).map_or(d, |s| { let v: Vec<u32> = s.split(',').map(|x| x.parse().unwrap()).collect(); (v[0], v[1]) });
    let (tr, te) = (years("--train=", (2015, 2020)), years("--test=", (2020, 2025)));
    let dir = format!("{}/../../apps/stocks/data/1D", env!("CARGO_MANIFEST_DIR"));
    let data: Vec<BTreeMap<String, f64>> = syms.iter().map(|s| opens_by_date(&dir, s)).collect();
    let mut dates: BTreeSet<String> = data[0].keys().cloned().collect();
    for d in &data[1..] { dates.retain(|x| d.contains_key(x)); }
    let period = |(a, b): (u32, u32)| -> (Vec<Vec<f64>>, String, String) {
        let ds: Vec<&String> = dates.iter().filter(|x| x[..4].parse::<u32>().unwrap() >= a && x[..4].parse::<u32>().unwrap() < b).collect();
        (data.iter().map(|d| ds.iter().map(|x| d[*x]).collect()).collect(), ds[0].clone(), ds[ds.len() - 1].clone())
    };
    let (train, t0, t1) = period(tr);
    let (test, s0, s1) = period(te);
    println!("split: {} | train {} to {} ({} bars) | test {} to {} ({} bars) | {} episodes", syms.join(","), t0, t1, train[0].len() - 1, s0, s1, test[0].len() - 1, episodes);

    let mut m = Machine::new(256, Policy::events_to_actions(), 1);
    let mut chans: Vec<(ChannelId, DimId, DimId)> = Vec::new();
    for s in &syms {
        let ch = m.declare_channel(s, Some(("move", 2)), Some(("position", 2)), &[]);
        chans.push((ch, m.channels[ch as usize].event.unwrap(), m.channels[ch as usize].action.unwrap()));
    }
    let show = |label: &str, ep: usize, r: &Result, m: &Machine, secs: f64| {
        let accs: Vec<String> = syms.iter().zip(&r.right).map(|(s, (a, b))| format!("{} {:.2}%", s, 100.0 * *a as f64 / (*b).max(1) as f64)).collect();
        let (a, b) = r.right.iter().fold((0, 0), |(a, b), x| (a + x.0, b + x.1));
        println!("{} {} | neurons {} | functions {} | right {} | average {:.2}% | net profit ${:.2}, ROI {:.2}% ({} trades) | {:.0}s",
            label, ep, m.alive, m.dict.len(), accs.join(", "), 100.0 * a as f64 / b.max(1) as f64, r.profit, r.profit / 150.0, r.trades, secs);
    };
    for ep in 1..=episodes {
        // training: the machine learns
        m.frozen = false;
        m.reset_context();
        let t = std::time::Instant::now();
        let r = run(&mut m, &syms, &chans, &train);
        show("TRAIN", ep, &r, &m, t.elapsed().as_secs_f64());
        // the test: frozen, it changes nothing of itself; checked by everything its neurons and dictionary hold
        let before = fingerprint(&m);
        m.frozen = true;
        let t = std::time::Instant::now();
        let r = run(&mut m, &syms, &chans, &test);
        let after = fingerprint(&m);
        show("TEST ", ep, &r, &m, t.elapsed().as_secs_f64());
        println!("      read-only check: {} (neurons, functions, history rows, connections, index, connection sum, dictionary) {:?}{}",
            if before == after { "unchanged" } else { "CHANGED" }, before, if before == after { String::new() } else { format!(" -> {:?}", after) });
    }
}
