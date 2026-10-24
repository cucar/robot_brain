//! Fit (D54), savings (D22) and the greedy cover of a row (D28): how a function is read against a row, and
//! what the election would write for a row from the table.

use crate::neuron::*;
use crate::types::*;

/// How one place of a function held in a row (D54).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Hold {
    /// A fixed node on the neighbor that stands there: covered.
    Covers(usize),
    /// A variable on the neighbor that stands there: its position covered, the neighbor bound (D10).
    Binds(usize),
    /// A fixed node held by a child that stands for its neuron there: credited nothing, not covered (D22).
    Under(usize),
}

/// One function measured against a row (D22).
#[derive(Clone, Debug)]
pub(crate) struct Measure {
    pub(crate) coverage: f64,
    pub(crate) price: f64,
    /// Per place besides the owner that held, how; a place missing here is an absence.
    pub(crate) holds: Vec<(usize, Hold)>,
}

impl Measure {
    pub(crate) fn savings(&self) -> f64 { self.coverage - self.price }
}

/// What is still uncovered in a row as a greedy cover runs (D28): the owner's activation, and per neighbor
/// its cost and the part of it that is its position.
#[derive(Clone, Debug)]
pub(crate) struct RowState {
    pub(crate) owner: f64,
    pub(crate) cost: Vec<f64>,
    pub(crate) pos: Vec<f64>,
}

impl RowState {
    pub(crate) fn of(row: &Row) -> Self {
        RowState { owner: row.own_cost, cost: row.nbs.iter().map(|nb| nb.cost).collect(), pos: row.nbs.iter().map(|nb| nb.pos).collect() }
    }

    /// Take what a measured function was paid for: the owner and what it covers leave, what it binds keeps
    /// its neuron (D28, D10).
    pub(crate) fn take(&mut self, m: &Measure) {
        self.owner = 0.0;
        for (_, h) in &m.holds {
            match h {
                Hold::Covers(j) => { self.cost[*j] = 0.0; self.pos[*j] = 0.0; }
                Hold::Binds(j) => { self.cost[*j] -= self.pos[*j]; self.pos[*j] = 0.0; }
                Hold::Under(_) => {}
            }
        }
    }

    pub(crate) fn left(&self) -> f64 { self.owner + self.cost.iter().sum::<f64>() }
}

impl Neuron {
    // ───────────────────────────── fit (D54) and savings (D22) ─────────────────────────────

    /// The neighbors standing in an offset's group (D6, D26).
    pub(crate) fn at(&self, row: &Row, off: &Offset) -> Vec<usize> {
        (0..row.nbs.len()).filter(|&j| self.round(&row.nbs[j].delta) == *off).collect()
    }

    /// Of several neighbors that could answer one place, the one not yet covered first, then the one nearest
    /// the owner, then the earlier coordinate (D31).
    fn pick(&self, row: &Row, cands: &[usize], state: Option<&RowState>) -> Option<usize> {
        cands.iter().copied().min_by(|&a, &b| {
            let free = |j: usize| state.map_or(true, |s| s.cost[j] > 0.0);
            let near = |j: usize| row.nbs[j].delta.iter().map(|x| x.abs()).sum::<i32>();
            free(b).cmp(&free(a)).then(near(a).cmp(&near(b))).then(row.nbs[a].delta.cmp(&row.nbs[b].delta))
        })
    }

    /// One function read against a row (D22, D54): each node holds or is an absence; the coverage is the
    /// owner's activation, what each fixed node that holds covers, and the position of each place a variable
    /// holds at, read from what is still uncovered where a greedy cover is running; the price is the call's
    /// record and an absence per node that did not hold, each at what a record cost when the row stood, as
    /// every neighbor in it is at what it cost then (D13).
    pub(crate) fn measure(&self, row: &Row, f: &Function, state: Option<&RowState>, _ctx: Ctx) -> Measure {
        let cost = |j: usize| state.map_or(row.nbs[j].cost, |s| s.cost[j]);
        let posb = |j: usize| state.map_or(row.nbs[j].pos, |s| s.pos[j]);
        let mut coverage = state.map_or(row.own_cost, |s| s.owner);
        let mut price = row.rec;
        let mut holds: Vec<(usize, Hold)> = Vec::new();
        let mut used: Vec<usize> = Vec::new();
        for (pi, (place, off)) in f.places.iter().enumerate().skip(1) {
            match place {
                Place::Owner | Place::Param(_) => {}
                Place::Fixed(n) => {
                    let direct: Vec<usize> = self.at(row, off).into_iter().filter(|&j| row.nbs[j].neuron == *n && !used.contains(&j)).collect();
                    if let Some(j) = self.pick(row, &direct, state) {
                        used.push(j);
                        coverage += cost(j);
                        holds.push((pi, Hold::Covers(j)));
                    } else if let Some(j) = (0..row.nbs.len()).find(|&j| row.nbs[j].stands_for.iter().any(|(m, d)| m == n && self.round(d) == *off)) {
                        holds.push((pi, Hold::Under(j)));
                    } else {
                        price += row.rec;
                    }
                }
                Place::Class(k) => {
                    let Some(c) = self.class(*k) else { continue };
                    let cands: Vec<usize> = self.at(row, off).into_iter().filter(|&j| c.has(row.nbs[j].neuron) && !used.contains(&j)).collect();
                    match self.pick(row, &cands, state) {
                        Some(j) => { used.push(j); coverage += posb(j); holds.push((pi, Hold::Binds(j))); }
                        None => { price += row.rec; }
                    }
                }
            }
        }
        // parameters: the value is the neuron at the earliest of their places (D44)
        for k in 0..f.params.len() as u32 {
            let mut places = f.param_places(k);
            places.sort_by(|a, b| f.places[*a].1.cmp(&f.places[*b].1));
            let value = places.iter().find_map(|&pi| {
                let cands: Vec<usize> = self.at(row, &f.places[pi].1).into_iter().filter(|j| !used.contains(j)).collect();
                self.pick(row, &cands, state).map(|j| row.nbs[j].neuron)
            });
            for &pi in &places {
                let cands: Vec<usize> = value.map_or(Vec::new(), |v| self.at(row, &f.places[pi].1).into_iter().filter(|&j| row.nbs[j].neuron == v && !used.contains(&j)).collect());
                match self.pick(row, &cands, state) {
                    Some(j) => { used.push(j); coverage += posb(j); holds.push((pi, Hold::Binds(j))); }
                    None => { price += row.rec; }
                }
            }
        }
        holds.sort_by_key(|(pi, _)| *pi);
        Measure { coverage, price, holds }
    }

    /// Whether a function fits a row: its savings there are positive (D54).
    pub(crate) fn fits(&self, row: &Row, f: &Function, ctx: Ctx) -> bool { self.measure(row, f, None, ctx).savings() > 0.0 }

    // ───────────────────────────── the cost of a row (D28, D30) ─────────────────────────────

    /// What the election would write for a row from a table (D30): the greedy cover (D28) of the row by the
    /// functions of it that fit the row, each taken one at its price there, and what none covers at its own
    /// cost. Ties go to the older function (D15).
    pub(crate) fn row_cost(&self, row: &Row, table: &[&Function], ctx: Ctx) -> f64 {
        let (paid, state) = self.cover_row(row, table, ctx);
        paid + state.left()
    }

    /// The greedy cover of a row by a table (D28): what the functions taken were paid, and what is left
    /// uncovered.
    pub(crate) fn cover_row(&self, row: &Row, table: &[&Function], ctx: Ctx) -> (f64, RowState) {
        let mut fitting: Vec<&Function> = table.iter().copied().filter(|f| self.fits(row, f, ctx)).collect();
        fitting.sort_by_key(|f| f.id);
        let mut state = RowState::of(row);
        let mut paid = 0.0;
        loop {
            let mut best: Option<(usize, Measure)> = None;
            for (x, f) in fitting.iter().enumerate() {
                let m = self.measure(row, f, Some(&state), ctx);
                if !(m.coverage > m.price) { continue; }
                let better = match &best {
                    None => true,
                    Some((_, b)) => m.coverage / m.price > b.coverage / b.price,
                };
                if better { best = Some((x, m)); }
            }
            let Some((x, m)) = best else { break };
            paid += m.price;
            state.take(&m);
            fitting.remove(x);
        }
        (paid, state)
    }
}
