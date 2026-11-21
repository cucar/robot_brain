/// One neuron (§4, §5): a history of its activations, a relation index over it, and its connections.
///
/// It records, counts and returns; it decides nothing about what is covered.
/// Every call reaches it through a batch its column runs: activate a level, take lifted relations, connect, infer,
/// forget.

use std::collections::VecDeque;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::types::*;

pub use crate::board::ActRef;

/// A relation as one neuron records it from its own activation (D47, D34): its own pose, the neighbor's pose, and
/// the offset.
pub type RelKey = (Sym, Sym, Offset);

/// One neighbor of a row (D26): the activation that stands there, its pose, and its exact offset from the row's
/// activation (D6).
#[derive(Clone, Copy, Debug)]
pub struct Nb {
    pub act: ActRef,
    pub sym: Sym,
    pub off: Offset,
}

/// One row of the history (D18): an activation, its pose, and the neighborhood it observed, with what followed it
/// as it comes.
#[derive(Clone, Debug)]
pub struct Row {
    pub act: ActRef,
    pub own: Sym,
    pub nbs: Vec<Nb>,
}

/// What the machine hands a neuron for one activation of the level (R20).
#[derive(Clone, Debug)]
pub struct RowInput {
    pub act: ActRef,
    pub own: Sym,
    pub nbs: Vec<Nb>,
}

/// What the call returns for one activation (R21): each neighbor, with the strength of its relation.
#[derive(Clone, Debug)]
pub struct ActReturn {
    pub act: ActRef,
    pub rels: Vec<(Nb, u32)>,
}

/// A call a row takes as a neighbor where a relation of its activation is lifted to it (D10, D55).
///
/// The machine resolves laps (D53), so it names every record that stands for the row and for the call.
/// `row` is the activation whose row takes the call; `row_records` the records of the row's frame that a lap
/// resolves to the same call as `row`.
/// `call_records` the records that resolve to the same call as `call.act`.
#[derive(Clone, Debug)]
pub struct LiftTask {
    pub row: ActRef,
    pub row_records: Vec<ActRef>,
    pub call_records: Vec<ActRef>,
    pub call: Nb,
}

/// One activation's level task (R20): its rows, and the calls lifted to them before they were written (D10).
#[derive(Clone, Debug, Default)]
pub struct ActivateTask {
    pub rows: Vec<RowInput>,
    pub lifts: Vec<LiftTask>,
}

/// One write into a neuron's connections (R31, R33, R37).
///
/// A write that counts an exposure adds one to the strength; any write adds its share to the earnings, a share
/// for an earlier frame only where the connection exists.
/// `walk` lists, per action dimension the target acted in, its base actions in declared order: where the estimate
/// turns negative, the first one with no connection at that offset is wired (R37).
#[derive(Clone, Debug)]
pub struct ConnectionWrite {
    pub own: Sym,
    pub target: Sym,
    pub off: Offset,
    pub share: f64,
    pub exposure: bool,
    pub walk: Arc<Vec<Vec<NeuronId>>>,
}

/// A count grouping asks of one pose's rows (D47, R50). Every count reads only the rows of the pose `own`.
///
/// `Ends`: the rows that hold the relation, from the relation index, and of those the rows of the latest frame, for a
/// relation to an earlier frame.
/// `Holding`: the rows that have decided the relation and hold it.
/// `Rows`: the rows of the pose.
/// `Answer`: over the rows that decided the relation and every condition and hold every condition, those that hold
/// the relation (`with`) and, where there is a condition, those that do not (`without`).
/// `Alone`: the rows that decided every offset of `offs` and hold the relation.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum CountQuery {
    Ends { own: Sym, far: Sym, off: Offset },
    Holding { own: Sym, far: Sym, off: Offset },
    Rows { own: Sym },
    Answer { own: Sym, far: Sym, off: Offset, cond: Vec<(Sym, Offset)> },
    Alone { own: Sym, far: Sym, off: Offset, offs: Vec<Offset> },
}

impl CountQuery {
    /// The pose whose rows the count reads.
    pub fn read_pose(&self) -> Sym {
        match self {
            CountQuery::Ends { own, .. } | CountQuery::Holding { own, .. } | CountQuery::Rows { own }
            | CountQuery::Answer { own, .. } | CountQuery::Alone { own, .. } => *own,
        }
    }
}

/// Answer a count over a history and its relation index, at frame `frame`.
///
/// A row has decided an offset backward in time when it is written, one forward only once it has taken what followed,
/// from the frame after the one the offset reaches (R50, D18).
pub fn count_rows(history: &VecDeque<Row>, index: &FxHashMap<RelKey, u32>, now_frame: FrameNumber, now_index: &FxHashMap<RelKey, u32>, frame: FrameNumber, q: &CountQuery) -> (u32, u32) {
    let decided = |row: &Row, off: &Offset| off[0] <= 0 || row.act.frame + (off[0] as i64) < frame;
    let holds = |row: &Row, far: Sym, off: &Offset| row.nbs.iter().any(|nb| nb.sym == far && nb.off == *off);
    let own = q.read_pose();
    let rows = history.iter().filter(|r| r.own == own);
    match q {
        CountQuery::Ends { own, far, off } => {
            let k = (*own, *far, *off);
            let now = if off[0] < 0 && now_frame == frame { now_index.get(&k).copied().unwrap_or(0) } else { 0 };
            (index.get(&k).copied().unwrap_or(0), now)
        }
        CountQuery::Holding { far, off, .. } => (rows.filter(|r| decided(r, off) && holds(r, *far, off)).count() as u32, 0),
        CountQuery::Rows { .. } => (rows.count() as u32, 0),
        CountQuery::Answer { far, off, cond, .. } => {
            let (mut with, mut without) = (0, 0);
            for row in rows {
                if !decided(row, off) || !cond.iter().all(|(_, o2)| decided(row, o2)) { continue; }
                if !cond.iter().all(|(f2, o2)| holds(row, *f2, o2)) { continue; }
                if holds(row, *far, off) { with += 1; } else if !cond.is_empty() { without += 1; }
            }
            (with, without)
        }
        CountQuery::Alone { far, off, offs, .. } => (rows.filter(|r| offs.iter().all(|o| decided(r, o)) && holds(r, *far, off)).count() as u32, 0),
    }
}

/// A connection (D25): exposures and earnings.
#[derive(Clone, Copy, Debug, Default)]
pub struct Conn {
    pub strength: f64,
    pub earnings: f64,
}

impl Conn {
    pub fn estimate(&self) -> f64 { if self.strength > 0.0 { self.earnings / self.strength } else { 0.0 } }
}

pub struct Neuron {
    pub id: NeuronId,
    pub alive: bool,
    pub history: VecDeque<Row>,
    pub h: usize,
    /// The relation index (D47): every relation its rows hold, with its strength.
    pub index: FxHashMap<RelKey, u32>,
    /// The same index over the rows of the latest frame a row was written for (a speed-up only).
    pub now_frame: FrameNumber,
    pub now_index: FxHashMap<RelKey, u32>,
    /// Per phase of its own less its laps, per pose and offset, one connection (D25).
    pub connections: FxHashMap<(Sym, Sym, Offset), Conn>,
}

/// The relations a row holds, each once.
fn read_keys(row: &Row) -> Vec<RelKey> {
    let mut k: Vec<RelKey> = row.nbs.iter().map(|nb| (row.own, nb.sym, nb.off)).collect();
    k.sort_unstable();
    k.dedup();
    k
}

impl Neuron {
    pub fn new(id: NeuronId, h: usize) -> Self {
        Neuron {
            id,
            alive: true,
            history: VecDeque::new(),
            h,
            index: FxHashMap::default(),
            now_frame: FrameNumber::MIN,
            now_index: FxHashMap::default(),
            connections: FxHashMap::default(),
        }
    }

    pub fn strength(&self, k: &RelKey) -> u32 { self.index.get(k).copied().unwrap_or(0) }

    /// Count keys a row of `frame` came to hold, in the index and in the frame's own index.
    ///
    /// A row of a later frame starts the frame's index afresh.
    fn add_keys(&mut self, keys: &[RelKey], frame: FrameNumber) {
        for k in keys { *self.index.entry(*k).or_insert(0) += 1; }
        if frame > self.now_frame { self.now_frame = frame; self.now_index.clear(); }
        if frame == self.now_frame { for k in keys { *self.now_index.entry(*k).or_insert(0) += 1; } }
    }

    fn remove_keys(&mut self, keys: &[RelKey], frame: FrameNumber) {
        for k in keys {
            if let Some(v) = self.index.get_mut(k) {
                *v -= 1;
                if *v == 0 { self.index.remove(k); }
            }
        }
        if frame != self.now_frame { return; }
        for k in keys {
            if let Some(v) = self.now_index.get_mut(k) {
                *v -= 1;
                if *v == 0 { self.now_index.remove(k); }
            }
        }
    }

    /// Activate (R20): evict the oldest rows as the frame needs, admit the frame's rows, and return each
    /// activation's relations with their strengths.
    ///
    /// The calls lifted to these rows before they were written are taken after the strengths are read (D10).
    pub fn activate(&mut self, task: ActivateTask) -> Vec<ActReturn> {
        let incoming = task.rows.len().min(self.h);
        while self.history.len() + incoming > self.h {
            let old = self.history.pop_front().unwrap();
            self.remove_keys(&read_keys(&old), old.act.frame);
        }
        let mut admitted: Vec<(ActRef, Sym, Vec<Nb>)> = Vec::new();
        for inp in task.rows.into_iter().take(incoming) {
            let row = Row { act: inp.act, own: inp.own, nbs: inp.nbs };
            self.add_keys(&read_keys(&row), row.act.frame);
            admitted.push((row.act, row.own, row.nbs.clone()));
            self.history.push_back(row);
        }
        let returns = admitted.into_iter()
            .map(|(act, own, nbs)| ActReturn { act, rels: nbs.iter().map(|nb| (*nb, self.strength(&(own, nb.sym, nb.off)))).collect() })
            .collect();
        for lift in &task.lifts { self.take_lift(lift); }
        returns
    }

    /// What `activate` would return, admitting nothing: the call of a brain whose learning is off.
    ///
    /// The strengths are the ones `activate` returns: the rows it would evict taken out, the rows it would admit
    /// counted in.
    pub fn peek(&self, task: ActivateTask) -> Vec<ActReturn> {
        let incoming = task.rows.len().min(self.h);
        let evict = (self.history.len() + incoming).saturating_sub(self.h);
        let mut delta: FxHashMap<RelKey, i64> = FxHashMap::default();
        for old in self.history.iter().take(evict) { for k in read_keys(old) { *delta.entry(k).or_insert(0) -= 1; } }
        let rows: Vec<RowInput> = task.rows.into_iter().take(incoming).collect();
        for inp in &rows {
            let mut k: Vec<RelKey> = inp.nbs.iter().map(|nb| (inp.own, nb.sym, nb.off)).collect();
            k.sort_unstable();
            k.dedup();
            for k in k { *delta.entry(k).or_insert(0) += 1; }
        }
        let at = |k: &RelKey| (self.strength(k) as i64 + delta.get(k).copied().unwrap_or(0)).max(0) as u32;
        rows.into_iter()
            .map(|inp| ActReturn { act: inp.act, rels: inp.nbs.iter().map(|nb| (*nb, at(&(inp.own, nb.sym, nb.off)))).collect() })
            .collect()
    }

    /// Take a lifted call (D10, D55): the row of the activation, or of the record a lap grew it from, holds the call at
    /// its offset where it does not hold it yet.
    ///
    /// Returns whether the row was there.
    pub fn take_lift(&mut self, lift: &LiftTask) -> bool {
        let found = self.history.iter().rposition(|r| r.act == lift.row)
            .or_else(|| self.history.iter().rposition(|r| r.act.frame == lift.row.frame && lift.row_records.contains(&r.act)));
        let Some(i) = found else { return false };
        let call = &lift.call;
        let holds = self.history[i].nbs.iter().any(|x| (x.act == call.act || lift.call_records.contains(&x.act)) && x.off == call.off);
        if holds { return true; }
        let row_act = self.history[i].act;
        self.add_followed(row_act, &[*call]);
        true
    }

    /// Take what followed the activation (D18): the row holds each neighbor it does not hold yet, and the index
    /// counts it.
    pub fn add_followed(&mut self, viewer: ActRef, nbs: &[Nb]) -> bool {
        let Some(pos) = self.history.iter().rposition(|r| r.act == viewer) else { return false };
        self.push_neighbors(pos, nbs);
        true
    }

    /// The row at `pos` takes each neighbor whose activation it does not hold yet, and the index counts each relation
    /// the row did not hold before.
    fn push_neighbors(&mut self, pos: usize, nbs: &[Nb]) {
        let row = &mut self.history[pos];
        let mut came: Vec<RelKey> = Vec::new();
        for nb in nbs {
            if row.nbs.iter().any(|x| x.act == nb.act) { continue; }
            let key = (row.own, nb.sym, nb.off);
            if !row.nbs.iter().any(|x| x.sym == nb.sym && x.off == nb.off) && !came.contains(&key) { came.push(key); }
            row.nbs.push(*nb);
        }
        let fr = row.act.frame;
        self.add_keys(&came, fr);
    }

    /// Write the frame's connections in order (R31, R33, R37).
    pub fn write_connections(&mut self, writes: &[ConnectionWrite]) {
        for w in writes {
            let key = (w.own, w.target, w.off);
            if w.exposure {
                self.connections.entry(key).or_default().strength += 1.0;
                if w.share != 0.0 { self.add_earnings(w); }
                continue;
            }
            if self.connections.contains_key(&key) { self.add_earnings(w); }
        }
    }

    /// Add a share of reward into one connection (R33); where its estimate turns negative, walk (R37).
    fn add_earnings(&mut self, w: &ConnectionWrite) {
        let c = self.connections.entry((w.own, w.target, w.off)).or_default();
        let before = c.estimate();
        c.earnings += w.share;
        if c.estimate() >= 0.0 || before < 0.0 { return; }
        // the next untried base action, in each dimension the target acted in
        for dim in w.walk.iter() {
            for &n in dim {
                let key = (w.own, n, w.off);
                if self.connections.contains_key(&key) { continue; }
                self.connections.insert(key, Conn { strength: 1.0, earnings: 0.0 });
                break;
            }
        }
    }

    /// Read the connections of one pose (R36): each target, offset, strength and estimate.
    pub fn read_connections(&self, own: Sym) -> Vec<(Sym, Offset, f64, f64)> {
        self.connections.iter().filter(|((o, _, _), _)| *o == own).map(|((_, t, off), c)| (*t, *off, c.strength, c.estimate())).collect()
    }

    /// Answer a count over the rows (D47, R50).
    pub fn count(&self, q: &CountQuery, frame: FrameNumber) -> (u32, u32) {
        count_rows(&self.history, &self.index, self.now_frame, &self.now_index, frame, q)
    }

    /// Read what the neuron holds, for inspection: its row count and its connections, in key order.
    pub fn inspect(&self) -> (usize, Vec<((Sym, Sym, Offset), Conn)>) {
        let mut conns: Vec<((Sym, Sym, Offset), Conn)> = self.connections.iter().map(|(k, c)| (*k, *c)).collect();
        conns.sort_by(|a, b| a.0.cmp(&b.0));
        (self.history.len(), conns)
    }

    /// Read everything the neuron holds, for a backup: its history depth, the latest frame a row was written for, its
    /// rows, and its connections in the order it holds them.
    pub fn export(&self) -> (usize, FrameNumber, Vec<Row>, Vec<((Sym, Sym, Offset), Conn)>) {
        (self.h, self.now_frame, self.history.iter().cloned().collect(), self.connections.iter().map(|(k, c)| (*k, *c)).collect())
    }

    /// Take what a backup holds: the rows, with the relation index and the frame's index recounted from them, and the
    /// connections in the order they were held.
    pub fn import(&mut self, now_frame: FrameNumber, alive: bool, rows: Vec<Row>, conns: Vec<((Sym, Sym, Offset), Conn)>) {
        self.alive = alive;
        self.now_frame = now_frame;
        for row in rows {
            let keys = read_keys(&row);
            for k in &keys { *self.index.entry(*k).or_insert(0) += 1; }
            if row.act.frame == now_frame { for k in &keys { *self.now_index.entry(*k).or_insert(0) += 1; } }
            self.history.push_back(row);
        }
        for (k, c) in conns { self.connections.insert(k, c); }
    }

    /// Clear a decommissioned neuron (R38).
    pub fn clear(&mut self) {
        self.alive = false;
        self.history.clear();
        self.index.clear();
        self.now_index.clear();
        self.connections.clear();
    }

    /// Forget decommissioned poses (§4.1): remove each, in every phase it had, from every row and relation that names
    /// it.
    pub fn forget(&mut self, dead: &[Sym]) {
        self.index.retain(|k, _| !dead.contains(&k.0) && !dead.contains(&k.1));
        self.now_index.retain(|k, _| !dead.contains(&k.0) && !dead.contains(&k.1));
        for row in self.history.iter_mut() { row.nbs.retain(|nb| !dead.contains(&nb.sym)); }
        self.connections.retain(|(o, t, _), _| !dead.contains(o) && !dead.contains(t));
    }
}
