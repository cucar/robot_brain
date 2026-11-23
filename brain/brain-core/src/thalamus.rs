/// Thalamus — the relay between the brain and its neurons.
///
/// Owns:
///   - Region[R] tree (each Region owns Column[C], each Column owns Neurons)
///   - the registry: each neuron's kind, dimension, bucket, channel, axes, and whether it is alive
///   - the dictionary (D36, D56) and the board (D9), the brain's maps of functions and open activations
///   - the neuron id allocator
///
/// The brain never reaches one neuron.
/// It hands the thalamus a list of tasks; the thalamus routes each to its region (id mod R) and column (id mod C),
/// runs the regions and their columns in parallel, and hands the results back in the order the tasks came.

use std::collections::VecDeque;
use std::ops::Index;

use rayon::prelude::*;

use crate::board::FrameRec;
use crate::dictionary::Dictionary;
use crate::neuron::*;
use crate::region::Region;
use crate::types::*;

/// A neuron's fixed facts, set when it is allotted; only `alive` changes, at decommissioning (R38).
#[derive(Clone, Debug)]
pub struct NeuronFacts {
    pub kind: Kind,
    pub dim: Option<DimensionId>,
    pub bucket: u32,
    pub channel: Option<ChannelId>,
    pub axes: Axes,
    pub alive: bool,
}

/// The registry: every neuron's facts, by id.
#[derive(Default)]
pub struct Registry {
    facts: Vec<NeuronFacts>,
}

impl Registry {
    pub fn len(&self) -> usize { self.facts.len() }
    pub fn get(&self, id: NeuronId) -> Option<&NeuronFacts> { self.facts.get(id as usize) }
    pub fn is_alive(&self, id: NeuronId) -> bool { self.get(id).map_or(false, |f| f.alive) }
    pub fn mark_dead(&mut self, id: NeuronId) { self.facts[id as usize].alive = false; }
    pub fn iter(&self) -> impl Iterator<Item = (NeuronId, &NeuronFacts)> + '_ { self.facts.iter().enumerate().map(|(i, f)| (i as NeuronId, f)) }
}

impl Index<NeuronId> for Registry {
    type Output = NeuronFacts;
    fn index(&self, id: NeuronId) -> &NeuronFacts { self.get(id).expect("no such neuron") }
}

/// One neuron's part of the connect call (§7.1): what followed its rows, then its connection writes.
///
/// `named` lists the interneurons that what followed and the writes name a pose of, for the neuron to note (R38).
#[derive(Default)]
pub struct ConnectTask {
    pub followed: Vec<(ActRef, Vec<Nb>)>,
    pub writes: Vec<ConnectionWrite>,
    pub named: Vec<NeuronId>,
}

pub struct Thalamus {
    regions: Vec<Region>,
    pub registry: Registry,
    /// The dictionary (D36, D56): every function and its interneuron.
    pub dict: Dictionary,
    /// The board (D9): the frames still open, with their activations.
    pub frames: VecDeque<FrameRec>,
    /// Neurons allotted since the last batch; each batch creates them in their columns before it runs.
    allotted: Vec<(NeuronId, usize)>,
}

impl Thalamus {
    pub fn new(regions: usize, columns: usize) -> Self {
        Thalamus {
            regions: (0..regions.max(1)).map(|_| Region::new(columns)).collect(),
            registry: Registry::default(),
            dict: Dictionary::default(),
            frames: VecDeque::new(),
            allotted: Vec::new(),
        }
    }

    /// Map a neuron id to its owning region.
    pub fn route_neuron(&self, neuron_id: NeuronId) -> usize {
        (neuron_id as usize) % self.regions.len()
    }

    // ── Allocation ──────────────────────────────────────────────────────────

    /// Allot the next neuron id with its facts; the neuron is created in its column by the next batch.
    pub fn allot_neuron(&mut self, facts: NeuronFacts, h: usize) -> NeuronId {
        let id = self.registry.facts.len() as NeuronId;
        self.registry.facts.push(facts);
        self.allotted.push((id, h));
        id
    }

    /// Create the allotted neurons in their columns, in one batch.
    fn create_allotted(&mut self) {
        if self.allotted.is_empty() { return; }
        let specs = std::mem::take(&mut self.allotted);
        let mut per_region: Vec<Vec<(NeuronId, usize)>> = (0..self.regions.len()).map(|_| Vec::new()).collect();
        for spec in specs { let r = self.route_neuron(spec.0); per_region[r].push(spec); }
        self.regions.par_iter_mut().zip(per_region.into_par_iter()).for_each(|(region, specs)| region.create_neurons(specs));
    }

    // ── Dispatch ────────────────────────────────────────────────────────────

    /// Run one task per entry on the neuron it names, the regions and columns in parallel, the results in task
    /// order.
    fn dispatch<T: Send, R: Send>(&mut self, tasks: Vec<(NeuronId, T)>, op: impl Fn(&mut Neuron, T) -> R + Sync) -> Vec<R> {
        self.create_allotted();
        let count = tasks.len();
        let mut per_region: Vec<Vec<(usize, NeuronId, T)>> = (0..self.regions.len()).map(|_| Vec::new()).collect();
        for (at, (id, task)) in tasks.into_iter().enumerate() { let r = self.route_neuron(id); per_region[r].push((at, id, task)); }
        let nested: Vec<Vec<(usize, R)>> = self.regions.par_iter_mut()
            .zip(per_region.into_par_iter())
            .map(|(region, tasks)| if tasks.is_empty() { Vec::new() } else { region.run(tasks, &op) })
            .collect();
        let mut slots: Vec<Option<R>> = (0..count).map(|_| None).collect();
        for (at, r) in nested.into_iter().flatten() { slots[at] = Some(r); }
        slots.into_iter().map(|r| r.expect("every task returns")).collect()
    }

    /// Run an operation on every living neuron, the regions and columns in parallel.
    fn dispatch_living<R: Send>(&mut self, op: impl Fn(&mut Neuron) -> R + Sync) -> Vec<R> {
        self.create_allotted();
        let nested: Vec<Vec<R>> = self.regions.par_iter_mut().map(|region| region.run_living(&op)).collect();
        nested.into_iter().flatten().collect()
    }

    // ── Batch calls ─────────────────────────────────────────────────────────

    /// Activate a level (R20): each neuron admits its rows and returns their relations; with learning off, it
    /// returns what it would and writes nothing.
    pub fn activate_neurons(&mut self, tasks: Vec<(NeuronId, ActivateTask)>, learning: bool) -> Vec<Vec<ActReturn>> {
        if learning { self.dispatch(tasks, |n, t| n.activate(t)) } else { self.dispatch(tasks, |n, t| n.peek(t)) }
    }

    /// Lift relations to rows (D10, D55): each row takes its call; per lift, whether the row was there.
    pub fn write_lifts(&mut self, lifts: Vec<(NeuronId, LiftTask)>) -> Vec<bool> {
        self.dispatch(lifts, |n, l| n.take_lift(&l))
    }

    /// Connect (§7.1): each row takes what followed it, then each neuron writes its connections in order.
    pub fn connect_neurons(&mut self, tasks: Vec<(NeuronId, ConnectTask)>) {
        self.dispatch(tasks, |n, t| {
            n.note_interneurons(&t.named);
            for (me, nbs) in &t.followed { n.add_followed(*me, nbs); }
            n.write_connections(&t.writes);
        });
    }

    /// Infer (§7.2): per activation, the connections of its pose.
    pub fn read_connections(&mut self, reads: Vec<(NeuronId, Sym)>) -> Vec<Vec<(Sym, Offset, f64, f64)>> {
        self.dispatch(reads, |n, own| n.read_connections(own))
    }

    /// Decommission (R38): clear the decommissioned neurons, then forget their poses in every living neuron.
    pub fn decommission_neurons(&mut self, ids: Vec<NeuronId>, dead: Vec<Sym>) {
        if ids.is_empty() { return; }
        self.dispatch(ids.iter().map(|id| (*id, ())).collect(), |n, _| n.clear());
        self.dispatch_living(|n| n.forget(&ids, &dead));
    }

    /// Answer counts over the neurons' rows (D47, R50): per query, its answer, in the order the queries came.
    pub fn count_relations(&mut self, queries: Vec<(NeuronId, CountQuery)>, frame: FrameNumber) -> Vec<(u32, u32)> {
        self.dispatch(queries, |n, q| n.count(&q, frame))
    }

    /// Read what each named neuron holds, for inspection: its row count and its connections, in key order.
    pub fn inspect_neurons(&mut self, ids: Vec<NeuronId>) -> Vec<(usize, Vec<((Sym, Sym, Offset), Conn)>)> {
        self.dispatch(ids.into_iter().map(|id| (id, ())).collect(), |n, _| n.inspect())
    }

    /// Read everything each named neuron holds, for a backup.
    pub fn export_neurons(&mut self, ids: Vec<NeuronId>) -> Vec<(usize, FrameNumber, Vec<Row>, Vec<((Sym, Sym, Offset), Conn)>)> {
        self.dispatch(ids.into_iter().map(|id| (id, ())).collect(), |n, _| n.export())
    }

    /// Restore neurons from a backup into a thalamus that holds none: each one's facts in the registry, then each one
    /// created in its column and handed its rows and connections, in one batch.
    pub fn import_neurons(&mut self, neurons: Vec<(NeuronFacts, usize, FrameNumber, Vec<Row>, Vec<((Sym, Sym, Offset), Conn)>, Vec<NeuronId>)>) {
        assert!(self.registry.len() == 0, "a backup loads only into a thalamus that holds no neuron");
        let mut tasks = Vec::with_capacity(neurons.len());
        for (facts, h, now, rows, conns, named) in neurons {
            let alive = facts.alive;
            let id = self.allot_neuron(facts, h);
            tasks.push((id, (now, alive, rows, conns, named)));
        }
        self.dispatch(tasks, |n, (now, alive, rows, conns, named)| n.import(now, alive, rows, conns, named));
    }

    /// Count the connections of every living neuron.
    pub fn count_connections(&mut self) -> usize {
        self.dispatch_living(|n| n.connections.len()).into_iter().sum()
    }
}
