/// N-API bindings for brain-core: the `Brain` class of the machine of docs/algorithm.md, for the jobs in apps/.
///
/// Conventions:
///   - options, channel specs, inputs and rewards come in as plain JS objects; results go out the same way
///   - every call reaches the core through its public calls: the brain itself decides how its neurons are reached
///   - a position is the list of a symbol's places on the spatial axes, in the order the axes were declared
///   - a pose is the number the core names a neuron in a phase by; `describePose` reads it as text
///   - frame numbers cross as JS numbers

use std::cell::RefCell;
use std::path::Path;

use napi::bindgen_prelude::*;
use napi_derive::napi;

use brain_core::board::{Act, ActRef, FrameRec};
use brain_core::brain::{Brain as CoreBrain, FrameStats, Winner};
use brain_core::dictionary::{canon, Func, Rel};
use brain_core::thalamus::NeuronFacts;
use brain_core::types::*;

// ── The shapes that cross ───────────────────────────────────────────────────

/// The brain's options, each optional.
#[napi(object)]
pub struct BrainOptions {
    /// The depth of every neuron's history (default 256).
    pub h: Option<u32>,
    /// The neighborhood policy: "eventsToActions" (default) or "allOn".
    pub policy: Option<String>,
    /// What an action dimension outputs when every candidate is negative: "leastBad" (default) or "nothing".
    pub action_policy: Option<String>,
    /// The dispatch: regions, and columns per region (one of each by default).
    pub regions: Option<u32>,
    pub columns: Option<u32>,
    /// Fixed for the life of the brain (default true); a brain built with it off is loaded from a backup to be tested.
    pub learning: Option<bool>,
}

/// One neuron dimension of a channel: its name and how many values it takes.
#[napi(object)]
pub struct DimSpec {
    pub name: String,
    pub resolution: u32,
}

/// A channel: its event dimension, its action dimension, and the spatial axes it is laid out over, by index.
#[napi(object)]
pub struct ChannelSpec {
    pub name: String,
    pub event: Option<DimSpec>,
    pub action: Option<DimSpec>,
    pub axes: Option<Vec<u32>>,
}

/// What declaring a channel returns: its id, and the ids of its event and action dimensions.
#[napi(object)]
pub struct ChannelIds {
    pub channel_id: u32,
    pub event_dim: Option<u32>,
    pub action_dim: Option<u32>,
}

/// A declared channel: its id, name, event and action dimensions, and the spatial axes it is laid out over.
#[napi(object)]
pub struct ChannelOut {
    pub channel_id: u32,
    pub name: String,
    pub event_dim: Option<u32>,
    pub action_dim: Option<u32>,
    pub axes: Vec<u32>,
}

/// How many activations stand on the apex in the last frame, by kind.
#[napi(object)]
pub struct ApexCounts {
    pub calls: u32,
    pub events: u32,
    pub actions: u32,
}

/// One symbol the world reports in a frame: its dimension, its value, and its position.
#[napi(object)]
pub struct InputSpec {
    pub dim: u32,
    pub bucket: u32,
    pub position: Option<Vec<i32>>,
}

/// A reward: its amount, the action channels it pays (all when absent), and its span of frames (the longest when
/// absent).
#[napi(object)]
pub struct RewardSpec {
    pub reward: f64,
    pub channels: Option<Vec<u32>>,
    pub frames: Option<u32>,
}

/// One output for the frame ahead: a dimension's winning value at a position.
#[napi(object)]
pub struct WinnerOut {
    pub dim: u32,
    pub bucket: u32,
    pub position: Vec<i32>,
    pub kind: String,
    pub estimate: f64,
    pub share: f64,
    pub voters: u32,
}

/// The counts of the last frame.
#[napi(object)]
pub struct StatsOut {
    pub frame: i64,
    pub levels: u32,
    pub apex_activations: u32,
    pub functions_added: u32,
    pub functions_retired: u32,
    pub voters: u32,
    pub neurons: u32,
    pub functions: u32,
    pub connections: u32,
}

/// A neuron's fixed facts.
#[napi(object)]
pub struct NeuronOut {
    pub id: u32,
    pub kind: String,
    pub dim: Option<u32>,
    pub bucket: u32,
    pub channel: Option<u32>,
    pub alive: bool,
}

/// A relation of the dictionary: two poses and the offset from the first to the second (time first).
#[napi(object)]
pub struct RelationOut {
    pub from: u32,
    pub to: u32,
    pub offset: Vec<i32>,
}

/// A function of the dictionary.
#[napi(object)]
pub struct FunctionOut {
    pub id: u32,
    pub description: String,
    pub nodes: Vec<u32>,
    pub relations: Vec<RelationOut>,
    pub interneuron: Option<u32>,
    pub is_cycle: bool,
    pub origin: Option<u32>,
}

/// A pose that names a call: its interneuron, and its phase from one node to another over a number of laps.
#[napi(object)]
pub struct PoseOut {
    pub neuron: u32,
    pub from: u32,
    pub to: u32,
    pub laps: u32,
}

/// An activation's place on the board: the frame it is recorded under and its index there.
#[napi(object)]
pub struct ActRefOut {
    pub frame: i64,
    pub index: u32,
}

/// A relation an activation's neuron returned at it: the other activation, its offset, and its strength.
#[napi(object)]
pub struct BoardRelationOut {
    pub other: ActRefOut,
    pub offset: Vec<i32>,
    pub strength: u32,
}

/// An open activation on the board: `frame` is the frame it is recorded under, `time` the frame it stands at, and
/// `reach` its reach in time first, then on each spatial axis.
#[napi(object)]
pub struct ActivationOut {
    pub index: u32,
    pub neuron: u32,
    pub pose: u32,
    pub kind: String,
    pub frame: i64,
    pub time: i64,
    pub position: Vec<i32>,
    pub reach: Vec<i32>,
    pub strong: bool,
    pub on_apex: bool,
    pub gone: bool,
    pub level: u32,
    pub covered_at: Option<u32>,
    pub covers: Vec<ActRefOut>,
    pub coverers: Vec<ActRefOut>,
    pub fired_by: Option<u32>,
    pub relations: Vec<BoardRelationOut>,
}

/// A frame still open on the board, with its activations in order.
#[napi(object)]
pub struct FrameOut {
    pub frame: i64,
    pub activations: Vec<ActivationOut>,
}

// ── Conversions ─────────────────────────────────────────────────────────────

fn make_policy(options: &BrainOptions) -> Result<Policy> {
    let mut policy = match options.policy.as_deref().unwrap_or("eventsToActions") {
        "eventsToActions" => Policy::events_to_actions(),
        "allOn" => Policy::all_on(),
        other => return Err(Error::from_reason(format!("unknown policy {}", other))),
    };
    policy.action = match options.action_policy.as_deref().unwrap_or("leastBad") {
        "leastBad" => ActionPolicy::LeastBad,
        "nothing" => ActionPolicy::Nothing,
        other => return Err(Error::from_reason(format!("unknown action policy {}", other))),
    };
    Ok(policy)
}

fn describe_kind(kind: Kind) -> String {
    match kind { Kind::Event => "event", Kind::Action => "action", Kind::Inter => "inter" }.to_string()
}

/// A position from its places on the spatial axes, time left at zero for the brain to set.
fn make_pos(position: &Option<Vec<i32>>) -> Pos {
    let places: Vec<i32> = position.iter().flatten().copied().collect();
    Pos::make(places.len() + 1, |d| if d == 0 { 0 } else { places[d - 1] })
}

/// The places of a position on the spatial axes.
fn read_places(pos: &Pos, dims: usize) -> Vec<i32> { (1..dims).map(|d| pos[d]).collect() }

fn make_input(spec: &InputSpec) -> Input { Input { dim: spec.dim, bucket: spec.bucket, pos: make_pos(&spec.position) } }

fn make_reward(spec: &RewardSpec) -> Reward {
    Reward { reward: spec.reward, channels: spec.channels.clone().unwrap_or_default(), frames: spec.frames.unwrap_or(0) }
}

fn describe_winner(w: &Winner, dims: usize) -> WinnerOut {
    WinnerOut { dim: w.dim, bucket: w.bucket, position: read_places(&w.pos, dims), kind: describe_kind(w.kind), estimate: w.estimate, share: w.share, voters: w.voters as u32 }
}

fn describe_stats(stats: &FrameStats) -> StatsOut {
    StatsOut {
        frame: stats.frame,
        levels: stats.levels as u32,
        apex_activations: stats.apex_acts as u32,
        functions_added: stats.added as u32,
        functions_retired: stats.retired as u32,
        voters: stats.voters as u32,
        neurons: stats.neurons as u32,
        functions: stats.patterns as u32,
        connections: stats.connections as u32,
    }
}

fn describe_neuron(id: u32, facts: &NeuronFacts) -> NeuronOut {
    NeuronOut { id, kind: describe_kind(facts.kind), dim: facts.dim, bucket: facts.bucket, channel: facts.channel, alive: facts.alive }
}

fn describe_channel(c: &Channel) -> ChannelOut {
    let axes = c.axes.space().iter().map(|d| *d as u32).collect();
    ChannelOut { channel_id: c.id, name: c.name.clone(), event_dim: c.event, action_dim: c.action, axes }
}

/// An offset from its values, time first; the dimensions not given are zero.
fn make_offset(values: &[i32]) -> Offset {
    Pos::from_slice(values)
}

fn describe_relation(r: &Rel, dims: usize) -> RelationOut { RelationOut { from: r.a, to: r.b, offset: r.off.to_vec(dims.max(r.off.len())) } }

fn describe_ref(r: &ActRef) -> ActRefOut { ActRefOut { frame: r.frame, index: r.index } }

/// An activation as JS reads it; its relations are copied only when asked, since a frame can hold many.
fn describe_activation(frame: i64, index: usize, a: &Act, with_relations: bool, dims: usize) -> ActivationOut {
    ActivationOut {
        index: index as u32,
        neuron: a.neuron,
        pose: a.sym,
        kind: describe_kind(a.kind),
        frame,
        time: a.pos[0] as i64,
        position: read_places(&a.pos, dims),
        reach: a.reach.to_vec(dims),
        strong: a.strong,
        on_apex: a.on_apex(),
        gone: a.gone,
        level: a.level,
        covered_at: a.covered_at,
        covers: a.covers.iter().map(describe_ref).collect(),
        coverers: a.coverers.iter().map(describe_ref).collect(),
        fired_by: a.fired_by.as_ref().map(|f| f.func),
        relations: if with_relations { a.rels.iter().map(|r| BoardRelationOut { other: describe_ref(&r.other), offset: r.off.to_vec(dims.max(r.off.len())), strength: r.strength }).collect() } else { Vec::new() },
    }
}

fn describe_frame(f: &FrameRec, with_relations: bool, dims: usize) -> FrameOut {
    FrameOut { frame: f.no, activations: f.acts.iter().enumerate().map(|(i, a)| describe_activation(f.no, i, a, with_relations, dims)).collect() }
}

/// A stamp of what a function is made of: a hash of its nodes, its relations and its interneuron, cut to 52 bits.
fn stamp_function(f: &Func) -> f64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(&(&f.nodes, &f.rels, f.inter), &mut hasher);
    (std::hash::Hasher::finish(&hasher) & ((1u64 << 52) - 1)) as f64
}

fn to_error(message: String) -> Error { Error::from_reason(message) }

// ── The Brain class ─────────────────────────────────────────────────────────

#[napi(js_name = "Brain")]
pub struct JsBrain {
    /// napi-rs hands methods `&self`; the brain changes as it runs, so it sits in a cell.
    inner: RefCell<CoreBrain>,
}

#[napi]
impl JsBrain {
    /// Create a brain from its options.
    #[napi(constructor)]
    pub fn new(options: Option<BrainOptions>) -> Result<Self> {
        let options = options.unwrap_or(BrainOptions { h: None, policy: None, action_policy: None, regions: None, columns: None, learning: None });
        let policy = make_policy(&options)?;
        let h = options.h.unwrap_or(256) as usize;
        let regions = options.regions.unwrap_or(1) as usize;
        let columns = options.columns.unwrap_or(1) as usize;
        let brain = CoreBrain::new(h, policy, regions, columns, options.learning.unwrap_or(true));
        Ok(JsBrain { inner: RefCell::new(brain) })
    }

    // ── Declaring the world ─────────────────────────────────────────────────

    /// Declare a spatial axis; returns its index, from 1.
    #[napi]
    pub fn declare_axis(&self, name: String) -> u32 { self.inner.borrow_mut().declare_axis(&name, 1) as u32 }

    /// Declare a channel; its base neurons are made in declaration order.
    #[napi]
    pub fn declare_channel(&self, spec: ChannelSpec) -> ChannelIds {
        let mut brain = self.inner.borrow_mut();
        let axes: Vec<usize> = spec.axes.iter().flatten().map(|a| *a as usize).collect();
        let event = spec.event.as_ref().map(|d| (d.name.as_str(), d.resolution));
        let action = spec.action.as_ref().map(|d| (d.name.as_str(), d.resolution));
        let id = brain.declare_channel(&spec.name, event, action, &axes);
        let channel = &brain.channels[id as usize];
        ChannelIds { channel_id: id, event_dim: channel.event, action_dim: channel.action }
    }

    /// The spatial axes declared, by name, in index order from 1.
    #[napi]
    pub fn list_axes(&self) -> Vec<String> { self.inner.borrow().act_dims.iter().skip(1).map(|d| d.name.clone()).collect() }

    /// The channels declared, in id order.
    #[napi]
    pub fn list_channels(&self) -> Vec<ChannelOut> { self.inner.borrow().channels.iter().map(describe_channel).collect() }

    // ── Running ─────────────────────────────────────────────────────────────

    /// Process one frame: what was reported and what ran, with the rewards; returns the output for the frame ahead.
    #[napi]
    pub fn process_frame(&self, inputs: Vec<InputSpec>, rewards: Vec<RewardSpec>) -> Vec<WinnerOut> {
        let inputs: Vec<Input> = inputs.iter().map(make_input).collect();
        let rewards: Vec<Reward> = rewards.iter().map(make_reward).collect();
        let mut brain = self.inner.borrow_mut();
        let winners = brain.process_frame(&inputs, rewards);
        let dims = brain.act_dims.len();
        winners.iter().map(|w| describe_winner(w, dims)).collect()
    }

    /// Clear every open activation; keep the dictionary and the connections.
    #[napi]
    pub fn reset_context(&self) { self.inner.borrow_mut().reset_context(); }

    // ── Reading the brain ───────────────────────────────────────────────────

    /// The counts of the last frame.
    #[napi]
    pub fn get_stats(&self) -> StatsOut { describe_stats(&self.inner.borrow().last_stats) }

    /// The frame number of the last frame.
    #[napi]
    pub fn get_frame_number(&self) -> i64 { self.inner.borrow().frame_no }

    /// How many neurons are alive.
    #[napi]
    pub fn count_living_neurons(&self) -> u32 { self.inner.borrow().alive as u32 }

    /// How many neurons were ever allotted, living or not.
    #[napi]
    pub fn count_neurons(&self) -> u32 { self.inner.borrow().get_registry().len() as u32 }

    /// The longest reach in time among the activations still open.
    #[napi]
    pub fn get_longest_reach(&self) -> i32 { self.inner.borrow().max_reach_t }

    /// The base neuron of a dimension's value.
    #[napi]
    pub fn find_base_neuron(&self, dim: u32, bucket: u32) -> u32 { self.inner.borrow().find_base_neuron(dim, bucket) }

    /// The fixed facts of the named neurons, in the order named.
    #[napi]
    pub fn read_neurons(&self, ids: Vec<u32>) -> Result<Vec<NeuronOut>> {
        let brain = self.inner.borrow();
        let registry = brain.get_registry();
        ids.iter().map(|&id| registry.get(id).map(|f| describe_neuron(id, f)).ok_or_else(|| to_error(format!("no neuron {}", id)))).collect()
    }

    /// Describe the named neurons: kind, rows and connections, the functions each is the interneuron of, and its
    /// first sixteen connections.
    #[napi]
    pub fn describe_neurons(&self, ids: Vec<u32>) -> Vec<String> { self.inner.borrow_mut().describe_neurons(&ids) }

    /// How many living neurons stand at each height.
    #[napi]
    pub fn count_heights(&self) -> Vec<u32> { self.inner.borrow().count_heights().into_iter().map(|n| n as u32).collect() }

    /// How many functions of the dictionary never took a call.
    #[napi]
    pub fn count_functions_never_called(&self) -> u32 { self.inner.borrow().count_functions_never_called() as u32 }

    /// Every function of the dictionary, in id order.
    #[napi]
    pub fn get_dictionary(&self) -> Vec<FunctionOut> {
        let brain = self.inner.borrow();
        let dict = brain.get_dictionary();
        let mut ids: Vec<u32> = dict.funcs.keys().copied().collect();
        ids.sort_unstable();
        ids.iter().map(|id| describe_function(&dict.funcs[id], dict.describe(&dict.funcs[id]), dict.dims)).collect()
    }

    /// A fingerprint of the dictionary: a hash of every function's description, sorted.
    /// Two fingerprints differ where a function started, left, or reads differently; it is read in one call, where
    /// comparing the dictionary itself would carry every function across to JS.
    #[napi]
    pub fn read_dictionary_fingerprint(&self) -> String {
        let brain = self.inner.borrow();
        let dict = brain.get_dictionary();
        let mut descriptions: Vec<String> = dict.funcs.values().map(|f| dict.describe(f)).collect();
        descriptions.sort();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&descriptions, &mut hasher);
        format!("{:016x}", std::hash::Hasher::finish(&hasher))
    }

    /// Each function's id and a stamp of what it is made of (its nodes, its relations and its interneuron), in id order,
    /// as one flat list: id, stamp, id, stamp.
    /// A stamp changes where what the function is made of changes, so a reader asks only those functions again.
    /// The stamp is a 52-bit hash, so it is exact as a JS number.
    #[napi]
    pub fn read_function_stamps(&self) -> Vec<f64> {
        let brain = self.inner.borrow();
        let dict = brain.get_dictionary();
        let mut ids: Vec<u32> = dict.funcs.keys().copied().collect();
        ids.sort_unstable();
        ids.iter().flat_map(|id| [*id as f64, stamp_function(&dict.funcs[id])]).collect()
    }

    /// The named functions of the dictionary, in the order named; a function no longer there is left out.
    #[napi]
    pub fn describe_functions(&self, ids: Vec<u32>) -> Vec<FunctionOut> {
        let brain = self.inner.borrow();
        let dict = brain.get_dictionary();
        ids.iter().filter_map(|id| dict.funcs.get(id)).map(|f| describe_function(f, dict.describe(f), dict.dims)).collect()
    }

    /// How many functions the dictionary holds.
    #[napi]
    pub fn count_functions(&self) -> u32 { self.inner.borrow().get_dictionary().len() as u32 }

    /// A pose as the dictionary writes it.
    #[napi]
    pub fn describe_pose(&self, pose: u32) -> String { self.inner.borrow().get_dictionary().describe_sym(pose) }

    /// The call a pose names, if it names one: its interneuron and its phase.
    #[napi]
    pub fn read_pose(&self, pose: u32) -> Option<PoseOut> {
        let brain = self.inner.borrow();
        let dict = brain.get_dictionary();
        dict.find_pose(pose).map(|p| PoseOut { neuron: dict.find_neuron_of(pose), from: p.from, to: p.to, laps: p.laps })
    }

    /// The functions an interneuron holds, in order.
    #[napi]
    pub fn list_functions_of(&self, interneuron: u32) -> Vec<u32> { self.inner.borrow().get_dictionary().list_of_inter(interneuron).to_vec() }

    /// A relation as the dictionary holds it: from `from` to `to` at the offset, or turned round where the dictionary keeps it the other way.
    #[napi]
    pub fn canon_relation(&self, from: u32, to: u32, offset: Vec<i32>) -> RelationOut { describe_relation(&canon(from, to, make_offset(&offset)), self.inner.borrow().get_dictionary().dims) }

    /// The function that holds a relation, as the dictionary holds it; none where no function holds it.
    #[napi]
    pub fn find_holder(&self, relation: RelationOut) -> Option<u32> {
        let rel = Rel { a: relation.from, b: relation.to, off: make_offset(&relation.offset) };
        self.inner.borrow().get_dictionary().find_holder(&rel).map(|f| f as u32)
    }

    /// Every frame still open on the board, oldest first; each activation's relations only when `with_relations` is true.
    #[napi]
    pub fn get_board(&self, with_relations: Option<bool>) -> Vec<FrameOut> {
        let with_relations = with_relations.unwrap_or(false);
        let brain = self.inner.borrow();
        let dims = brain.act_dims.len();
        brain.get_board().iter().map(|f| describe_frame(f, with_relations, dims)).collect()
    }

    /// How many activations stand on the apex in the last frame, by kind; none where the board is empty.
    #[napi]
    pub fn count_apex(&self) -> ApexCounts {
        let brain = self.inner.borrow();
        let mut counts = ApexCounts { calls: 0, events: 0, actions: 0 };
        for a in brain.get_board().back().iter().flat_map(|f| f.acts.iter()).filter(|a| a.on_apex()) {
            match a.kind { Kind::Inter => counts.calls += 1, Kind::Event => counts.events += 1, Kind::Action => counts.actions += 1 }
        }
        counts
    }

    /// The last frame on the board; each activation's relations only when `with_relations` is true.
    #[napi]
    pub fn get_last_frame(&self, with_relations: Option<bool>) -> Option<FrameOut> {
        let with_relations = with_relations.unwrap_or(false);
        let brain = self.inner.borrow();
        let dims = brain.act_dims.len();
        brain.get_board().back().map(|f| describe_frame(f, with_relations, dims))
    }

    // ── Backups ─────────────────────────────────────────────────────────────

    /// Save the brain to `<jobDir>/backups/<label>/`; returns the folder written.
    #[napi]
    pub fn save(&self, job_dir: String, label: String) -> Result<String> {
        let folder = self.inner.borrow_mut().save(Path::new(&job_dir), &label).map_err(to_error)?;
        Ok(folder.display().to_string())
    }

    /// Load `<jobDir>/backups/<label>/` into a brain that has not run.
    #[napi]
    pub fn load(&self, job_dir: String, label: String) -> Result<()> { self.inner.borrow_mut().load(Path::new(&job_dir), &label).map_err(to_error) }

    /// Save the board to `<jobDir>/contexts/<label>/`; returns the folder written.
    #[napi]
    pub fn save_context(&self, job_dir: String, label: String) -> Result<String> {
        let folder = self.inner.borrow().save_context(Path::new(&job_dir), &label).map_err(to_error)?;
        Ok(folder.display().to_string())
    }

    /// Load `<jobDir>/contexts/<label>/` into a brain loaded from the backup saved with it.
    #[napi]
    pub fn load_context(&self, job_dir: String, label: String) -> Result<()> {
        self.inner.borrow_mut().load_context(Path::new(&job_dir), &label).map_err(to_error)
    }
}

fn describe_function(f: &Func, description: String, dims: usize) -> FunctionOut {
    FunctionOut {
        id: f.id,
        description,
        nodes: f.nodes.clone(),
        relations: f.rels.iter().map(|r| describe_relation(r, dims)).collect(),
        interneuron: f.inter,
        is_cycle: f.has_cycle(),
        origin: f.find_origin(),
    }
}
