//! N-API bindings for brain-core: the `Brain` class, the UCAR machine of docs/algorithm.md.
//!
//! Frames cross the boundary as flat typed arrays: `inputs` is an Int32Array of
//! `[dim, bucket, p1, p2]` per reported symbol (p1, p2 are positions on the spatial axes, 0 when
//! the channel has none); `rewards` is an array of `{ reward, channels?, frames? }`.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::cell::RefCell;

use brain_core::machine::Machine;
use brain_core::types::*;

#[napi(object)]
pub struct BrainOptions {
    /// The depth of every neuron's history (R4).
    pub h: Option<u32>,
    /// The radius in time (R4).
    pub time_radius: Option<i32>,
    /// "all" (every entry on) or "eventsToActions" (the stocks/MNIST policy) (D50).
    pub policy: Option<String>,
    /// "leastBad" or "nothing" (D51).
    pub action_policy: Option<String>,
}

#[napi(object)]
pub struct DimSpec {
    pub name: String,
    pub resolution: u32,
}

#[napi(object)]
pub struct ChannelSpec {
    pub name: String,
    pub event: Option<DimSpec>,
    pub action: Option<DimSpec>,
    pub axes: Option<Vec<u32>>,
}

#[napi(object)]
pub struct ChannelIds {
    pub channel_id: u32,
    pub event_dim: Option<u32>,
    pub action_dim: Option<u32>,
}

#[napi(object)]
pub struct RewardSpec {
    pub reward: f64,
    pub channels: Option<Vec<u32>>,
    pub frames: Option<u32>,
}

#[napi(object)]
pub struct WinnerOut {
    pub dim: u32,
    pub bucket: u32,
    pub p1: i32,
    pub p2: i32,
    pub kind: String,
    pub estimate: f64,
    pub share: f64,
    pub voters: u32,
}

#[napi(object)]
pub struct StatsOut {
    pub frame: f64,
    pub base_acts: u32,
    pub apex_acts: u32,
    pub apex_cost: f64,
    pub waves: u32,
    pub bids: u32,
    pub accepted: u32,
    pub minted: u32,
    pub reused: u32,
    pub retired: u32,
    pub added: u32,
    pub neurons: u32,
    pub patterns: u32,
    pub connections: u32,
    pub voters: u32,
    pub voters_above_base: u32,
    pub elapsed_ms: f64,
}

#[napi(object)]
pub struct FrameOut {
    pub winners: Vec<WinnerOut>,
    pub stats: StatsOut,
}

#[napi(js_name = "Brain")]
pub struct JsBrain {
    inner: RefCell<Machine>,
}

fn stats_out(m: &Machine) -> StatsOut {
    let s = &m.last_stats;
    StatsOut {
        frame: s.frame as f64, base_acts: s.base_acts as u32, apex_acts: s.apex_acts as u32, apex_cost: s.apex_cost,
        waves: s.waves as u32, bids: s.bids as u32, accepted: s.accepted as u32, minted: s.minted as u32, reused: s.reused as u32,
        retired: s.retired as u32, added: s.added as u32, neurons: s.neurons as u32, patterns: s.patterns as u32,
        connections: s.connections as u32, voters: s.voters as u32, voters_above_base: s.voters_above_base as u32,
        elapsed_ms: s.elapsed_ms,
    }
}

#[napi]
impl JsBrain {
    #[napi(constructor)]
    pub fn new(options: Option<BrainOptions>) -> Result<Self> {
        let o = options.unwrap_or(BrainOptions { h: None, time_radius: None, policy: None, action_policy: None });
        let mut policy = match o.policy.as_deref() {
            Some("eventsToActions") => Policy::events_to_actions(),
            _ => Policy::all_on(),
        };
        policy.action = match o.action_policy.as_deref() {
            Some("nothing") => ActionPolicy::Nothing,
            _ => ActionPolicy::LeastBad,
        };
        let m = Machine::new(o.h.unwrap_or(256) as usize, policy, o.time_radius.unwrap_or(1));
        Ok(Self { inner: RefCell::new(m) })
    }

    /// Declare a spatial activation dimension; returns its axis index (1 or 2).
    #[napi(js_name = "declareAxis")]
    pub fn declare_axis(&self, name: String, radius: i32) -> Result<u32> {
        Ok(self.inner.borrow_mut().declare_axis(&name, radius) as u32)
    }

    #[napi(js_name = "declareChannel")]
    pub fn declare_channel(&self, spec: ChannelSpec) -> Result<ChannelIds> {
        let mut m = self.inner.borrow_mut();
        let axes: Vec<usize> = spec.axes.unwrap_or_default().iter().map(|a| *a as usize).collect();
        let ev = spec.event.as_ref().map(|d| (d.name.as_str(), d.resolution));
        let ac = spec.action.as_ref().map(|d| (d.name.as_str(), d.resolution));
        let id = m.declare_channel(&spec.name, ev, ac, &axes);
        let ch = &m.channels[id as usize];
        Ok(ChannelIds { channel_id: id, event_dim: ch.event, action_dim: ch.action })
    }

    #[napi(js_name = "processFrame")]
    pub fn process_frame(&self, inputs: Int32Array, rewards: Option<Vec<RewardSpec>>) -> Result<FrameOut> {
        let mut m = self.inner.borrow_mut();
        let raw: &[i32] = &inputs;
        let mut ins = Vec::with_capacity(raw.len() / 4);
        for c in raw.chunks_exact(4) {
            if c[0] < 0 || c[1] < 0 { continue; }
            ins.push(Input { dim: c[0] as u32, bucket: c[1] as u32, pos: [0, c[2], c[3]] });
        }
        let rws: Vec<Reward> = rewards.unwrap_or_default().into_iter()
            .map(|r| Reward { reward: r.reward, channels: r.channels.unwrap_or_default(), frames: r.frames.unwrap_or(0) })
            .collect();
        let winners = m.process_frame(&ins, rws);
        let out: Vec<WinnerOut> = winners.iter().map(|w| WinnerOut {
            dim: w.dim, bucket: w.bucket, p1: w.pos[1], p2: w.pos[2],
            kind: if w.kind == Kind::Action { "action".into() } else { "event".into() },
            estimate: w.estimate, share: w.share, voters: w.voters as u32,
        }).collect();
        Ok(FrameOut { winners: out, stats: stats_out(&m) })
    }

    /// Clear every open activation and standing inference; keep the tables and connections.
    #[napi(js_name = "resetContext")]
    pub fn reset_context(&self) -> Result<()> {
        self.inner.borrow_mut().reset_context();
        Ok(())
    }

    /// How many living neurons stand at each height.
    #[napi(js_name = "heightCounts")]
    pub fn height_counts(&self) -> Result<Vec<u32>> {
        Ok(self.inner.borrow().height_counts().iter().map(|c| *c as u32).collect())
    }

    #[napi(js_name = "describeNeuron")]
    pub fn describe_neuron(&self, id: u32) -> Result<String> {
        let m = self.inner.borrow();
        if id as usize >= m.neurons.len() { return Ok(String::new()); }
        Ok(m.describe_neuron(id))
    }

    #[napi]
    pub fn stats(&self) -> Result<StatsOut> {
        Ok(stats_out(&self.inner.borrow()))
    }

    /// Base neuron id of a (dim, bucket) pair (D52).
    #[napi(js_name = "baseNeuron")]
    pub fn base_neuron(&self, dim: u32, bucket: u32) -> Result<u32> {
        Ok(self.inner.borrow().base_neuron(dim, bucket))
    }

    /// Dump every live interneuron and every base neuron holding a table, as text.
    #[napi(js_name = "dumpTables")]
    pub fn dump_tables(&self, limit: u32) -> Result<String> {
        let m = self.inner.borrow();
        let mut s = String::new();
        for n in m.neurons.iter().filter(|n| n.alive && (!n.table.is_empty() || n.kind == Kind::Inter)).take(limit as usize) {
            s.push_str(&m.describe_neuron(n.id));
        }
        Ok(s)
    }
}
