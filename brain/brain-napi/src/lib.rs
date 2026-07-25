/// N-API bindings for brain-core.
///
/// Exposes the Brain struct as a JavaScript class with the same API shape as
/// the JS Brain class, so the host code (job.js, test jobs) can swap in the
/// native addon without changing call sites.
///
/// Marshalling conventions:
///   - JS Maps (channelId→dimMap, etc.) are received as JsObject entries and
///     converted to FxHashMaps on the Rust side.
///   - Return values are plain JS objects with camelCase property names matching
///     the existing JS Brain API so the host-side renderer and job code work unchanged.
///   - Options are received as a JS object with optional properties.

use napi::bindgen_prelude::*;
use napi::{JsFunction, JsNumber, JsObject, JsString, JsUnknown};
use napi_derive::napi;
use rustc_hash::FxHashMap;
use std::cell::RefCell;

use brain_core::brain::{Brain as CoreBrain, FrameResult};
use brain_core::types::{ConsensusMode, Coordinate, GroupMode, NeuronType};

// ── Helper: JS Map iteration ────────────────────────────────────────────────

/// Read a JS Map<number, V> by calling its entries() iterator and collecting.
/// Returns (key, JsUnknown) pairs so the caller can interpret V.
fn read_js_map(_env: &Env, map_obj: &JsObject) -> Result<Vec<(i64, JsUnknown)>> {
    let size: u32 = map_obj.get_named_property::<JsNumber>("size")?.get_uint32()?;
    if size == 0 { return Ok(Vec::new()); }

    let entries_fn: JsFunction = map_obj.get_named_property("entries")?;
    let iterator: JsObject = entries_fn.call_without_args(Some(map_obj))?.coerce_to_object()?;

    let mut result = Vec::with_capacity(size as usize);
    loop {
        let next_fn: JsFunction = iterator.get_named_property("next")?;
        let entry: JsObject = next_fn.call_without_args(Some(&iterator))?.coerce_to_object()?;
        let done: bool = entry.get_named_property::<napi::JsBoolean>("done")?.get_value()?;
        if done { break; }
        let value: JsObject = entry.get_named_property::<JsObject>("value")?;
        let key: JsNumber = value.get_element(0)?;
        let val: JsUnknown = value.get_element(1)?;
        result.push((key.get_int64()?, val));
    }
    Ok(result)
}

/// Read a JS Map<number, number> into an FxHashMap.
fn read_number_map(env: &Env, map_obj: &JsObject) -> Result<FxHashMap<u32, f64>> {
    let entries = read_js_map(env, map_obj)?;
    let mut result = FxHashMap::default();
    for (key, val) in entries {
        let v: f64 = val.coerce_to_number()?.get_double()?;
        result.insert(key as u32, v);
    }
    Ok(result)
}

/// Read a JS Map<number, Map<number, number>> into nested FxHashMaps.
fn read_nested_map(env: &Env, map_obj: &JsObject) -> Result<FxHashMap<u32, FxHashMap<u32, f64>>> {
    let entries = read_js_map(env, map_obj)?;
    let mut result = FxHashMap::default();
    for (key, val) in entries {
        let inner_obj: JsObject = val.coerce_to_object()?;
        let inner = read_number_map(env, &inner_obj)?;
        result.insert(key as u32, inner);
    }
    Ok(result)
}

/// Read a JS Map<number, number> as a Vec<(f64, f64)>, preserving f64 precision on the key side.
/// Used for the (value, reward) inner maps in the `learn()` actions payload — value is the action scalar
/// (continuous in stocks, integer in MNIST) so we can't truncate it to int64 like read_number_map does.
fn read_value_reward_pairs(_env: &Env, map_obj: &JsObject) -> Result<Vec<(f64, f64)>> {
    let size: u32 = map_obj.get_named_property::<JsNumber>("size")?.get_uint32()?;
    if size == 0 { return Ok(Vec::new()); }

    let entries_fn: JsFunction = map_obj.get_named_property("entries")?;
    let iterator: JsObject = entries_fn.call_without_args(Some(map_obj))?.coerce_to_object()?;

    let mut result = Vec::with_capacity(size as usize);
    loop {
        let next_fn: JsFunction = iterator.get_named_property("next")?;
        let entry: JsObject = next_fn.call_without_args(Some(&iterator))?.coerce_to_object()?;
        let done: bool = entry.get_named_property::<napi::JsBoolean>("done")?.get_value()?;
        if done { break; }
        let value: JsObject = entry.get_named_property::<JsObject>("value")?;
        let key: JsNumber = value.get_element(0)?;
        let val: JsNumber = value.get_element(1)?;
        result.push((key.get_double()?, val.get_double()?));
    }
    Ok(result)
}

/// Read a JS Map<channelId, Map<dimId, Map<value, reward>>> — the `learn()` actions payload shape.
/// Outer two layers are integer-keyed (channelId, dimId); the innermost is value-keyed (action scalar) with reward as the value.
fn read_actions_map(
    env: &Env,
    map_obj: &JsObject,
) -> Result<FxHashMap<u32, FxHashMap<u32, Vec<(f64, f64)>>>> {
    let entries = read_js_map(env, map_obj)?;
    let mut result = FxHashMap::default();
    for (key, val) in entries {
        let inner_obj: JsObject = val.coerce_to_object()?;
        let inner_entries = read_js_map(env, &inner_obj)?;
        let mut inner = FxHashMap::default();
        for (dim_key, dim_val) in inner_entries {
            let dim_obj: JsObject = dim_val.coerce_to_object()?;
            let pairs = read_value_reward_pairs(env, &dim_obj)?;
            inner.insert(dim_key as u32, pairs);
        }
        result.insert(key as u32, inner);
    }
    Ok(result)
}

/// Create a JS Map from entries and set it on a parent object.
fn create_js_map(env: &Env, entries: Vec<(u32, JsUnknown)>) -> Result<JsObject> {
    let global = env.get_global()?;
    let map_constructor: JsFunction = global.get_named_property("Map")?;
    let map: JsObject = map_constructor.new_instance(&[] as &[JsUnknown])?.coerce_to_object()?;
    let set_fn: JsFunction = map.get_named_property("set")?;
    for (key, value) in entries {
        let js_key = env.create_uint32(key)?;
        set_fn.call(Some(&map), &[js_key.into_unknown(), value])?;
    }
    Ok(map)
}

// ── Brain JS class ──────────────────────────────────────────────────────────

#[napi(js_name = "Brain")]
pub struct JsBrain {
    /// Interior mutability — napi-rs passes &self to methods, but Brain needs &mut self.
    inner: RefCell<CoreBrain>,
}

#[napi]
impl JsBrain {
    /// Create a new Brain instance from an options object.
    ///
    /// Options shape (all optional, with defaults matching JS Brain):
    ///   contextLength: number (default 10)
    ///   groupThreshold: number (default 0.5) — the single grouping coefficient θ, shared by spatial and temporal.
    ///     It SEEDS one adaptive per-unit threshold: recognition fires at similarity ≥ θ and correction at < θ
    ///     until a unit has error history, after which both float together off its stats (see groupMode).
    ///   groupMode: string (default 'neutral') — how the live grouping threshold adapts from error stats
    ///     ('static' pins it at the θ seed; conservative/neutral/aggressive shift it by mean ± σ)
    ///   patternForgetRate: number (default 0.01)
    ///   regions: number (default 1)
    ///   columns: number (default 1)
    ///   consensus: string 'democratic' | 'nb' (default 'democratic')
    ///   debug: boolean (default false)
    ///   learning: boolean (default true) — fixed for the life of the instance; construct with false for frozen evaluation
    ///   spatialRefine: boolean (default true) — refine the served entry toward what it serves each frame
    ///   spatialDelete: boolean (default true) — run the delete pass that retires children below their cost
    ///
    /// The retired per-phase `mergeThreshold` / `errorCorrectionThreshold` knobs and their `spatial*` / `temporal*`
    /// variants collapsed into the single `groupThreshold` — `error = 1 − merge` is one Jaccard test read from
    /// opposite sides, so there is one number, not six. Passing any retired key logs a one-time warning and is ignored.
    #[napi(constructor)]
    pub fn new(_env: Env, options: Option<JsObject>) -> Result<Self> {
        let (context_length, group_threshold, group_mode,
             pattern_forget_rate, regions, columns, consensus_mode, debug) = match options {
            Some(ref opts) => {
                let cl = get_opt_u32(opts, "contextLength")?.unwrap_or(10);

                warn_retired_grouping_keys(opts)?;

                let group_threshold = get_opt_f64(opts, "groupThreshold")?.unwrap_or(0.5);
                let mode_str = get_opt_string(opts, "groupMode")?
                    .unwrap_or_else(|| "neutral".to_string());
                let mode = parse_group_mode(&mode_str)?;

                let pfr = get_opt_f64(opts, "patternForgetRate")?.unwrap_or(0.01);
                let r = get_opt_u32(opts, "regions")?.unwrap_or(1) as usize;
                let c = get_opt_u32(opts, "columns")?.unwrap_or(1) as usize;
                let consensus_str = get_opt_string(opts, "consensus")?
                    .unwrap_or_else(|| "democratic".to_string());
                let consensus = parse_consensus_mode(&consensus_str)?;
                let d = get_opt_bool(opts, "debug")?.unwrap_or(false);
                (cl, group_threshold, mode, pfr, r, c, consensus, d)
            }
            None => (10, 0.5, GroupMode::Neutral, 0.01, 1, 1, ConsensusMode::Democratic, false),
        };

        let apex_coverage = opt_bool(options.as_ref(), "apexCoverage")?.unwrap_or(false);

        // The learning state is fixed at construction: a frozen evaluation is a separate brain
        // instance loaded from a backup, not a toggled one.
        let learning = opt_bool(options.as_ref(), "learning")?.unwrap_or(true);

        // Spatial refinement and the delete pass, both on by design. Exposed so an experiment can
        // isolate their contribution without a rebuild.
        let refine = opt_bool(options.as_ref(), "spatialRefine")?.unwrap_or(true);
        let delete = opt_bool(options.as_ref(), "spatialDelete")?.unwrap_or(true);
        let inner = CoreBrain::new(
            context_length, group_threshold, group_mode,
            pattern_forget_rate, regions, columns, consensus_mode, debug, learning,
            apex_coverage, refine, delete,
        );

        Ok(Self { inner: RefCell::new(inner) })
    }

    /// Register a channel spec. Accepts the same spec shape as JS Brain.
    ///
    /// spec: { name, learnActionSequences?, dimensions: [...] }
    /// Returns: { channelId: number, dimensionIds: {name: id, ...} }
    #[napi(js_name = "registerChannelSpec")]
    pub fn register_channel_spec(&self, env: Env, spec: JsObject) -> Result<JsObject> {
        let name: String = spec.get_named_property::<JsString>("name")?.into_utf8()?.into_owned()?;
        let learn_action_sequences = get_opt_bool(&spec, "learnActionSequences")?.unwrap_or(false);

        // Parse dimensions array
        let dims_array: JsObject = spec.get_named_property("dimensions")?;
        let dims_len = dims_array.get_array_length()?;
        let mut dimensions = Vec::with_capacity(dims_len as usize);

        for i in 0..dims_len {
            let dim: JsObject = dims_array.get_element(i)?;
            let dim_name: String = dim.get_named_property::<JsString>("name")?.into_utf8()?.into_owned()?;
            let kind_str: String = dim.get_named_property::<JsString>("kind")?.into_utf8()?.into_owned()?;
            let kind = match kind_str.as_str() {
                "action" => brain_core::brain::DimKind::Action,
                _ => brain_core::brain::DimKind::Input,
            };
            let resolution = dim.get_named_property::<JsNumber>("resolution")?.get_uint32()?;
            let mode = get_opt_string(&dim, "mode")?;
            let boundaries = get_opt_f64_array(&dim, "boundaries")?;
            let actions = get_opt_i32_array(&dim, "actions")?;
            let default_action = get_opt_i32(&dim, "defaultAction")?;
            let warmup_samples = get_opt_u32(&dim, "warmupSamples")?.map(|v| v as usize);

            dimensions.push(brain_core::brain::DimSpecInput {
                name: dim_name,
                kind,
                resolution,
                mode,
                boundaries,
                actions,
                default_action,
                warmup_samples,
            });
        }

        let reg = self.inner.borrow_mut().register_channel_spec(
            &name, dimensions, learn_action_sequences,
        );

        // Build return object: { channelId, dimensionIds: { name: id, ... } }
        let mut result = env.create_object()?;
        result.set_named_property("channelId", env.create_uint32(reg.channel_id)?)?;

        let mut dim_ids_obj = env.create_object()?;
        for (dim_name, dim_id) in &reg.dimension_ids {
            dim_ids_obj.set_named_property(dim_name, env.create_uint32(*dim_id)?)?;
        }
        result.set_named_property("dimensionIds", dim_ids_obj)?;

        Ok(result)
    }

    /// Process a single frame. Accepts Maps matching JS Brain.processFrame.
    ///
    /// inputs: Map<channelId, Map<dimId, scalar>>
    /// rewards: Map<channelId, reward>
    /// Returns: { inferences: Map<channelId, [...]>, votes: Array<...>, frame: { elapsed, timings } }
    #[napi(js_name = "processFrame")]
    pub fn process_frame(&self, env: Env, inputs: JsObject, rewards: JsObject) -> Result<JsObject> {
        // Marshal inputs: Map<number, Map<number, number>> → FxHashMap
        let rust_inputs = read_nested_map(&env, &inputs)?;
        let rust_rewards = read_number_map(&env, &rewards)?;

        // Call core
        let frame_result = self.inner.borrow_mut().process_frame(&rust_inputs, &rust_rewards);

        // Marshal result back to JS
        build_frame_result(&env, &frame_result)
    }

    /// Get active neurons in context with their levels.
    /// Returns an array of { neuronId, temporalLevel, suppressed } objects.
    /// suppressed=true means the neuron activated a higher-level pattern and
    /// should not be counted as an independent voter.
    #[napi(js_name = "getActiveNeurons")]
    pub fn get_active_neurons(&self, env: Env) -> Result<JsObject> {
        let brain = self.inner.borrow();
        let snapshot = brain.get_context_snapshot();
        let mut arr = env.create_array_with_length(snapshot.len())?;
        for (i, (neuron_id, _frame, temporal_level, state)) in snapshot.iter().enumerate() {
            let mut obj = env.create_object()?;
            obj.set_named_property("neuronId", env.create_uint32(*neuron_id as u32)?)?;
            obj.set_named_property("temporalLevel", env.create_uint32(*temporal_level as u32)?)?;
            obj.set_named_property("suppressed", env.get_boolean(state.activated_pattern_id.is_some())?)?;
            arr.set_element(i as u32, obj)?;
        }
        Ok(arr)
    }

    /// Inspect one neuron: returns { neuronId, temporalLevel, spatialLevel, channelId | null,
    /// parentId | null, context: [{ neuronId, distance, strength }, ...] }.
    /// channelId is the neuron's own anchor channel — for sensory neurons their registered
    /// channel, for spatial-correction neurons the founding pixel's channel (inherited at mint).
    /// Context entries come from the parent neuron's routing-table entry
    /// for this child pattern. Level-0 sensory neurons have parent_id=null
    /// and empty context.
    #[napi(js_name = "inspectNeuron")]
    pub fn inspect_neuron(&self, env: Env, neuron_id: u32) -> Result<JsObject> {
        let brain = self.inner.borrow();
        let info = brain.inspect_neuron(neuron_id as u64);
        let mut obj = env.create_object()?;
        obj.set_named_property("neuronId", env.create_uint32(info.neuron_id as u32)?)?;
        obj.set_named_property("temporalLevel", env.create_uint32(info.temporal_level as u32)?)?;
        obj.set_named_property("spatialLevel", env.create_uint32(info.spatial_level as u32)?)?;
        match info.channel_id {
            Some(c) => obj.set_named_property("channelId", env.create_uint32(c as u32)?)?,
            None => obj.set_named_property("channelId", env.get_null()?)?,
        }
        match info.parent_id {
            Some(p) => obj.set_named_property("parentId", env.create_uint32(p as u32)?)?,
            None => obj.set_named_property("parentId", env.get_null()?)?,
        }
        let mut ctx_arr = env.create_array_with_length(info.context.len())?;
        for (i, (nid, dist, strength)) in info.context.iter().enumerate() {
            let mut e = env.create_object()?;
            e.set_named_property("neuronId", env.create_uint32(*nid as u32)?)?;
            e.set_named_property("distance", env.create_uint32(*dist as u32)?)?;
            e.set_named_property("strength", env.create_double(*strength)?)?;
            ctx_arr.set_element(i as u32, e)?;
        }
        obj.set_named_property("context", ctx_arr)?;
        Ok(obj)
    }

    /// Dump a neuron's outgoing connections.
    /// Returns [{ distance, targetId, strength, reward }, ...].
    #[napi(js_name = "getNeuronConnections")]
    pub fn get_neuron_connections(&self, env: Env, neuron_id: u32) -> Result<JsObject> {
        let brain = self.inner.borrow();
        let conns = brain.get_neuron_connections(neuron_id as u64);
        let mut arr = env.create_array_with_length(conns.len())?;
        for (i, (dist, target, strength, reward)) in conns.iter().enumerate() {
            let mut obj = env.create_object()?;
            obj.set_named_property("distance", env.create_uint32(*dist as u32)?)?;
            obj.set_named_property("targetId", env.create_uint32(*target as u32)?)?;
            obj.set_named_property("strength", env.create_double(*strength)?)?;
            obj.set_named_property("reward", env.create_double(*reward)?)?;
            arr.set_element(i as u32, obj)?;
        }
        Ok(arr)
    }

    /// Toggle per-vote resolution on processFrame's `votes` payload. Off by
    /// default — per-vote resolution allocates strings and walks parent
    /// chains, so training runs that don't consume votes pay nothing.
    #[napi(js_name = "setEmitVotes")]
    pub fn set_emit_votes(&self, enabled: bool) -> Result<()> {
        self.inner.borrow_mut().set_emit_votes(enabled);
        Ok(())
    }

    /// Supervised wiring step that sits on top of the last `processFrame` call.
    /// `actions: Map<channelId, Map<dimId, Map<value, reward>>>` names every action target with its per-value reward.
    /// Each `value` is quantized to the corresponding action neuron; reward is applied to that connection
    /// via smoothed accumulation (strength += 1, reward = running mean). Callers typically supply every action value
    /// on the dim — correct value with reward=1, others with reward=0 — so `conn.reward` converges to P(target|voter).
    /// `distance` is the connection-table slot at which to wire and read back.
    /// Wires every currently-active age-0 voter to every supplied action target at the given distance.
    /// Then runs a post-wire inference sweep at age (distance - 1) and returns the resulting FrameResult.
    /// Single-frame supervised harnesses (MNIST) pass distance=1 to match the existing temporal voting slot.
    #[napi(js_name = "learn")]
    pub fn learn(&self, env: Env, actions: JsObject, distance: u32) -> Result<JsObject> {
        let rust_actions = read_actions_map(&env, &actions)?;
        let frame_result = self.inner.borrow_mut().learn(&rust_actions, distance);
        build_frame_result(&env, &frame_result)
    }

    /// Reset brain memory state for a clean episode start.
    #[napi(js_name = "resetContext")]
    pub fn reset_context(&self) -> Result<()> {
        self.inner.borrow_mut().reset_context();
        Ok(())
    }

    /// Hard reset: clears ALL learned data.
    #[napi(js_name = "resetBrain")]
    pub fn reset_brain(&self) -> Result<()> {
        self.inner.borrow_mut().reset_brain();
        Ok(())
    }

    /// Reset accuracy and reward stats for a new episode.
    #[napi(js_name = "resetAccuracyStats")]
    pub fn reset_accuracy_stats(&self) -> Result<()> {
        self.inner.borrow_mut().reset_accuracy_stats();
        Ok(())
    }

    /// Cumulative count of spatial corrections minted since brain start (or last hard reset).
    #[napi(js_name = "getSpatialCorrectionCount")]
    pub fn get_spatial_correction_count(&self) -> Result<u32> {
        Ok(self.inner.borrow().get_spatial_correction_count() as u32)
    }

    /// Cumulative count of spatial children retired by the one test's delete pass since brain start.
    /// Paired with getSpatialCorrectionCount, this is the cold-start churn the Phase-1 gate watches.
    #[napi(js_name = "getSpatialDeletionCount")]
    pub fn get_spatial_deletion_count(&self) -> Result<u32> {
        Ok(self.inner.borrow().get_spatial_deletion_count() as u32)
    }

    /// Number of correction neurons currently sitting above the base spatial level.
    #[napi(js_name = "countActiveSpatialCorrections")]
    pub fn count_active_spatial_corrections(&self) -> Result<u32> {
        Ok(self.inner.borrow().count_active_spatial_corrections() as u32)
    }

    /// Per-level count of correction neurons. Returns array where index 0 is level 1, etc.
    /// Length tells you the maximum spatial level reached.
    #[napi(js_name = "spatialLevelCounts")]
    pub fn spatial_level_counts(&self) -> Result<Vec<u32>> {
        Ok(self.inner.borrow().spatial_level_counts())
    }

    /// Per-level count of PAID correction neurons — patterns that may fire, as opposed to unpaid
    /// hypotheses still accumulating evidence toward their price. Same indexing as spatialLevelCounts.
    #[napi(js_name = "spatialLevelPaidCounts")]
    pub fn spatial_level_paid_counts(&self) -> Result<Vec<u32>> {
        Ok(self.inner.borrow().spatial_level_paid_counts())
    }

    /// Declare the SPATIAL (d=0 co-activation) neighbor channel set for a registered channel.
    /// This is the set a channel may co-fire with in the same frame to form a spatial pattern.
    /// Names not in the registry are silently ignored; an empty list shrinks the spatial
    /// neighborhood to {itself}; channels with no call retain the default all-pairs spatial
    /// neighborhood. Call AFTER registering all channels — neighbor names are resolved at this call.
    #[napi(js_name = "setSpatialNeighbors")]
    pub fn set_spatial_neighbors(&self, name: String, neighbor_names: Vec<String>) -> Result<()> {
        self.inner.borrow_mut().set_spatial_neighbors(&name, &neighbor_names);
        Ok(())
    }

    /// Declare per-level SPATIAL neighbor sets for a registered channel — the level-based radius.
    /// `neighborNamesByLevel[l]` is the neighbor list a level-l neuron of this channel uses (e.g. the
    /// radius-(l+1) neighborhood of a retinotopic pixel); levels past the end reuse the last set. Each list is
    /// used verbatim like `setSpatialNeighbors`. Call AFTER registering all channels.
    #[napi(js_name = "setSpatialNeighborLevels")]
    pub fn set_spatial_neighbor_levels(&self, name: String, neighbor_names_by_level: Vec<Vec<String>>) -> Result<()> {
        self.inner.borrow_mut().set_spatial_neighbor_levels(&name, &neighbor_names_by_level);
        Ok(())
    }

    /// Declare the TEMPORAL (d>0 sequence) neighbor channel set for a registered channel.
    /// This is the set whose past a channel may sequence against to predict the future.
    /// Same name-resolution and all-pairs-default semantics as `setSpatialNeighbors`.
    #[napi(js_name = "setTemporalNeighbors")]
    pub fn set_temporal_neighbors(&self, name: String, neighbor_names: Vec<String>) -> Result<()> {
        self.inner.borrow_mut().set_temporal_neighbors(&name, &neighbor_names);
        Ok(())
    }

    /// Declare the same neighbor set for BOTH phases — convenience for channels whose spatial and
    /// temporal neighbors coincide (e.g. retinotopic pixels). Equivalent to calling
    /// `setSpatialNeighbors` and `setTemporalNeighbors` with the same list.
    #[napi(js_name = "setChannelNeighbors")]
    pub fn set_channel_neighbors(&self, name: String, neighbor_names: Vec<String>) -> Result<()> {
        self.inner.borrow_mut().set_channel_neighbors(&name, &neighbor_names);
        Ok(())
    }

    /// Look up a dimension ID by its registered name.
    #[napi(js_name = "getDimensionIdByName")]
    pub fn get_dimension_id_by_name(&self, env: Env, name: String) -> Result<JsUnknown> {
        match self.inner.borrow().get_dimension_id_by_name(&name) {
            Some(id) => Ok(env.create_uint32(id)?.into_unknown()),
            None => Ok(env.get_null()?.into_unknown()),
        }
    }

    /// Look up a neuron ID by its (dimId, bucketId) coordinate.
    #[napi(js_name = "getNeuronIdByCoordinate")]
    pub fn get_neuron_id_by_coordinate(&self, env: Env, dim_id: u32, bucket_id: i32) -> Result<JsUnknown> {
        let coord = brain_core::types::Coordinate { dim_id, bucket_id };
        match self.inner.borrow().get_neuron_id_by_coordinate(&coord) {
            Some(id) => Ok(env.create_uint32(id as u32)?.into_unknown()),
            None => Ok(env.get_null()?.into_unknown()),
        }
    }

    /// Save brain state to a labeled backup folder.
    #[napi]
    pub fn save(&self, job_dir: String, label: String) -> Result<()> {
        self.inner.borrow_mut()
            .save(std::path::Path::new(&job_dir), &label)
            .map_err(|e| Error::from_reason(e))?;
        Ok(())
    }

    /// Load a backup by label.
    #[napi]
    pub fn load(&self, job_dir: String, label: String) -> Result<()> {
        self.inner.borrow_mut()
            .load(std::path::Path::new(&job_dir), &label)
            .map_err(|e| Error::from_reason(e))?;
        Ok(())
    }

    /// Save brain context (active neurons, frame number, rewards) to a labeled folder.
    #[napi(js_name = "saveContext")]
    pub fn save_context(&self, job_dir: String, label: String) -> Result<()> {
        self.inner.borrow()
            .save_context(std::path::Path::new(&job_dir), &label)
            .map_err(|e| Error::from_reason(e))?;
        Ok(())
    }

    /// Load brain context from a labeled folder.
    #[napi(js_name = "loadContext")]
    pub fn load_context(&self, job_dir: String, label: String) -> Result<()> {
        self.inner.borrow_mut()
            .load_context(std::path::Path::new(&job_dir), &label)
            .map_err(|e| Error::from_reason(e))?;
        Ok(())
    }

    /// Get the per-frame summary for the renderer.
    #[napi(js_name = "getFrameSummary")]
    pub fn get_frame_summary(&self, env: Env) -> Result<JsObject> {
        let summary = self.inner.borrow().get_frame_summary();
        let mut obj = env.create_object()?;
        obj.set_named_property("frameNumber", env.create_int64(summary.frame_number as i64)?)?;
        obj.set_named_property("neuronCount", env.create_uint32(summary.neuron_count as u32)?)?;
        obj.set_named_property("maxTemporalLevel", env.create_uint32(summary.max_temporal_level)?)?;
        obj.set_named_property("maxSpatialLevel", env.create_uint32(summary.max_spatial_level)?)?;

        // Spread diagnostic stats with camelCase names matching JS
        set_diagnostic_stats(&env, &mut obj, &summary.stats)?;

        Ok(obj)
    }

    /// Get episode summary with all diagnostic information.
    #[napi(js_name = "getEpisodeSummary")]
    pub fn get_episode_summary(&self, env: Env) -> Result<JsObject> {
        let summary = self.inner.borrow().get_episode_summary();
        let mut obj = env.create_object()?;
        obj.set_named_property("frameNumber", env.create_int64(summary.frame_number as i64)?)?;

        // Nest accuracy + mispredictions under their original keys
        let mut accuracy_obj = env.create_object()?;
        accuracy_obj.set_named_property("correct", env.create_int64(summary.stats.accuracy_correct as i64)?)?;
        accuracy_obj.set_named_property("total", env.create_int64(summary.stats.accuracy_total as i64)?)?;
        obj.set_named_property("accuracy", accuracy_obj)?;

        let mut mispredictions = env.create_array_with_length(summary.stats.mispredictions.len())?;
        for (i, mp) in summary.stats.mispredictions.iter().enumerate() {
            let mut mp_obj = env.create_object()?;
            mp_obj.set_named_property("channelId", env.create_uint32(mp.channel_id)?)?;
            set_coordinate(&env, &mut mp_obj, "predicted", &mp.predicted)?;
            set_coordinate(&env, &mut mp_obj, "actual", &mp.actual)?;
            mispredictions.set_element(i as u32, mp_obj)?;
        }
        obj.set_named_property("mispredictions", mispredictions)?;

        Ok(obj)
    }

    /// Get start-of-frame diagnostic snapshot. Returns null when frame was empty.
    #[napi(js_name = "getStartFrameInfo")]
    pub fn get_start_frame_info(&self, env: Env) -> Result<JsUnknown> {
        let info = self.inner.borrow().get_start_frame_info();
        match info {
            None => Ok(env.get_null()?.into_unknown()),
            Some(info) => {
                let mut obj = env.create_object()?;
                obj.set_named_property("frameNumber", env.create_int64(info.frame_number as i64)?)?;

                // rewards as Map<channelId, reward>
                let reward_entries: Vec<(u32, JsUnknown)> = info.rewards.iter()
                    .map(|(&ch, &r)| Ok((ch, env.create_double(r)?.into_unknown())))
                    .collect::<Result<Vec<_>>>()?;
                let rewards_map = create_js_map(&env, reward_entries)?;
                obj.set_named_property("rewards", rewards_map)?;

                // frame points array
                let mut frame_arr = env.create_array_with_length(info.frame.len())?;
                for (i, point) in info.frame.iter().enumerate() {
                    let mut pt = env.create_object()?;
                    let mut coord = env.create_object()?;
                    coord.set_named_property("dimId", env.create_uint32(point.coordinate.dim_id)?)?;
                    coord.set_named_property("bucketId", env.create_int32(point.coordinate.bucket_id)?)?;
                    pt.set_named_property("coordinate", coord)?;
                    pt.set_named_property("channelId", env.create_uint32(point.channel_id)?)?;
                    let type_str = match point.neuron_type {
                        NeuronType::Event => "event",
                        NeuronType::Action => "action",
                    };
                    pt.set_named_property("type", env.create_string(type_str)?)?;
                    frame_arr.set_element(i as u32, pt)?;
                }
                obj.set_named_property("frame", frame_arr)?;

                // dimensionIdToName as plain object { dimId: name }
                let mut dim_names = env.create_object()?;
                for (&dim_id, name) in &info.dimension_id_to_name {
                    dim_names.set_property(
                        env.create_uint32(dim_id)?,
                        env.create_string(name)?,
                    )?;
                }
                obj.set_named_property("dimensionIdToName", dim_names)?;

                Ok(obj.into_unknown())
            }
        }
    }
}

// ── Helper: build processFrame result ───────────────────────────────────────

/// Convert FrameResult into the JS return shape:
/// { inferences: Map<channelId, [...]>, votes: Array<...>, frame: { elapsed, timings } }
fn build_frame_result(env: &Env, result: &FrameResult) -> Result<JsObject> {
    let mut obj = env.create_object()?;

    // Build inferences Map<channelId, Array<{dimId, kind, winner, continuous}>>
    let mut inference_entries: Vec<(u32, JsUnknown)> = Vec::new();
    for (&channel_id, dims) in &result.inferences {
        let mut arr = env.create_array_with_length(dims.len())?;
        for (i, dim) in dims.iter().enumerate() {
            let mut inf_obj = env.create_object()?;
            inf_obj.set_named_property("dimId", env.create_uint32(dim.dim_id)?)?;
            let kind_str = match dim.kind {
                NeuronType::Event => "event",
                NeuronType::Action => "action",
            };
            inf_obj.set_named_property("kind", env.create_string(kind_str)?)?;

            // winner: { neuronId, value, strength, score }
            let mut winner_obj = env.create_object()?;
            winner_obj.set_named_property("neuronId", env.create_uint32(dim.winner.neuron_id as u32)?)?;
            match dim.winner.value {
                Some(v) => winner_obj.set_named_property("value", env.create_double(v)?)?,
                None => winner_obj.set_named_property("value", env.get_null()?)?,
            };
            winner_obj.set_named_property("strength", env.create_double(dim.winner.strength)?)?;
            winner_obj.set_named_property("score", env.create_double(dim.winner.score)?)?;
            inf_obj.set_named_property("winner", winner_obj)?;

            // continuous
            match dim.continuous {
                Some(c) => inf_obj.set_named_property("continuous", env.create_double(c)?)?,
                None => inf_obj.set_named_property("continuous", env.get_null()?)?,
            };

            arr.set_element(i as u32, inf_obj)?;
        }
        inference_entries.push((channel_id, arr.into_unknown()));
    }
    let inferences_map = create_js_map(env, inference_entries)?;
    obj.set_named_property("inferences", inferences_map)?;

    // votes: Array<{voterId, voterLabel, targetId, targetType, channelId, dimId, value, distance, strength, reward}>
    let mut votes_arr = env.create_array_with_length(result.votes.len())?;
    for (i, v) in result.votes.iter().enumerate() {
        let mut v_obj = env.create_object()?;
        v_obj.set_named_property("voterId", env.create_uint32(v.voter_id as u32)?)?;
        v_obj.set_named_property("voterLabel", env.create_string(&v.voter_label)?)?;
        v_obj.set_named_property("voterTemporalLevel", env.create_uint32(v.voter_temporal_level as u32)?)?;
        v_obj.set_named_property("targetId", env.create_uint32(v.target_id as u32)?)?;
        let kind_str = match v.target_type {
            NeuronType::Event => "event",
            NeuronType::Action => "action",
        };
        v_obj.set_named_property("targetType", env.create_string(kind_str)?)?;
        v_obj.set_named_property("channelId", env.create_uint32(v.channel_id as u32)?)?;
        v_obj.set_named_property("dimId", env.create_uint32(v.dim_id as u32)?)?;
        v_obj.set_named_property("value", env.create_int32(v.value)?)?;
        v_obj.set_named_property("distance", env.create_uint32(v.distance as u32)?)?;
        v_obj.set_named_property("strength", env.create_double(v.strength)?)?;
        v_obj.set_named_property("reward", env.create_double(v.reward)?)?;
        votes_arr.set_element(i as u32, v_obj)?;
    }
    obj.set_named_property("votes", votes_arr)?;

    // frame: { elapsed, timings }
    let mut frame_obj = env.create_object()?;
    // elapsed in milliseconds (Rust stores seconds)
    frame_obj.set_named_property("elapsed", env.create_double(result.elapsed * 1000.0)?)?;

    // timings: per-section wall-clock (seconds). Nested under `frame` alongside elapsed.
    let t = &result.timings;
    let mut timings_obj = env.create_object()?;
    timings_obj.set_named_property("total", env.create_double(t.total)?)?;
    timings_obj.set_named_property("buildFrame", env.create_double(t.build_frame)?)?;
    timings_obj.set_named_property("createSensory", env.create_double(t.create_sensory)?)?;
    timings_obj.set_named_property("cleanupDead", env.create_double(t.cleanup_dead)?)?;
    timings_obj.set_named_property("ageContext", env.create_double(t.age_context)?)?;
    timings_obj.set_named_property("activate", env.create_double(t.activate)?)?;
    timings_obj.set_named_property("processSpatial", env.create_double(t.process_spatial)?)?;
    timings_obj.set_named_property("apexHandoff", env.create_double(t.apex_handoff)?)?;
    timings_obj.set_named_property("processTemporal", env.create_double(t.process_temporal)?)?;
    timings_obj.set_named_property("applyResults", env.create_double(t.apply_results)?)?;
    timings_obj.set_named_property("infer", env.create_double(t.infer)?)?;
    timings_obj.set_named_property("trackError", env.create_double(t.track_error)?)?;
    timings_obj.set_named_property("neuronLearnConnections",  env.create_double(t.neuron_learn_connections)?)?;
    timings_obj.set_named_property("neuronRecognizePatterns", env.create_double(t.neuron_recognize_patterns)?)?;
    timings_obj.set_named_property("neuronCorrectErrors",     env.create_double(t.neuron_correct_errors)?)?;
    timings_obj.set_named_property("neuronGenerateVotes",     env.create_double(t.neuron_generate_votes)?)?;
    timings_obj.set_named_property("recognizeCandidateSearch",     env.create_double(t.recognize_candidate_search)?)?;
    timings_obj.set_named_property("recognizeCandidateEval",       env.create_double(t.recognize_candidate_eval)?)?;
    timings_obj.set_named_property("recognizeCandidatesEvaluated", env.create_uint32(t.recognize_candidates_evaluated as u32)?)?;
    timings_obj.set_named_property("orchGetLevelTasks",      env.create_double(t.orch_get_level_tasks)?)?;
    timings_obj.set_named_property("orchDispatchFrame",      env.create_double(t.orch_dispatch_frame)?)?;
    timings_obj.set_named_property("orchCollectActivations", env.create_double(t.orch_collect_activations)?)?;
    timings_obj.set_named_property("orchCollectVotes",       env.create_double(t.orch_collect_votes)?)?;
    timings_obj.set_named_property("memGetLevelNeurons",        env.create_double(t.mem_get_level_neurons)?)?;
    timings_obj.set_named_property("memWriteBackLevelNeurons",  env.create_double(t.mem_write_back_level_neurons)?)?;
    timings_obj.set_named_property("memActivatePatterns",       env.create_double(t.mem_activate_patterns)?)?;
    frame_obj.set_named_property("timings", timings_obj)?;

    obj.set_named_property("frame", frame_obj)?;

    Ok(obj)
}

// ── Helper: diagnostic stats → JS object ────────────────────────────────────

/// Set the camelCase diagnostic stat properties on a JS object, matching the
/// shape JS Brain.getFrameSummary() produces via `...this.diagnostics.getStats()`.
fn set_diagnostic_stats(env: &Env, obj: &mut JsObject, stats: &brain_core::brain::DiagnosticStats) -> Result<()> {
    match stats.base_accuracy {
        Some(v) => obj.set_named_property("baseAccuracy", env.create_double(v)?)?,
        None => obj.set_named_property("baseAccuracy", env.get_null()?)?,
    };
    obj.set_named_property("accuracyCorrect", env.create_int64(stats.accuracy_correct as i64)?)?;
    obj.set_named_property("accuracyTotal", env.create_int64(stats.accuracy_total as i64)?)?;

    match stats.avg_reward {
        Some(v) => obj.set_named_property("avgReward", env.create_double(v)?)?,
        None => obj.set_named_property("avgReward", env.get_null()?)?,
    };
    obj.set_named_property("rewardCount", env.create_int64(stats.reward_count as i64)?)?;
    obj.set_named_property("totalReward", env.create_double(stats.total_reward)?)?;

    match stats.mape {
        Some(v) => obj.set_named_property("mape", env.create_double(v)?)?,
        None => obj.set_named_property("mape", env.get_null()?)?,
    };
    obj.set_named_property("mapeCount", env.create_int64(stats.mape_count as i64)?)?;

    Ok(())
}

/// Set a coordinate sub-object { dimId, bucketId } on a parent JS object.
fn set_coordinate(env: &Env, parent: &mut JsObject, key: &str, coord: &Coordinate) -> Result<()> {
    let mut obj = env.create_object()?;
    obj.set_named_property("dimId", env.create_uint32(coord.dim_id)?)?;
    obj.set_named_property("bucketId", env.create_int32(coord.bucket_id)?)?;
    parent.set_named_property(key, obj)?;
    Ok(())
}

// ── Helper: read optional JS object properties ─────────────────────────────

/// The grouping knobs retired when `mergeThreshold` / `errorCorrectionThreshold` (and their per-phase
/// `spatial*` / `temporal*` variants) collapsed into the single `groupThreshold`, and the shared
/// `errorCorrectionMode` was renamed to `groupMode`.
const RETIRED_GROUPING_KEYS: &[&str] = &[
    "mergeThreshold", "errorCorrectionThreshold", "errorCorrectionMode",
    "spatialMergeThreshold", "spatialErrorCorrectionThreshold", "spatialErrorCorrectionMode",
    "temporalMergeThreshold", "temporalErrorCorrectionThreshold", "temporalErrorCorrectionMode",
];

/// Warn (once, to stderr) if a caller still passes any retired grouping key, so a stale config or CLI flag
/// fails loudly-enough rather than silently doing nothing. The keys are ignored either way.
fn warn_retired_grouping_keys(opts: &JsObject) -> Result<()> {
    let present: Vec<&str> = RETIRED_GROUPING_KEYS.iter().copied()
        .filter(|k| is_present(opts, k).unwrap_or(false))
        .collect();
    if !present.is_empty() {
        eprintln!(
            "⚠️  Brain: ignoring retired grouping option(s) [{}] — they collapsed into the single `groupThreshold`. \
             Set `groupThreshold` (and `groupMode`) instead.",
            present.join(", "),
        );
    }
    Ok(())
}

/// True if `key` is present on `obj` with a non-undefined, non-null value.
fn is_present(obj: &JsObject, key: &str) -> Result<bool> {
    match obj.get_named_property::<JsUnknown>(key) {
        Ok(val) => {
            let t = val.get_type()?;
            Ok(t != napi::ValueType::Undefined && t != napi::ValueType::Null)
        }
        Err(_) => Ok(false),
    }
}

fn get_opt_string(obj: &JsObject, key: &str) -> Result<Option<String>> {
    match obj.get_named_property::<JsUnknown>(key) {
        Ok(val) => {
            if val.get_type()? == napi::ValueType::Undefined || val.get_type()? == napi::ValueType::Null {
                Ok(None)
            } else {
                Ok(Some(val.coerce_to_string()?.into_utf8()?.into_owned()?))
            }
        }
        Err(_) => Ok(None),
    }
}

fn get_opt_f64(obj: &JsObject, key: &str) -> Result<Option<f64>> {
    match obj.get_named_property::<JsUnknown>(key) {
        Ok(val) => {
            if val.get_type()? == napi::ValueType::Undefined || val.get_type()? == napi::ValueType::Null {
                Ok(None)
            } else {
                Ok(Some(val.coerce_to_number()?.get_double()?))
            }
        }
        Err(_) => Ok(None),
    }
}

fn get_opt_u32(obj: &JsObject, key: &str) -> Result<Option<u32>> {
    match obj.get_named_property::<JsUnknown>(key) {
        Ok(val) => {
            if val.get_type()? == napi::ValueType::Undefined || val.get_type()? == napi::ValueType::Null {
                Ok(None)
            } else {
                Ok(Some(val.coerce_to_number()?.get_uint32()?))
            }
        }
        Err(_) => Ok(None),
    }
}

fn get_opt_bool(obj: &JsObject, key: &str) -> Result<Option<bool>> {
    match obj.get_named_property::<JsUnknown>(key) {
        Ok(val) => {
            if val.get_type()? == napi::ValueType::Undefined || val.get_type()? == napi::ValueType::Null {
                Ok(None)
            } else {
                let b: napi::JsBoolean = val.try_into()?;
                Ok(Some(b.get_value()?))
            }
        }
        Err(_) => Ok(None),
    }
}

fn get_opt_f64_array(obj: &JsObject, key: &str) -> Result<Option<Vec<f64>>> {
    match obj.get_named_property::<JsUnknown>(key) {
        Ok(val) => {
            if val.get_type()? == napi::ValueType::Undefined || val.get_type()? == napi::ValueType::Null {
                Ok(None)
            } else {
                let arr: JsObject = val.coerce_to_object()?;
                let len = arr.get_array_length()?;
                let mut result = Vec::with_capacity(len as usize);
                for i in 0..len {
                    let v: JsNumber = arr.get_element(i)?;
                    result.push(v.get_double()?);
                }
                Ok(Some(result))
            }
        }
        Err(_) => Ok(None),
    }
}

fn get_opt_i32(obj: &JsObject, key: &str) -> Result<Option<i32>> {
    match obj.get_named_property::<JsUnknown>(key) {
        Ok(val) => {
            if val.get_type()? == napi::ValueType::Undefined || val.get_type()? == napi::ValueType::Null {
                Ok(None)
            } else {
                Ok(Some(val.coerce_to_number()?.get_int32()?))
            }
        }
        Err(_) => Ok(None),
    }
}

fn get_opt_i32_array(obj: &JsObject, key: &str) -> Result<Option<Vec<i32>>> {
    match obj.get_named_property::<JsUnknown>(key) {
        Ok(val) => {
            if val.get_type()? == napi::ValueType::Undefined || val.get_type()? == napi::ValueType::Null {
                Ok(None)
            } else {
                let arr: JsObject = val.coerce_to_object()?;
                let len = arr.get_array_length()?;
                let mut result = Vec::with_capacity(len as usize);
                for i in 0..len {
                    let v: JsNumber = arr.get_element(i)?;
                    result.push(v.get_int32()?);
                }
                Ok(Some(result))
            }
        }
        Err(_) => Ok(None),
    }
}

/// Option-aware wrappers over the get_opt_* readers for options that may be absent entirely.
fn opt_bool(opts: Option<&JsObject>, key: &str) -> Result<Option<bool>> {
    match opts { Some(o) => get_opt_bool(o, key), None => Ok(None) }
}

fn parse_group_mode(s: &str) -> Result<GroupMode> {
    match s {
        "static" => Ok(GroupMode::Static),
        "conservative" => Ok(GroupMode::Conservative),
        "neutral" => Ok(GroupMode::Neutral),
        "aggressive" => Ok(GroupMode::Aggressive),
        _ => Err(Error::from_reason(format!(
            "Invalid groupMode '{}'. Expected one of: static, conservative, neutral, aggressive", s
        ))),
    }
}

fn parse_consensus_mode(s: &str) -> Result<ConsensusMode> {
    match s {
        "democratic" => Ok(ConsensusMode::Democratic),
        "nb" => Ok(ConsensusMode::Nb),
        _ => Err(Error::from_reason(format!(
            "Invalid consensus '{}'. Expected one of: democratic, nb", s
        ))),
    }
}
