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
use brain_core::types::{Coordinate, ErrorMode, NeuronType};

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
    ///   errorCorrectionMode: string (default 'conservative')
    ///   errorCorrectionThreshold: number (default 0.5)
    ///   mergeThreshold: number (default 0.5)
    ///   patternForgetRate: number (default 0.01)
    ///   regions: number (default 1)
    ///   columns: number (default 1)
    ///   debug: boolean (default false)
    #[napi(constructor)]
    pub fn new(_env: Env, options: Option<JsObject>) -> Result<Self> {
        let (context_length, error_mode, error_threshold, merge_threshold,
             pattern_forget_rate, regions, columns, debug) = match options {
            Some(ref opts) => {
                let cl = get_opt_u32(opts, "contextLength")?.unwrap_or(10);
                let mode_str = get_opt_string(opts, "errorCorrectionMode")?
                    .unwrap_or_else(|| "conservative".to_string());
                let mode = parse_error_mode(&mode_str)?;
                let et = get_opt_f64(opts, "errorCorrectionThreshold")?.unwrap_or(0.5);
                let mt = get_opt_f64(opts, "mergeThreshold")?.unwrap_or(0.5);
                let pfr = get_opt_f64(opts, "patternForgetRate")?.unwrap_or(0.01);
                let r = get_opt_u32(opts, "regions")?.unwrap_or(1) as usize;
                let c = get_opt_u32(opts, "columns")?.unwrap_or(1) as usize;
                let d = get_opt_bool(opts, "debug")?.unwrap_or(false);
                (cl, mode, et, mt, pfr, r, c, d)
            }
            None => (10, ErrorMode::Conservative, 0.5, 0.5, 0.01, 1, 1, false),
        };

        Ok(Self {
            inner: RefCell::new(CoreBrain::new(
                context_length, error_mode, error_threshold, merge_threshold,
                pattern_forget_rate, regions, columns, debug,
            )),
        })
    }

    /// Register a channel spec. Accepts the same spec shape as JS Brain.
    ///
    /// spec: { name, emitsReward?, learnActionSequences?, dimensions: [...] }
    /// Returns: { channelId: number, dimensionIds: {name: id, ...} }
    #[napi(js_name = "registerChannelSpec")]
    pub fn register_channel_spec(&self, env: Env, spec: JsObject) -> Result<JsObject> {
        let name: String = spec.get_named_property::<JsString>("name")?.into_utf8()?.into_owned()?;
        let emits_reward = get_opt_bool(&spec, "emitsReward")?.unwrap_or(false);
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
            &name, dimensions, emits_reward, learn_action_sequences,
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
    /// Returns: { inferences: Map<channelId, [...]>, frame: { elapsed, voteDebug } }
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

    /// Save brain state to disk.
    #[napi]
    pub fn save(&self, job_dir: String) -> Result<()> {
        self.inner.borrow_mut()
            .save(std::path::Path::new(&job_dir))
            .map_err(|e| Error::from_reason(e))?;
        Ok(())
    }

    /// Load the most recent backup from disk.
    #[napi]
    pub fn load(&self, job_dir: String) -> Result<()> {
        self.inner.borrow_mut()
            .load(std::path::Path::new(&job_dir))
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
        obj.set_named_property("maxLevel", env.create_uint32(summary.max_level)?)?;

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
/// { inferences: Map<channelId, [...]>, frame: { elapsed, voteDebug } }
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

            // winner: { value, strength, score }
            let mut winner_obj = env.create_object()?;
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

    // frame: { elapsed, voteDebug }
    let mut frame_obj = env.create_object()?;
    // elapsed in milliseconds (Rust stores seconds)
    frame_obj.set_named_property("elapsed", env.create_double(result.elapsed * 1000.0)?)?;

    match &result.vote_debug {
        Some(_debug) => {
            // voteDebug is expensive to marshal and only used for --debug rendering.
            // For now, pass null — the debug renderer will get a null check.
            // Full marshalling can be added when needed.
            frame_obj.set_named_property("voteDebug", env.get_null()?)?;
        }
        None => {
            frame_obj.set_named_property("voteDebug", env.get_null()?)?;
        }
    }
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

fn get_opt_u32_array(obj: &JsObject, key: &str) -> Result<Option<Vec<u32>>> {
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
                    result.push(v.get_uint32()?);
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

fn parse_error_mode(s: &str) -> Result<ErrorMode> {
    match s {
        "static" => Ok(ErrorMode::Static),
        "conservative" => Ok(ErrorMode::Conservative),
        "neutral" => Ok(ErrorMode::Neutral),
        "aggressive" => Ok(ErrorMode::Aggressive),
        _ => Err(Error::from_reason(format!(
            "Invalid errorCorrectionMode '{}'. Expected one of: static, conservative, neutral, aggressive", s
        ))),
    }
}
