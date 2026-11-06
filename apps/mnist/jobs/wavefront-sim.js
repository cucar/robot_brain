// ============================================================================
//  Neuron-reuse — corrected reference simulation (recognize → mispredict → reuse)
// ============================================================================
//
// A deterministic, dependency-free model of the CORRECTED neuron-reuse mechanism
// from docs/neuron-reuse.md §3 and the spec docs/neuron-reuse-simulation.md. It
// replaces the obsolete cold-start / group-by-observed-set model that lived here.
//
// The mechanism, per frame and per level:
//
//   recognize (context match ≥ merge threshold)  →  predict L0
//     →  where the L0 prediction is wrong (≥ error threshold): a correction request
//        →  cluster requests by transitive merge over neighborhoods (≥1-neighbor rule)
//           →  reuse a matched pattern (reverse index) or mint one; matched patterns expand
//              →  install (fires a later frame, once its context recurs)
//
// with MERGE = clustering + reuse + expansion and SPLIT = refinement. There is no decay/forgetting:
// refinement weakens references, and a pattern reaped once its last parent reference drains is the
// only culling — the multi-parent reference-count corollary, matching the brain (decay was removed).
//
// Faithful to the Rust brain's actual seeding (read from brain-core):
//   - The base layer is per-position BLACK/WHITE — the WHOLE field is active, one
//     neuron per position (not just the stroke). A base neuron is persistent and
//     keyed by `pos:value`; it carries learned connections to its neighbors.
//   - A unit PREDICTS L0 via its connections, taking the per-position MODAL value
//     (the strongest-voted bucket per neighbor position), exactly as the brain's
//     mint_spatial_corrections does. Error = (missing + novel) / union over the
//     predicted-vs-observed base-id sets — a Hamming distance from the modal patch.
//   - Frame 1 only learns connections (no predictions yet → no mint); frame 2+
//     mispredicts where the image deviates and mints / reuses.
//   - Recognition score is the Jaccard common/(common+missing+novel) ≥ threshold,
//     mirroring SpatialContext::match_observed. The same score gates reuse.
//
// Brain mapping (camelCase here → snake_case there), so the port stays a transliteration:
//   Brain.processFrame           ~ Brain::process_frame / process_spatial
//   Brain.runLevelLoop           ~ process_spatial_levels (the settling level loop)
//   Brain.collectRequests        ~ the predict-L0 + error eval inside the loop
//   Brain.clusterReuseMint       ~ thalamus mint_spatial_corrections + reuse lookup
//   Brain.recognizeHigher        ~ recognize_spatial_patterns (the climb)
//   scoreContextMatch            ~ SpatialContext::match_observed (Jaccard)
//   reverse target index         ~ the reuse reverse index (target → predicting patterns)
//
// Run (ASCII shapes):  node apps/mnist/jobs/wavefront-sim.js
// Run (MNIST struct):  node apps/mnist/jobs/wavefront-sim.js mnist [count] [imageSize] [radius] [frames]
// Run (MNIST readout): node apps/mnist/jobs/wavefront-sim.js acc [train] [test] [imageSize] [radius]
// Run (sweeps):        node apps/mnist/jobs/wavefront-sim.js sweep [train] [test] [imageSize]

import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadImages, loadLabels } from '../loader.js';
import { MNISTPixelChannelsEncoder } from '../encoder.js';

const SIM_DIR = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
//  1. Geometry — the base neighbor graph
// ---------------------------------------------------------------------------

// Connectivity of the base neighbor graph: 4 = von Neumann (orthogonal),
// 8 = Moore (orthogonal + diagonal). Set per run by the runners below.
let connectivity = 8;
// Radius of the base neighbor graph (Chebyshev for 8-conn, Manhattan for 4-conn).
// MNIST mirrors the brain's encoder radius (1 = 3×3, 2 = 5×5); ASCII shapes use 1.
let radius = 1;

// Memoized neighbor lists per position id — positions repeat every frame and the
// neighbor ring never changes, so caching turns the hot adjacency probe into a lookup.
let neighborCache = new Map();

/**
 * Reset the per-position neighbor cache.
 * Called whenever connectivity or radius changes so stale rings are never reused.
 */
function resetGeometry() {
	neighborCache = new Map();
}

/**
 * List the coordinate neighbors of a base position under the current connectivity.
 * These edges define the base neighbor graph; footprint adjacency is built on top.
 */
function listPixelNeighbors(pixelId) {
	const cached = neighborCache.get(pixelId);
	if (cached) return cached;
	const [c, r] = pixelId.split(',').map(Number);
	const neighbors = [];
	for (let dr = -radius; dr <= radius; dr++) {
		for (let dc = -radius; dc <= radius; dc++) {
			if (dc === 0 && dr === 0) continue;
			// 8-conn (Moore) takes the whole (2r+1)² window; 4-conn (von Neumann) keeps Manhattan ≤ r.
			if (connectivity === 4 && Math.abs(dc) + Math.abs(dr) > radius) continue;
			neighbors.push(`${c + dc},${r + dr}`);
		}
	}
	neighborCache.set(pixelId, neighbors);
	return neighbors;
}

/**
 * Build an inverted index base-position → ids of the items whose footprint covers it.
 * Turns footprint adjacency into a local lookup instead of an O(N²) all-pairs scan — the
 * JS analog of the Rust bitset dilate + AND. `items` are { id, footprint } records.
 */
function buildPosIndex(items) {
	const index = new Map();
	for (const item of items) {
		for (const pos of item.footprint) {
			let list = index.get(pos);
			if (!list) { list = []; index.set(pos, list); }
			list.push(item.id);
		}
	}
	return index;
}

/**
 * The ids whose footprint TOUCHES the given footprint, via the inverted index: dilate the
 * footprint by the neighbor ring and collect every id covering a probed position. `excludeId`
 * drops self. This is exactly doFootprintsTouch, computed once per footprint rather than per pair.
 */
function touchingIds(footprint, posIndex, excludeId) {
	const out = new Set();
	for (const pos of footprint) {
		const here = posIndex.get(pos);
		if (here) for (const id of here) if (id !== excludeId) out.add(id);
		for (const np of listPixelNeighbors(pos)) {
			const there = posIndex.get(np);
			if (there) for (const id of there) if (id !== excludeId) out.add(id);
		}
	}
	return out;
}

// ---------------------------------------------------------------------------
//  2. Base-id helpers — a base neuron is (position, value)
// ---------------------------------------------------------------------------

/**
 * The id of a base neuron at a position with a quantized value.
 * Black/white share a position but are distinct neurons, so the value is in the id.
 */
function baseId(pos, value) {
	return `${pos}=${value}`;
}

/**
 * Split a base-neuron id back into its position and value.
 */
function splitBaseId(id) {
	const eq = id.lastIndexOf('=');
	return { pos: id.slice(0, eq), value: Number(id.slice(eq + 1)) };
}

// ---------------------------------------------------------------------------
//  3. Context / target scoring — mirrors SpatialContext::match_observed
// ---------------------------------------------------------------------------

/**
 * Score a stored id→strength map against an observed id set, returning the Jaccard
 * common/(common+missing+novel) plus the common/missing/novel id lists for refinement.
 * Returns null when the score is below the merge threshold (no match), exactly as the
 * brain's match_observed does. `known` is a Map<id, strength>; `observed` is a Set<id>.
 */
function scoreContextMatch(known, observed, mergeThreshold) {
	const common = [];
	const missing = [];
	for (const [id, strength] of known) {
		if (strength <= 0) continue;
		if (observed.has(id)) common.push(id); else missing.push(id);
	}
	if (common.length === 0 && missing.length === 0) return null;
	const novel = [];
	for (const id of observed) if (!known.has(id)) novel.push(id);
	const unionSize = common.length + missing.length + novel.length;
	if (unionSize === 0) return null;
	const score = common.length / unionSize;
	if (score < mergeThreshold) return null;
	return { score, common, missing, novel };
}

/**
 * Collapse an id→strength target map to its per-position MODAL base-id set — the
 * strongest-voted value at each position, ties broken on smaller id for determinism.
 * This is the brain's per-position competition (one prediction per neighbor position).
 */
function modalPrediction(targets) {
	const winnerByPos = new Map(); // pos -> { id, strength }
	for (const [id, strength] of targets) {
		if (strength <= 0) continue;
		const { pos } = splitBaseId(id);
		const cur = winnerByPos.get(pos);
		if (!cur || strength > cur.strength || (strength === cur.strength && id < cur.id)) {
			winnerByPos.set(pos, { id, strength });
		}
	}
	const out = new Set();
	for (const { id } of winnerByPos.values()) out.add(id);
	return out;
}

/**
 * The (missing + novel) / union error between a predicted base-id set and the observed
 * base-id set — the brain's spatial error rate, a Hamming distance from the modal patch.
 */
function predictionError(predicted, observed) {
	let missing = 0;
	for (const id of predicted) if (!observed.has(id)) missing++;
	let novel = 0;
	for (const id of observed) if (!predicted.has(id)) novel++;
	const union = predicted.size + novel; // |predicted ∪ observed| = |predicted| + (observed \ predicted)
	if (union === 0) return 0;
	return (missing + novel) / union;
}

// ---------------------------------------------------------------------------
//  4. The Brain — persistent across frames, mirrors the spatial frame flow
// ---------------------------------------------------------------------------

class Brain {

	constructor(opts = {}) {
		// Tunable rates — the merge/split levers the experiment sweeps.
		this.mergeThreshold = opts.mergeThreshold ?? 0.5;   // recognition + reuse (θ); 1.0 disables both
		this.errorThreshold = opts.errorThreshold ?? 0.3;   // static L0-misprediction tolerance
		this.reuse = opts.reuse ?? true;                    // cross-frame reverse-index reuse + expansion
		this.refine = opts.refine ?? false;                 // split force: consolidate + specialize, and drain stale references
		this.adaptiveError = opts.adaptiveError ?? false;   // adapt the per-unit error threshold from Welford stats
		this.errorMode = opts.errorMode ?? 'conservative';  // adaptive direction: conservative | neutral | aggressive
		this.reliabilityFloor = opts.reliabilityFloor ?? 0.6; // drop a footprint position predicted right < this often
		this.refineMinSamples = opts.refineMinSamples ?? 5; // min per-position samples before a position may be pruned
		this.expansionOverlapFloor = opts.expansionOverlapFloor ?? 0.0; // min overlap to absorb a neighbor
		// Parent reference strengths (multi-parent refcount). Same integer-counter arithmetic as the
		// brain's connection/context entries: created at 1 when a parent is wired, +1 when the parent
		// is active as the pattern fires, −1 when absent, deleted at 0, no cap. Refinement draining
		// these references to zero is what reaps a pattern — not decay.
		this.maxLevels = opts.maxLevels ?? 12;

		// Persistent base neurons, keyed `pos=value`. Each carries learned neighbor connections.
		this.bases = new Map();
		// Persistent pattern neurons (corrections), keyed by minted id.
		this.patterns = new Map();
		// Reverse index: L0 target base-id → set of pattern ids predicting it (membership-only).
		this.targetIndex = new Map();
		// Context index: source id → set of pattern ids whose context references it (recognition candidates).
		this.contextIndex = new Map();
		// Child index: parent id → set of pattern ids that reference it as a parent — the reverse of
		// `parents`, so reaping a pattern can find the children it leaves parentless without a full scan.
		this.childIndex = new Map();
		// Per-unit Welford error stats for the adaptive threshold, keyed by unit id.
		this.errorStats = new Map();

		this.nextId = 0;
		this.frame = 0;

		// Per-run diagnostics, accumulated across frames.
		this.diag = newDiagnostics();
	}

	/**
	 * Allocate a fresh coordinate-less pattern id at the given spatial level.
	 */
	allocatePatternId(spatialLevel) {
		return `L${spatialLevel}#${this.nextId++}`;
	}

	/**
	 * Process one frame: activate the whole base field, then run the settling level loop
	 * (which also refines matched patterns and reaps any left unreferenced). `learning` gates
	 * all structural mutation so an eval run is reproducible (no mint / expand / refine / reap / learn).
	 * Returns the per-level record plus the set of pattern ids that fired this frame.
	 */
	processFrame(bits, imageSize, learning = true) {
		this.frame++;
		const active = this.activateBase(bits, imageSize);
		return this.runLevelLoop(active, learning);
	}

	/**
	 * Activate the base sensory field: one neuron per position carrying this image's
	 * value (black or white). Base neurons are persistent — created lazily, reused
	 * across frames so their learned connections accumulate. Records the frame's
	 * value-at-position map so neighbors can be read back during prediction.
	 * Returns the active base neurons in row-major (deterministic) order.
	 */
	activateBase(bits, imageSize) {
		this.valueByPos = new Map();
		const active = [];
		for (let p = 0; p < bits.length; p++) {
			const pos = `${p % imageSize},${Math.floor(p / imageSize)}`;
			const value = bits[p];
			const id = baseId(pos, value);
			let neuron = this.bases.get(id);
			if (!neuron) {
				neuron = {
					id,
					spatialLevel: 0,
					pos,
					value,
					footprint: new Set([pos]),
					connections: new Map(), // neighbor base-id → strength (its learned L0 prediction)
				};
				this.bases.set(id, neuron);
			}
			this.valueByPos.set(pos, value);
			active.push(neuron);
		}
		return active;
	}

	/**
	 * The settling level loop. At each level the active units predict L0 and the
	 * mispredictors become correction requests; requests cluster and reuse/mint
	 * corrections that install for a later frame. In parallel, existing patterns one
	 * level up are RECOGNIZED and become the next level's active set — that is the climb.
	 * The climb stops when no higher level is recognized (a cold brain stops at level 0).
	 * Mirrors process_spatial_levels (the fired/subsumed bookkeeping).
	 */
	runLevelLoop(activeBase, learning) {
		const levels = [];
		const firedPatternIds = new Set();
		const doomed = []; // patterns whose last parent reference drained this frame (refcount → 0)
		let active = activeBase;

		for (let level = 0; level < this.maxLevels; level++) {
			// Predict L0 from each active unit; collect the mispredictors as requests.
			const requests = this.collectRequests(active, level, learning);

			// Cluster requests and reuse-or-mint one correction per connected cluster.
			let minted = [], reused = [], expanded = [];
			if (learning) {
				const out = this.clusterReuseMint(requests, level);
				minted = out.minted; reused = out.reused; expanded = out.expanded;
			}

			// Recognize existing patterns one level up — they fire this frame and climb.
			const recognized = this.recognizeHigher(active, level, learning);
			for (const r of recognized) firedPatternIds.add(r.pattern.id);

			// Split force on the patterns we matched this frame: refinement strengthens the
			// present parent references and weakens the absent ones; a pattern that drains its
			// last reference is doomed (the multi-parent refcount corollary — reaped below).
			if (learning && this.refine) {
				const activeIds = new Set(active.map(u => u.id));
				for (const m of recognized) if (this.refinePattern(m.pattern, m.match, activeIds)) doomed.push(m.pattern.id);
			}

			levels.push({
				level,
				activeCount: active.length,
				requests: requests.length,
				minted: minted.length,
				reused: reused.length,
				expanded: expanded.length,
				recognized: recognized.length,
			});

			this.accumulateLevelDiagnostics(level, requests, minted, reused, expanded, recognized, active);

			if (recognized.length === 0) break;
			active = recognized.map(r => r.pattern);
		}

		// Reap unreferenced patterns, cascading to any child left parentless by the reap.
		if (doomed.length) this.reapWithCascade(doomed);

		return { levels, firedPatternIds };
	}

	// ── Predict L0 + error → requests ─────────────────────────────────────────

	/**
	 * For every active unit at this level, predict L0 via its connections (per-position
	 * modal), compare to the observed L0 in its neighborhood, and emit a correction
	 * request when the error clears the threshold and the unit has ≥1 neighbor for
	 * context. Base units also LEARN their neighbor connections here (the only place
	 * the L0 predictor is built). Returns the request descriptors for clustering.
	 */
	collectRequests(active, level, learning) {
		const idToUnit = new Map(active.map(u => [u.id, u]));
		const posIndex = buildPosIndex(active);
		const funnel = this.funnel(level);
		funnel.active += active.length;
		const requests = [];
		for (const unit of active) {
			const observedL0 = this.observeL0(unit, level);
			const neighbors = this.neighborsOf(unit, level, posIndex, idToUnit);

			if (level === 0 && learning) this.learnBaseConnections(unit);

			const predicted = level === 0
				? modalPrediction(unit.connections)
				: modalPrediction(unit.targets);

			// Bootstrap: a unit with no learned prediction yet has nothing to be wrong about.
			if (predicted.size === 0) continue;

			const error = predictionError(predicted, observedL0);
			if (learning) this.recordError(unit.id, error);

			const threshold = this.thresholdFor(unit.id);
			if (error <= threshold) continue;
			funnel.mispredict++;
			if (neighbors.length === 0) { funnel.neighborless++; continue; } // ≥1-neighbor rule — no context, no correction

			funnel.requests++;
			requests.push({ unit, level, observedL0, neighbors, footprint: unit.footprint });
		}
		return requests;
	}

	/**
	 * Lazily fetch the climb-funnel counter bucket for a level (active / mispredict / neighborless
	 * / requests / minted / reused), so a stalled level is localizable in the diagnostics.
	 */
	funnel(level) {
		let f = this.diag.funnelByLevel.get(level);
		if (!f) { f = { active: 0, mispredict: 0, neighborless: 0, requests: 0, minted: 0, reused: 0 }; this.diag.funnelByLevel.set(level, f); }
		return f;
	}

	/**
	 * The observed L0 base-id set a unit is accountable for: at the base level, the
	 * actual values at the unit's coordinate-neighbor positions; at higher levels,
	 * the actual values over the unit's footprint region. This is what the prediction
	 * is scored against (the correct L0 reality the unit should have predicted).
	 */
	observeL0(unit, level) {
		const observed = new Set();
		if (level === 0) {
			for (const np of listPixelNeighbors(unit.pos)) {
				const v = this.valueByPos.get(np);
				if (v !== undefined) observed.add(baseId(np, v));
			}
		} else {
			for (const pos of unit.footprint) {
				const v = this.valueByPos.get(pos);
				if (v !== undefined) observed.add(baseId(pos, v));
			}
		}
		return observed;
	}

	/**
	 * Strengthen a base neuron's connections to the actual values at its neighbor
	 * positions this frame — the running tally whose per-position mode is its L0 prediction.
	 */
	learnBaseConnections(unit) {
		for (const np of listPixelNeighbors(unit.pos)) {
			const v = this.valueByPos.get(np);
			if (v === undefined) continue;
			const id = baseId(np, v);
			unit.connections.set(id, (unit.connections.get(id) || 0) + 1);
		}
	}

	/**
	 * The neighbors of a unit among the active set at this level: coordinate adjacency
	 * at the base, footprint-touch at higher levels (the §2.2 footprint neighborhood).
	 * The unit is never its own neighbor.
	 */
	neighborsOf(unit, level, posIndex, idToUnit) {
		const out = [];
		if (level === 0) {
			for (const np of listPixelNeighbors(unit.pos)) {
				const here = posIndex.get(np);
				if (here) for (const id of here) if (id !== unit.id) out.push(idToUnit.get(id));
			}
		} else {
			for (const id of touchingIds(unit.footprint, posIndex, unit.id)) out.push(idToUnit.get(id));
		}
		return out;
	}

	// ── Cluster + reuse/mint/expand ───────────────────────────────────────────

	/**
	 * Cluster the requests by transitive merge over the neighbor relation and, per
	 * connected cluster, either reuse/expand a matched pattern or mint a fresh one
	 * predicting the correct L0. Reuse lookup runs first (per request, against the
	 * reverse target index); matched patterns join the clustering pool so an adjacent
	 * matched pattern absorbs new requests by expansion. Installs fire a later frame.
	 * Returns { minted, reused, expanded } pattern lists for diagnostics.
	 */
	clusterReuseMint(requests, level) {
		if (requests.length === 0) return { minted: [], reused: [], expanded: [] };

		// Reuse lookup — find, per request, the best existing pattern predicting its L0.
		const matchedByRequest = new Map(); // request -> pattern
		if (this.reuse) {
			for (const req of requests) {
				const pattern = this.findReusable(req);
				if (pattern) matchedByRequest.set(req, pattern);
			}
		}

		// Build the unified clustering pool: requests plus the distinct matched patterns.
		const matchedPatterns = new Map(); // patternId -> pattern (deduped)
		for (const p of matchedByRequest.values()) matchedPatterns.set(p.id, p);

		const clusters = this.transitiveMerge(requests, [...matchedPatterns.values()], level);

		const minted = [], reused = [], expanded = [];
		for (const cluster of clusters) {
			if (cluster.neighbors.size === 0) continue; // need ≥1 neighbor for context

			if (cluster.patterns.length > 0) {
				// Reuse: wire the cluster's requests to the (smallest-id) matched pattern, expand it.
				const target = cluster.patterns.sort((a, b) => (a.id < b.id ? -1 : 1))[0];
				this.expandPattern(target, cluster);
				reused.push(target);
				if (cluster.requests.length > 0) expanded.push(target);
			} else {
				// Mint one coordinate-less correction over the connected cluster.
				minted.push(this.mintCorrection(cluster, level));
			}
		}
		const f = this.funnel(level);
		f.minted += minted.length;
		f.reused += reused.length;
		return { minted, reused, expanded };
	}

	/**
	 * Query the reverse target index for an existing pattern whose L0 prediction matches
	 * the request's correct-L0 ≥ the merge threshold. Candidates are the patterns indexed
	 * under any of the request's L0 target ids; scoring is the strength-blind Jaccard over
	 * their modal predictions. Returns the best (smaller id tie-break) or null. Self is
	 * not a concern here — requests are units, candidates are patterns.
	 */
	findReusable(req) {
		const candidates = new Set();
		for (const targetId of req.observedL0) {
			const ids = this.targetIndex.get(targetId);
			if (ids) for (const id of ids) candidates.add(id);
		}
		let best = null, bestScore = -1, bestId = null;
		for (const id of [...candidates].sort()) {
			const pattern = this.patterns.get(id);
			if (!pattern) continue;
			const predicted = modalPrediction(pattern.targets);
			const known = new Map([...predicted].map(t => [t, 1]));
			const match = scoreContextMatch(known, req.observedL0, this.mergeThreshold);
			if (!match) continue;
			if (match.score > bestScore || (match.score === bestScore && id < bestId)) {
				best = pattern; bestScore = match.score; bestId = id;
			}
		}
		return best;
	}

	/**
	 * Connected components over the unified pool (requests ∪ matched patterns) by the
	 * neighbor relation at this level. Each component reports its requests, its matched
	 * patterns, the union footprint, and the neighbor-source set (context) it conditions
	 * on. Pool members are joined when their footprints touch (coordinate ring at base).
	 * Iteration is in sorted-id order for determinism.
	 */
	transitiveMerge(requests, matchedPatterns) {
		const members = [
			...requests.map(r => ({ kind: 'request', id: r.unit.id, footprint: r.footprint, req: r })),
			...matchedPatterns.map(p => ({ kind: 'pattern', id: p.id, footprint: p.footprint, pattern: p })),
		].sort((a, b) => (a.id < b.id ? -1 : 1));

		const parent = new Map(members.map(m => [m.id, m.id]));
		const find = x => { while (parent.get(x) !== x) { parent.set(x, parent.get(parent.get(x))); x = parent.get(x); } return x; };
		const union = (a, b) => { const ra = find(a), rb = find(b); if (ra !== rb) parent.set(ra < rb ? rb : ra, ra < rb ? ra : rb); };

		// Join any two pool members whose footprints touch (transitive over the relation), via the
		// inverted index — each member dilates once and unions with the members it lands on.
		const posIndex = buildPosIndex(members);
		for (const m of members) {
			for (const otherId of touchingIds(m.footprint, posIndex, m.id)) union(m.id, otherId);
		}

		const byRoot = new Map();
		for (const m of members) {
			const root = find(m.id);
			if (!byRoot.has(root)) byRoot.set(root, { requests: [], patterns: [], footprint: new Set(), neighbors: new Set() });
			const cluster = byRoot.get(root);
			if (m.kind === 'request') {
				cluster.requests.push(m.req);
				for (const pos of m.footprint) cluster.footprint.add(pos);
				for (const nb of m.req.neighbors) cluster.neighbors.add(nb.id);
			} else {
				cluster.patterns.push(m.pattern);
				for (const pos of m.footprint) cluster.footprint.add(pos);
			}
		}
		return [...byRoot.values()];
	}

	/**
	 * Mint one coordinate-less correction over a connected cluster: footprint = the
	 * cluster's coverage, context = the cluster's neighbor sources, targets = the
	 * correct L0 the cluster's requests should have predicted. Every request in the
	 * cluster is wired as a parent. The correction is registered in both indexes so it
	 * is a recognition candidate (by context) and a reuse candidate (by target) next frame.
	 */
	mintCorrection(cluster, level) {
		const targets = new Map();
		for (const req of cluster.requests) {
			for (const id of req.observedL0) targets.set(id, (targets.get(id) || 0) + 1);
		}
		const id = this.allocatePatternId(level + 1);
		const pattern = {
			id,
			spatialLevel: level + 1,
			footprint: new Set(cluster.footprint),
			context: new Map([...cluster.neighbors].map(nb => [nb, 1])),
			targets,
			parents: new Map(), // parentId → reference strength (multi-parent refcount)
			posStats: new Map(), // pos → { hit, total } — per-position reliability for specialization
		};
		for (const r of cluster.requests) this.addParentRef(pattern, r.unit.id);
		this.patterns.set(id, pattern);
		this.indexContext(pattern);
		this.indexTargets(pattern, modalPrediction(targets));
		this.diag.births++;
		return pattern;
	}

	/**
	 * Expand a matched pattern to absorb an adjacent cluster of new requests: grow its
	 * footprint and context to cover the new region, fold in the requests' correct L0,
	 * and wire the requests as new parents. An optional overlap floor blocks absorbing a
	 * region that barely touches (the open expansion gate). Re-indexes any new targets.
	 */
	expandPattern(pattern, cluster) {
		if (cluster.requests.length === 0) return;

		// Overlap gate: skip absorption when the new coverage barely overlaps the pattern.
		if (this.expansionOverlapFloor > 0) {
			let shared = 0;
			for (const pos of cluster.footprint) if (pattern.footprint.has(pos)) shared++;
			const overlap = shared / cluster.footprint.size;
			if (overlap < this.expansionOverlapFloor) return;
		}

		for (const pos of cluster.footprint) pattern.footprint.add(pos);
		for (const nb of cluster.neighbors) pattern.context.set(nb, (pattern.context.get(nb) || 0) + 1);
		for (const req of cluster.requests) {
			this.addParentRef(pattern, req.unit.id);
			for (const id of req.observedL0) pattern.targets.set(id, (pattern.targets.get(id) || 0) + 1);
		}
		this.indexContext(pattern);
		this.indexTargets(pattern, modalPrediction(pattern.targets));
		this.diag.expansions++;
	}

	// ── Recognition (the climb) ───────────────────────────────────────────────

	/**
	 * Recognize the patterns one level up whose stored context matches this level's
	 * active set ≥ the merge threshold — they fire this frame and become the next
	 * level's active units. Candidates come from the context index (patterns sharing
	 * ≥1 active source); a context match ≥ threshold fires the pattern. Returns
	 * { pattern, match, observedL0 } so refinement can consolidate the matched patterns.
	 */
	recognizeHigher(active, level) {
		const targetLevel = level + 1;

		// Narrow to patterns at the target level that reference an active source.
		const candidates = new Set();
		for (const u of active) {
			const ids = this.contextIndex.get(u.id);
			if (ids) for (const id of ids) candidates.add(id);
		}

		// One inverted index over the active set, reused for every candidate's local context.
		const activePosIndex = level === 0 ? null : buildPosIndex(active);

		const recognized = [];
		for (const id of [...candidates].sort()) {
			const pattern = this.patterns.get(id);
			if (!pattern || pattern.spatialLevel !== targetLevel) continue;
			// Match the pattern's stored context against its LOCAL observed context — the active
			// ids in its own neighborhood — never the whole field. With the base field wholly
			// active, a global match would drown every local pattern in novel co-actives.
			const observed = this.localObservedForRecognition(pattern, level, activePosIndex);
			const match = scoreContextMatch(pattern.context, observed, this.mergeThreshold);
			if (!match) continue;
			recognized.push({ pattern, match, observedL0: this.observeL0(pattern, targetLevel) });
		}
		return recognized;
	}

	/**
	 * The local observed context a candidate pattern is matched against during recognition:
	 * at the base, the actually-active base id at each position its context references (so a
	 * mismatched position contributes one missing + one novel — a true local Jaccard); higher
	 * up, the active level-below patterns whose footprints touch the candidate's footprint.
	 */
	localObservedForRecognition(pattern, level, activePosIndex) {
		if (level === 0) {
			const observed = new Set();
			for (const id of pattern.context.keys()) {
				const { pos } = splitBaseId(id);
				const v = this.valueByPos.get(pos);
				if (v !== undefined) observed.add(baseId(pos, v));
			}
			return observed;
		}
		return touchingIds(pattern.footprint, activePosIndex, pattern.id);
	}

	// ── Split force — refinement (consolidate, specialize, drain references) ───

	/**
	 * Consolidate a matched pattern toward the common core of the configs it matches:
	 * strengthen common context, add novel context, weaken context missing from the
	 * match, pull the targets toward the observed L0, and prune unreliable positions.
	 * Also drains parent references: a parent active as the pattern fires is reinforced,
	 * an absent one is weakened and dropped at zero. Returns true when the LAST reference
	 * drains — the pattern is then unreferenced and the caller reaps it (the split force's
	 * culling, a corollary of the multi-parent refcount; no time-based decay).
	 */
	refinePattern(pattern, match, activeIds) {
		// Context (sources): strengthen the matched core, add novel, decay what was missing.
		for (const id of match.common) pattern.context.set(id, (pattern.context.get(id) || 0) + 1);
		for (const id of match.novel) pattern.context.set(id, (pattern.context.get(id) || 0) + 1);
		for (const id of match.missing) {
			const next = (pattern.context.get(id) || 0) - 1;
			if (next <= 0) pattern.context.delete(id); else pattern.context.set(id, next);
		}

		// Targets (prediction): per position, strengthen the observed value, decay the off-modal
		// ones, and track whether the pattern's modal prediction was right at that position.
		const predicted = modalPrediction(pattern.targets);
		const predictedByPos = new Map();
		for (const id of predicted) predictedByPos.set(splitBaseId(id).pos, id);
		for (const pos of pattern.footprint) {
			const v = this.valueByPos.get(pos);
			if (v === undefined) continue;
			const observedId = baseId(pos, v);
			pattern.targets.set(observedId, (pattern.targets.get(observedId) || 0) + 1);
			let stat = pattern.posStats.get(pos);
			if (!stat) { stat = { hit: 0, total: 0 }; pattern.posStats.set(pos, stat); }
			stat.total++;
			if (predictedByPos.get(pos) === observedId) stat.hit++;
		}
		for (const [id, strength] of [...pattern.targets]) {
			const { pos, value } = splitBaseId(id);
			const v = this.valueByPos.get(pos);
			if (v === undefined || value !== v) {
				const next = strength - 1;
				if (next <= 0) pattern.targets.delete(id); else pattern.targets.set(id, next);
			}
		}

		// Specialize (the split force): drop footprint positions the pattern predicts unreliably,
		// once they have enough samples. The pattern shrinks toward the region it gets right; the
		// dropped region is where it was too vague — exactly what drives a more-specific child higher up.
		if (pattern.footprint.size > 1) {
			for (const [pos, stat] of [...pattern.posStats]) {
				if (stat.total < this.refineMinSamples) continue;
				if (stat.hit / stat.total >= this.reliabilityFloor) continue;
				pattern.footprint.delete(pos);
				pattern.posStats.delete(pos);
				for (const id of [...pattern.targets.keys()]) if (splitBaseId(id).pos === pos) pattern.targets.delete(id);
				this.diag.prunes++;
			}
		}

		// References (parents): strengthen the parents present as the pattern fires, weaken the
		// absent ones. A reference that drains to zero is dropped; when none remain the pattern is
		// no longer referenced and is reaped. This is what removes a pattern that has drifted away
		// from the contexts that once minted it — refinement, not forgetting, does the culling.
		for (const [pid, strength] of [...pattern.parents]) {
			if (activeIds.has(pid)) {
				pattern.parents.set(pid, strength + 1);
			} else {
				const next = strength - 1;
				if (next <= 0) this.dropParentRef(pattern, pid); else pattern.parents.set(pid, next);
			}
		}

		this.indexContext(pattern);
		this.indexTargets(pattern, modalPrediction(pattern.targets));
		this.diag.refinements++;
		return pattern.parents.size === 0;
	}

	/**
	 * Wire a parent reference onto a pattern (mint / reuse / expansion): create it at refInit,
	 * or strengthen an existing one, and record the reverse edge in the child index.
	 */
	addParentRef(pattern, parentId) {
		const existing = pattern.parents.get(parentId);
		pattern.parents.set(parentId, existing === undefined ? 1 : existing + 1);
		let kids = this.childIndex.get(parentId);
		if (!kids) { kids = new Set(); this.childIndex.set(parentId, kids); }
		kids.add(pattern.id);
	}

	/**
	 * Drop a drained parent reference from a pattern and from the reverse child index.
	 */
	dropParentRef(pattern, parentId) {
		pattern.parents.delete(parentId);
		const kids = this.childIndex.get(parentId);
		if (kids) { kids.delete(pattern.id); if (kids.size === 0) this.childIndex.delete(parentId); }
	}

	/**
	 * Reap unreferenced patterns, cascading to any child the reap leaves parentless. Removing a
	 * pattern scrubs it from both indexes, drops every parent-reference edge it held, removes it
	 * as a context source from the patterns that referenced it, and removes it as a parent from
	 * its children — a child whose last parent was this pattern is then itself unreferenced.
	 */
	reapWithCascade(seedIds) {
		const queue = [...seedIds];
		while (queue.length) {
			const id = queue.shift();
			const pattern = this.patterns.get(id);
			if (!pattern) continue;

			this.deindexContext(pattern);
			this.deindexTargets(pattern);
			for (const pid of pattern.parents.keys()) this.dropParentRef(pattern, pid);
			this.patterns.delete(id);
			this.diag.deaths++;

			// Scrub this pattern as a context source from the patterns that recognized it.
			const referrers = this.contextIndex.get(id);
			if (referrers) for (const rid of [...referrers]) {
				const r = this.patterns.get(rid);
				if (r) r.context.delete(id);
			}
			this.contextIndex.delete(id);

			// Drop this pattern as a parent from its children; a child left parentless cascades.
			const kids = this.childIndex.get(id);
			if (kids) for (const childId of [...kids]) {
				const child = this.patterns.get(childId);
				if (!child) continue;
				child.parents.delete(id);
				if (child.parents.size === 0) queue.push(childId);
			}
			this.childIndex.delete(id);
		}
	}

	// ── Error stats + index maintenance ───────────────────────────────────────

	/**
	 * Fold an error sample into a unit's running Welford stats for the adaptive threshold.
	 */
	recordError(unitId, error) {
		let s = this.errorStats.get(unitId);
		if (!s) { s = { n: 0, mean: 0, m2: 0 }; this.errorStats.set(unitId, s); }
		s.n++;
		const delta = error - s.mean;
		s.mean += delta / s.n;
		s.m2 += delta * (error - s.mean);
	}

	/**
	 * The error threshold for a unit: the static value, or — with adaptive error on and
	 * ≥3 samples — the conservative mean + σ (the hoped-for split lever that lets a
	 * generalizing pattern start requesting corrections where it has become too vague).
	 */
	thresholdFor(unitId) {
		if (!this.adaptiveError) return this.errorThreshold;
		const s = this.errorStats.get(unitId);
		if (!s || s.n < 3) return this.errorThreshold;
		const sigma = Math.sqrt(s.m2 / s.n);
		// Conservative (mean+σ) tolerates more — fewer children; aggressive (mean−σ) makes a
		// usually-right pattern start erroring on its high-error tail — more specific children, the climb.
		if (this.errorMode === 'aggressive') return Math.max(0, s.mean - sigma);
		if (this.errorMode === 'neutral') return s.mean;
		return s.mean + sigma;
	}

	/**
	 * Add a pattern to the context index under every source it references.
	 */
	indexContext(pattern) {
		for (const sourceId of pattern.context.keys()) {
			let set = this.contextIndex.get(sourceId);
			if (!set) { set = new Set(); this.contextIndex.set(sourceId, set); }
			set.add(pattern.id);
		}
	}

	/**
	 * Remove a pattern from the context index everywhere it is referenced.
	 */
	deindexContext(pattern) {
		for (const sourceId of pattern.context.keys()) {
			const set = this.contextIndex.get(sourceId);
			if (set) { set.delete(pattern.id); if (set.size === 0) this.contextIndex.delete(sourceId); }
		}
	}

	/**
	 * Index a pattern under its current modal L0 targets, tracking the indexed set so a
	 * later re-index can scrub stale entries when the modal prediction shifts.
	 */
	indexTargets(pattern, modalSet) {
		if (pattern.indexedTargets) this.deindexTargets(pattern);
		for (const targetId of modalSet) {
			let set = this.targetIndex.get(targetId);
			if (!set) { set = new Set(); this.targetIndex.set(targetId, set); }
			set.add(pattern.id);
		}
		pattern.indexedTargets = modalSet;
	}

	/**
	 * Remove a pattern from the reverse target index for its currently-indexed targets.
	 */
	deindexTargets(pattern) {
		if (!pattern.indexedTargets) return;
		for (const targetId of pattern.indexedTargets) {
			const set = this.targetIndex.get(targetId);
			if (set) { set.delete(pattern.id); if (set.size === 0) this.targetIndex.delete(targetId); }
		}
		pattern.indexedTargets = null;
	}

	// ── Diagnostics ───────────────────────────────────────────────────────────

	/**
	 * Accumulate the per-frame merge/split/reuse signals the experiment reports: pattern
	 * counts and footprint sizes per level, plus merge (clusters + expansions) vs reuse
	 * vs recognition rates. Footprint sizes are sampled from the minted/reused patterns.
	 */
	accumulateLevelDiagnostics(level, requests, minted, reused, expanded, recognized, active) {
		const d = this.diag.byLevel.get(level) || newLevelDiagnostics();
		d.frames++;
		d.requests += requests.length;
		d.minted += minted.length;
		d.reused += reused.length;
		d.expanded += expanded.length;
		d.recognized += recognized.length;
		d.activeUnits += active.length;
		for (const p of [...minted, ...reused]) d.footprintSizes.push(p.footprint.size);
		this.diag.byLevel.set(level, d);
	}

	/**
	 * Snapshot the live structural state — pattern count and per-level footprint-size
	 * distribution — the headline "one big blurry pattern vs many regional patterns" signal.
	 */
	snapshot() {
		const byLevel = new Map();
		for (const p of this.patterns.values()) {
			const arr = byLevel.get(p.spatialLevel) || [];
			arr.push(p.footprint.size);
			byLevel.set(p.spatialLevel, arr);
		}
		return { patternCount: this.patterns.size, baseCount: this.bases.size, byLevel };
	}
}

/**
 * A fresh whole-run diagnostics accumulator.
 */
function newDiagnostics() {
	// Lifetime event counters track the merge/split balance: births + expansions are the
	// merge force, deaths + prunes are the split force. `history` snapshots them at checkpoints.
	// `funnelByLevel` traces the climb: per level, how active units flow active → mispredict →
	// (neighbor-gated) request → mint vs reuse, so a stalled level (e.g. no L3) is localizable.
	return { byLevel: new Map(), births: 0, deaths: 0, expansions: 0, refinements: 0, prunes: 0, history: [], funnelByLevel: new Map() };
}

/**
 * A fresh per-level diagnostics accumulator.
 */
function newLevelDiagnostics() {
	return { frames: 0, requests: 0, minted: 0, reused: 0, expanded: 0, recognized: 0, activeUnits: 0, footprintSizes: [] };
}

// ---------------------------------------------------------------------------
//  5. Reporting helpers
// ---------------------------------------------------------------------------

/**
 * Summarize a list of footprint sizes as min / median / max / mean — the regional-vs-blurry signal.
 */
function summarizeSizes(sizes) {
	if (sizes.length === 0) return 'none';
	const sorted = [...sizes].sort((a, b) => a - b);
	const median = sorted[Math.floor(sorted.length / 2)];
	const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length;
	return `min ${sorted[0]} / med ${median} / max ${sorted[sorted.length - 1]} / mean ${mean.toFixed(1)}`;
}

/**
 * Render a list of values as a unicode block sparkline, scaled to the run's max.
 * Used to show a metric's trajectory (grow then shrink) inline in the terminal.
 */
function spark(values) {
	const blocks = '▁▂▃▄▅▆▇█';
	if (values.length === 0) return '';
	const max = Math.max(...values, 1);
	return values.map(v => blocks[Math.min(blocks.length - 1, Math.round((v / max) * (blocks.length - 1)))]).join('');
}

/**
 * Print the merge/split diagnostics gathered over a run, per level.
 */
function reportDiagnostics(brain) {
	console.log('\n  ── diagnostics (per level, summed over frames) ──');
	const levels = [...brain.diag.byLevel.keys()].sort((a, b) => a - b);
	for (const level of levels) {
		const d = brain.diag.byLevel.get(level);
		const reuseRate = d.requests > 0 ? (100 * d.reused / (d.minted + d.reused || 1)).toFixed(0) : '0';
		console.log(`  L${level}: requests ${d.requests} | minted ${d.minted} | reused ${d.reused} (expanded ${d.expanded}) | recognized ${d.recognized} | reuse-share ${reuseRate}% | footprints ${summarizeSizes(d.footprintSizes)}`);
	}
	const snap = brain.snapshot();
	console.log(`  live: ${snap.patternCount} patterns, ${snap.baseCount} base neurons`);
	for (const level of [...snap.byLevel.keys()].sort((a, b) => a - b)) {
		console.log(`    level ${level}: ${snap.byLevel.get(level).length} patterns | footprints ${summarizeSizes(snap.byLevel.get(level))}`);
	}
}

// ---------------------------------------------------------------------------
//  6. ASCII shapes — quick whole-field visual checks over a few frames
// ---------------------------------------------------------------------------

/**
 * Parse a multi-line ASCII grid into a row-major binary bit array and its side length.
 * 'X' or '#' is value 1 (black/stroke); every other cell is value 0 (white) — the WHOLE
 * field is encoded, matching the per-position black/white base layer.
 */
function parseGrid(text) {
	const rows = text.split('\n').filter(line => line.length > 0);
	const size = Math.max(rows.length, ...rows.map(r => r.length));
	const bits = new Uint8Array(size * size);
	rows.forEach((line, r) => {
		for (let c = 0; c < size; c++) bits[r * size + c] = (line[c] === 'X' || line[c] === '#') ? 1 : 0;
	});
	return { bits, size };
}

const shapes = {
	'plus': [
		'.........',
		'....X....',
		'....X....',
		'....X....',
		'.XXXXXXX.',
		'....X....',
		'....X....',
		'....X....',
		'.........',
	],
	'two blobs': [
		'..........',
		'.XX....XX.',
		'.XXX..XXX.',
		'.XX....XX.',
		'..........',
	],
	'L shape': [
		'......',
		'.X....',
		'.X....',
		'.X....',
		'.XXXX.',
		'......',
	],
};

/**
 * Run one ASCII shape for several frames (so cross-frame minting can fire) and print
 * the structural diagnostics. The same image repeats — frame 1 learns, frame 2+ mints.
 */
function runShape(name, gridText, opts = {}) {
	connectivity = opts.connectivity || 8;
	radius = opts.radius || 1;
	resetGeometry();
	const frames = opts.frames || 6;
	const { bits, size } = parseGrid(gridText);

	console.log(`\n================ ${name}  (${connectivity}-conn, ${frames} frames) ================`);
	console.log(gridText);

	const brain = new Brain(opts);
	for (let f = 0; f < frames; f++) brain.processFrame(bits, size, true);
	reportDiagnostics(brain);
}

// ---------------------------------------------------------------------------
//  7. MNIST — feed real binary digit fields through the corrected mechanism
// ---------------------------------------------------------------------------

/**
 * Load the MNIST train/test split, or report and return null if the data is absent.
 */
function loadMnist() {
	const dataDir = path.join(SIM_DIR, '..', 'data');
	try {
		return {
			trainImages: loadImages(path.join(dataDir, 'train-images-idx3-ubyte.gz')),
			trainLabels: loadLabels(path.join(dataDir, 'train-labels-idx1-ubyte.gz')),
			testImages: loadImages(path.join(dataDir, 't10k-images-idx3-ubyte.gz')),
			testLabels: loadLabels(path.join(dataDir, 't10k-labels-idx1-ubyte.gz')),
			dataDir,
		};
	} catch (err) {
		console.log(`MNIST data not found (run jobs/download.js first): ${err.message}`);
		return null;
	}
}

/**
 * Stream `count` MNIST images through one persistent brain and report the structural
 * diagnostics — how patterns accrue, reuse, and settle (the merge/split experiment).
 * No supervised readout here; this is the structure-only run.
 */
function runMnist(opts = {}) {
	const count = opts.count || 200;
	const imageSize = opts.imageSize || 14;
	connectivity = 8;
	radius = opts.radius || 1;
	resetGeometry();

	const data = loadMnist();
	if (!data) return;
	const encoder = new MNISTPixelChannelsEncoder(2, imageSize, radius);

	console.log(`\n################ MNIST structure (${count} images, ${imageSize}×${imageSize} binary, radius ${radius}) ################`);
	console.log(`  knobs: merge ${opts.mergeThreshold ?? 0.5}, error ${opts.errorThreshold ?? 0.3}, reuse ${opts.reuse ?? true}, refine ${opts.refine ?? false}`);

	const brain = new Brain(opts);
	const start = Date.now();
	for (let i = 0; i < count; i++) {
		const bits = encoder.buildBits(data.trainImages[i]);
		brain.processFrame(bits, imageSize, true);
		if ((i + 1) % Math.max(1, Math.floor(count / 5)) === 0) {
			const snap = brain.snapshot();
			console.log(`  after ${i + 1}: ${snap.patternCount} patterns (${[...snap.byLevel.keys()].sort((a, b) => a - b).map(l => `L${l}:${snap.byLevel.get(l).length}`).join(' ')})`);
		}
	}
	console.log(`  ${((Date.now() - start) / 1000).toFixed(1)}s total`);
	reportDiagnostics(brain);
}

// ---------------------------------------------------------------------------
//  8. Supervised readout — the MNIST oracle (accuracy)
// ---------------------------------------------------------------------------

/**
 * Train a per-pattern digit-count table by streaming the train split through one
 * persistent brain (structure grows as it learns), then freeze and decode the test
 * split by NB-product over the patterns each image fires. The voter key is the stable
 * pattern id, so a pattern that fires on many images is one cross-image voter — the
 * §7 readout. Returns nothing; prints train/test accuracy and the diagnostics.
 */
function runAccuracy(opts = {}) {
	const trainCount = opts.trainCount || 2000;
	const testCount = opts.testCount || 1000;
	const imageSize = opts.imageSize || 14;
	connectivity = 8;
	radius = opts.radius || 1;
	resetGeometry();
	const EPS = 1e-3; // mirrors the brain's NB Laplace floor

	const data = loadMnist();
	if (!data) return;
	const encoder = new MNISTPixelChannelsEncoder(2, imageSize, radius);

	console.log(`\n################ MNIST readout (train ${trainCount}, test ${testCount}, ${imageSize}×${imageSize}, radius ${radius}) ################`);
	console.log(`  knobs: merge ${opts.mergeThreshold ?? 0.5}, error ${opts.errorThreshold ?? 0.3}, reuse ${opts.reuse ?? true}, refine ${opts.refine ?? false}`);

	const brain = new Brain(opts);

	// TRAIN — grow structure (learning on) and tally per-pattern digit counts on fired patterns.
	const digitCounts = new Map(); // patternId -> Int32Array(10)
	const trainStart = Date.now();
	let deepest = 0;
	for (let i = 0; i < trainCount; i++) {
		const bits = encoder.buildBits(data.trainImages[i]);
		const { levels, firedPatternIds } = brain.processFrame(bits, imageSize, true);
		deepest = Math.max(deepest, levels.length);
		const label = data.trainLabels[i];
		for (const id of firedPatternIds) {
			let counts = digitCounts.get(id);
			if (!counts) { counts = new Int32Array(10); digitCounts.set(id, counts); }
			counts[label]++;
		}
		if ((i + 1) % Math.max(1, Math.floor(trainCount / 5)) === 0) {
			console.error(`  [m${opts.mergeThreshold ?? 0.5} e${opts.errorThreshold ?? 0.3}] trained ${i + 1}/${trainCount} | ${brain.patterns.size} patterns | deepest ${deepest}`);
		}
	}
	console.log(`  trained: ${brain.patterns.size} patterns, ${digitCounts.size} of them voted, deepest level ${deepest} (${((Date.now() - trainStart) / 1000).toFixed(1)}s)`);

	// EVAL — freeze structure (learning off → no mint/refine/reap) and NB-product decode.
	const evaluate = (images, labels, n, name) => {
		let correct = 0, noMatch = 0;
		for (let i = 0; i < n; i++) {
			const bits = encoder.buildBits(images[i]);
			const { firedPatternIds } = brain.processFrame(bits, imageSize, false);
			const logScores = new Float64Array(10);
			let matched = 0;
			for (const id of firedPatternIds) {
				const counts = digitCounts.get(id);
				if (!counts) continue;
				matched++;
				let total = 0;
				for (let d = 0; d < 10; d++) total += counts[d];
				for (let d = 0; d < 10; d++) logScores[d] += Math.log(counts[d] / total + EPS);
			}
			if (matched === 0) noMatch++;
			let best = 0;
			for (let d = 1; d < 10; d++) if (logScores[d] > logScores[best]) best = d;
			if (best === labels[i]) correct++;
		}
		const acc = 100 * correct / n;
		console.log(`  ${name}: ${acc.toFixed(2)}% (${correct}/${n})${noMatch ? `  [${noMatch} had no matching pattern]` : ''}`);
		return { acc, noMatch };
	};

	const trainEval = evaluate(data.trainImages, data.trainLabels, Math.min(testCount, trainCount), 'train eval');
	const testEval = evaluate(data.testImages, data.testLabels, testCount, 'test  eval ');
	if (!opts.quiet) reportDiagnostics(brain);
	return { trainAcc: trainEval.acc, testAcc: testEval.acc, testNoMatch: testEval.noMatch, patterns: brain.patterns.size, voters: digitCounts.size, maxLevel: deepest };
}

// ---------------------------------------------------------------------------
//  9. Sweeps — the merge/split equilibrium experiment
// ---------------------------------------------------------------------------

/**
 * Sweep the merge and error thresholds and tabulate the MNIST readout for each — test accuracy,
 * pattern count, deepest level, and no-match rate — to see how θ (recognition/reuse) and the error
 * tolerance trade off. Reuse and refinement are on (the full model); pass merges/errors to vary the grid.
 */
function runSweep(opts = {}) {
	const trainCount = opts.trainCount || 500;
	const testCount = opts.testCount || 500;
	const imageSize = opts.imageSize || 14;
	const merges = opts.merges || [0.3, 0.5, 0.7, 0.9];
	const errors = opts.errors || [0.2, 0.4];
	const refine = opts.refine ?? true;

	console.log(`\n################ SWEEP merge × error (train ${trainCount}, test ${testCount}, ${imageSize}×${imageSize}, reuse on, refine ${refine}) ################`);
	const rows = [];
	for (const mergeThreshold of merges) {
		for (const errorThreshold of errors) {
			const r = runAccuracy({ trainCount, testCount, imageSize, mergeThreshold, errorThreshold, reuse: true, refine, quiet: true });
			if (r) rows.push({ mergeThreshold, errorThreshold, ...r });
		}
	}

	console.log(`\n################ SWEEP SUMMARY ################`);
	console.log('  merge | error | test acc | train acc | patterns | maxLvl | no-match');
	for (const r of rows) {
		console.log(`   ${r.mergeThreshold.toFixed(2)} |  ${r.errorThreshold.toFixed(2)} |  ${r.testAcc.toFixed(2)}%  |  ${r.trainAcc.toFixed(2)}%  | ${String(r.patterns).padStart(8)} | ${String(r.maxLevel).padStart(6)} | ${r.testNoMatch}/${testCount}`);
	}
	const best = rows.slice().sort((a, b) => b.testAcc - a.testAcc)[0];
	if (best) console.log(`  best test: ${best.testAcc.toFixed(2)}% at merge ${best.mergeThreshold}, error ${best.errorThreshold}`);
}

/**
 * Test the "one grouping coefficient" hypothesis: sweep a single threshold θ with the error
 * threshold pinned to 1 − θ (the consequence of recognition and correction being one sameness
 * operation). Reports each θ and flags the best, to compare against the decoupled optimum.
 */
function runCoupledSweep(opts = {}) {
	const trainCount = opts.trainCount || 400;
	const testCount = opts.testCount || 400;
	const imageSize = opts.imageSize || 14;
	const thetas = opts.thetas || [0.5, 0.6, 0.7, 0.8, 0.9];

	console.log(`\n################ COUPLED SWEEP θ with error = 1−θ (train ${trainCount}, test ${testCount}, ${imageSize}×${imageSize}) ################`);
	const rows = [];
	for (const mergeThreshold of thetas) {
		const errorThreshold = Math.round((1 - mergeThreshold) * 100) / 100;
		const r = runAccuracy({ trainCount, testCount, imageSize, mergeThreshold, errorThreshold, reuse: true, refine: true, quiet: true });
		if (r) rows.push({ mergeThreshold, errorThreshold, ...r });
	}

	console.log(`\n################ COUPLED SUMMARY (θ / error=1−θ) ################`);
	console.log('  θ (merge) | error | test acc | train acc | patterns | maxLvl | no-match');
	for (const r of rows) {
		console.log(`     ${r.mergeThreshold.toFixed(2)}   |  ${r.errorThreshold.toFixed(2)} |  ${r.testAcc.toFixed(2)}%  |  ${r.trainAcc.toFixed(2)}%  | ${String(r.patterns).padStart(8)} | ${String(r.maxLevel).padStart(6)} | ${r.testNoMatch}/${testCount}`);
	}
	const best = rows.slice().sort((a, b) => b.testAcc - a.testAcc)[0];
	if (best) console.log(`  best coupled: ${best.testAcc.toFixed(2)}% at θ=${best.mergeThreshold} (error ${best.errorThreshold})`);
}

// ---------------------------------------------------------------------------
//  10. Lifecycle — watch the merge/split forces fight over a stream
// ---------------------------------------------------------------------------

/**
 * Stream images through two brains — MERGE-ONLY (reuse, no split) and MERGE+SPLIT
 * (reuse + refinement + aggressive adaptive error, no forgetting) — and print a
 * per-checkpoint table for each: live patterns, L1 footprint median/max, L2 count,
 * max level, and births/deaths/expansions/prunes since the last checkpoint. The
 * sparklines show L1 max-footprint and live count over time. The hypothesis holds if
 * merge-only footprints grow without bound while merge+split footprints GROW THEN
 * SHRINK, deaths and prunes turn the population over, and the max level climbs.
 */
function runLifecycle(opts = {}) {
	const trainCount = opts.trainCount || 1500;
	const imageSize = opts.imageSize || 14;
	const interval = opts.interval || Math.max(1, Math.floor(trainCount / 25));
	connectivity = 8;
	radius = opts.radius || 1;
	resetGeometry();

	const data = loadMnist();
	if (!data) return;
	const encoder = new MNISTPixelChannelsEncoder(2, imageSize, radius);

	const regimes = [
		{ name: 'merge only', knobs: { reuse: true, refine: false } },
		{ name: 'merge + refine', knobs: {
			reuse: true, refine: true, reliabilityFloor: 0.75, refineMinSamples: 3,
			expansionOverlapFloor: 0.34, mergeThreshold: 0.55, adaptiveError: true, errorMode: 'aggressive',
		} },
	];
	const warmup = opts.warmup || Math.floor(trainCount / 5); // let a few patterns establish before watching
	// Optionally drive a single regime (e.g. just 'refine' for a long run, since merge-only is characterized).
	const selected = opts.only ? regimes.filter(r => r.name.includes(opts.only)) : regimes;

	console.log(`\n################ LIFECYCLE (train ${trainCount}, ${imageSize}×${imageSize}, checkpoint every ${interval}) ################`);

	for (const regime of selected) {
		const brain = new Brain(regime.knobs);
		const rows = [];
		let watch = null;                 // ids of a few patterns we follow through their whole life
		const watchHistory = new Map();   // id → [{ frame, size }]  (size 0 once reaped)
		let prev = { births: 0, deaths: 0, expansions: 0, prunes: 0 };

		for (let i = 0; i < trainCount; i++) {
			brain.processFrame(encoder.buildBits(data.trainImages[i]), imageSize, true);

			// Lock the watchlist once at warmup: the largest L1 patterns then, so we follow
			// units that have already started growing and watch whether split carves them back.
			if (watch === null && i + 1 >= warmup) {
				// Sample across the size range (every k-th of the size-sorted L1 set), not just the
				// biggest — the biggest are reap-bait, the mid-sized are where shrink-and-survive shows.
				const sorted = [...brain.patterns.values()].filter(p => p.spatialLevel === 1).sort((a, b) => b.footprint.size - a.footprint.size);
				const step = Math.max(1, Math.floor(sorted.length / 5));
				watch = sorted.filter((_, i) => i % step === 0).slice(0, 5).map(p => p.id);
				for (const id of watch) watchHistory.set(id, []);
			}

			if ((i + 1) % interval !== 0 && i + 1 !== trainCount) continue;
			const snap = brain.snapshot();
			const l1 = snap.byLevel.get(1) || [];
			const sorted = [...l1].sort((a, b) => a - b);
			const d = brain.diag;
			rows.push({
				frame: i + 1,
				live: snap.patternCount,
				l1: l1.length,
				med: sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0,
				max: l1.length ? Math.max(...l1) : 0,
				l2: (snap.byLevel.get(2) || []).length,
				maxLevel: Math.max(0, ...snap.byLevel.keys()),
				births: d.births - prev.births,
				deaths: d.deaths - prev.deaths,
				expansions: d.expansions - prev.expansions,
				prunes: d.prunes - prev.prunes,
			});
			prev = { births: d.births, deaths: d.deaths, expansions: d.expansions, prunes: d.prunes };
			if (watch) for (const id of watch) {
				const p = brain.patterns.get(id);
				watchHistory.get(id).push({ frame: i + 1, size: p ? p.footprint.size : 0 });
			}
			// Live progress on stderr (unbuffered, unlike block-buffered stdout to a file) so a long run is observable.
			const last = rows[rows.length - 1];
			console.error(`  [${regime.name}] f${last.frame} | live ${last.live} | L1 med/max ${last.med}/${last.max} | L2 ${last.l2} | deaths ${d.deaths}`);
		}

		console.log(`\n── ${regime.name} (${Object.entries(regime.knobs).map(([k, v]) => `${k}=${v}`).join(' ')}) ──`);
		console.log('   frame | live | L1# | L1 med/max | L2# | maxLvl | +born -died | expand prune');
		for (const r of rows) {
			console.log(`   ${String(r.frame).padStart(5)} | ${String(r.live).padStart(4)} | ${String(r.l1).padStart(3)} | ${String(r.med).padStart(3)}/${String(r.max).padStart(3)}    | ${String(r.l2).padStart(3)} | ${String(r.maxLevel).padStart(6)} | +${String(r.births).padStart(4)} -${String(r.deaths).padStart(4)} | ${String(r.expansions).padStart(5)} ${String(r.prunes).padStart(5)}`);
		}
		console.log(`   L1 max-footprint: ${spark(rows.map(r => r.max))}  (${rows[0].max} → peak ${Math.max(...rows.map(r => r.max))} → ${rows[rows.length - 1].max})`);
		console.log(`   live patterns:    ${spark(rows.map(r => r.live))}  (${rows[0].live} → ${rows[rows.length - 1].live})`);

		// Per-pattern tracers: each watched pattern's footprint over its life, and the children
		// it spawned at higher levels — the individual grow → shrink → climb story.
		console.log('   watched patterns (footprint over life | children by level):');
		for (const id of watch || []) {
			const hist = watchHistory.get(id);
			const sizes = hist.map(h => h.size);
			const peak = Math.max(...sizes);
			const alive = sizes.filter(s => s > 0);
			const trough = alive.length ? Math.min(...alive) : 0; // smallest it shrank to while alive
			const end = sizes[sizes.length - 1];
			const childrenByLevel = new Map();
			for (const p of brain.patterns.values()) {
				if (p.parents.has(id)) childrenByLevel.set(p.spatialLevel, (childrenByLevel.get(p.spatialLevel) || 0) + 1);
			}
			const kids = [...childrenByLevel.entries()].sort((a, b) => a[0] - b[0]).map(([lvl, n]) => `L${lvl}:${n}`).join(' ') || 'none';
			const fate = end === 0 ? 'REAPED' : `alive ${end}px`;
			console.log(`     ${id.padEnd(8)} ${spark(sizes)} peak ${peak} → trough ${trough}px → ${fate} | children ${kids}`);
		}
	}
}

// ---------------------------------------------------------------------------
//  11. Climb experiment — does L3 form, and if not, where does the funnel stall?
// ---------------------------------------------------------------------------

/**
 * Print the per-level climb funnel: how active units at each level flow active → mispredict
 * → (after the ≥1-neighbor rule) request → mint vs reuse. A level whose `requests` collapse to
 * zero (all mispredictors are neighborless) or whose `minted` is zero while `reused` is high
 * tells us exactly why the next level never forms.
 */
function reportFunnel(brain) {
	console.log('   level | active | mispredict | neighborless | requests | minted | reused');
	for (const level of [...brain.diag.funnelByLevel.keys()].sort((a, b) => a - b)) {
		const f = brain.diag.funnelByLevel.get(level);
		console.log(`   L${level}→L${level + 1} | ${String(f.active).padStart(6)} | ${String(f.mispredict).padStart(10)} | ${String(f.neighborless).padStart(12)} | ${String(f.requests).padStart(8)} | ${String(f.minted).padStart(6)} | ${String(f.reused).padStart(6)}`);
	}
}

/**
 * Test whether the hierarchy can climb past L2. Streams MNIST through one merge+refine brain and
 * reports the per-level pattern counts, the climb funnel, the deepest level reached, and a sample
 * of any L3+ patterns. Run with reuse OFF (the default here) to force mints — if L3 appears only
 * then, reuse short-circuits the climb (no bug); if L3 never appears, the funnel localizes the stall.
 */
function runClimb(opts = {}) {
	const trainCount = opts.trainCount || 400;
	const imageSize = opts.imageSize || 14;
	const reuse = opts.reuse ?? false;
	connectivity = 8;
	radius = opts.radius || 1;
	resetGeometry();

	const data = loadMnist();
	if (!data) return;
	const encoder = new MNISTPixelChannelsEncoder(2, imageSize, radius);

	const knobs = { reuse, refine: true, reliabilityFloor: 0.75, refineMinSamples: 3, expansionOverlapFloor: 0.34, mergeThreshold: 0.55, adaptiveError: true, errorMode: 'aggressive' };
	console.log(`\n################ CLIMB (train ${trainCount}, ${imageSize}×${imageSize}, reuse=${reuse}) ################`);
	console.log(`  knobs: ${Object.entries(knobs).map(([k, v]) => `${k}=${v}`).join(' ')}`);

	const brain = new Brain(knobs);
	let deepest = 0;
	for (let i = 0; i < trainCount; i++) {
		const { levels } = brain.processFrame(encoder.buildBits(data.trainImages[i]), imageSize, true);
		deepest = Math.max(deepest, levels.length);
		if ((i + 1) % Math.max(1, Math.floor(trainCount / 10)) === 0) {
			const snap = brain.snapshot();
			const perLevel = [...snap.byLevel.keys()].sort((a, b) => a - b).map(l => `L${l}:${snap.byLevel.get(l).length}`).join(' ');
			console.error(`  f${i + 1} | deepest level ${deepest} | ${perLevel}`);
		}
	}

	const snap = brain.snapshot();
	console.log(`\n  live: ${snap.patternCount} patterns, ${snap.baseCount} base; deepest level reached ${deepest}`);
	for (const level of [...snap.byLevel.keys()].sort((a, b) => a - b)) {
		console.log(`    level ${level}: ${snap.byLevel.get(level).length} patterns | footprints ${summarizeSizes(snap.byLevel.get(level))}`);
	}
	console.log('');
	reportFunnel(brain);

	// Surface the actual L3+ patterns, if any — the thing we want to see.
	const high = [...brain.patterns.values()].filter(p => p.spatialLevel >= 3).sort((a, b) => (a.id < b.id ? -1 : 1));
	console.log(`\n  L3+ patterns: ${high.length}`);
	for (const p of high.slice(0, 10)) {
		const kids = [...brain.patterns.values()].filter(q => q.parents.has(p.id)).length;
		console.log(`    ${p.id}: footprint ${p.footprint.size}px | context ${p.context.size} | parents ${p.parents.size} | children ${kids}`);
	}
}

// ---------------------------------------------------------------------------
//  12. CLI dispatch
// ---------------------------------------------------------------------------

const argv = process.argv.slice(2);
if (argv[0] === 'mnist') {
	runMnist({ count: Number(argv[1]) || 200, imageSize: Number(argv[2]) || 14, radius: Number(argv[3]) || 1 });
} else if (argv[0] === 'acc') {
	// node wavefront-sim.js acc [train] [test] [imageSize] [merge] [error] [refine|norefine]
	runAccuracy({
		trainCount: Number(argv[1]) || 2000,
		testCount: Number(argv[2]) || 1000,
		imageSize: Number(argv[3]) || 14,
		mergeThreshold: argv[4] ? Number(argv[4]) : 0.5,
		errorThreshold: argv[5] ? Number(argv[5]) : 0.3,
		refine: argv[6] !== 'norefine',
		reuse: true,
	});
} else if (argv[0] === 'sweep') {
	runSweep({ trainCount: Number(argv[1]) || 1000, testCount: Number(argv[2]) || 500, imageSize: Number(argv[3]) || 14 });
} else if (argv[0] === 'couple') {
	// node wavefront-sim.js couple [train] [test] [imageSize]   — sweep θ with error pinned to 1−θ
	runCoupledSweep({ trainCount: Number(argv[1]) || 400, testCount: Number(argv[2]) || 400, imageSize: Number(argv[3]) || 14 });
} else if (argv[0] === 'lifecycle') {
	// node wavefront-sim.js lifecycle [frames] [imageSize] [only]   (only = 'refine' | 'merge only' to pick one regime)
	runLifecycle({ trainCount: Number(argv[1]) || 1500, imageSize: Number(argv[2]) || 14, only: argv[3] });
} else if (argv[0] === 'climb') {
	// node wavefront-sim.js climb [frames] [imageSize] [reuse]   (reuse = 'reuse' to enable; default off, to force mints)
	runClimb({ trainCount: Number(argv[1]) || 400, imageSize: Number(argv[2]) || 14, reuse: argv[3] === 'reuse' });
} else {
	for (const [name, lines] of Object.entries(shapes)) runShape(name, lines.join('\n'));
}
