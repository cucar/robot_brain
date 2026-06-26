// ============================================================================
//  Wave-front neuron-reuse — sequential reference simulation (proof of concept)
// ============================================================================
//
//  ⚠ OBSOLETE — THIS SIMULATION MODELS THE WRONG MECHANISM. DO NOT BUILD ON IT.
//  It is cold-start, single-frame, no recognition, active = ON pixels, and groups
//  by identical observed-set. The correct model is recognize → predict L0 → on
//  misprediction transitively-merge-cluster the requests → reuse/expand or mint,
//  across many frames. See docs/neuron-reuse.md §3 and the rebuild spec
//  docs/neuron-reuse-simulation.md. Kept only as a historical reference.
//
// A small, deterministic, dependency-free model of the wave-front grouping +
// reuse algorithm described in docs/neuron-reuse.md (§2 footprints, §3 reuse).
// It is the *executable spec* / oracle for the incremental migration described
// in docs/neuron-reuse-wavefront-implementation.md: validate the algorithm here,
// then transliterate into the Rust brain (brain/brain-core).
//
// What it models (one frame, spatial distance d=0, every unit novel → erroring):
//   - A NEURON has an id, a spatial_level, and a FOOTPRINT: the set of base
//     pixels it covers. Base sensory neurons are single active pixels
//     (footprint = {self}, with a coordinate); corrections are coordinate-less
//     (footprint only — the union of what they bind).
//   - NEIGHBOUR = footprints TOUCH in the base neighbour graph (adjacency), NOT
//     set-overlap. Two disjoint footprints are still neighbours if a base pixel
//     in one is adjacent to a base pixel in the other.
//   - OBSERVED-SET of a unit = the other active units its footprint touches.
//   - CO-FAILERS = units with the IDENTICAL observed-set. They group and mint
//     ONE shared correction (within-frame sharing / multi-parent).
//   - correction.footprint = union(co-failers' footprints) ∪ observed-set's
//     footprints — the whole bound cluster (so it stays connected even when the
//     co-failers' own footprints are disjoint, e.g. arms around a centre).
//   - ISOLATED unit (empty observed-set) is its own group; empties never merge.
//   - The minted corrections become the active units of the next level. Repeat
//     until the set of footprints stops changing.
//
// What it does NOT model yet (deliberately — these come with later stages):
//   - prediction / recognition of existing patterns,
//   - the cross-frame reuse INDEX lookup (Phase D) — here every unit errors,
//   - the merge/agreement THRESHOLD — grouping is exact observed-set match only,
//   - reward / actions / temporal distances (d>0), MNIST input + readout.
//
// Mapping to the Rust brain (camelCase here transliterates to snake_case there):
//   Brain.processFrame        ~ Brain::process_frame
//   Brain.activateSensoryEvents ~ activate_sensory_events
//   Brain.processSpatial      ~ process_spatial
//   Brain.processSpatialLevels~ process_spatial_levels (the settling sweep)
//   Brain.processSpatialLevel ~ process_spatial_level
//   Brain.mintSpatialCorrections / installSpatialCorrections ~ the Phase-C mint path
//   doFootprintsTouch         ~ footprint-adjacency test (Rust: bitset dilate + AND)
//
// Cold-start collapse: the Rust brain mints corrections that fire NEXT frame, so
// it grows one level per frame and reaches a fixed point over many frames. This
// reference computes that fixed point in ONE synchronous pass — each level's
// mints become the next level's active set immediately. Same hierarchy, no
// per-frame deferral, which is correct here because every unit is novel (nothing
// to recognise, only to mint).
//
// Run (ASCII shapes):  node apps/mnist/jobs/wavefront-sim.js
// Run (real MNIST):     node apps/mnist/jobs/wavefront-sim.js mnist [count] [imageSize] [radius]
// Edit the `shapes` map at the bottom to try other inputs.

import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadImages, loadLabels } from '../loader.js';
import { MNISTPixelChannelsEncoder } from '../encoder.js';

const SIM_DIR = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
//  1. Geometry — the base neighbour graph
// ---------------------------------------------------------------------------

// Connectivity of the base neighbour graph: 4 = von Neumann (orthogonal),
// 8 = Moore (orthogonal + diagonal). Set per run by runSimulation / runMnist.
// MNIST strokes are diagonal-heavy, so 4 vs 8 changes whether they bind.
let connectivity = 4;
// Radius of the base neighbour graph (Chebyshev for 8-conn, Manhattan for 4-conn).
// MNIST mirrors the brain's encoder radius (1 = 3×3, 2 = 5×5); ASCII shapes use 1.
let radius = 1;

/**
 * Parse a multi-line ASCII grid into the set of active base-pixel ids.
 * 'X' or '#' marks an active pixel; an id is the string "col,row".
 */
function parseGrid(text) {
	const active = new Set();
	const rows = text.split('\n').filter(line => line.length > 0);
	rows.forEach((line, r) => {
		for (let c = 0; c < line.length; c++) {
			// Two glyphs so grids can be drawn with either marker.
			if (line[c] === 'X' || line[c] === '#') active.add(`${c},${r}`);
		}
	});
	return active;
}

/**
 * List the coordinate neighbours of a base pixel under the current connectivity.
 * These edges define the base neighbour graph; footprint adjacency is built on top.
 */
function listPixelNeighbors(pixelId) {
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
	return neighbors;
}

/**
 * Decide whether two footprints TOUCH in the base neighbour graph.
 * True if some base pixel in one is equal to, or adjacent to, a base pixel in the other.
 * This is adjacency, not set-intersection — disjoint-but-abutting footprints touch.
 * Rust port: footprints are bitsets; this is dilate-by-neighbour-ring + AND, nonzero.
 */
function doFootprintsTouch(footprintA, footprintB) {
	for (const a of footprintA) {
		// Shared pixel — footprints overlap, which trivially counts as touching.
		if (footprintB.has(a)) return true;
		// Otherwise look for an adjacency across the boundary.
		for (const b of listPixelNeighbors(a)) {
			if (footprintB.has(b)) return true;
		}
	}
	return false;
}

/**
 * Merge several footprints into one set (the union of all their base pixels).
 */
function mergeFootprints(footprints) {
	const out = new Set();
	for (const fp of footprints) for (const pixel of fp) out.add(pixel);
	return out;
}

/**
 * Compute every active unit's observed-set in one pass, via an inverted
 * pixel → units index, so adjacency is a local lookup rather than an O(N²)
 * all-pairs scan — the JS analog of the Rust bitset dilate + AND. For each unit,
 * the observed-set is the other units whose footprint TOUCHES its own.
 * Returns Map<unitId, unit[]>, each list in active order for determinism.
 */
function computeObservedSets(active) {
	// Inverted index: base pixel → the units whose footprint covers it.
	const pixelToUnits = new Map();
	for (const unit of active) {
		for (const pixel of unit.footprint) {
			let list = pixelToUnits.get(pixel);
			if (!list) { list = []; pixelToUnits.set(pixel, list); }
			list.push(unit);
		}
	}

	// Each unit's position in `active`, so observed-sets stay in a deterministic order.
	const order = new Map();
	active.forEach((unit, i) => order.set(unit.id, i));

	const observedByUnit = new Map();
	for (const unit of active) {
		// Candidate touchers: units covering a pixel equal or adjacent to one of ours.
		const candidateIds = new Set();
		for (const pixel of unit.footprint) {
			for (const probe of [pixel, ...listPixelNeighbors(pixel)]) {
				const list = pixelToUnits.get(probe);
				if (!list) continue;
				for (const other of list) if (other.id !== unit.id) candidateIds.add(other.id);
			}
		}
		// Materialize in active order; confirm with the touch predicate (the index already
		// yields exactly the touchers — the confirm just keeps doFootprintsTouch authoritative).
		const observed = [...candidateIds]
			.sort((a, b) => order.get(a) - order.get(b))
			.map(id => active[order.get(id)])
			.filter(other => doFootprintsTouch(unit.footprint, other.footprint));
		observedByUnit.set(unit.id, observed);
	}
	return observedByUnit;
}

// ---------------------------------------------------------------------------
//  2. The Brain — mirrors the Rust decomposition and the spatial frame flow
// ---------------------------------------------------------------------------
//
// Rust ownership chain: Brain -> Thalamus -> Region[R] -> Column[C] -> Neuron,
// with neurons sharded across columns by `neuron_id % C`. That sharding is a
// parallelism layer; this single-threaded reference folds it into one neuron
// store and keeps the rest of the shape faithful.

class Brain {

	constructor() {
		// The neuron store (Column's `neurons` map, flattened — sharding not modelled).
		this.neurons = new Map();
		// Central metadata mirrors (Thalamus): spatial level and parents per neuron.
		this.neuronSpatialLevels = new Map();
		this.neuronParents = new Map();
		// Monotonic id suffix per spatial level, for readable correction ids.
		// Rust port: a single numeric `next_neuron_id` allocator instead.
		this.mintCounts = new Map();
	}

	/**
	 * Process one frame: activate the base sensory neurons, then run the spatial phase.
	 * Mirrors Brain::process_frame -> process_spatial.
	 */
	processFrame(activeBasePixels, maxLevels) {
		this.activateSensoryEvents(activeBasePixels);
		return this.processSpatial(maxLevels);
	}

	/**
	 * Create one base sensory neuron per active pixel at spatial level 0.
	 * Base neurons carry a coordinate and a footprint of {self}; corrections are
	 * coordinate-less (footprint only), per the wave-front design.
	 * Mirrors Brain::activate_sensory_events.
	 */
	activateSensoryEvents(activeBasePixels) {
		for (const pixel of activeBasePixels) {
			this.neurons.set(pixel, {
				id: pixel,
				spatialLevel: 0,
				footprint: new Set([pixel]),
				coordinate: pixel,
				parentIds: [],
				observedIds: [],
			});
			this.neuronSpatialLevels.set(pixel, 0);
		}
	}

	/**
	 * The spatial phase: run the level sweep (and surface its apex).
	 * Mirrors Brain::process_spatial.
	 */
	processSpatial(maxLevels) {
		return this.processSpatialLevels(maxLevels);
	}

	/**
	 * The settling level sweep. The active set is fixed at the start of each level;
	 * a level's minted corrections become the next level's active set, and the sweep
	 * stops when a level adds nothing new (a single pattern, or footprints stable).
	 * Mirrors Brain::process_spatial_levels (fired_set / subsumed_set / activations / max_active_level).
	 */
	processSpatialLevels(maxLevels) {
		const firedSet = new Set();
		const subsumedSet = new Set();
		const levels = [];
		// Level-0 active set: the base neurons, fixed at frame start.
		let active = [...this.neurons.values()];
		let previousSignature = null;

		for (let spatialLevel = 0; spatialLevel < maxLevels; spatialLevel++) {
			// Every active unit fires this level.
			for (const unit of active) firedSet.add(unit.id);

			// Process the level: group co-failers and mint one correction per group.
			const corrections = this.processSpatialLevel(active, spatialLevel);

			// Each correction's parents are subsumed (absorbed by the higher-level pattern).
			for (const c of corrections) for (const parentId of c.parentIds) subsumedSet.add(parentId);

			// Converged when one pattern covers everything or nothing changed this level.
			const signature = footprintSignature(corrections);
			const converged = corrections.length <= 1 || signature === previousSignature;
			levels.push({ spatialLevel: spatialLevel + 1, corrections, converged });
			if (converged) break;

			previousSignature = signature;
			// The mints activate at level+1 — they are the next level's active set.
			active = corrections;
		}

		// The top level's corrections fired but were never re-processed (the sweep ended);
		// add them so the apex (fired \ subsumed) surfaces the top-level patterns.
		const topCorrections = levels.length ? levels[levels.length - 1].corrections : [];
		for (const c of topCorrections) firedSet.add(c.id);

		// Apex: fired neurons not absorbed by any parent (what would hand off to temporal).
		const apex = [...firedSet].filter(id => !subsumedSet.has(id));
		return { levels, firedSet, subsumedSet, apex };
	}

	/**
	 * Process a single spatial level: mint correction specs, then install them as neurons.
	 * Mirrors process_spatial_level -> mint_spatial_corrections + install_spatial_corrections.
	 */
	processSpatialLevel(active, spatialLevel) {
		const specs = this.mintSpatialCorrections(active);
		return this.installSpatialCorrections(specs, spatialLevel);
	}

	/**
	 * Group co-failers by observed-set and build one correction spec per group.
	 * For each unit, observed-set = the other active units whose footprint touches its own;
	 * a unit predicts its neighbours, never itself, so it is excluded from its own set.
	 * Units with the identical observed-set are co-failers and share one correction.
	 * An isolated unit (empty observed-set) is its own group — empties never merge.
	 * footprint = union(co-failers) ∪ observed-set (the whole bound cluster).
	 * Mirrors Thalamus::mint_spatial_corrections (group_by observed-set).
	 */
	mintSpatialCorrections(active) {
		// Observed-sets for all active units in one indexed pass (scales to MNIST).
		const observedByUnit = computeObservedSets(active);
		const errors = active.map(unit => {
			const observed = observedByUnit.get(unit.id);
			// Isolated units get a UNIQUE key so empties never group together.
			// Rust port: the grouping key is a sorted Vec<NeuronId>, not a delimited string.
			const key = observed.length === 0
				? `ISOLATED:${unit.id}`
				: observed.map(o => o.id).sort().join('|');
			return { unit, observed, key };
		});

		// Bucket by identical observed-set — the co-failer groups.
		const groups = new Map();
		for (const error of errors) {
			if (!groups.has(error.key)) groups.set(error.key, []);
			groups.get(error.key).push(error);
		}

		// One spec per group: parents (co-failers), observed-set, and the union∪observed footprint.
		const specs = [];
		for (const [, groupErrors] of groups) {
			const coFailers = groupErrors.map(e => e.unit);
			// Every member of a group shares the same observed-set by construction.
			const observed = groupErrors[0].observed;
			const footprint = mergeFootprints([
				...coFailers.map(p => p.footprint),
				...observed.map(o => o.footprint),
			]);
			specs.push({ footprint, parentIds: coFailers.map(p => p.id), observedIds: observed.map(o => o.id) });
		}
		return specs;
	}

	/**
	 * Install correction specs as neurons. Specs that share a footprint are folded
	 * into one neuron (same pattern = same neuron — a stand-in for the recognition /
	 * reuse-index match that lands in a later stage). Each surviving spec gets an id
	 * and spatial level, registers its parents, and is stored.
	 * Mirrors Column::install_spatial_corrections (allocate id, set spatial_level, wire parents).
	 */
	installSpatialCorrections(specs, spatialLevel) {
		// Fold specs with the identical footprint, merging their parents/observed.
		const bySignature = new Map();
		for (const spec of specs) {
			const signature = [...spec.footprint].sort().join('|');
			if (!bySignature.has(signature)) {
				bySignature.set(signature, spec);
			} else {
				const kept = bySignature.get(signature);
				kept.parentIds = [...new Set([...kept.parentIds, ...spec.parentIds])];
				kept.observedIds = [...new Set([...kept.observedIds, ...spec.observedIds])];
			}
		}

		// Allocate ids and store the surviving corrections.
		const corrections = [];
		let n = this.mintCounts.get(spatialLevel) || 0;
		for (const spec of bySignature.values()) {
			const id = `L${spatialLevel + 1}#${n++}`;
			const neuron = {
				id,
				spatialLevel: spatialLevel + 1,
				footprint: spec.footprint,
				coordinate: null, // corrections are coordinate-less
				parentIds: spec.parentIds,
				observedIds: spec.observedIds,
			};
			this.neurons.set(id, neuron);
			this.neuronSpatialLevels.set(id, spatialLevel + 1);
			this.neuronParents.set(id, spec.parentIds);
			corrections.push(neuron);
		}
		this.mintCounts.set(spatialLevel, n);
		return corrections;
	}
}

/**
 * A stable signature of a level's footprints, used to detect convergence
 * (the set of footprints is unchanged from the previous level).
 */
function footprintSignature(corrections) {
	return corrections.map(c => [...c.footprint].sort().join('|')).sort().join(' ;; ');
}

// ---------------------------------------------------------------------------
//  3. Reporting
// ---------------------------------------------------------------------------

/**
 * Render a footprint on its OWN bounding box, so a local pattern prints small.
 * The bounding-box size is the at-a-glance signal for "local vs whole-shape".
 */
function renderFootprint(footprint) {
	const coords = [...footprint].map(p => p.split(',').map(Number));
	const cols = coords.map(x => x[0]);
	const rows = coords.map(x => x[1]);
	const minC = Math.min(...cols), maxC = Math.max(...cols);
	const minR = Math.min(...rows), maxR = Math.max(...rows);
	const lines = [];
	for (let r = minR; r <= maxR; r++) {
		let line = '';
		for (let c = minC; c <= maxC; c++) line += footprint.has(`${c},${r}`) ? 'X' : '.';
		lines.push('        ' + line);
	}
	return lines.join('\n');
}

/**
 * Print one level: a stats line (count, max footprint, reuse-groups, singletons)
 * plus the largest few footprints. Reuse-groups (parents ≥ 2) are the within-frame
 * sharing events — watch how few there are on irregular shapes.
 */
function reportLevel(record, shapeSize, renderCap) {
	const { spatialLevel, corrections, converged } = record;
	const sizes = corrections.map(c => c.footprint.size);
	const maxSize = Math.max(...sizes);
	const reuseGroups = corrections.filter(c => c.parentIds.length >= 2).length;
	const singletons = corrections.length - reuseGroups;

	console.log(`\n--- Level ${spatialLevel}: ${corrections.length} pattern(s) | max footprint ${maxSize}px/${shapeSize}px | reuse-groups ${reuseGroups} | singletons ${singletons} ---`);

	// Only render footprints on the first level (the locality question) and the
	// converged level (the final result); middle levels just show stats.
	if (spatialLevel === 1 || converged) {
		const sorted = [...corrections].sort((a, b) => b.footprint.size - a.footprint.size);
		for (const c of sorted.slice(0, renderCap)) {
			const reuse = c.parentIds.length >= 2 ? `  [REUSE: ${c.parentIds.length} parents]` : '';
			console.log(`  ${c.id}: ${c.footprint.size}px, ${c.parentIds.length} parent(s)${reuse}`);
			console.log(renderFootprint(c.footprint));
		}
		if (sorted.length > renderCap) console.log(`  ... and ${sorted.length - renderCap} more`);
	}
}

/**
 * Run the whole simulation for one named shape and print a per-shape report,
 * ending with the headline: did level 1 stay local?
 */
function runSimulation(name, gridText, opts = {}) {
	connectivity = opts.connectivity || 4;
	radius = opts.radius || 1;
	const maxLevels = opts.maxLevels || 10;
	const renderCap = opts.renderCap || 8;

	console.log(`\n================ ${name}  (connectivity=${connectivity}) ================`);
	console.log(gridText);

	const active = parseGrid(gridText);
	console.log(`Level 0: ${active.size} base pixels active`);

	const { levels } = new Brain().processFrame(active, maxLevels);
	for (const record of levels) reportLevel(record, active.size, renderCap);

	const last = levels[levels.length - 1];
	const level1Max = Math.max(...levels[0].corrections.map(c => c.footprint.size));
	const verdict = last.converged
		? (last.corrections.length <= 1 ? 'single pattern spans the structure' : 'footprints stable')
		: `hit maxLevels=${maxLevels}`;
	console.log(`(converged at level ${last.spatialLevel}: ${verdict})`);
	console.log(`SUMMARY: level-1 max footprint = ${level1Max}px of ${active.size}px  ->  ${level1Max < active.size ? 'LEVEL 1 STAYED LOCAL' : 'level 1 = whole shape'}`);
}

// ---------------------------------------------------------------------------
//  4. Test shapes
// ---------------------------------------------------------------------------

// Each shape is drawn as ASCII; 'X' is an active pixel. The interesting axes are
// symmetry (drives within-frame sharing), connectedness (drives whether parts bind),
// and irregularity (stresses the exact observed-set grouping).
const shapes = {
	// Symmetric: sharing fires strongly (arms share an observed centre).
	'plus of pluses': [
		'.........',
		'....X....',
		'...XXX...',
		'..X.X.X..',
		'.XXXXXXX.',
		'..X.X.X..',
		'...XXX...',
		'....X....',
		'.........',
	],
	// Irregular: almost every pixel sees a unique neighbour set → little sharing.
	'uneven blob': [
		'........',
		'..XX....',
		'..XXX.X.',
		'...XXXX.',
		'.X.XXX..',
		'..XXX...',
		'....X...',
		'........',
	],
	'ragged patch': [
		'.........',
		'..X.XX...',
		'.XXXX.X..',
		'.XX.XXX..',
		'..XXXX...',
		'.X..XX.X.',
		'..XXX....',
		'.........',
	],
	// Diagonal: disconnected under 4-conn (no orthogonal adjacency), connected under 8.
	'diagonal (4-conn)': [
		'.......',
		'.X.....',
		'..X....',
		'...X...',
		'....X..',
		'.....X.',
		'.......',
	],
	'L shape': [
		'......',
		'.X....',
		'.X....',
		'.X....',
		'.XXXX.',
		'......',
	],
	// Solid: connected but asymmetric locally → zero within-frame sharing.
	'solid 4x4 block': [
		'......',
		'.XXXX.',
		'.XXXX.',
		'.XXXX.',
		'.XXXX.',
		'......',
	],
	// Two components: must stay separate (locality must not bind across the gap).
	'two separate blobs': [
		'..........',
		'.XX....XX.',
		'.XXX..XXX.',
		'.XX....XX.',
		'..........',
	],
};

// ---------------------------------------------------------------------------
//  5. MNIST input — feed real binary digit patches through the wave-front
// ---------------------------------------------------------------------------

/**
 * Convert a quantized bit array (row-major, imageSize²) into the set of active
 * base-pixel ids. A pixel is active if its bucket value is non-zero — for binary
 * MNIST this is the ON/stroke set, the direct analog of an ASCII 'X'.
 * (Whether background/value-0 pixels should also be neurons is a Stage-C parity
 * decision against the brain; for now the foreground stroke is the active set.)
 */
function bitsToActivePixels(bits, imageSize) {
	const active = [];
	for (let p = 0; p < bits.length; p++) {
		if (bits[p] !== 0) active.push(`${p % imageSize},${Math.floor(p / imageSize)}`);
	}
	return active;
}

/**
 * Feed real MNIST test images through the wave-front grouping and report
 * structural diagnostics per image — depth, per-level pattern counts, the
 * level-1 max footprint, neuron count, apex size, and timing. No supervised
 * readout yet (that is stage A3); this stage proves the sim runs on real digits
 * and that the inverted index keeps it fast.
 */
function runMnist(opts = {}) {
	const count = opts.count || 20;
	const imageSize = opts.imageSize || 14;
	const neighborhoodRadius = opts.radius || 1;
	connectivity = 8;            // the brain's encoder uses square (Moore) neighborhoods
	radius = neighborhoodRadius;

	const dataDir = path.join(SIM_DIR, '..', 'data');
	let images, labels;
	try {
		images = loadImages(path.join(dataDir, 't10k-images-idx3-ubyte.gz'));
		labels = loadLabels(path.join(dataDir, 't10k-labels-idx1-ubyte.gz'));
	} catch (err) {
		console.log(`MNIST data not found in ${dataDir} (run jobs/download.js first): ${err.message}`);
		return;
	}
	const encoder = new MNISTPixelChannelsEncoder(2, imageSize, neighborhoodRadius);

	console.log(`\n################ MNIST (${count} test images, ${imageSize}×${imageSize} binary, radius ${neighborhoodRadius}, ${connectivity}-conn) ################`);
	let totalMs = 0, sumDepth = 0, sumNeurons = 0, sumL1Max = 0, sumActive = 0;
	for (let i = 0; i < count; i++) {
		const bits = encoder.buildBits(images[i]);
		const activePixels = bitsToActivePixels(bits, imageSize);

		const start = process.hrtime.bigint();
		const brain = new Brain();
		const { levels, apex } = brain.processFrame(activePixels, 20);
		const ms = Number(process.hrtime.bigint() - start) / 1e6;

		const depth = levels.length;
		const level1Max = levels[0].corrections.length ? Math.max(...levels[0].corrections.map(c => c.footprint.size)) : 0;
		const perLevel = levels.map(l => l.corrections.length).join('/');
		totalMs += ms; sumDepth += depth; sumNeurons += brain.neurons.size; sumL1Max += level1Max; sumActive += activePixels.length;
		console.log(`  #${i} digit ${labels[i]}: ${activePixels.length}px | depth ${depth} | L-counts ${perLevel} | L1 max ${level1Max}px | ${brain.neurons.size} neurons | apex ${apex.length} | ${ms.toFixed(1)}ms`);
	}
	console.log('  ---');
	console.log(`  avg over ${count}: ${(sumActive / count).toFixed(0)}px active | depth ${(sumDepth / count).toFixed(1)} | L1 max ${(sumL1Max / count).toFixed(1)}px | ${(sumNeurons / count).toFixed(0)} neurons | ${(totalMs / count).toFixed(1)}ms/img`);
}

// ---------------------------------------------------------------------------
//  6. Supervised readout — the MNIST oracle (accuracy)
// ---------------------------------------------------------------------------

/**
 * The footprint signature of a pattern neuron — its sorted base-pixel set.
 * For binary MNIST this is the bound ON-pixel configuration, so it doubles as
 * the exact-match recognition key: two images share a pattern iff identical sig.
 */
function correctionSignature(correction) {
	return [...correction.footprint].sort().join('|');
}

/**
 * Collect the voter signatures for one image: every pattern neuron its
 * wave-front hierarchy produces (across all levels). Base pixels do not vote on
 * their own — an isolated pixel votes via its singleton correction.
 */
function votersForImage(encoder, image, imageSize) {
	const bits = encoder.buildBits(image);
	const active = bitsToActivePixels(bits, imageSize);
	const { levels } = new Brain().processFrame(active, 20);
	const signatures = [];
	for (const level of levels) for (const c of level.corrections) signatures.push(correctionSignature(c));
	return signatures;
}

/**
 * Train a per-pattern digit-count map on MNIST images, then decode test digits
 * by NB-product over the matching patterns — the supervised readout that turns
 * the sim into a full MNIST oracle. Every pattern is a voter keyed by its
 * footprint signature; a test image's patterns recognise a trained pattern iff
 * the signature matches EXACTLY (threshold = 1.0 — the exact-match baseline; the
 * merge threshold that loosens this to partial-overlap reuse lands in stage D).
 */
function runMnistAccuracy(opts = {}) {
	const trainCount = opts.trainCount || 2000;
	const testCount = opts.testCount || 1000;
	const imageSize = opts.imageSize || 14;
	const neighborhoodRadius = opts.radius || 1;
	connectivity = 8;          // square (Moore) neighbourhoods, as the brain's encoder
	radius = neighborhoodRadius;
	const EPS = 1e-3;          // mirrors the brain's NB Laplace floor

	const dataDir = path.join(SIM_DIR, '..', 'data');
	let trainImages, trainLabels, testImages, testLabels;
	try {
		trainImages = loadImages(path.join(dataDir, 'train-images-idx3-ubyte.gz'));
		trainLabels = loadLabels(path.join(dataDir, 'train-labels-idx1-ubyte.gz'));
		testImages = loadImages(path.join(dataDir, 't10k-images-idx3-ubyte.gz'));
		testLabels = loadLabels(path.join(dataDir, 't10k-labels-idx1-ubyte.gz'));
	} catch (err) {
		console.log(`MNIST data not found in ${dataDir} (run jobs/download.js first): ${err.message}`);
		return;
	}
	const encoder = new MNISTPixelChannelsEncoder(2, imageSize, neighborhoodRadius);

	console.log(`\n################ MNIST readout (train ${trainCount}, test ${testCount}, ${imageSize}×${imageSize} binary, radius ${neighborhoodRadius}) ################`);

	// TRAIN — accumulate per-signature digit counts (P(d|voter) = counts[d] / Σ counts).
	const digitCounts = new Map(); // signature -> Int32Array(10)
	const trainStart = Date.now();
	for (let i = 0; i < trainCount; i++) {
		const label = trainLabels[i];
		for (const sig of votersForImage(encoder, trainImages[i], imageSize)) {
			let counts = digitCounts.get(sig);
			if (!counts) { counts = new Int32Array(10); digitCounts.set(sig, counts); }
			counts[label]++;
		}
	}
	console.log(`  trained: ${digitCounts.size} distinct pattern signatures from ${trainCount} images (${((Date.now() - trainStart) / 1000).toFixed(1)}s)`);

	// EVAL — NB-product decode: argmax_d Σ_voter log(P(d|voter) + eps), frozen counts.
	const evaluate = (images, labels, count, name) => {
		let correct = 0, noMatch = 0;
		for (let i = 0; i < count; i++) {
			const logScores = new Float64Array(10);
			let matched = 0;
			for (const sig of votersForImage(encoder, images[i], imageSize)) {
				const counts = digitCounts.get(sig);
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
		console.log(`  ${name}: ${(100 * correct / count).toFixed(2)}% (${correct}/${count})${noMatch ? `  [${noMatch} images had no matching pattern]` : ''}`);
	};

	evaluate(trainImages, trainLabels, Math.min(testCount, trainCount), 'train eval');
	evaluate(testImages, testLabels, testCount, 'test  eval ');
}

// ---------------------------------------------------------------------------
//  7. CLI dispatch
// ---------------------------------------------------------------------------

const argv = process.argv.slice(2);
if (argv[0] === 'mnist') {
	// node wavefront-sim.js mnist [count] [imageSize] [radius]
	runMnist({ count: Number(argv[1]) || 20, imageSize: Number(argv[2]) || 14, radius: Number(argv[3]) || 1 });
} else if (argv[0] === 'acc') {
	// node wavefront-sim.js acc [trainCount] [testCount] [imageSize] [radius]
	runMnistAccuracy({ trainCount: Number(argv[1]) || 2000, testCount: Number(argv[2]) || 1000, imageSize: Number(argv[3]) || 14, radius: Number(argv[4]) || 1 });
} else {
	// Default: run every ASCII shape under 4-connectivity, then the diagonal under 8-conn
	// (4-conn shatters it into points, 8-conn binds it into one pattern).
	for (const [name, lines] of Object.entries(shapes)) {
		runSimulation(name, lines.join('\n'));
	}
	runSimulation('diagonal (8-conn)', shapes['diagonal (4-conn)'].join('\n'), { connectivity: 8 });
}
