// ============================================================================
//  Wave-front neuron-reuse — sequential reference simulation (proof of concept)
// ============================================================================
//
// A small, deterministic, dependency-free model of the wave-front grouping +
// reuse algorithm described in docs/neuron-reuse.md (§2 footprints, §3 reuse).
// It is the *executable spec*: a place to validate the algorithm's behaviour on
// hand-drawn shapes before porting it into the brain (brain/brain-core + the
// MNIST test job). Keep it faithful to the docs so it maps cleanly to the port.
//
// What it models (one frame, spatial distance d=0, every unit novel → erroring):
//   - A UNIT has an id and a FOOTPRINT: the set of base pixels it covers.
//     Base units are single active pixels (footprint = {self}); higher units are
//     minted corrections (footprint = union of what they bind).
//   - NEIGHBOUR = footprints TOUCH in the base neighbour graph (adjacency), NOT
//     set-overlap. Two disjoint footprints are still neighbours if a base pixel
//     in one is adjacent to a base pixel in the other.
//   - OBSERVED-SET of a unit = the other active units its footprint touches.
//   - CO-FAILERS = units with the IDENTICAL observed-set. They group and mint
//     ONE shared correction (this is within-frame reuse / multi-parent).
//   - correction.footprint = union(co-failers' footprints) ∪ observed-set's
//     footprints — the whole bound cluster (so it stays connected even when the
//     co-failers' own footprints are disjoint, e.g. arms around a centre).
//   - ISOLATED unit (empty observed-set) is its own group; empties never merge.
//   - The minted corrections become the active units of the next level. Repeat
//     until the set of footprints stops changing.
//
// What it does NOT model yet (deliberately — these come with the port):
//   - prediction (existing patterns suppressing errors next frame),
//   - the cross-frame reuse INDEX lookup (Phase D) — here every unit errors,
//   - the merge/agreement THRESHOLD — grouping is exact observed-set match only,
//   - reward / actions / temporal distances (d>0).
//
// Mapping to the brain:
//   runOneLevel       ~ one iteration of process_spatial_levels (the settling sweep)
//   doFootprintsTouch ~ footprint-adjacency test replacing channel-neighbour filtering
//   groupCoFailers    ~ Phase C batched mint: group_by (distance, observed-set)
//   mintCorrection    ~ mint_one(observed, distance); footprint = ⋃errs ∪ observed
//   (a reuse-index lookup would slot in front of mintCorrection — Phase D)
//
// Run:  node apps/mnist/jobs/wavefront-sim.js
// Edit the `shapes` map at the bottom to try other inputs.

// ---------------------------------------------------------------------------
//  1. Geometry — the base neighbour graph
// ---------------------------------------------------------------------------

// Connectivity of the base neighbour graph: 4 = von Neumann (orthogonal),
// 8 = Moore (orthogonal + diagonal). Set per run by runSimulation.
// MNIST strokes are diagonal-heavy, so 4 vs 8 changes whether they bind.
let connectivity = 4;

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
	// Orthogonal neighbours always count.
	const neighbors = [`${c - 1},${r}`, `${c + 1},${r}`, `${c},${r - 1}`, `${c},${r + 1}`];
	// Diagonals only under 8-connectivity.
	if (connectivity === 8) {
		neighbors.push(`${c - 1},${r - 1}`, `${c + 1},${r - 1}`, `${c - 1},${r + 1}`, `${c + 1},${r + 1}`);
	}
	return neighbors;
}

/**
 * Decide whether two footprints TOUCH in the base neighbour graph.
 * True if some base pixel in one is equal to, or adjacent to, a base pixel in the other.
 * This is adjacency, not set-intersection — disjoint-but-abutting footprints touch.
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

// ---------------------------------------------------------------------------
//  2. The wave-front operation — one level
// ---------------------------------------------------------------------------

/**
 * Compute the observed-set of a unit: the other active units whose footprint
 * touches its own. A unit predicts its neighbours, never itself, so the unit is
 * excluded from its own observed-set.
 */
function computeObservedSet(unit, units) {
	return units.filter(other => other.id !== unit.id && doFootprintsTouch(unit.footprint, other.footprint));
}

/**
 * Mint one correction binding a group of co-failers to their shared observed-set.
 * The footprint is the whole bound cluster: the co-failers' footprints UNION the
 * observed-set's footprints. Including the observed-set keeps the footprint
 * connected even when the co-failers' own footprints are disjoint.
 */
function mintCorrection(id, coFailers, observed) {
	const footprint = mergeFootprints([
		...coFailers.map(p => p.footprint),
		...observed.map(o => o.footprint),
	]);
	return {
		id,
		footprint,
		parentIds: coFailers.map(p => p.id), // the units wired to this correction (multi-parent under reuse)
		observedIds: observed.map(o => o.id), // what this correction is "about"
	};
}

/**
 * Run one wave-front level: group co-failers by observed-set, mint one correction
 * per group. The minted corrections are the active units of the next level.
 */
function runOneLevel(units, level) {
	// For every unit, record its observed-set and a grouping key.
	const errors = units.map(unit => {
		const observed = computeObservedSet(unit, units);
		// An isolated unit (no neighbours) has nothing to bind: give it a UNIQUE key
		// so empties never group together — otherwise disconnected units would merge.
		const key = observed.length === 0
			? `ISOLATED:${unit.id}`
			: observed.map(o => o.id).sort().join('|');
		return { unit, observed, key };
	});

	// Bucket errors by identical observed-set — these buckets are the co-failer groups.
	const groups = new Map();
	for (const error of errors) {
		if (!groups.has(error.key)) groups.set(error.key, []);
		groups.get(error.key).push(error);
	}

	// Mint exactly one correction per group.
	let n = 0;
	const corrections = [];
	for (const [, groupErrors] of groups) {
		const coFailers = groupErrors.map(e => e.unit);
		// Every member of a group shares the same observed-set by construction.
		const observed = groupErrors[0].observed;
		corrections.push(mintCorrection(`L${level + 1}#${n++}`, coFailers, observed));
	}
	return corrections;
}

/**
 * Merge corrections that ended up with the identical footprint — a stand-in for
 * "same pattern = same neuron". Real dedup is by inference-output match (the
 * reuse index); footprint identity is a faithful-enough proxy for this POC.
 */
function dedupByFootprint(corrections) {
	const bySignature = new Map();
	for (const correction of corrections) {
		const signature = [...correction.footprint].sort().join('|');
		if (!bySignature.has(signature)) {
			bySignature.set(signature, correction);
		} else {
			// Fold the duplicate's parents/observed into the one we keep.
			const kept = bySignature.get(signature);
			kept.parentIds = [...new Set([...kept.parentIds, ...correction.parentIds])];
			kept.observedIds = [...new Set([...kept.observedIds, ...correction.observedIds])];
		}
	}
	return [...bySignature.values()];
}

// ---------------------------------------------------------------------------
//  3. Driver — stack levels until the footprints stop changing
// ---------------------------------------------------------------------------

/**
 * A stable signature of a level's footprints, used to detect convergence
 * (the set of footprints is unchanged from the previous level).
 */
function footprintSignature(corrections) {
	return corrections.map(c => [...c.footprint].sort().join('|')).sort().join(' ;; ');
}

/**
 * Build the spatial hierarchy for one frame: run levels until a single pattern
 * spans everything, or the footprints stabilise, or maxLevels is hit.
 * Returns the per-level record for reporting.
 */
function buildHierarchy(active, maxLevels) {
	let units = [...active].map(pixel => ({ id: pixel, footprint: new Set([pixel]) }));
	const levels = [];
	let previousSignature = null;

	for (let level = 0; level < maxLevels; level++) {
		const corrections = dedupByFootprint(runOneLevel(units, level));
		const signature = footprintSignature(corrections);
		// Converged when one pattern covers everything or nothing changed this level.
		const converged = corrections.length <= 1 || signature === previousSignature;

		levels.push({ level: level + 1, corrections, converged });
		if (converged) break;

		previousSignature = signature;
		units = corrections; // the corrections become next level's active units
	}
	return levels;
}

// ---------------------------------------------------------------------------
//  4. Reporting
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
 * reuse events — watch how few there are on irregular shapes.
 */
function reportLevel(record, shapeSize, renderCap) {
	const { level, corrections, converged } = record;
	const sizes = corrections.map(c => c.footprint.size);
	const maxSize = Math.max(...sizes);
	const reuseGroups = corrections.filter(c => c.parentIds.length >= 2).length;
	const singletons = corrections.length - reuseGroups;

	console.log(`\n--- Level ${level}: ${corrections.length} pattern(s) | max footprint ${maxSize}px/${shapeSize}px | reuse-groups ${reuseGroups} | singletons ${singletons} ---`);

	// Only render footprints on the first level (the locality question) and the
	// converged level (the final result); middle levels just show stats.
	if (level === 1 || converged) {
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
	const maxLevels = opts.maxLevels || 10;
	const renderCap = opts.renderCap || 8;

	console.log(`\n================ ${name}  (connectivity=${connectivity}) ================`);
	console.log(gridText);

	const active = parseGrid(gridText);
	console.log(`Level 0: ${active.size} base pixels active`);

	const levels = buildHierarchy(active, maxLevels);
	for (const record of levels) reportLevel(record, active.size, renderCap);

	const last = levels[levels.length - 1];
	const level1Max = Math.max(...levels[0].corrections.map(c => c.footprint.size));
	const verdict = last.converged
		? (last.corrections.length <= 1 ? 'single pattern spans the structure' : 'footprints stable')
		: `hit maxLevels=${maxLevels}`;
	console.log(`(converged at level ${last.level}: ${verdict})`);
	console.log(`SUMMARY: level-1 max footprint = ${level1Max}px of ${active.size}px  ->  ${level1Max < active.size ? 'LEVEL 1 STAYED LOCAL' : 'level 1 = whole shape'}`);
}

// ---------------------------------------------------------------------------
//  5. Test shapes
// ---------------------------------------------------------------------------

// Each shape is drawn as ASCII; 'X' is an active pixel. The interesting axes are
// symmetry (drives within-frame reuse), connectedness (drives whether parts bind),
// and irregularity (stresses the exact observed-set grouping).
const shapes = {
	// Symmetric: reuse fires strongly (arms share an observed centre).
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
	// Irregular: almost every pixel sees a unique neighbour set → little reuse.
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
	// Solid: connected but asymmetric locally → zero within-frame reuse.
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

// Run every shape under default 4-connectivity.
for (const [name, lines] of Object.entries(shapes)) {
	runSimulation(name, lines.join('\n'));
}

// Re-run the diagonal under 8-connectivity to contrast with the 4-conn result:
// 4-conn shatters it into points, 8-conn binds it into one pattern.
runSimulation('diagonal (8-conn)', shapes['diagonal (4-conn)'].join('\n'), { connectivity: 8 });
