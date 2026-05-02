import fs from 'node:fs';
import path from 'node:path';
import { Neuron } from './neuron.js';
import { Thalamus } from './thalamus.js';

// Hard cap on retained backup folders. The 11th save evicts the oldest by
// folder-name sort (timestamps sort lexicographically thanks to the format).
const MAX_BACKUPS = 10;

/**
 * Backup / restore for brain state.
 *
 * Snapshots are written as a folder of CSVs (no header rows) that mirror the
 * MySQL schema, so the same files can be bulk-loaded with `LOAD DATA INFILE`
 * by the apps/db/import job.
 *
 * Layout: <jobDir>/backups/<YYYY-MM-DD_HH-mm-ss>/
 *   channels.csv      id,name
 *   dimensions.csv    id,name
 *   neurons.csv       id,level
 *   base_neurons.csv  neuron_id,channel_id,type,dimension_id,val
 *   connections.csv   from_neuron_id,to_neuron_id,distance,strength,reward
 *   patterns.csv      pattern_neuron_id,parent_neuron_id,strength
 *   contexts.csv     pattern_neuron_id,context_neuron_id,context_age,strength
 *   neuron_error_stats.csv  neuron_id,age,n,mean,m2
 */
export class Backup {

	/**
	 * Stash the Neuron-construction hyperparameters so loadLatest() can hand them
	 * to each freshly-rehydrated Neuron. The snapshot itself doesn't carry these
	 * (they're brain-wide, not per-neuron), so they have to be passed in here.
	 */
	constructor(patternForgetRate, mergeThreshold, contextLength, errorMode, errorThreshold) {
		this.patternForgetRate = patternForgetRate;
		this.mergeThreshold = mergeThreshold;
		this.contextLength = contextLength;
		this.errorMode = errorMode;
		this.errorThreshold = errorThreshold;
	}

	/**
	 * Save a snapshot under <jobDir>/backups/<timestamp>/. Returns the folder path.
	 * Errors are caught and logged so a save failure during shutdown can never
	 * mask the original exit.
	 */
	save(jobDir, snapshot) {
		try {
			// Ensure the per-job backups root exists. recursive:true is a no-op if it does.
			const backupsDir = path.join(jobDir, 'backups');
			fs.mkdirSync(backupsDir, { recursive: true });

			// Each save gets its own timestamped folder — sortable lexicographically so
			// "latest" and "oldest" both fall out of a plain string sort.
			const timestamp = formatTimestamp(new Date());
			const folder = path.join(backupsDir, timestamp);
			fs.mkdirSync(folder, { recursive: true });

			// Write each table in dependency order: channels/dimensions are referenced by
			// base_neurons; neurons are referenced by connections/patterns/contexts.
			// Order doesn't matter for files (only for SQL FK loading), but we keep it
			// stable so diffs across backups are predictable.
			this.writeChannels(folder, snapshot);
			this.writeDimensions(folder, snapshot);
			this.writeNeurons(folder, snapshot);
			this.writeBaseNeurons(folder, snapshot);
			this.writeConnections(folder, snapshot);
			this.writePatterns(folder, snapshot);
			this.writeContexts(folder, snapshot);
			this.writeNeuronErrorStats(folder, snapshot);

			console.log(`💾 Backup saved: ${folder} (${snapshot.neurons.length} neurons)`);

			// Evict any folders past MAX_BACKUPS *after* the new one is on disk —
			// otherwise a crash mid-prune could leave fewer backups than intended.
			this.pruneOldBackups(backupsDir);
			return folder;
		}
		catch (err) {
			// Save is called from shutdown handlers (incl. the SIGINT/uncaught-exception
			// path), so it must never throw — the original failure has to surface.
			console.error('Backup save failed:', err);
			return null;
		}
	}

	/**
	 * Load the latest backup (newest timestamp folder) under <jobDir>/backups/.
	 * Throws if no backups exist — callers using --load expect a backup to be there.
	 *
	 * Returns a snapshot in the same shape as Thalamus.getSnapshot(), ready for
	 * Thalamus.restoreSnapshot().
	 */
	loadLatest(jobDir, channelActionIds, actionIds) {
		const backupsDir = path.join(jobDir, 'backups');
		const folder = this.findLatestBackup(backupsDir);
		// --load is opt-in; if the user asked for it and there's nothing, that's a hard
		// error (silently starting from a fresh brain would mask the misconfiguration).
		if (!folder) throw new Error(`--load requested but no backups found in ${backupsDir}`);

		console.log(`📂 Loading backup: ${folder}`);

		// Channels: rebuild both directions of the lookup. The id→name side is needed
		// below to translate base_neurons rows; the name→id side is part of the snapshot.
		const channelIdToName = {};
		const channelNameToId = {};
		for (const [idStr, name] of readCsv(path.join(folder, 'channels.csv'))) {
			const id = Number(idStr);
			channelIdToName[id] = name;
			channelNameToId[name] = id;
		}

		// Dimensions: only name→id is exposed back to Thalamus (the id→name direction
		// isn't needed during restore since base_neurons already carries dimId numerically).
		const dimensionNameToId = {};
		for (const [idStr, name] of readCsv(path.join(folder, 'dimensions.csv'))) {
			dimensionNameToId[name] = Number(idStr);
		}

		// Neurons: create the empty Neuron objects first so that connections/patterns
		// loaded later can reference them by id without ordering constraints.
		const neurons = new Map();
		const levels = new Map();
		for (const [idStr, levelStr] of readCsv(path.join(folder, 'neurons.csv'))) {
			const id = Number(idStr);
			const level = Number(levelStr);
			const forgetRate = Thalamus.effectiveForgetRate(this.patternForgetRate, this.contextLength, level);
			const neuron = new Neuron(id, forgetRate, this.mergeThreshold, this.errorMode, this.errorThreshold, channelActionIds, actionIds);
			neurons.set(id, neuron);
			levels.set(id, level);
		}

		// Base neurons: level-0 sensory metadata. File may be absent if the snapshot
		// only has interneurons (rare, but cheap to guard for).
		const baseNeurons = new Map();
		const baseFile = path.join(folder, 'base_neurons.csv');
		if (fs.existsSync(baseFile)) {
			for (const [neuronId, channelId, type, dimId, val] of readCsv(baseFile)) {
				baseNeurons.set(Number(neuronId), {
					channel: channelIdToName[Number(channelId)],
					type,
					coordinate: { dimId: Number(dimId), bucketId: Number(val) }
				});
			}
		}

		// Connections: each row reattaches a directed (from→to, distance) link with its
		// strength and reward. Both endpoints must already be in the neuron map.
		const connFile = path.join(folder, 'connections.csv');
		if (fs.existsSync(connFile)) {
			for (const [fromId, toId, distance, strength, reward] of readCsv(connFile)) {
				const fromNeuron = neurons.get(Number(fromId));
				if (!fromNeuron) throw new Error(`Connection source neuron not found: ${fromId}`);
				if (!neurons.has(Number(toId))) throw new Error(`Connection target neuron not found: ${toId}`);
				fromNeuron.createConnection(Number(distance), Number(toId), Number(strength), Number(reward));
			}
		}

		// Patterns: register each pattern as a child of its parent. We also remember
		// pattern→parent here so the contexts loader below can look up the parent
		// without re-reading the patterns file.
		const neuronParents = new Map();
		const patternsFile = path.join(folder, 'patterns.csv');
		if (fs.existsSync(patternsFile)) {
			for (const [patternId, parentId, strength] of readCsv(patternsFile)) {
				const pid = Number(patternId);
				const ppid = Number(parentId);
				neuronParents.set(pid, ppid);
				const parent = neurons.get(ppid);
				if (!parent) throw new Error(`Pattern parent not found: ${parentId}`);
				parent.addChild(pid, Number(strength));
			}
		}

		// Contexts (context entries): for each pattern, restore the context neurons
		// and ages that define when it should activate. addContext lives on the parent;
		// addContextRef lives on the context neuron — both halves are needed for the
		// bidirectional lookup the runtime relies on.
		const contextsFile = path.join(folder, 'contexts.csv');
		if (fs.existsSync(contextsFile)) {
			for (const [patternId, contextId, contextAge, strength] of readCsv(contextsFile)) {
				const pid = Number(patternId);
				const cid = Number(contextId);
				const parentId = neuronParents.get(pid);
				const parent = neurons.get(parentId);
				const contextNeuron = neurons.get(cid);
				if (!parent) throw new Error(`contexts parent not found for pattern ${patternId}`);
				if (!contextNeuron) throw new Error(`contexts context neuron not found: ${contextId}`);
				parent.addContext(pid, cid, Number(contextAge), Number(strength));
				contextNeuron.addContextRef(parentId, Number(contextAge));
			}
		}

		// Per-(neuron, age) Welford error stats. Optional — older snapshots predating
		// dynamic error thresholds won't have this file, in which case neurons start
		// with empty errorStats and warmup-fallback to errorThreshold for their first
		// 3 samples per age. New snapshots restore the stats verbatim so the dynamic
		// threshold picks up exactly where it left off.
		const errorStatsFile = path.join(folder, 'neuron_error_stats.csv');
		if (fs.existsSync(errorStatsFile)) {
			for (const [neuronId, age, n, mean, m2] of readCsv(errorStatsFile)) {
				const neuron = neurons.get(Number(neuronId));
				if (!neuron) throw new Error(`neuron_error_stats neuron not found: ${neuronId}`);
				neuron.loadErrorStats(Number(age), Number(n), Number(mean), Number(m2));
			}
		}

		// Flatten neurons back into the snapshot shape Thalamus expects: an array of
		// per-neuron entries, each with optional baseNeuron (level 0) and parentId
		// (patterns). The ordering is the iteration order of the Map, which matches
		// the file order on most engines but isn't strictly guaranteed — Thalamus
		// shouldn't depend on it.
		const neuronEntries = [];
		for (const [neuronId, neuron] of neurons) {
			const level = levels.get(neuronId);
			const entry = { neuron, level };
			if (level === 0) entry.baseNeuron = baseNeurons.get(neuronId);
			const parentId = neuronParents.get(neuronId);
			if (parentId) entry.parentId = parentId;
			neuronEntries.push(entry);
		}

		console.log(`   Loaded ${neuronEntries.length} neurons`);
		return { neurons: neuronEntries, channelNameToId, dimensionNameToId };
	}

	/* ---------- writers ---------- */

	/**
	 * Write the channel id↔name table. Sorted by id so the file order is stable
	 * across runs — keeps diffs between consecutive backups clean and lets bulk
	 * loaders see deterministic input.
	 */
	writeChannels(folder, snapshot) {
		// Object.entries gives [name, id] pairs; flip them to [id, name] for the file.
		const rows = Object.entries(snapshot.channelNameToId)
			.sort((a, b) => a[1] - b[1])
			.map(([name, id]) => [id, name]);
		writeCsv(path.join(folder, 'channels.csv'), rows);
	}

	/**
	 * Write the dimension id↔name table. Same id-sort discipline as writeChannels;
	 * dimensions and channels are the two parent tables every other CSV references
	 * by id, so deterministic ordering matters most here.
	 */
	writeDimensions(folder, snapshot) {
		const rows = Object.entries(snapshot.dimensionNameToId)
			.sort((a, b) => a[1] - b[1])
			.map(([name, id]) => [id, name]);
		writeCsv(path.join(folder, 'dimensions.csv'), rows);
	}

	/**
	 * Write the neuron id+level table. Just the bare (id, level) tuple — every
	 * other per-neuron field lives in base_neurons / connections / patterns /
	 * contexts, joined by neuron id on import.
	 */
	writeNeurons(folder, snapshot) {
		const rows = snapshot.neurons.map(({ neuron, level }) => [neuron.id, level]);
		// Sort by id for a stable file order, matching the parent-table convention.
		rows.sort((a, b) => a[0] - b[0]);
		writeCsv(path.join(folder, 'neurons.csv'), rows);
	}

	/**
	 * Write the level-0 sensory metadata table (channel, type, dim, bucket). Skips
	 * interneurons since they have no coordinate. Channel names are translated to
	 * channel ids so the file is fully numeric — friendly to MySQL FKs on import.
	 */
	writeBaseNeurons(folder, snapshot) {
		const rows = [];
		for (const { neuron, level, baseNeuron } of snapshot.neurons) {
			// Level > 0 neurons are patterns/interneurons — they live only in patterns.csv.
			if (level !== 0) continue;
			const { channel, type, coordinate } = baseNeuron;
			rows.push([neuron.id, snapshot.channelNameToId[channel], type, coordinate.dimId, coordinate.bucketId]);
		}
		rows.sort((a, b) => a[0] - b[0]);
		writeCsv(path.join(folder, 'base_neurons.csv'), rows);
	}

	/**
	 * Write the directed-connection table (from, to, distance, strength, reward).
	 * Walks the per-neuron nested map distance → (toNeuronId → conn) and emits one
	 * row per leaf. No sort — connections is typically the largest table and order
	 * doesn't change semantics, so the cycles are better spent elsewhere.
	 */
	writeConnections(folder, snapshot) {
		const rows = [];
		for (const { neuron } of snapshot.neurons)
			for (const [distance, targets] of neuron.connections)
				for (const [toNeuronId, conn] of targets)
					// reward defaults to 0 when the connection has never carried one — keeps
					// the file column count fixed.
					rows.push([neuron.id, toNeuronId, distance, conn.strength, conn.reward || 0]);
		writeCsv(path.join(folder, 'connections.csv'), rows);
	}

	/**
	 * Write the pattern→parent table with each pattern's current activation
	 * strength. The strength lives on the parent's routing table (not the child),
	 * so we look it up parent-side rather than reading it off the pattern neuron.
	 */
	writePatterns(folder, snapshot) {
		// Build an id→neuron lookup once so the inner loop is O(1) per pattern.
		const neuronMap = new Map();
		for (const entry of snapshot.neurons) neuronMap.set(entry.neuron.id, entry.neuron);

		const rows = [];
		for (const { neuron, level, parentId } of snapshot.neurons) {
			// Level-0 neurons aren't patterns — they have no parent and don't belong here.
			if (level === 0) continue;
			let strength = 0;
			const parent = neuronMap.get(parentId);
			if (parent) {
				// Pull the child's current activation strength from the parent's routing
				// table. If the entry has been evicted, default to 0 — restore will
				// recompute on first activation.
				const ctx = parent.routingTable.get(neuron.id);
				if (ctx) strength = ctx.activationStrength;
			}
			rows.push([neuron.id, parentId, strength]);
		}
		writeCsv(path.join(folder, 'patterns.csv'), rows);
	}

	/**
	 * Write the pattern context-entry table (pattern, context neuron, age,
	 * strength). Each neuron exposes its routing table as a flat iterable of
	 * context entries — emit one row per tuple. This is the table that defines
	 * *when* each pattern fires.
	 */
	writeContexts(folder, snapshot) {
		const rows = [];
		for (const { neuron } of snapshot.neurons)
			for (const { patternId, neuronId, distance, strength } of neuron.getRoutingTable())
				rows.push([patternId, neuronId, distance, strength]);
		writeCsv(path.join(folder, 'contexts.csv'), rows);
	}

	/**
	 * Write the per-(neuron, age) Welford error-stats table. Sparse: only neurons
	 * that have observed at least one error sample at a given age contribute a row.
	 * Sorted by (neuron_id, age) for stable file output, matching the parent-table
	 * convention. Used only by the dynamic error-correction modes — for `static`
	 * mode the file is written empty (the stats are still maintained but never
	 * consulted, so persisting them is harmless).
	 */
	writeNeuronErrorStats(folder, snapshot) {
		const rows = [];
		for (const { neuron } of snapshot.neurons)
			for (const [age, { n, mean, M2 }] of neuron.errorStats)
				rows.push([neuron.id, age, n, mean, M2]);
		rows.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
		writeCsv(path.join(folder, 'neuron_error_stats.csv'), rows);
	}

	/* ---------- folder management ---------- */

	/**
	 * Find the most recent backup folder under `backupsDir` (or null if there are
	 * none). Folder names follow the timestamp regex below; anything else (README,
	 * .DS_Store, hand-edited dumps) is ignored so it can't be picked as "latest".
	 */
	findLatestBackup(backupsDir) {
		// First-run guard: no backups dir means nothing to load.
		if (!fs.existsSync(backupsDir)) return null;
		const folders = fs.readdirSync(backupsDir, { withFileTypes: true })
			.filter(e => e.isDirectory())
			.map(e => e.name)
			.filter(name => /^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$/.test(name))
			.sort();
		if (folders.length === 0) return null;
		// Lex-sort puts newest at the end thanks to zero-padded timestamps — no
		// need to parse them as dates.
		return path.join(backupsDir, folders[folders.length - 1]);
	}

	/**
	 * Delete backup folders past the MAX_BACKUPS retention cap. Re-scans the
	 * directory each call rather than tracking state in memory — also cleans up
	 * stragglers left behind by previous processes that crashed mid-prune.
	 */
	pruneOldBackups(backupsDir) {
		const folders = fs.readdirSync(backupsDir, { withFileTypes: true })
			.filter(e => e.isDirectory())
			.map(e => e.name)
			.filter(name => /^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$/.test(name))
			.sort();
		// Keep the last MAX_BACKUPS entries; everything before that slice is stale.
		const stale = folders.slice(0, Math.max(0, folders.length - MAX_BACKUPS));
		for (const name of stale) {
			// recursive+force so we don't choke on partially-written folders.
			fs.rmSync(path.join(backupsDir, name), { recursive: true, force: true });
			console.log(`   Pruned old backup: ${name}`);
		}
	}
}

/**
 * Build a `YYYY-MM-DD_HH-mm-ss` timestamp matching the folder regex above. Uses
 * local time (not UTC) so the folder names line up with what the user sees in
 * their shell history.
 */
function formatTimestamp(d) {
	const pad = n => String(n).padStart(2, '0');
	return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}_${pad(d.getHours())}-${pad(d.getMinutes())}-${pad(d.getSeconds())}`;
}

/**
 * Write `rows` (array of arrays) to `filepath` as CSV. Each cell is escaped via
 * escapeField, fields joined by commas, lines by '\n'. Empty input produces a
 * zero-byte file rather than a lone newline so MySQL's LOAD DATA and `wc -l`
 * both report the right thing.
 */
function writeCsv(filepath, rows) {
	const lines = rows.map(row => row.map(escapeField).join(','));
	fs.writeFileSync(filepath, lines.length ? lines.join('\n') + '\n' : '');
}

/**
 * Escape a single CSV cell. Wraps in double-quotes (and doubles up any embedded
 * quotes) when the value contains a delimiter, quote, or newline; otherwise the
 * value passes through raw. Mirrors the rules parseCsvLine expects.
 */
function escapeField(value) {
	const s = String(value);
	if (s.includes(',') || s.includes('"') || s.includes('\n')) return `"${s.replace(/"/g, '""')}"`;
	return s;
}

/**
 * Read a CSV file written by writeCsv, returning an array of row arrays. Splits
 * on '\n' (matching the writer) so behavior is symmetric on Windows where the
 * platform default would be CRLF.
 */
function readCsv(filepath) {
	const content = fs.readFileSync(filepath, 'utf-8');
	const rows = [];
	for (const line of content.split('\n')) {
		// Skip blank lines: trailing newline at EOF, or stray empty rows from edits.
		if (!line) continue;
		rows.push(parseCsvLine(line));
	}
	return rows;
}

/**
 * Parse one CSV line into an array of field strings. Hand-rolled because we
 * only need to handle the same quoting rules escapeField produces — no multi-
 * line quoted fields, no RFC 4180 edge cases.
 */
function parseCsvLine(line) {
	const fields = [];
	let i = 0;
	while (i < line.length) {
		if (line[i] === '"') {
			// Quoted field: scan until the closing quote, treating "" as a literal ".
			let value = '';
			i++;
			while (i < line.length) {
				if (line[i] === '"' && line[i + 1] === '"') { value += '"'; i += 2; }
				else if (line[i] === '"') { i++; break; }
				else { value += line[i]; i++; }
			}
			fields.push(value);
			// Consume the trailing comma after the closing quote, if any.
			if (line[i] === ',') i++;
		} else {
			// Unquoted field: scan up to the next comma or EOL.
			let value = '';
			while (i < line.length && line[i] !== ',') { value += line[i]; i++; }
			fields.push(value);
			if (line[i] === ',') i++;
		}
	}
	return fields;
}
