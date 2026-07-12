import { executeJob, parseBrainArgs } from 'robot-brain';
import MNISTTestJob from './test.js';

const NB_EPS = 1e-3; // mirrors the brain's baked NB Laplace floor (test.js scoreDigitsNB)

/**
 * Diagnostic: restrict the held-out test pass to a chosen set of digits (--only-digits 8,9,7)
 * and, for every miss, attribute the winner-vs-true log-score gap back to the individual voters
 * that caused it. Each voter resolves to an anchor pixel via brain.inspectNeuron().channelId —
 * sensory neurons report their own registered channel, spatial-correction neurons the founding
 * pixel's channel they inherited at mint time (thalamus.rs allocate_spatial_pattern_neuron).
 * Contributions accumulate into a per-pixel heatmap: which regions of the image systematically
 * vote wrong. Digits run sequentially against the SAME loaded brain — the 1.7M-neuron backup load
 * dominates wall time, so paying it once for a whole digit list beats one process per digit.
 */
class DebugVotesJob extends MNISTTestJob {

	async configureChannels() {
		await super.configureChannels();
		this.fullTestImages = this.testImages;
		this.fullTestLabels = this.testLabels;
		this.fullTestBits = this.testBits;

		const idx = process.argv.indexOf('--only-digits');
		const arg = idx !== -1 ? process.argv[idx + 1] : null;
		this.digitsToRun = arg ? arg.split(',').map(Number) : null;

		this.channelToPixel = new Map();
		this.encoder.pixelChannelIds.forEach((channelId, p) => this.channelToPixel.set(channelId, p));
		this.neuronPixelCache = new Map(); // voterId -> {x,y,spatialLevel} | null — persists across digits, neurons are shared
	}

	/**
	 * Narrow the active test set to one digit's images and reset the per-digit accumulators.
	 */
	selectDigit(target) {
		const keep = [];
		for (let i = 0; i < this.fullTestLabels.length; i++) if (this.fullTestLabels[i] === target) keep.push(i);
		this.testImages = keep.map(i => this.fullTestImages[i]);
		this.testLabels = keep.map(i => this.fullTestLabels[i]);
		this.testBits = keep.map(i => this.fullTestBits[i]);
		console.log(`\n${'='.repeat(70)}\n  DIGIT ${target}: ${this.testLabels.length} test images\n${'='.repeat(70)}`);

		const sz = this.config.imageSize;
		this.heat = Array.from({ length: sz }, () => new Array(sz).fill(0));
		this.levelHeat = new Map(); // spatialLevel -> heat grid, so L1 vs L2 vs L3 contributions can be told apart
		this.topContribs = [];
		this.missImages = 0;
		this.episodeResults = [];
	}

	/**
	 * Runs each requested digit's evaluation pass in turn against the one loaded brain instance.
	 */
	async executeJob() {
		if (!this.digitsToRun) { await super.executeJob(); return; }
		for (const digit of this.digitsToRun) {
			this.selectDigit(digit);
			await super.executeJob();
			await super.showResults();
			this.reportVoteHeatmap();
			if (this.isShuttingDown) return;
		}
	}

	/**
	 * The base class's own end-of-run report; already emitted per-digit inside executeJob() above
	 * when running a digit list, so skip the redundant final call.
	 */
	async showResults() {
		if (this.digitsToRun) return;
		await super.showResults();
		this.reportVoteHeatmap();
	}

	/**
	 * Resolve a voter's anchor pixel + spatial level via inspectNeuron, caching by voterId since
	 * the same pattern neurons fire across many images (and across digits, within one process).
	 */
	resolveVoter(voterId) {
		if (this.neuronPixelCache.has(voterId)) return this.neuronPixelCache.get(voterId);
		const info = this.brain.inspectNeuron(voterId);
		const p = info.channelId != null ? this.channelToPixel.get(info.channelId) : undefined;
		const resolved = p == null ? null : { x: p % this.config.imageSize, y: Math.floor(p / this.config.imageSize), spatialLevel: info.spatialLevel };
		this.neuronPixelCache.set(voterId, resolved);
		return resolved;
	}

	/**
	 * Runs after the base class's own --debug-miss text report for this miss. Groups the raw
	 * votes by voter, resolves each voter's anchor pixel, and adds its (winner − true) log-score
	 * contribution into the spatial heatmap (overall and per spatial level).
	 */
	analyzeMiss(miss, label, predicted, bits) {
		super.analyzeMiss(miss, label, predicted, bits);
		if (!this.heat || predicted === label) return;
		this.missImages++;

		const byVoter = new Map();
		for (const v of this._lastVotes || []) {
			if (v.targetType !== 'action' || v.channelId !== this.encoder.digitChannelId) continue;
			if (v.value !== predicted && v.value !== label) continue;
			if (!byVoter.has(v.voterId)) byVoter.set(v.voterId, {});
			const rec = byVoter.get(v.voterId);
			// v.strength is the connection's reinforcement count — how many training frames actually
			// backed this specific reward estimate (see neuron.rs strengthen_connection: strength += 1.0
			// per observation, reward is the running mean over exactly that many observations).
			rec[v.value === predicted ? 'winnerReward' : 'trueReward'] = v.reward;
			rec[v.value === predicted ? 'winnerStrength' : 'trueStrength'] = v.strength;
		}

		for (const [voterId, rec] of byVoter) {
			const loc = this.resolveVoter(voterId);
			if (!loc) continue; // coordinate-less voter (shouldn't happen for MNIST's spatial-only hierarchy)
			const wr = rec.winnerReward ?? 0;
			const tr = rec.trueReward ?? 0;
			const contribution = Math.log(wr + NB_EPS) - Math.log(tr + NB_EPS);
			this.heat[loc.y][loc.x] += contribution;
			if (!this.levelHeat.has(loc.spatialLevel)) {
				const sz = this.config.imageSize;
				this.levelHeat.set(loc.spatialLevel, Array.from({ length: sz }, () => new Array(sz).fill(0)));
			}
			this.levelHeat.get(loc.spatialLevel)[loc.y][loc.x] += contribution;
			// Report whichever side's strength actually drove the contribution: the winner's count when
			// the vote favored the (wrong) winner, the true digit's count when it favored the true digit.
			const strength = contribution >= 0 ? (rec.winnerStrength ?? 0) : (rec.trueStrength ?? 0);
			this.topContribs.push({ voterId, x: loc.x, y: loc.y, spatialLevel: loc.spatialLevel, contribution, strength, predicted, label });
		}
	}

	/**
	 * Tests the fragmentation hypothesis directly: are the near-ceiling (|contrib|~6.9, meaning the
	 * connection's reward is a hard 0/1 split) voters backed by thin observation counts (few training
	 * frames ever activated that pattern neuron), or do they have healthy counts and are genuinely,
	 * durably single-class? `strength` is the connection's exact reinforcement count (neuron.rs
	 * strengthen_connection: += 1.0 per observation, reward is the running mean over that many).
	 */
	reportStrengthStats() {
		const strengths = this.topContribs.map(t => t.strength).filter(s => s > 0);
		if (!strengths.length) return;
		const ceiling = this.topContribs.filter(t => Math.abs(t.contribution) > 6.5).map(t => t.strength);
		const rest = this.topContribs.filter(t => Math.abs(t.contribution) <= 6.5).map(t => t.strength);
		const stats = arr => {
			if (!arr.length) return 'n/a';
			const sorted = [...arr].sort((a, b) => a - b);
			const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
			return `n=${arr.length} mean=${mean.toFixed(1)} median=${sorted[Math.floor(sorted.length / 2)]} min=${sorted[0]} max=${sorted[sorted.length - 1]}`;
		};
		console.log(`  Observation-count (strength) behind each contributing connection:`);
		console.log(`    near-ceiling voters (|contrib|>6.5, hard 0/1 reward split): ${stats(ceiling)}`);
		console.log(`    other contributing voters:                                 ${stats(rest)}\n`);
	}

	reportVoteHeatmap() {
		if (!this.heat) return;
		this.renderHeat(this.heat, `VOTE HEATMAP — all spatial levels — sum of (log P(pred|voter) − log P(true|voter)) per anchor pixel, over ${this.missImages} misses`);
		for (const [level, grid] of [...this.levelHeat.entries()].sort((a, b) => a[0] - b[0]))
			this.renderHeat(grid, `  ↳ spatial level ${level} only`);

		this.reportStrengthStats();

		const top = [...this.topContribs].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, 20);
		console.log('  Top individual voter contributions:');
		for (const t of top) {
			console.log(`    voter=${t.voterId} L${t.spatialLevel} anchor=(${t.x},${t.y}) contrib=${t.contribution.toFixed(2)} strength=${t.strength} (pred=${t.predicted} true=${t.label})`);
		}
	}

	/**
	 * ASCII-render one heat grid: magnitude via a brightness ramp, sign via case.
	 */
	renderHeat(grid, title) {
		const sz = this.config.imageSize;
		console.log(`\n${title}`);
		console.log('  positive = region favors the WRONG digit; negative = favors the TRUE digit (UPPER=wrong, lower=true)\n');
		const ramp = ' .:-=+*#%@';
		let maxAbs = 0;
		for (let y = 0; y < sz; y++) for (let x = 0; x < sz; x++) maxAbs = Math.max(maxAbs, Math.abs(grid[y][x]));
		for (let y = 0; y < sz; y++) {
			let row = '  ';
			for (let x = 0; x < sz; x++) {
				const v = grid[y][x];
				const mag = maxAbs > 0 ? Math.abs(v) / maxAbs : 0;
				const ch = ramp[Math.min(ramp.length - 1, Math.floor(mag * (ramp.length - 1)))];
				row += v > 0 ? ch : (v < 0 ? ch.toLowerCase() : ' ');
			}
			console.log(row);
		}
	}
}

// getJobDir() reads backups from <test.js dir>/test/ — reuse test.js's own backup namespace so --load-brain finds mnist28.
DebugVotesJob.moduleUrl = new URL('./test.js', import.meta.url).href;
await executeJob(DebugVotesJob, parseBrainArgs());
