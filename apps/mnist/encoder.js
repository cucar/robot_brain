/**
 * MNIST Encoder — single-channel binary pixel stream.
 *
 * Each image is fed pixel-by-pixel as one event per frame through a single
 * channel "px" with two dimensions:
 *   - px_val:  input, resolution = buckets (default 2 → binary)
 *   - px_digit: action, resolution 10, one bucket per digit (0–9)
 *
 * Optional trim drops leading/trailing all-zero pixel runs so the temporal
 * stream starts at the first non-black pixel and ends at the last one. This
 * removes the long uniform preamble/postamble whose only role is to confuse
 * the per-age error-correction patterns at the first transition.
 *
 * The classification idea: feed the bit stream as events to build context,
 * then call brain.learn({digit: label}, {reward: +1}) so every votable
 * context neuron wires to the correct digit-action neuron. At test time,
 * replay the bit stream and call brain.infer() — the action winner is the
 * predicted digit.
 */

const DIGITS = 10;
const DIGIT_ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

export class MNISTEncoder {

	/**
	 * @param {number} buckets — quantization levels per pixel (2 = binary)
	 * @param {boolean} trim — strip leading/trailing all-black pixel runs
	 */
	constructor(buckets = 2, trim = true) {
		this.buckets = buckets;
		this.trim = trim;

		this.channelId = null;
		this.inputDimId = null;
		this.actionDimId = null;

		// Last digit the brain inferred (or that was forced). Reset per image.
		this.lastAction = -1;
	}

	/**
	 * Register the single pixel channel with the brain.
	 *
	 * @param {Brain} brain
	 */
	registerChannels(brain) {
		const { channelId, dimensionIds } = brain.registerChannelSpec({
			name: 'px',
			emitsReward: true,
			learnActionSequences: false,
			dimensions: [
				{
					name: 'px_val',
					kind: 'input',
					resolution: this.buckets,
					mode: 'passthrough'
				},
				{
					name: 'px_digit',
					kind: 'action',
					resolution: DIGITS,
					mode: 'passthrough',
					actions: DIGIT_ACTIONS,
					defaultAction: 0
				}
			]
		});
		this.channelId = channelId;
		this.inputDimId = dimensionIds['px_val'];
		this.actionDimId = dimensionIds['px_digit'];
	}

	/**
	 * Quantize a pixel value (0–255) into a bucket index.
	 *
	 * @param {number} value
	 * @returns {number}
	 */
	quantize(value) {
		const bucket = Math.floor(value * this.buckets / 256);
		return Math.min(bucket, this.buckets - 1);
	}

	/**
	 * Build the (optionally trimmed) bit/bucket stream for an image.
	 *
	 * @param {Uint8Array} pixels — 784 raw values (row-major)
	 * @returns {Uint8Array} encoded stream (length ≤ 784)
	 */
	buildBits(pixels) {
		const all = new Uint8Array(784);
		for (let i = 0; i < 784; i++) all[i] = this.quantize(pixels[i]);
		if (!this.trim) return all;
		let start = 0, end = 784;
		while (start < end && all[start] === 0) start++;
		while (end > start && all[end - 1] === 0) end--;
		return all.subarray(start, end);
	}

	/**
	 * Build the inputs Map for a single pixel frame.
	 *
	 * @param {number} bit — quantized pixel value
	 * @returns {Map<number, Map<number, number>>}
	 */
	encodePixel(bit) {
		const inputs = new Map();
		const dimMap = new Map();
		dimMap.set(this.inputDimId, bit);
		inputs.set(this.channelId, dimMap);
		return inputs;
	}

	/**
	 * Build the actions Map for the digit label (used by brain.learn()).
	 *
	 * @param {number} label — correct digit (0–9)
	 * @returns {Map<number, Map<number, number>>}
	 */
	encodeAction(label) {
		const actions = new Map();
		const dimMap = new Map();
		dimMap.set(this.actionDimId, label);
		actions.set(this.channelId, dimMap);
		return actions;
	}

	/**
	 * Build the rewards Map for the single channel.
	 *
	 * @param {number} label — correct digit (0–9)
	 * @param {number} correctReward
	 * @param {number} incorrectReward
	 * @returns {Map<number, number>} channelId → reward
	 */
	buildRewards(label, correctReward = 1, incorrectReward = -1) {
		const rewards = new Map();
		if (this.lastAction >= 0) {
			rewards.set(this.channelId, this.lastAction === label ? correctReward : incorrectReward);
		} else {
			rewards.set(this.channelId, correctReward);
		}
		return rewards;
	}

	/**
	 * Read the predicted digit from brain.infer() / brain.learn() inferences.
	 *
	 * @param {Map} inferences
	 * @returns {{ predicted: number, score: number, strength: number }}
	 */
	applyInferences(inferences) {
		const arr = inferences?.get(this.channelId);
		if (!arr) return { predicted: -1, score: 0, strength: 0 };
		for (const inf of arr) {
			if (inf.kind === 'action') {
				const predicted = inf.winner?.value ?? -1;
				this.lastAction = predicted;
				return {
					predicted,
					score: inf.winner?.score ?? 0,
					strength: inf.winner?.strength ?? 0
				};
			}
		}
		return { predicted: -1, score: 0, strength: 0 };
	}

	/**
	 * Record a forced digit (used by supervised training so buildRewards()
	 * delivers the correct reward on the next frame).
	 *
	 * @param {number} label
	 */
	setForcedAction(label) {
		this.lastAction = label;
	}

	/**
	 * Clear per-image action tracking — call between images.
	 */
	resetActions() {
		this.lastAction = -1;
	}
}

// ─────────────────────────────────────────────────────────────────────────────

/**
 * MNIST 28-Channel Encoder — one channel per row.
 *
 * Each image becomes 28 frames. At frame t, each row-channel emits its
 * pixel at column t. So the whole 28×28 image is presented in 28 frames
 * (column-by-column scan, with all 28 rows fed in parallel each frame).
 *
 * With context_length=30 the whole image fits in the memory window, so
 * L1 patterns naturally encode column transitions and L2+ patterns
 * encode whole-image column-sequences.
 *
 * Action neuron lives on a separate "digit" channel to keep the row
 * channels uniform. No trimming — every image is exactly 28 frames.
 */
export class MNISTRowChannelsEncoder {

	constructor(buckets = 2) {
		this.buckets = buckets;
		this.rowChannelIds = [];     // 28 entries, channel id per row
		this.rowDimIds = [];         // 28 entries, px_val dim id per row
		this.digitChannelId = null;
		this.digitDimId = null;
		this.lastAction = -1;
	}

	registerChannels(brain) {
		for (let r = 0; r < 28; r++) {
			const { channelId, dimensionIds } = brain.registerChannelSpec({
				name: `row${r}`,
				emitsReward: false,
				learnActionSequences: false,
				dimensions: [
					{ name: 'px_val', kind: 'input', resolution: this.buckets, mode: 'passthrough' }
				]
			});
			this.rowChannelIds.push(channelId);
			this.rowDimIds.push(dimensionIds['px_val']);
		}
		const { channelId, dimensionIds } = brain.registerChannelSpec({
			name: 'digit',
			emitsReward: true,
			learnActionSequences: false,
			dimensions: [
				{
					name: 'px_digit',
					kind: 'action',
					resolution: DIGITS,
					mode: 'passthrough',
					actions: DIGIT_ACTIONS,
					defaultAction: 0
				}
			]
		});
		this.digitChannelId = channelId;
		this.digitDimId = dimensionIds['px_digit'];
	}

	quantize(value) {
		const bucket = Math.floor(value * this.buckets / 256);
		return Math.min(bucket, this.buckets - 1);
	}

	/**
	 * Build a 28×28 quantized image. Returns Uint8Array length 784, row-major.
	 * No trimming — every image is exactly 28×28.
	 */
	buildBits(pixels) {
		const out = new Uint8Array(784);
		for (let i = 0; i < 784; i++) out[i] = this.quantize(pixels[i]);
		return out;
	}

	/**
	 * Build the events Map for column `col` (0..27) of an image's bit grid.
	 * Each of the 28 row-channels gets its pixel at this column.
	 */
	encodeColumn(bits, col) {
		const inputs = new Map();
		for (let r = 0; r < 28; r++) {
			const v = bits[r * 28 + col];
			const dimMap = new Map();
			dimMap.set(this.rowDimIds[r], v);
			inputs.set(this.rowChannelIds[r], dimMap);
		}
		return inputs;
	}

	encodeAction(label) {
		const actions = new Map();
		const dimMap = new Map();
		dimMap.set(this.digitDimId, label);
		actions.set(this.digitChannelId, dimMap);
		return actions;
	}

	buildRewards(label, correctReward = 1, incorrectReward = -1) {
		const rewards = new Map();
		if (this.lastAction >= 0) {
			rewards.set(this.digitChannelId, this.lastAction === label ? correctReward : incorrectReward);
		} else {
			rewards.set(this.digitChannelId, correctReward);
		}
		return rewards;
	}

	applyInferences(inferences) {
		const arr = inferences?.get(this.digitChannelId);
		if (!arr) return { predicted: -1, score: 0, strength: 0 };
		for (const inf of arr) {
			if (inf.kind === 'action') {
				const predicted = inf.winner?.value ?? -1;
				this.lastAction = predicted;
				return { predicted, score: inf.winner?.score ?? 0, strength: inf.winner?.strength ?? 0 };
			}
		}
		return { predicted: -1, score: 0, strength: 0 };
	}

	/**
	 * Sum-of-evidence consensus from raw actionVoteStats. Picks the digit
	 * with the highest totalStrength × avgReward (= weighted_total).
	 * Use this instead of brain's per-voter-mean consensus.
	 */
	predictBySumConsensus(actionVoteStats) {
		let bestDigit = -1, bestScore = -Infinity;
		for (const s of actionVoteStats || []) {
			const score = s.totalStrength * s.avgReward;
			if (score > bestScore) { bestScore = score; bestDigit = s.value; }
		}
		this.lastAction = bestDigit;
		return bestDigit;
	}

	setForcedAction(label) { this.lastAction = label; }
	resetActions() { this.lastAction = -1; }
}
