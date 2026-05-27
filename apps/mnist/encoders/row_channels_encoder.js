import { DIGITS, DIGIT_ACTIONS } from './digits.js';

/**
 * 28-channel column-scan encoder — one channel per row. An image becomes
 * 28 frames: at frame t every row-channel emits its pixel at column t, so
 * the whole 28×28 is presented column-by-column with rows fed in parallel.
 *
 * With context_length=30 the whole image fits in the memory window — L1
 * patterns naturally encode column transitions and L2+ encode whole-image
 * column sequences. The digit action lives on a separate "digit" channel
 * to keep the row channels uniform.
 */
export class MNISTRowChannelsEncoder {

	/**
	 * @param {number} buckets - pixel quantization levels (2 = binary)
	 */
	constructor(buckets = 2) {
		this.buckets = buckets;
		// 28 entries each — parallel arrays so encodeColumn() does O(1) lookup per row.
		this.rowChannelIds = [];
		this.rowDimIds = [];
		this.digitChannelId = null;
		this.digitDimId = null;
		this.lastAction = -1;
	}

	/**
	 * Register 28 row input channels plus the digit action channel.
	 */
	registerChannels(brain) {
		for (let r = 0; r < 28; r++) {
			const { channelId, dimensionIds } = brain.registerChannelSpec({
				name: `row${r}`,
				emitsReward: false,             // pixels don't emit reward; the digit channel does
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

	/**
	 * Map a raw 0–255 pixel value into a bucket id [0, buckets).
	 */
	quantize(value) {
		const bucket = Math.floor(value * this.buckets / 256);
		return Math.min(bucket, this.buckets - 1);
	}

	/**
	 * Quantize a full 28×28 image into a 784-length row-major Uint8Array.
	 * No trimming — every image is exactly 28×28 so the column scan stays
	 * aligned across all images.
	 */
	buildBits(pixels) {
		const out = new Uint8Array(784);
		for (let i = 0; i < 784; i++) out[i] = this.quantize(pixels[i]);
		return out;
	}

	/**
	 * Build the events Map for column `col` (0..27) of an image — each of
	 * the 28 row-channels emits its pixel at that column.
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

	/**
	 * Build the actions Map naming the correct digit for brain.learn().
	 */
	encodeAction(label) {
		const actions = new Map();
		const dimMap = new Map();
		dimMap.set(this.digitDimId, label);
		actions.set(this.digitChannelId, dimMap);
		return actions;
	}

	/**
	 * Build the rewards Map for the digit channel. Positive reward when the
	 * last predicted action matched the label, negative when it didn't.
	 */
	buildRewards(label, correctReward = 1, incorrectReward = -1) {
		const rewards = new Map();
		if (this.lastAction >= 0) {
			rewards.set(this.digitChannelId, this.lastAction === label ? correctReward : incorrectReward);
		} else {
			rewards.set(this.digitChannelId, correctReward);
		}
		return rewards;
	}

	/**
	 * Read the predicted digit out of the inferences Map. Returns
	 * { predicted, score, strength } with predicted=-1 when no action
	 * inference exists for the digit channel.
	 */
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
	 * with the highest totalStrength × avgReward — preserves the frequency
	 * signal that the brain's per-voter-mean consensus collapses away.
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

	/**
	 * Force lastAction to a label — supervised training uses this so the
	 * next buildRewards() delivers the correct reward.
	 */
	setForcedAction(label) { this.lastAction = label; }

	/**
	 * Clear per-image action tracking between images.
	 */
	resetActions() { this.lastAction = -1; }
}
