import { DIGITS, DIGIT_ACTIONS } from './digits.js';

/**
 * Single-channel pixel-stream encoder. Each image is fed pixel-by-pixel
 * as one event per frame on a single channel "px" with two dimensions:
 *   - px_val:   input, resolution = buckets (default 2 = binary)
 *   - px_digit: action, resolution 10 (one bucket per digit)
 *
 * Optional trim drops leading/trailing all-zero pixel runs so the temporal
 * stream starts at the first non-black pixel — removes the long uniform
 * preamble/postamble whose only role is to confuse per-age error patterns
 * at the first transition.
 */
export class MNISTEncoder {

	/**
	 * @param {number} buckets - quantization levels per pixel (2 = binary)
	 * @param {boolean} trim - strip leading/trailing all-black pixel runs
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
	 * Register the "px" channel with one input dimension and one action
	 * dimension on the same channel.
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
	 * Map a raw 0–255 pixel value into a bucket id [0, buckets).
	 */
	quantize(value) {
		const bucket = Math.floor(value * this.buckets / 256);
		// Cap so 255 doesn't overflow when buckets evenly divides 256.
		return Math.min(bucket, this.buckets - 1);
	}

	/**
	 * Build the (optionally trimmed) bit/bucket stream for an image.
	 * Returns a view that may be shorter than 784 when trim removes
	 * leading/trailing black runs.
	 *
	 * @param {Uint8Array} pixels - 784 raw values (row-major)
	 * @returns {Uint8Array} encoded stream (length ≤ 784)
	 */
	buildBits(pixels) {
		const all = new Uint8Array(784);
		for (let i = 0; i < 784; i++) all[i] = this.quantize(pixels[i]);
		if (!this.trim) return all;
		// Walk both ends inward, skipping all-zero buckets.
		let start = 0, end = 784;
		while (start < end && all[start] === 0) start++;
		while (end > start && all[end - 1] === 0) end--;
		return all.subarray(start, end);
	}

	/**
	 * Build the inputs Map for a single pixel frame.
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
	 */
	encodeAction(label) {
		const actions = new Map();
		const dimMap = new Map();
		dimMap.set(this.actionDimId, label);
		actions.set(this.channelId, dimMap);
		return actions;
	}

	/**
	 * Build the rewards Map for the single channel. Positive reward when
	 * the last predicted action matched the label, negative when it didn't.
	 * On the first frame of an image lastAction is -1 — treat as fresh and
	 * emit the positive reward.
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
	 * Returns { predicted, score, strength } — predicted is -1 when no
	 * action inference exists on this channel.
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
	 * Force lastAction to a label — supervised training uses this so the
	 * next buildRewards() delivers the correct reward.
	 */
	setForcedAction(label) {
		this.lastAction = label;
	}

	/**
	 * Clear per-image action tracking between images so state doesn't bleed.
	 */
	resetActions() {
		this.lastAction = -1;
	}
}
