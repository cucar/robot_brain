import { DIGITS, DIGIT_ACTIONS } from './digits.js';

/**
 * One channel per pixel at the configured imageSize (28 → 784 channels,
 * 14 → 196, 7 → 49). Source MNIST images are always 28×28; smaller sizes
 * are produced by block-average downsampling before quantization. Exactly
 * `pixels` sensory neurons fire per frame. A whole image is one frame —
 * intra-frame co-activation builds the image pattern. Action neuron lives
 * on a separate "digit" channel.
 */
export class MNISTPixelChannelsEncoder {

	/**
	 * @param {number} buckets - pixel quantization levels (2 = binary)
	 * @param {number} imageSize - output image side length. Source images are
	 *   always 28×28; smaller sizes are produced by block-average downsampling
	 *   (28 must be divisible by imageSize). Used for scaling experiments.
	 */
	constructor(buckets = 2, imageSize = 28) {
		if (28 % imageSize !== 0) {
			throw new Error(`imageSize must divide 28 evenly, got ${imageSize}`);
		}
		this.buckets = buckets;
		this.imageSize = imageSize;
		this.pixels = imageSize * imageSize;
		this.downsampleFactor = 28 / imageSize;
		// Parallel arrays so encodeImage() can do an O(1) channel/dim lookup per pixel.
		this.pixelChannelIds = [];
		this.pixelDimIds = [];
		this.digitChannelId = null;
		this.digitDimId = null;
		// Tracks the last predicted/forced digit so buildRewards() can attribute
		// the next reward to the right action neuron.
		this.lastAction = -1;
	}

	/**
	 * Register the 784 pixel input channels plus the digit action channel.
	 * Called once at job startup, before any frames are processed.
	 */
	registerChannels(brain) {
		// One channel per pixel — names "px_0".."px_N-1" line up with row-major
		// order at the current imageSize.
		for (let p = 0; p < this.pixels; p++) {
			const { channelId, dimensionIds } = brain.registerChannelSpec({
				name: `px_${p}`,
				emitsReward: false,             // pixels don't emit reward; the digit channel does
				learnActionSequences: false,
				dimensions: [
					{ name: 'px_val', kind: 'input', resolution: this.buckets, mode: 'passthrough' }
				]
			});
			this.pixelChannelIds.push(channelId);
			this.pixelDimIds.push(dimensionIds['px_val']);
		}
		// Separate action channel — keeps pixel channels uniform and pixel
		// neurons distinct from action neurons in the voting graph.
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
		// Cap at the top bucket so 255 doesn't overflow when buckets evenly divides 256.
		return Math.min(bucket, this.buckets - 1);
	}

	/**
	 * Quantize a full 28x28 image into an (imageSize × imageSize) row-major
	 * Uint8Array. When imageSize < 28, block-average the source 2×2 / 4×4 /
	 * 7×7 region into one output pixel before quantizing — preserves the
	 * "average intensity" of each block.
	 */
	buildBits(pixels) {
		const out = new Uint8Array(this.pixels);
		const f = this.downsampleFactor;
		if (f === 1) {
			for (let i = 0; i < this.pixels; i++) out[i] = this.quantize(pixels[i]);
			return out;
		}
		const blockArea = f * f;
		for (let r = 0; r < this.imageSize; r++) {
			for (let c = 0; c < this.imageSize; c++) {
				let sum = 0;
				const srcRow0 = r * f;
				const srcCol0 = c * f;
				for (let dr = 0; dr < f; dr++) {
					const srcRow = srcRow0 + dr;
					for (let dc = 0; dc < f; dc++) {
						sum += pixels[srcRow * 28 + srcCol0 + dc];
					}
				}
				out[r * this.imageSize + c] = this.quantize(sum / blockArea);
			}
		}
		return out;
	}

	/**
	 * Build the events Map for one frame — every pixel channel fires its
	 * bucket value simultaneously. Shape: channelId → (dimId → bucketValue).
	 */
	encodeImage(bits) {
		const inputs = new Map();
		for (let p = 0; p < this.pixels; p++) {
			// One-entry dim map per channel — each pixel has a single input dimension.
			const dimMap = new Map();
			dimMap.set(this.pixelDimIds[p], bits[p]);
			inputs.set(this.pixelChannelIds[p], dimMap);
		}
		return inputs;
	}

	/**
	 * Build the actions Map naming the correct digit. Passed to brain.learn()
	 * so it knows which action neuron to wire the active voters to.
	 */
	encodeAction(label) {
		const actions = new Map();
		const dimMap = new Map();
		dimMap.set(this.digitDimId, label);
		actions.set(this.digitChannelId, dimMap);
		return actions;
	}

	/**
	 * Build the rewards Map for the digit channel. Uses lastAction to decide
	 * whether the previous prediction was right (positive reward) or wrong
	 * (negative). On the first call per image lastAction is -1 — treat it as
	 * a fresh start and emit the positive reward unconditionally.
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
	 * Per-voter reward-normalized consensus.
	 *
	 * Reads infer().actionVotes (one entry per voter→action vote). For each
	 * voter, normalizes its rewards so they sum to 1 (each voter contributes
	 * total weight 1 regardless of how many digits it votes for), then sums
	 * the normalized contributions per digit and returns argmax.
	 *
	 * Why `reward` not `strength * reward`: rewards are already additive in
	 * the brain's learn() call, so multiplying by strength (which also tracks
	 * frequency) would double-count the same signal.
	 *
	 * @param {Array<{voterNeuronId, value, reward}>} actionVotes
	 * @returns {number} predicted digit (-1 if no positive votes)
	 */
	predictByVoterNormalizedConsensus(actionVotes) {
		if (!actionVotes || actionVotes.length === 0) {
			this.lastAction = -1;
			return -1;
		}

		// Group votes by voter so we can normalize each voter's slate independently.
		const byVoter = new Map();
		for (const v of actionVotes) {
			let arr = byVoter.get(v.voterNeuronId);
			if (!arr) { arr = []; byVoter.set(v.voterNeuronId, arr); }
			arr.push(v);
		}

		// For each voter: sum its rewards, then divide each vote by that sum so
		// the voter's contributions add up to 1. Skip voters whose total is <= 0
		// (no positive signal to distribute).
		const digitScores = new Map();
		for (const votes of byVoter.values()) {
			let total = 0;
			for (const v of votes) total += v.reward;
			if (total <= 0) continue;
			for (const v of votes) {
				const contrib = v.reward / total;
				digitScores.set(v.value, (digitScores.get(v.value) ?? 0) + contrib);
			}
		}

		// Argmax across digits.
		let bestDigit = -1, bestScore = -Infinity;
		for (const [digit, score] of digitScores) {
			if (score > bestScore) { bestScore = score; bestDigit = digit; }
		}
		// Cache the prediction so the next buildRewards() can score it.
		this.lastAction = bestDigit;
		return bestDigit;
	}

	/**
	 * Force lastAction to a specific label — used by supervised training
	 * paths so buildRewards() delivers the correct reward next frame.
	 */
	setForcedAction(label) { this.lastAction = label; }

	/**
	 * Clear lastAction between images so leftover state doesn't bleed across
	 * unrelated examples.
	 */
	resetActions() { this.lastAction = -1; }
}
