const DIGITS = 10;
const DIGIT_ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

/**
 * Retinotopic per-pixel encoder for the sensory-only (Naive Bayes) MNIST app.
 *
 * One channel per pixel position at the configured imageSize (28→784, 14→196, 7→49).
 * Source MNIST images are always 28×28; smaller sizes are produced by block-average downsampling before quantization.
 * A whole image is one frame — every pixel channel fires its quantized value concurrently.
 * Supervision lands on a separate `digit` action channel via brain.learn(), not through processFrame's reward map.
 */
export class MNISTPixelChannelsEncoder {

	/**
	 * @param {number} buckets - pixel quantization levels (2 = binary)
	 * @param {number} imageSize - output image side length.
	 *   Source images are always 28×28; smaller sizes are produced by block-average downsampling.
	 *   28 must be divisible by imageSize.
	 */
	constructor(buckets = 2, imageSize = 28) {
		if (28 % imageSize !== 0) throw new Error(`imageSize must divide 28 evenly, got ${imageSize}`);
		this.buckets = buckets;
		this.imageSize = imageSize;
		this.pixels = imageSize * imageSize;
		this.downsampleFactor = 28 / imageSize;
		// Parallel arrays so encodeImage() can do an O(1) channel/dim lookup per pixel.
		this.pixelChannelIds = [];
		this.pixelDimIds = [];
		this.digitChannelId = null;
		this.digitDimId = null;
	}

	/**
	 * Register one input channel per pixel plus the shared digit action channel.
	 * Called once at job startup, before any frames are processed.
	 */
	registerChannels(brain) {
		for (let p = 0; p < this.pixels; p++) {
			const { channelId, dimensionIds } = brain.registerChannelSpec({
				name: `px_${p}`,
				learnActionSequences: false,
				dimensions: [
					{ name: 'px_val', kind: 'input', resolution: this.buckets, mode: 'passthrough' }
				]
			});
			this.pixelChannelIds.push(channelId);
			this.pixelDimIds.push(dimensionIds['px_val']);
		}
		const { channelId, dimensionIds } = brain.registerChannelSpec({
			name: 'digit',
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
	 * Quantize a full 28×28 image into a (imageSize × imageSize) row-major Uint8Array.
	 * When imageSize < 28, block-average the source 2×2 / 4×4 / 7×7 region into one output pixel before quantizing.
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
	 * Build the inputs Map for one frame — every pixel channel fires its bucket value simultaneously.
	 * Shape: channelId → (dimId → bucketValue).
	 */
	encodeImage(bits) {
		const inputs = new Map();
		for (let p = 0; p < this.pixels; p++) {
			const dimMap = new Map();
			dimMap.set(this.pixelDimIds[p], bits[p]);
			inputs.set(this.pixelChannelIds[p], dimMap);
		}
		return inputs;
	}

	/**
	 * Build the actions Map for brain.learn() — every digit is named, correct gets reward=1, others get reward=0.
	 * Wiring all ten digits per image lets the brain's smoothed-reward update converge `conn.reward(V,d)` to
	 * the per-voter posterior P(d|V) = K(V,d)/N_V (the running mean of 1s when label=d and 0s when label≠d).
	 * Shape: channelId → (dimId → Map(digitValue → reward)).
	 */
	encodeAction(correctLabel) {
		const actions = new Map();
		const dimMap = new Map();
		const valueRewardMap = new Map();
		for (let d = 0; d < DIGITS; d++) {
			valueRewardMap.set(d, d === correctLabel ? 1 : 0);
		}
		dimMap.set(this.digitDimId, valueRewardMap);
		actions.set(this.digitChannelId, dimMap);
		return actions;
	}

	/**
	 * Decode the brain's digit-channel inference back into a digit 0–9.
	 * Reads the action winner's value on the digit channel; returns -1 if no inference is present.
	 */
	decodeDigit(inferences) {
		const dimInfs = inferences?.get?.(this.digitChannelId);
		if (!dimInfs) return -1;
		const actionInf = dimInfs.find(inf => inf.kind === 'action');
		if (!actionInf || actionInf.winner.value == null) return -1;
		return actionInf.winner.value;
	}
}
