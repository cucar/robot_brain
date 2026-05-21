/**
 * MNIST Encoder — manages 784 pixel channels for digit recognition.
 *
 * Each pixel position (row, col) in the 28×28 grid becomes an independent
 * brain channel with one input dimension (quantized pixel value) and one
 * action dimension (digit prediction 0–9). This retinotopic layout mirrors
 * cortical column organization: each spatial position has a dedicated column
 * that learns cross-channel correlations through temporal co-activation.
 *
 * Unlike the stock encoder (one instance per symbol), a single MNISTEncoder
 * manages all 784 channels. This is practical because every pixel channel
 * shares identical structure — they differ only in their spatial position
 * and the neuron IDs the brain allocates for them.
 *
 * Neuron budget at binary quantization (buckets=2):
 *   784 channels × 2 event neurons  =  1,568 sensory neurons
 *   784 channels × 10 action neurons =  7,840 action neurons
 *   Total base neurons: 9,408
 */

const PIXELS = 784;
const DIGITS = 10;
const DIGIT_ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

export class MNISTEncoder {

	/**
	 * Create a new encoder with the specified quantization level.
	 * IDs remain unset until registerChannels() wires them to the brain.
	 *
	 * @param {number} buckets — number of quantization levels per pixel
	 *   (2 = binary black/white, 4 = Phase B, 8/16 = Phase C)
	 */
	constructor(buckets = 2) {
		this.buckets = buckets;

		// Brain-allocated IDs — one entry per pixel position (0–783).
		// Populated by registerChannels() after brain.registerChannelSpec().
		this.channelIds = new Array(PIXELS);
		this.inputDimIds = new Array(PIXELS);
		this.actionDimIds = new Array(PIXELS);

		// Per-channel last-action tracker: -1 = hasn't acted yet, 0–9 = digit.
		// Used by buildRewards() to deliver per-channel reward based on whether
		// that channel's most recent digit prediction matched the label.
		this.lastActions = new Int8Array(PIXELS).fill(-1);
	}

	/**
	 * Register all 784 pixel channels with the brain. Each channel gets:
	 *   - one input dim: quantized pixel brightness (passthrough, resolution = buckets)
	 *   - one action dim: digit prediction (passthrough, 10 actions, default = 0)
	 *
	 * Channel names encode grid position (px_14_14) so diagnostic output
	 * is human-readable. The brain allocates unique IDs for each channel
	 * and dimension; we store them for fast lookup during frame encoding.
	 *
	 * @param {Brain} brain — brain instance to register with
	 */
	registerChannels(brain) {
		for (let p = 0; p < PIXELS; p++) {
			const row = Math.floor(p / 28);
			const col = p % 28;
			const name = `px_${row}_${col}`;

			const { channelId, dimensionIds } = brain.registerChannelSpec({
				name,
				emitsReward: true,
				learnActionSequences: false,
				dimensions: [
					{
						name: `${name}_val`,
						kind: 'input',
						resolution: this.buckets,
						mode: 'passthrough'
					},
					{
						name: `${name}_digit`,
						kind: 'action',
						resolution: DIGITS,
						mode: 'passthrough',
						actions: DIGIT_ACTIONS,
						defaultAction: 0
					}
				]
			});

			this.channelIds[p] = channelId;
			this.inputDimIds[p] = dimensionIds[`${name}_val`];
			this.actionDimIds[p] = dimensionIds[`${name}_digit`];
		}
	}

	/**
	 * Quantize a raw pixel value (0–255) into a bucket index (0 to buckets-1).
	 * Uses equal-width bins so the boundaries scale with the bucket count:
	 *   2 buckets: [0–127]=0, [128–255]=1
	 *   4 buckets: [0–63]=0, [64–127]=1, [128–191]=2, [192–255]=3
	 *
	 * The Math.min clamp guards against value=255 landing in an out-of-range
	 * bucket due to floor(255 * N / 256) equaling N when N is a power of two.
	 *
	 * @param {number} value — raw grayscale pixel value (0–255)
	 * @returns {number} bucket index (0 to buckets-1)
	 */
	quantize(value) {
		const bucket = Math.floor(value * this.buckets / 256);
		return Math.min(bucket, this.buckets - 1);
	}

	/**
	 * Build the inputs map for all 784 channels from a single image. The
	 * returned map has the shape brain.processFrame() expects:
	 *   Map<channelId, Map<dimId, bucketValue>>
	 *
	 * Called once per image and reused across all frames of the same episode
	 * (the image is identical each frame — only the brain's internal context
	 * changes between frames).
	 *
	 * @param {Uint8Array} pixels — 784 raw pixel values (0–255), row-major
	 * @returns {Map<number, Map<number, number>>}
	 */
	encodeImage(pixels) {
		const inputs = new Map();
		for (let p = 0; p < PIXELS; p++) {
			const dimMap = new Map();
			dimMap.set(this.inputDimIds[p], this.quantize(pixels[p]));
			inputs.set(this.channelIds[p], dimMap);
		}
		return inputs;
	}

	/**
	 * Build the per-channel rewards map based on each channel's last digit
	 * prediction versus the correct label. Each channel is rewarded or
	 * punished independently — channel (14,14) might correctly predict "7"
	 * even when the aggregate vote is wrong, and it should be reinforced.
	 *
	 * Channels that haven't acted yet (lastActions[p] === -1) are skipped
	 * so we don't reward neurons that weren't responsible for any prediction.
	 *
	 * @param {number} label — correct digit (0–9)
	 * @param {number} correctReward — reward for matching the label (default +1)
	 * @param {number} incorrectReward — reward for not matching (default -1)
	 * @returns {Map<number, number>} channelId → reward
	 */
	buildRewards(label, correctReward = 1, incorrectReward = -1) {
		const rewards = new Map();
		for (let p = 0; p < PIXELS; p++) {
			if (this.lastActions[p] >= 0) {
				rewards.set(
					this.channelIds[p],
					this.lastActions[p] === label ? correctReward : incorrectReward
				);
			}
		}
		return rewards;
	}

	/**
	 * Extract action inferences from the brain's output, update the per-channel
	 * last-action tracker, and compute the aggregate digit prediction.
	 *
	 * Aggregation: each channel independently votes for a digit. We sum the
	 * winner's score per digit across all 784 channels. The digit with the
	 * highest total score is the aggregate prediction. This mirrors the stock
	 * architecture where multiple channels vote on a shared action space and
	 * the consensus mechanism extracts signal from the aggregate.
	 *
	 * Uninformative channels (e.g. corners that are always black) produce weak
	 * or default-action votes that wash out in the sum — the voting pool
	 * naturally weights contributions by connection strength.
	 *
	 * @param {Map} inferences — per-channel inference arrays from brain.processFrame()
	 * @returns {{ predicted: number, scores: Float64Array, confidence: number }}
	 */
	applyInferences(inferences) {
		const scores = new Float64Array(DIGITS);
		const voteCounts = new Uint16Array(DIGITS);

		for (let p = 0; p < PIXELS; p++) {
			const channelInfs = inferences.get(this.channelIds[p]);
			if (!channelInfs) continue;

			// Each channel produces input and action inferences; we only care
			// about the action (digit prediction) for classification.
			const actionInf = channelInfs.find(inf => inf.kind === 'action');
			if (!actionInf) continue;

			// Track this channel's prediction so buildRewards() can evaluate it.
			this.lastActions[p] = actionInf.winner.value;

			// Accumulate into the aggregate score. winner.score is the voting
			// consensus strength; fall back to winner.strength if score is absent.
			const digit = actionInf.winner.value;
			scores[digit] += actionInf.winner.score ?? actionInf.winner.strength ?? 0;
			voteCounts[digit]++;
		}

		// Argmax over the 10 digit scores.
		let predicted = 0;
		let maxScore = scores[0];
		for (let d = 1; d < DIGITS; d++) {
			if (scores[d] > maxScore) {
				maxScore = scores[d];
				predicted = d;
			}
		}

		return { predicted, scores, voteCounts, confidence: maxScore };
	}

	/**
	 * Clear per-image action tracking. Called between images (after resetContext)
	 * so that buildRewards() doesn't carry stale predictions from the previous
	 * image into the new episode.
	 */
	resetActions() {
		this.lastActions.fill(-1);
	}
}
