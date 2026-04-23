/**
 * Diagnostics - per-frame and per-episode stats for the brain core.
 *
 * This module is pure data: it accumulates counters and exposes them via getters.
 * It does NOT print, format, or otherwise render. All presentation lives in the
 * host-side renderer (libs/node/src/renderer.js) so that when the brain moves to
 * Rust the counters port cleanly (integers, floats, small structs) while the
 * rendering stays in JS where I/O and app-layer composition belong.
 */
export class Diagnostics {

	constructor() {
		this.accuracyStats = { correct: 0, total: 0 };
		this.rewardStats = { totalReward: 0, count: 0 };
		this.continuousPredictionMetrics = { totalError: 0, count: 0 };
		this.mispredictions = []; // [{ predicted: coordinate, actual: coordinate, channel }]
	}

	/**
	 * Reset per-episode counters. Cumulative across an episode but wiped when the
	 * Job kicks off a new one — mispredictions are reset too so retrospective
	 * tooling only sees the current episode.
	 */
	resetAccuracyStats() {
		this.accuracyStats = { correct: 0, total: 0 };
		this.rewardStats = { totalReward: 0, count: 0 };
		this.continuousPredictionMetrics = { totalError: 0, count: 0 };
		this.mispredictions = [];
	}

	/**
	 * Accumulate MAPE (Mean Absolute Percentage Error) from scalar-space inferences.
	 * Compares continuous (score-weighted) event predictions against the actual
	 * input scalars for the same (channelId, dimId). Actions are skipped — reward
	 * from next frame is the ground-truth signal for those.
	 *
	 * Dims not registered with the quantizer belong to a legacy channel that tracks
	 * its own prediction error via channel.calculatePredictionError; skipping here
	 * avoids double-counting.
	 *
	 * @param {Map<number, Array<{dimId, kind, continuous}>>} inferencesByChannel
	 * @param {Map<number, Map<number, number>>} inputs - channelId → (dimId → actual scalar)
	 * @param {Quantizer} quantizer - to skip dims owned by channels that haven't migrated;
	 *   those are still accounted via channel.calculatePredictionError in trackInferencePerformance
	 */
	trackContinuousError(inferencesByChannel, inputs, quantizer) {
		for (const [channelId, dimInferences] of inferencesByChannel) {
			const actuals = inputs.get(channelId);
			if (!actuals) continue;
			for (const { dimId, kind, continuous } of dimInferences) {
				if (kind !== 'event') continue;
				if (!quantizer.has(dimId)) continue; // channel still owns bucketization - skip to avoid double-counting
				if (continuous === null) continue; // brain had no observed-bucket data to produce a scalar prediction
				const actual = actuals.get(dimId);
				if (actual === undefined || actual === 0) continue; // skip undefined and avoid divide-by-zero
				this.continuousPredictionMetrics.totalError += Math.abs((actual - continuous) / actual) * 100;
				this.continuousPredictionMetrics.count++;
			}
		}
	}

	/**
	 * Legacy-path inference tracker: event accuracy, action rewards, per-channel
	 * continuous prediction errors (via channel callbacks), and misprediction log.
	 * Used by the Channel-class flow; spec-based flow routes through
	 * trackContinuousError + reward bookkeeping only.
	 *
	 * @param {Array<{neuronId, strength, channel, type, coordinate}>} inferences
	 * @param {Set<number>} activeNeuronIds
	 * @param {Map<string, Array>} actualEventCoordsByChannel
	 * @param {Map<string, number>} rewards
	 * @param {IterableIterator<[string, object]>} channels
	 */
	trackInferencePerformance(inferences, activeNeuronIds, actualEventCoordsByChannel, rewards, channels) {
		let eventCorrect = 0;
		let eventTotal = 0;
		let actionReward = 0;
		let actionCount = 0;

		// Group event predictions by channel for continuous error calculation
		const predictionsByChannel = new Map();

		for (const { neuronId, strength, channel, type, coordinate } of inferences) {

			// Track event prediction accuracy
			if (type === 'event') {
				eventTotal++;
				const isCorrect = activeNeuronIds.has(neuronId);
				if (isCorrect) eventCorrect++;

				// Group for continuous error calculation
				if (!predictionsByChannel.has(channel))
					predictionsByChannel.set(channel, []);
				predictionsByChannel.get(channel).push({ coordinate, strength, isCorrect });
			}
			// Track action reward from the action's channel
			else if (type === 'action') {
				const reward = rewards.get(channel) || 0;
				actionReward += reward;
				actionCount++;
			}
		}

		// Track mispredictions for each channel
		for (const [channelName] of channels) {
			const predictions = predictionsByChannel.get(channelName) || [];
			const actuals = actualEventCoordsByChannel.get(channelName) || [];

			// Find wrong predictions and record the misprediction with predicted and actual coordinates
			for (const { coordinate, isCorrect } of predictions)
				if (!isCorrect && actuals.length > 0)
					this.mispredictions.push({
						channel: channelName,
						predicted: coordinate,
						actual: actuals[0]
					});
		}

		// Calculate continuous prediction errors for each channel
		for (const [channelName, channel] of channels) {
			const predictions = predictionsByChannel.get(channelName) || [];
			const actuals = actualEventCoordsByChannel.get(channelName) || [];
			if (predictions.length === 0) continue;

			const error = channel.calculatePredictionError(predictions, actuals);
			if (error !== null) {
				this.continuousPredictionMetrics.totalError += error;
				this.continuousPredictionMetrics.count++;
			}
		}

		// Update cumulative stats
		this.accuracyStats.correct += eventCorrect;
		this.accuracyStats.total += eventTotal;
		this.rewardStats.totalReward += actionReward;
		this.rewardStats.count += actionCount;
	}

	/**
	 * Return episode-to-date stats in a stable shape for host-side rendering.
	 * All rates are returned as numbers (not pre-formatted strings) so the renderer
	 * decides precision and units. `null` means "no data yet" — callers should
	 * render "N/A" rather than 0.
	 */
	getStats() {
		return {
			baseAccuracy: this.accuracyStats.total > 0
				? this.accuracyStats.correct / this.accuracyStats.total
				: null,
			accuracyCorrect: this.accuracyStats.correct,
			accuracyTotal: this.accuracyStats.total,

			avgReward: this.rewardStats.count > 0
				? this.rewardStats.totalReward / this.rewardStats.count
				: null,
			rewardCount: this.rewardStats.count,
			totalReward: this.rewardStats.totalReward,

			mape: this.continuousPredictionMetrics.count > 0
				? this.continuousPredictionMetrics.totalError / this.continuousPredictionMetrics.count
				: null,
			mapeCount: this.continuousPredictionMetrics.count,

			mispredictions: this.mispredictions
		};
	}
}
