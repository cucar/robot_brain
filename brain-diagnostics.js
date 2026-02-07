import { createInterface } from 'node:readline';
import { stdin, stdout } from 'node:process';

/**
 * BrainDiagnostics - Handles diagnostic output and metrics for Brain
 * Receives data as parameters and returns formatted output
 */
export class BrainDiagnostics {
	constructor(debug) {

		// Diagnostic state
		this.accuracyStats = { correct: 0, total: 0 };
		this.rewardStats = { totalReward: 0, count: 0 };
		this.continuousPredictionMetrics = { totalError: 0, count: 0 };

		// Flags
		this.debug = debug;
		this.frameSummary = true;

		// Readline interface for debugging
		this.rl = createInterface({ input: stdin, output: stdout });
	}

	/**
	 * Wait for user input to continue - used for debugging
	 * @param {string} message - Message to display
	 * @returns {Promise}
	 */
	waitForUser(message) {
		return new Promise(resolve => this.rl.question(`\n${message}...`, resolve));
	}

	/**
	 * Reset accuracy and reward stats for a new episode
	 */
	resetAccuracyStats() {
		this.accuracyStats = { correct: 0, total: 0 };
		this.rewardStats = { totalReward: 0, count: 0 };
	}

	/**
	 * Track inference performance for both events and actions.
	 * Also calculates continuous prediction errors via channel callbacks.
	 * @param {Array} inferences - Array of inferred neurons {neuron, strength}
	 * @param {Map} neuronsAtAge0 - Map of neurons active at age 0 (reality)
	 * @param {Map} rewards - Map of channel name to reward value
	 * @param {IterableIterator<[string, object]>} channels - Iterator of [channelName, channel] pairs
	 */
	trackInferencePerformance(inferences, neuronsAtAge0, rewards, channels) {
		let eventCorrect = 0;
		let eventTotal = 0;
		let actionReward = 0;
		let actionCount = 0;

		// Group event predictions by channel for continuous error calculation
		const predictionsByChannel = new Map();
		const actualsByChannel = new Map();

		for (const { neuron, strength } of inferences) {

			// Track event prediction accuracy
			if (neuron.type === 'event') {
				eventTotal++;
				if (neuronsAtAge0.has(neuron)) eventCorrect++;

				// Group for continuous error calculation
				if (!predictionsByChannel.has(neuron.channel))
					predictionsByChannel.set(neuron.channel, []);
				predictionsByChannel.get(neuron.channel).push({ neuron, strength });
			}
			// Track action reward from the action's channel
			else if (neuron.type === 'action') {
				const reward = rewards.get(neuron.channel);
				if (reward !== undefined) {
					actionReward += reward;
					actionCount++;
				}
			}
		}

		// Group actual event neurons by channel
		for (const [neuron] of neuronsAtAge0) {
			if (neuron.type !== 'event') continue;
			if (!actualsByChannel.has(neuron.channel))
				actualsByChannel.set(neuron.channel, []);
			actualsByChannel.get(neuron.channel).push(neuron);
		}

		// Calculate continuous prediction errors for each channel
		for (const [channelName, channel] of channels) {
			const predictions = predictionsByChannel.get(channelName) || [];
			const actuals = actualsByChannel.get(channelName) || [];
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

		if (this.debug) {
			if (eventTotal > 0) {
				const accuracy = (eventCorrect / eventTotal * 100).toFixed(1);
				console.log(`Event predictions: ${eventCorrect}/${eventTotal} (${accuracy}%)`);
			}
			if (actionCount > 0)
				console.log(`Action rewards: ${actionReward.toFixed(3)} for ${actionCount} actions`);
		}
	}

	/**
	 * Display diagnostic frame header with frame number and observations
	 * @param {number} frameNumber - Current frame number
	 * @param {Map} rewards - Map of channel name to reward value
	 * @param {Array} frame - Frame data points
	 */
	startFrame(frameNumber, rewards, frame) {
		if (!this.debug) return;

		// Display reward information
		if (rewards.size > 0) {
			const rewardParts = [];
			for (const [channelName, reward] of rewards)
				rewardParts.push(`${channelName}:${reward.toFixed(3)}x`);
			console.log(`  Rewards: ${rewardParts.join(', ')}`);
		}

		// Build observation string from frame
		const observations = [];
		for (const point of frame)
			for (const [dim, val] of Object.entries(point.coordinates))
				observations.push(`${dim}=${val}`);

		console.log(`\nF${frameNumber} | Obs: ${observations.join(', ')}`);
	}

	/**
	 * Print one-line summary of frame processing
	 * @param {number} frameNumber - Current frame number
	 * @param {number} frameElapsed - Time elapsed for frame processing (ms)
	 * @param {IterableIterator<[string, object]>} channels - Iterator of [channelName, channel] pairs
	 */
	endFrame(frameNumber, frameElapsed, channels) {
		// Get base level (level 0) accuracy
		let baseAccuracy = 'N/A';
		if (this.accuracyStats.total > 0)
			baseAccuracy = `${(this.accuracyStats.correct / this.accuracyStats.total * 100).toFixed(1)}%`;

		// Get average action reward
		let avgReward = 'N/A';
		if (this.rewardStats.count > 0)
			avgReward = `${(this.rewardStats.totalReward / this.rewardStats.count).toFixed(3)} (${this.rewardStats.count})`;

		// Calculate average MAPE (Mean Absolute Percentage Error) and format with count
		let mapeDisplay = 'N/A';
		if (this.continuousPredictionMetrics.count > 0) {
			const avgMAPE = (this.continuousPredictionMetrics.totalError / this.continuousPredictionMetrics.count).toFixed(2);
			mapeDisplay = `${avgMAPE}% (${this.continuousPredictionMetrics.count})`;
		}

		// Collect output performance metrics from channels
		const outputMetrics = [];
		for (const [_, channel] of channels) {
			if (typeof channel.getOutputPerformanceMetrics === 'function') {
				const metrics = channel.getOutputPerformanceMetrics();
				if (metrics) outputMetrics.push(metrics);
			}
		}

		// Format output performance display
		let outputDisplay = 'N/A';
		if (outputMetrics.length > 0) {
			outputDisplay = outputMetrics.map(m => {
				const formatted = m.format === 'currency'
					? `${m.value >= 0 ? '+' : ''}${m.value.toFixed(2)}`
					: m.value.toFixed(2);
				return outputMetrics.length > 1 ? `${m.label}:${formatted}` : formatted;
			}).join(', ');
		}

		if (this.frameSummary)
			console.log(`Frame ${frameNumber} | Accuracy: ${baseAccuracy} | Reward: ${avgReward} | MAPE: ${mapeDisplay} | P&L: ${outputDisplay} | Time: ${frameElapsed.toFixed(2)}ms`);
	}
}

