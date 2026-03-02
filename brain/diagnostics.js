import { createInterface } from 'node:readline';
import { stdin, stdout } from 'node:process';

/**
 * Diagnostics - Handles diagnostic output and metrics for Brain
 * Receives data as parameters and returns formatted output
 */
export class Diagnostics {
	constructor(debug, frameSummary) {

		// Diagnostic state
		this.accuracyStats = { correct: 0, total: 0 };
		this.rewardStats = { totalReward: 0, count: 0 };
		this.continuousPredictionMetrics = { totalError: 0, count: 0 };
		this.previousProfit = 0;

		// Flags
		this.debug = debug;
		this.frameSummary = frameSummary;

		// Readline interface for debugging (created lazily only when needed)
		this.rl = null;
	}

	/**
	 * Wait for user input to continue - used for debugging
	 * @param {string} message - Message to display
	 * @returns {Promise}
	 */
	waitForUser(message) {
		// Create readline interface lazily only when actually needed
		if (!this.rl) this.rl = createInterface({ input: stdin, output: stdout });
		return new Promise(resolve => this.rl.question(`\n${message}...`, resolve));
	}

	/**
	 * Reset accuracy and reward stats for a new episode
	 */
	resetAccuracyStats() {
		this.accuracyStats = { correct: 0, total: 0 };
		this.rewardStats = { totalReward: 0, count: 0 };
		this.continuousPredictionMetrics = { totalError: 0, count: 0 };
	}

	/**
	 * Show debug information for the inference votes
	 */
	debugVotes(votes, winners, channels) {

		if (votes.length === 0) return;
		console.log(`Collected ${votes.length} votes`);

		// Build winner set for quick lookup
		const winnerIds = new Set(winners.map(w => w.neuron_id));

		// Group votes by channel
		const votesByChannel = new Map();
		for (const vote of votes) {
			const channelName = vote.neuron.channel;
			if (!votesByChannel.has(channelName)) votesByChannel.set(channelName, []);
			votesByChannel.get(channelName).push(vote);
		}

		// Debug votes for each channel
		for (const [channelName, channelVotes] of votesByChannel) {
			const channel = channels.get(channelName);
			if (!channel) continue;

			// Debug event votes and action votes separately
			this.debugEventVotes(channelVotes, winnerIds, channel);
			this.debugActionVotes(channelVotes, winnerIds, channel);
		}
	}

	/**
	 * Debug helper: show votes for event neurons
	 * Shows which patterns/connections are voting for which event predictions
	 * @param {Array} allVotes - Array of all votes for this channel
	 * @param {Set} winnerIds - Set of winning neuron IDs from determineConsensus
	 * @param {Object} channel - Channel instance for formatting
	 */
	debugEventVotes(allVotes, winnerIds, channel) {
		const eventVotes = allVotes.filter(v => v.neuron.type === 'event');
		if (eventVotes.length === 0) return;

		// Group votes by event neuron and build metadata
		const votesByNeuron = this.groupVotesByNeuron(eventVotes);

		// Aggregate votes by voter for each neuron
		const aggregatedByNeuron = new Map();
		for (const [neuronId, data] of votesByNeuron)
			aggregatedByNeuron.set(neuronId, this.aggregateVotesBySource(data.votes));

		// Group by dimension
		const byDimension = this.groupByDimension(votesByNeuron);

		// Display results
		console.log(`\n=== ${channel.name} EVENT VOTES ===`);
		for (const [dimName, candidates] of byDimension) {
			candidates.sort((a, b) => b.totalStrength - a.totalStrength);
			console.log(`  ${dimName} (${candidates.length} candidates):`);

			for (const candidate of candidates) {
				const marker = winnerIds.has(candidate.neuronId) ? '★ WINNER' : '';
				const coordsFormatted = this.formatCoordinates(candidate.coordsStr, channel);
				const aggVotes = aggregatedByNeuron.get(candidate.neuronId);
				console.log(`    ${coordsFormatted} (n${candidate.neuronId}) str=${candidate.totalStrength.toFixed(1)} ${marker}`);
				console.log(this.formatAggregatedVotes(aggVotes, '', false, channel));
			}
		}
		console.log(`===================\n`);
	}

	/**
	 * Group votes by neuron ID and build metadata for each neuron
	 * @param {Array} votes - Array of votes to group
	 * @returns {Map} Map of neuronId -> {neuronId, coordsStr, dimensions, votes, totalStrength}
	 */
	groupVotesByNeuron(votes) {
		const votesByNeuron = new Map();
		for (const v of votes) {
			if (!votesByNeuron.has(v.neuron.id)) {
				const coords = v.neuron.coordinates;
				const coordsStr = Object.entries(coords).sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => `${k}=${v}`).join(', ');
				votesByNeuron.set(v.neuron.id, {
					neuronId: v.neuron.id,
					coordsStr,
					dimensions: coords,
					votes: [],
					totalStrength: 0
				});
			}
			const data = votesByNeuron.get(v.neuron.id);
			data.votes.push(v);
			data.totalStrength += v.strength;
		}
		return votesByNeuron;
	}

	/**
	 * Group neurons by dimension for winner determination
	 * @param {Map} votesByNeuron - Map from groupVotesByNeuron
	 * @returns {Map} Map of dimension -> array of candidates
	 */
	groupByDimension(votesByNeuron) {
		const byDimension = new Map();
		for (const [_, data] of votesByNeuron) {
			for (const [dimName, _] of Object.entries(data.dimensions)) {
				if (!byDimension.has(dimName))
					byDimension.set(dimName, []);
				byDimension.get(dimName).push(data);
			}
		}
		return byDimension;
	}

	/**
	 * Debug helper: show votes for action neurons
	 * Shows which patterns/connections are voting for which actions
	 * @param {Array} allVotes - Array of all votes for this channel
	 * @param {Set} winnerIds - Set of winning neuron IDs from determineConsensus
	 * @param {Object} channel - Channel instance for formatting
	 */
	debugActionVotes(allVotes, winnerIds, channel) {
		const actionVotes = allVotes.filter(v => v.neuron.type === 'action');
		if (actionVotes.length === 0) return;

		// Group by action label and aggregate
		const actionGroups = this.groupActionsByLabel(actionVotes, channel);
		const aggregatedByAction = new Map();
		const totalsByAction = new Map();

		for (const [label, votes] of actionGroups) {
			const aggregated = this.aggregateVotesBySource(votes);
			aggregatedByAction.set(label, aggregated);
			totalsByAction.set(label, this.calculateActionTotals(aggregated));
		}

		// Find which action label won (check if any neuron in the group is a winner)
		const winningLabel = this.findWinningActionLabel(actionGroups, winnerIds);

		// Display results
		console.log(`\n=== ${channel.name} ACTION VOTES ===`);
		for (const [label, aggregated] of aggregatedByAction) {
			const total = totalsByAction.get(label);
			const winnerMarker = label === winningLabel ? ' ★' : '';
			const header = `${label} (${aggregated.length} voters, str=${total.str}, avgRwd=${total.rwd})${winnerMarker}`;
			console.log(this.formatAggregatedVotes(aggregated, header, true, channel));
		}
		console.log(`  SELECTION: ${winningLabel} (highest reward)`);
		console.log(`===================\n`);
	}

	/**
	 * Group action votes by action label
	 * @param {Array} actionVotes - Array of action votes
	 * @param {Object} channel - Channel instance for formatting
	 * @returns {Map} Map of label -> votes array
	 */
	groupActionsByLabel(actionVotes, channel) {
		const actionGroups = new Map();
		for (const v of actionVotes) {
			const coords = v.neuron.coordinates;
			const label = channel.formatActionLabel ? channel.formatActionLabel(coords) : JSON.stringify(coords);
			if (!actionGroups.has(label))
				actionGroups.set(label, []);
			actionGroups.get(label).push(v);
		}
		return actionGroups;
	}

	/**
	 * Calculate totals for an action group
	 * @param {Array} aggregated - Aggregated votes
	 * @returns {Object} {str, weightedRewardSum, rwd}
	 */
	calculateActionTotals(aggregated) {
		const total = {
			str: aggregated.reduce((s, a) => s + a.strength, 0),
			weightedRewardSum: aggregated.reduce((s, a) => s + a.weightedRewardSum, 0)
		};
		total.rwd = total.str > 0 ? total.weightedRewardSum / total.str : 0;
		return total;
	}

	/**
	 * Find which action label won based on winner IDs
	 * @param {Map<string, Array>} actionGroups - Map of label -> votes array
	 * @param {Set} winnerIds - Set of winning neuron IDs
	 * @returns {string} Winning label
	 */
	findWinningActionLabel(actionGroups, winnerIds) {
		for (const [label, votes] of actionGroups)
			for (const vote of votes)
				if (winnerIds.has(vote.neuron.id)) return label;
		throw new Error('Cannot find winning action label');
	}

	/**
	 * Aggregate votes by source neuron - sum strengths, strength-weighted average reward
	 * @param {Array} votes - Array of votes to aggregate
	 * @returns {Array} Array of aggregated votes by source
	 */
	aggregateVotesBySource(votes) {
		if (votes.length === 0) return [];

		// Aggregate by (voter neuron, distance) - same neuron at different ages generates
		// separate entries so multi-distance voting is visible as distinct rows in the output
		const bySource = new Map();
		for (const v of votes) {
			const coords = this.formatNeuronCoords(v.voter);
			const level = v.voter.level;
			const key = `${v.voter.id}:${v.distance}`;
			if (!bySource.has(key))
				bySource.set(key, { voterId: v.voter.id, strength: 0, weightedRewardSum: 0, coords, level, distance: v.distance });
			const agg = bySource.get(key);
			agg.strength += v.strength;
			agg.weightedRewardSum += v.strength * v.reward;
		}

		// Calculate strength-weighted average reward per source
		for (const [_, agg] of bySource)
			agg.reward = agg.strength > 0 ? agg.weightedRewardSum / agg.strength : 0;

		return [...bySource.values()];
	}

	/**
	 * Format neuron coordinates for display (brain.js version)
	 * @param {Object} neuron - Neuron object
	 * @returns {string} Formatted coordinates string
	 */
	formatNeuronCoords(neuron) {

		// Pattern neurons (level > 0) have a parent neuron instead of direct coordinates
		if (neuron.level > 0 && neuron.parent)
			return this.formatNeuronCoords(neuron.parent);

		// Sensory neurons (level 0) have coordinates
		if (!neuron.coordinates) return `n${neuron.id}`;
		return Object.entries(neuron.coordinates)
			.sort(([a], [b]) => a.localeCompare(b))
			.map(([k, v]) => `${k}=${v}`)
			.join(', ');
	}

	/**
	 * Format aggregated votes with source info
	 * @param {Array} aggVotes - Array of aggregated votes
	 * @param {string} label - Optional label for the section
	 * @param {boolean} includeReward - Whether to include reward in output
	 * @param {Object} channel - Channel instance for formatting
	 * @returns {string} Formatted string
	 */
	formatAggregatedVotes(aggVotes, label, includeReward, channel) {
		if (aggVotes.length === 0) {
			if (label) return `  ${label}: no votes`;
			return '    no votes';
		}

		const lines = label ? [`  ${label}:`] : [];
		for (const agg of aggVotes) {
			const coordsFormatted = this.formatCoordinates(agg.coords, channel);
			const rewardStr = includeReward ? `, avgRwd=${agg.reward.toFixed(2)}` : '';
			const levelStr = agg.level > 0 ? ` L${agg.level}` : '';
			const typeStr = agg.level > 0 ? ' [P]' : '';
			lines.push(`    ${coordsFormatted}${levelStr}${typeStr} (d=${agg.distance}) → str=${agg.strength.toFixed(1)}${rewardStr}`);
		}
		return lines.join('\n');
	}

	/**
	 * Format coordinates string with channel-specific formatting if available
	 * @param {string} coordsStr - Coordinates string (e.g., "dim1=val1, dim2=val2")
	 * @param {Object} channel - Channel instance
	 * @returns {string} Formatted coordinates
	 */
	formatCoordinates(coordsStr, channel) {
		if (!coordsStr) return '(no coords)';
		// Use channel's formatCoordinates if available, otherwise return as-is
		if (channel && typeof channel.formatCoordinates === 'function')
			return channel.formatCoordinates(coordsStr);
		return coordsStr;
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
				const reward = rewards.get(neuron.channel) || 0;
				actionReward += reward;
				actionCount++;
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
	 * @param {Array<Channel>} channels - array of [channelName, channel]
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

		// Get holdings display - show which stocks are being held
		let holdingsDisplay = 'None';
		const holdings = [];
		for (const [_, channel] of channels) {
			const info = channel.getHoldingsInfo();
			if (info.shares > 0) holdings.push(info);
		}
		if (holdings.length > 0) holdingsDisplay = holdings.map(h => `${h.symbol}:${h.shares}sh`).join(', ');

		// Get portfolio metrics if any stock channels exist
		let portfolioDisplay = '';
		if (channels.length > 0 && channels[0][1].constructor.getPortfolioMetrics) {
			const portfolioMetrics = channels[0][1].constructor.getPortfolioMetrics(channels);
			const profitDelta = portfolioMetrics.totalProfit - this.previousProfit;
			this.previousProfit = portfolioMetrics.totalProfit;
			const totalPL = portfolioMetrics.totalProfit >= 0 ? '+' : '';
			const deltaPL = profitDelta >= 0 ? '+' : '';
			portfolioDisplay = ` | Cash:${portfolioMetrics.cash.toFixed(0)} | Holdings: ${holdingsDisplay} | P&L:${totalPL}${portfolioMetrics.totalProfit.toFixed(2)} (${deltaPL}${profitDelta.toFixed(2)})`;
		}

		if (this.frameSummary)
			console.log(`Frame ${frameNumber} | Accuracy: ${baseAccuracy} | Reward: ${avgReward} | MAPE: ${mapeDisplay}${portfolioDisplay} | Time: ${frameElapsed.toFixed(2)}ms`);
	}
}