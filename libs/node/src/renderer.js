/**
 * Host-side renderer for brain per-frame output.
 *
 * The brain core (JS today, Rust later) is pure compute: it tracks stats as numbers
 * and exposes them via getters. This module owns all the console.log and string
 * formatting — presentation concerns that the core shouldn't carry across the FFI
 * boundary. Jobs call these functions (usually through Job.render* hooks) to surface
 * per-frame diagnostics, debug dumps, and the one-line summary.
 */

/**
 * Build the per-frame summary line from brain stats and an optional job-supplied tail.
 * The tail is where app-layer state (e.g. stocks portfolio P&L) lives; the brain has
 * no way to compute it, so the host composes it in. Returns null if the summary is
 * suppressed (no-op helper so callers don't branch).
 *
 * @param {object} summary - cumulative/episode stats from Brain.getFrameSummary()
 * @param {number} elapsed - per-frame elapsed ms (from the frame object returned by
 *                           Brain.processInputs / processFrame — NOT in summary because
 *                           it's a per-call byproduct, not cumulative state)
 * @param {string} [tail]  - extra " | ..." suffix produced by the job
 * @returns {string}
 */
export function formatFrameSummary(summary, elapsed, tail = '') {
	const { frameNumber, neuronCount, maxTemporalLevel, maxSpatialLevel } = summary;

	// "N/A" rather than 0 when the counter is empty — zero is a real measurement
	// (a perfectly wrong predictor scores 0%) and shouldn't be confused with "no data".
	const accuracy = summary.accuracyTotal > 0
		? `${(summary.baseAccuracy * 100).toFixed(1)}%`
		: 'N/A';

	const reward = summary.rewardCount > 0
		? `${summary.avgReward.toFixed(3)} (${summary.rewardCount})`
		: 'N/A';

	const mape = summary.mapeCount > 0
		? `${summary.mape.toFixed(2)}% (${summary.mapeCount})`
		: 'N/A';

	// Job tail (portfolio P&L, etc.) goes between MAPE and Time so the brain-owned
	// numbers stay grouped on the left and the timing stays anchored on the right.
	const suffix = tail ? ` | ${tail}` : '';

	return `Frame ${frameNumber} | Neurons: ${neuronCount} (T${maxTemporalLevel} S${maxSpatialLevel}) | Accuracy: ${accuracy} | Reward: ${reward} | MAPE: ${mape}${suffix} | Time: ${elapsed.toFixed(2)}ms`;
}

/**
 * Build the --diagnostic "start of frame" header: frame number, incoming rewards,
 * and the raw observations being sensed. Input is the structured snapshot from
 * Brain.getStartFrameInfo() — raw ids translated to names via the supplied map.
 *
 * Returns null when there's nothing to show (empty frame) so the caller can skip.
 *
 * @param {object} info - { frameNumber, rewards: Map, frame: [], dimensionIdToName }
 * @returns {string|null}
 */
export function formatStartFrame(info) {
	if (!info) return null;
	const { frameNumber, rewards, frame, dimensionIdToName } = info;

	const lines = [];

	// Rewards landed BEFORE the inputs they're crediting — display them first so
	// the reader sees "what the brain just got paid" before "what it now sees".
	if (rewards && rewards.size > 0) {
		const parts = [];
		for (const [channelName, reward] of rewards)
			parts.push(`${channelName}:${reward.toFixed(3)}x`);
		lines.push(`  Rewards: ${parts.join(', ')}`);
	}

	// Resolve numeric dimIds back to human names via the supplied table; fall back
	// to `dimN` if a dim isn't registered (shouldn't happen, but never crash a debug log).
	const observations = [];
	for (const point of frame) {
		const { dimId, bucketId } = point.coordinate;
		const label = dimensionIdToName?.[dimId] ?? `dim${dimId}`;
		observations.push(`${label}=${bucketId}`);
	}

	lines.push(`\nF${frameNumber} | Obs: ${observations.join(', ')}`);
	return lines.join('\n');
}

/**
 * Render the per-frame vote dump. Takes the raw vote list emitted by the brain
 * (always populated, regardless of debug flag) plus the consensus inferences
 * (which name the winning neuron per dim), and walks per-channel/per-kind to
 * produce a multi-line dump.
 *
 * The `formatters` map is keyed by channelId and supplies label/coord
 * formatters per channel (encoders for the spec path, Channel instances for
 * the legacy path). When a channel has no formatter, an inline default is used
 * so the dump still surfaces — never silently dropped. `dimensionIdToName`
 * resolves numeric dimIds to human-readable names; missing entries fall back
 * to `dim<N>`.
 *
 * @param {Array<object>} votes - vote list from FrameResult
 * @param {Map<number, Array<object>>} inferences - consensus per channel, per dim
 * @param {Map<number, {name, formatActionLabel?, formatCoordinates?}>} formatters
 * @param {object} dimensionIdToName - { [dimId]: name }
 * @returns {string|null}
 */
export function formatVotes(votes, inferences, formatters, dimensionIdToName) {
	if (!votes || votes.length === 0) return null;

	// Pre-build a Set of winner neuron ids from inferences so the per-vote loops
	// can mark winners in O(1). Each dim's winner neuron id is on its WinnerOutput.
	const winnerIds = new Set();
	for (const dims of inferences.values())
		for (const dim of dims) winnerIds.add(dim.winner.neuronId);

	// Partition votes by channel once — event vs. action split happens inside each.
	const votesByChannel = new Map();
	for (const vote of votes) {
		if (!votesByChannel.has(vote.channelId)) votesByChannel.set(vote.channelId, []);
		votesByChannel.get(vote.channelId).push(vote);
	}

	const out = [`Collected ${votes.length} votes`];
	for (const [channelId, channelVotes] of votesByChannel) {
		// Fall back to a name-only stub so unknown channels still print rather than
		// disappearing — the renderer's job is to show data, not gate-keep it.
		const channel = formatters?.get(channelId) ?? { name: `channel${channelId}` };

		// Each channel gets up to two sections (events + actions). Either may be
		// absent (event-only channels like text), so skip nulls instead of pushing them.
		const eventOut = formatEventVotes(channelVotes, winnerIds, channel, dimensionIdToName);
		if (eventOut) out.push(eventOut);

		const actionOut = formatActionVotes(channelVotes, winnerIds, channel, dimensionIdToName);
		if (actionOut) out.push(actionOut);
	}
	return out.join('\n');
}

/**
 * Format event votes for one channel. Groups by event neuron (each a candidate
 * bucket on some dimension), then groups candidates by dimension so dimensions
 * compete amongst themselves. Inside each candidate, voters are shown sorted by
 * strength — mirroring how consensus resolution ranks them.
 */
function formatEventVotes(allVotes, winnerIds, channel, dimensionIdToName) {
	const eventVotes = allVotes.filter(v => v.targetType === 'event');
	if (eventVotes.length === 0) return null;

	// Two indexes off the same vote list: by-neuron for the per-candidate voter
	// breakdown, by-dimension for the outer "candidates competing in dim X" grouping.
	const votesByNeuron = groupVotesByNeuron(eventVotes, dimensionIdToName);

	const aggregatedByNeuron = new Map();
	for (const [neuronId, data] of votesByNeuron)
		aggregatedByNeuron.set(neuronId, aggregateVotesBySource(data.votes));

	const byDimension = groupByDimension(votesByNeuron);

	const lines = [`\n=== ${channel.name} EVENT VOTES ===`];
	for (const [dimName, candidates] of byDimension) {
		// Strongest candidate first — matches how consensus picks the winner so the
		// reader can scan top-down and stop reading once they've seen "why X won".
		candidates.sort((a, b) => b.totalStrength - a.totalStrength);
		lines.push(`  ${dimName} (${candidates.length} candidates):`);

		for (const candidate of candidates) {
			const marker = winnerIds.has(candidate.neuronId) ? '★ WINNER' : '';
			const coordsFormatted = formatCoordinates(candidate.coordsStr, channel);
			const aggVotes = aggregatedByNeuron.get(candidate.neuronId);
			lines.push(`    ${coordsFormatted} (n${candidate.neuronId}) str=${candidate.totalStrength.toFixed(1)} ${marker}`);
			// Empty label + reward=false: this is an event candidate's voter list,
			// no per-row reward to show (events aren't rewarded — actions are).
			lines.push(formatAggregatedVotes(aggVotes, '', false, channel));
		}
	}
	lines.push('===================\n');
	return lines.join('\n');
}

/**
 * Format action votes for one channel. Actions are grouped by label (the human-
 * readable name the channel gives each action bucket), and the winning label is
 * the one whose neuron group contains a winner id.
 */
function formatActionVotes(allVotes, winnerIds, channel, dimensionIdToName) {
	const actionVotes = allVotes.filter(v => v.targetType === 'action');
	if (actionVotes.length === 0) return null;

	// Group by human label (e.g. "OWN"/"OUT") — different neuron ids can map to the
	// same action bucket, and we want them aggregated under one heading per label.
	const actionGroups = groupActionsByLabel(actionVotes, channel, dimensionIdToName);
	const aggregatedByAction = new Map();
	const totalsByAction = new Map();

	for (const [label, votes] of actionGroups) {
		const aggregated = aggregateVotesBySource(votes);
		aggregatedByAction.set(label, aggregated);
		totalsByAction.set(label, calculateActionTotals(aggregated));
	}

	// Walk the winners back to the label they came from so the matching block
	// can be marked ★ — there's exactly one action winner per channel per frame.
	const winningLabel = findWinningActionLabel(actionGroups, winnerIds);

	const lines = [`\n=== ${channel.name} ACTION VOTES ===`];
	for (const [label, aggregated] of aggregatedByAction) {
		const total = totalsByAction.get(label);
		const winnerMarker = label === winningLabel ? ' ★' : '';
		const header = `${label} (${aggregated.length} voters, str=${total.str.toFixed(1)}, avgRwd=${total.rwd.toFixed(2)})${winnerMarker}`;
		// reward=true: action voters carry the rewards that drive selection, so each
		// row shows its avgRwd contribution to the label's score.
		lines.push(formatAggregatedVotes(aggregated, header, true, channel));
	}
	lines.push(`  SELECTION: ${winningLabel} (highest reward)`);
	lines.push('===================\n');
	return lines.join('\n');
}

/* ---------- vote grouping / aggregation helpers (pure data transforms) ---------- */

/**
 * Resolve a vote's target into a coordinate-like { dimension, value } using
 * the dimensionIdToName map. Falls back to "dim<N>" when a name is missing.
 */
function voteCoordinate(vote, dimensionIdToName) {
	return {
		dimension: dimensionIdToName?.[vote.dimId] ?? `dim${vote.dimId}`,
		value: vote.value,
	};
}

/**
 * Index votes by target neuron id, accumulating each candidate's total strength
 * and stashing a pre-built coords string + dimension name for downstream grouping.
 * Returned Map's iteration order = insertion order = order votes first appeared.
 */
function groupVotesByNeuron(votes, dimensionIdToName) {
	const votesByNeuron = new Map();
	for (const v of votes) {
		if (!votesByNeuron.has(v.targetId)) {
			const coord = voteCoordinate(v, dimensionIdToName);
			votesByNeuron.set(v.targetId, {
				neuronId: v.targetId,
				coordsStr: `${coord.dimension}=${coord.value}`,
				dimension: coord.dimension,
				votes: [],
				totalStrength: 0
			});
		}
		const data = votesByNeuron.get(v.targetId);
		data.votes.push(v);
		data.totalStrength += v.strength;
	}
	return votesByNeuron;
}

/**
 * Re-bucket the per-neuron index by dimension so the event view can show
 * "candidates competing within dim X" — each dimension is its own race.
 */
function groupByDimension(votesByNeuron) {
	const byDimension = new Map();
	for (const data of votesByNeuron.values()) {
		if (!byDimension.has(data.dimension)) byDimension.set(data.dimension, []);
		byDimension.get(data.dimension).push(data);
	}
	return byDimension;
}

/**
 * Group action votes by their human label. Channels with a formatActionLabel
 * supply readable names ("OWN"/"OUT"); without one we fall back to JSON of the
 * raw coordinate so the dump still partitions correctly even if it's ugly.
 */
function groupActionsByLabel(actionVotes, channel, dimensionIdToName) {
	const actionGroups = new Map();
	for (const v of actionVotes) {
		const coord = voteCoordinate(v, dimensionIdToName);
		const label = channel.formatActionLabel
			? channel.formatActionLabel(coord)
			: JSON.stringify(coord);
		if (!actionGroups.has(label)) actionGroups.set(label, []);
		actionGroups.get(label).push(v);
	}
	return actionGroups;
}

/**
 * Strength-weighted average reward across a label's aggregated voters. Mirrors
 * the brain's action consensus math: each voter's reward is weighted by the
 * strength it brought, and the label with the highest avgRwd wins.
 */
function calculateActionTotals(aggregated) {
	const total = {
		str: aggregated.reduce((s, a) => s + a.strength, 0),
		weightedRewardSum: aggregated.reduce((s, a) => s + a.weightedRewardSum, 0)
	};
	total.rwd = total.str > 0 ? total.weightedRewardSum / total.str : 0;
	return total;
}

/**
 * Reverse-lookup: which label did the winning neuron belong to? Throws if no
 * winner is found among the action groups — a brain that produced action votes
 * MUST have picked one, so absence is a real bug worth surfacing loudly.
 */
function findWinningActionLabel(actionGroups, winnerIds) {
	for (const [label, votes] of actionGroups)
		for (const vote of votes)
			if (winnerIds.has(vote.targetId)) return label;
	throw new Error('Cannot find winning action label');
}

/**
 * Aggregate votes by (source neuron, distance). Same neuron voting at two different
 * ages produces two rows so the multi-distance voting pattern stays visible; the
 * renderer doesn't collapse them. Strength-weighted average reward per row.
 */
function aggregateVotesBySource(votes) {
	if (votes.length === 0) return [];
	const bySource = new Map();
	for (const v of votes) {
		// Key by (voter, distance) not just voter: the same neuron can vote at two
		// different ages (e.g. d=1 and d=3) and we want both rows visible so the
		// multi-distance pattern stays readable in the dump.
		const key = `${v.voterId}:${v.distance}`;
		if (!bySource.has(key))
			bySource.set(key, { voterId: v.voterId, strength: 0, weightedRewardSum: 0, coords: v.voterLabel, level: v.voterTemporalLevel, distance: v.distance });
		const agg = bySource.get(key);
		agg.strength += v.strength;
		agg.weightedRewardSum += v.strength * v.reward;
	}
	// Second pass: divide weighted sum by total strength to get the per-row avgRwd.
	// Done after aggregation so we don't recompute on every vote we add.
	for (const [_, agg] of bySource)
		agg.reward = agg.strength > 0 ? agg.weightedRewardSum / agg.strength : 0;
	return [...bySource.values()];
}

/**
 * Render an aggregated voter list as indented lines. `label` adds a header line
 * and bumps indentation; `includeReward` toggles the avgRwd suffix (events skip
 * it, actions include it). Levels > 0 mark pattern neurons with [P] + L tag.
 */
function formatAggregatedVotes(aggVotes, label, includeReward, channel) {
	// Empty-state: still emit a "no votes" placeholder so missing data is visible
	// rather than the section silently collapsing into nothing.
	if (aggVotes.length === 0) {
		if (label) return `  ${label}: no votes`;
		return '    no votes';
	}
	const lines = label ? [`  ${label}:`] : [];
	for (const agg of aggVotes) {
		const coordsFormatted = formatCoordinates(agg.coords, channel);
		const rewardStr = includeReward ? `, avgRwd=${agg.reward.toFixed(2)}` : '';
		// Level/type tags only on pattern neurons (L>0); base-level voters get no
		// suffix to keep the common case visually quiet.
		const levelStr = agg.level > 0 ? ` L${agg.level}` : '';
		const typeStr = agg.level > 0 ? ' [P]' : '';
		lines.push(`    ${coordsFormatted}${levelStr}${typeStr} (d=${agg.distance}) → str=${agg.strength.toFixed(1)}${rewardStr}`);
	}
	return lines.join('\n');
}

/**
 * Defer to the channel's formatCoordinates if it has one (encoders attach bucket
 * percent ranges, etc.); otherwise pass the raw "dim=value" string through.
 * Empty input → "(no coords)" placeholder so debug dumps don't show dangling spaces.
 */
function formatCoordinates(coordsStr, channel) {
	if (!coordsStr) return '(no coords)';
	if (channel && typeof channel.formatCoordinates === 'function')
		return channel.formatCoordinates(coordsStr);
	return coordsStr;
}
