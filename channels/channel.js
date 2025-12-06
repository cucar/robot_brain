/**
 * channels handle the interaction between the brain and a device. a device can be anything that has a consistent input/output mechanism.
 * examples of devices:
 * - eyes (input: visual data, output: eye movements - saccades)
 * - tongue (input: taste, output: tongue movements)
 * - ears (input: audio, output: ear movements - auriculomotor)
 * - arm/leg (input: touch, output: muscles)
 * - stocks (input: prices, output: buy/sell)
 * - text (input: characters, output: characters)
 * channels are like the adapters for the devices, similar to device drivers.
 * they do that by translating the streaming inputs and returning streaming outputs translated from brain action neurons.
 * this is the base class for all such channels. child classes implement different mediums like vision, audio, etc.
 */
export default class Channel {

	constructor(name) {
		this.name = name; // just for descriptions in debugging
		this.frameNumber = 0; // frame counter for channel-specific operations
		this.debug = false; // controls verbosity of channel output
		this.debug2 = false; // more detailed, verbose debug mode
		this.diagnostic = false; // diagnostic mode - shows detailed inference/conflict resolution info
		this.inferredActions = []; // actions selected by resolveConflicts
	}

	/**
	 * Initialize channel - override in subclasses if needed
	 * Called once during brain initialization
	 */
	async initialize() {
		// Default implementation does nothing
		// Child classes can override for channel-specific initialization
	}

	/**
	 * Execute outputs based on brain predictions - override in subclasses
	 * Invalid actions should be filtered during conflict resolution, so only valid actions should be received
	 */
	async executeOutputs() {
		throw new Error('Channel must implement executeOutputs() method');
	}

	/**
	 * Get event dimension names - override in subclasses
	 */
	getEventDimensions() {
		throw new Error('Channel must implement getEventDimensions() method');
	}

	/**
	 * Get state dimension names - override in subclasses - default implementation returns empty array
	 */
	getStateDimensions() {
		return [];
	}

	/**
	 * Get output dimension names - override in subclasses  
	 */
	getOutputDimensions() {
		throw new Error('Channel must implement getOutputDimensions() method');
	}

	/**
	 * Get frame events data - override in subclasses
	 * Returns array of input neuron objects: [{ [input-dim]: value }]
	 */
	async getFrameEvents() {
		throw new Error('Channel must implement getFrameEvents() method');
	}

	/**
	 * Get frame state data - override in subclasses
	 * Returns array of input neuron objects: [{ [input-dim]: value }]
	 */
	async getFrameState() {
		throw new Error('Channel must implement getFrameState() method');
	}

	/**
	 * Get inferred actions scheduled for current frame
	 * Returns actions selected by resolveConflicts in previous frame
	 */
	async getFrameOutputs() {
		if (this.inferredActions.length === 0) return [];
		const actions = this.inferredActions;
		this.inferredActions = [];
		return actions;
	}

	/**
	 * Get feedback based on previous actions and current state - override in subclasses
	 * Returns reward factor: number (1.0 = neutral, >1.0 = positive, <1.0 = negative)
	 * This factor will be multiplied with existing neuron reward factors
	 */
	async getRewards() {
		return 1.0; // Neutral feedback by default
	}

	/**
	 * Returns an exploration action based on current channel state
	 * Child classes should override this to provide context-aware exploration
	 * Examples:
	 * - Stock: Can't sell if not owned
	 * - Arm: Can't move beyond joint limits
	 * - Eyes: Can't saccade outside visual field
	 */
	getExplorationAction() {
		// Child classes must implement this with state-aware logic
		throw new Error('Channel must implement getExplorationAction() method');
	}

	/**
	 * Resolve conflicts between multiple inferred base neurons for this channel
	 * Base implementation calls getResolvedInference and stores result automatically
	 * Child classes override getResolvedInference, not this method
	 * @param {Array} inference - Array of inference objects with structure: [{ neuron_id, coordinates: {dim1: val1, dim2: val2}, strength: number }]
	 * @returns {Array} - Array of selected predictions to execute (can be 0, 1, or many)
	 */
	resolveConflicts(inference) {

		// ask the child class to resolve conflicts and return the selected neurons
		const resolvedInference = this.getResolvedInference(inference);

		// store only output neurons for next frame execution
		const outputDims = new Set(this.getOutputDimensions());
		const inferredOutput = resolvedInference.filter(pred => Object.keys(pred.coordinates).some(dim => outputDims.has(dim)));
		this.inferredActions = inferredOutput.map(pred => pred.coordinates);

		return resolvedInference;
	}

	/**
	 * returns event and action inferences
	 * @param {Array} inferences - all inferences
	 * @returns {Object} - { events, actions }
	 */
	getEventsAndActions(inferences) {
		const eventDims = new Set(this.getEventDimensions());
		const outputDims = new Set(this.getOutputDimensions());
		const events = inferences.filter(inf => Object.keys(inf.coordinates).some(dim => eventDims.has(dim)));
		const actions = inferences.filter(inf => Object.keys(inf.coordinates).some(dim => outputDims.has(dim)));
		return { events, actions };
	}

	/**
	 * Get resolved inference for the channel
	 * Resolves conflicts for both event predictions and action inferences
	 * Conflict resolution logic is implemented in channel subclasses
	 * State neurons exist only to help guide actions, so they are filtered out
	 * @param {Array} inferences - all inferred neurons for this channel
	 * @returns {Array} - resolved neurons according to channel specific logic
	 */
	getResolvedInference(inferences) {

		// if there are no inferences, nothing to resolve
		if (!inferences || inferences.length === 0) return [];

		// Separate into input predictions and output inferences
		const { events, actions } = this.getEventsAndActions(inferences);

		// Resolve event predictions: select strongest for each input dimension
		const resolvedEvents = this.resolveEventPredictions(events);

		// Resolve action inferences: select strongest action
		const resolvedActions = this.resolveActionInferences(actions);

		// Combine and return
		const resolved = [...resolvedEvents, ...resolvedActions];
		if (this.debug) console.log('getResolvedInference', resolved);
		if (this.debug) this.logResolution(inferences.length, events.length, actions.length, resolved.length);
		return resolved;
	}

	/**
	 * Log resolution results for debugging
	 */
	logResolution(totalCount, inputCount, outputCount, resolvedCount) {
		console.log(`${this.name}: Resolved ${totalCount} inferences (${inputCount} inputs, ${outputCount} outputs) → ${resolvedCount} selected`);
	}

	/**
	 * Display diagnostic information for inferences
	 * @param {Array} inferenceDetails - detailed inference info from brain.getInferenceDetails()
	 * @param {Array} resolvedInferences - resolved inferences after conflict resolution
	 */
	displayDiagnostics(inferenceDetails, resolvedInferences) {
		if (!this.diagnostic || inferenceDetails.length === 0) return;

		// Build a map of resolved neuron IDs for quick lookup
		const resolvedIds = new Set(resolvedInferences.map(inf => inf.neuron_id));

		// Group inferences by dimension for cleaner output
		const outputDims = new Set(this.getOutputDimensions());
		const eventDims = new Set(this.getEventDimensions());

		const outputs = [];
		const events = [];

		for (const inf of inferenceDetails) {
			const dimNames = Object.keys(inf.coordinates);
			const isOutput = dimNames.some(dim => outputDims.has(dim));
			const isEvent = dimNames.some(dim => eventDims.has(dim));

			if (isOutput) outputs.push(inf);
			else if (isEvent) events.push(inf);
		}

		// Display output inferences
		if (outputs.length > 0) {
			const parts = [];
			for (const inf of outputs) {
				const coordStr = this.formatCoordinates(inf.coordinates);
				const sourceStr = this.formatSources(inf.sources);
				const strength = inf.strength.toFixed(0);
				const resolved = resolvedIds.has(inf.neuron_id) ? '✓' : '✗';
				parts.push(`${coordStr}(${sourceStr} → ${strength}) ${resolved}`);
			}
			console.log(`  ${this.name} Actions: ${parts.join(' | ')}`);
		}

		// Display event predictions
		if (events.length > 0) {
			const parts = [];
			for (const inf of events) {
				const coordStr = this.formatCoordinates(inf.coordinates);
				const sourceStr = this.formatSources(inf.sources);
				const strength = inf.strength.toFixed(0);
				const resolved = resolvedIds.has(inf.neuron_id) ? '✓' : '✗';
				parts.push(`${coordStr}(${sourceStr}→${strength}) ${resolved}`);
			}
			console.log(`  ${this.name} Events: ${parts.join(' | ')}`);
		}
	}

	/**
	 * Format coordinates for diagnostic display
	 */
	formatCoordinates(coordinates) {
		const parts = [];
		for (const [dim, val] of Object.entries(coordinates)) {
			// Remove channel prefix for cleaner display
			const shortDim = dim.replace(`${this.name}_`, '');
			parts.push(`${shortDim}=${val}`);
		}
		return parts.join(',');
	}

	/**
	 * Format source information for diagnostic display
	 */
	formatSources(sources) {
		if (sources.length === 0) return 's:0';

		const parts = [];
		for (const src of sources) {
			if (src.type === 'connection' && src.sources) {
				const connParts = [];
				for (const conn of src.sources) {
					const s = conn.connection_strength.toFixed(0);
					const r = conn.connection_reward.toFixed(2);
					const h = conn.connection_habituation.toFixed(2);
					const eff = conn.prediction_strength.toFixed(0);
					connParts.push(`s${s}×r${r}×h${h}=${eff}`);
				}
				parts.push(`C:${connParts.join(' + ')}`);
			}
			else if (src.type === 'pattern' && src.sources) {
				const patternParts = [];
				for (const pat of src.sources) {
					const ps = pat.pattern_strength.toFixed(0);
					const pr = pat.pattern_reward.toFixed(2);
					const ph = pat.pattern_habituation.toFixed(2);
					const cs = pat.connection_strength.toFixed(0);
					const cr = pat.connection_reward.toFixed(2);
					const eff = pat.prediction_strength.toFixed(0);
					patternParts.push(`ps${ps}×pr${pr}×ph${ph}×cs${cs}×cr${cr}=${eff}`);
				}
				parts.push(`P:${patternParts.join('+')}`);
			}
		}
		return parts.join('+');
	}

	/**
	 * Get resolved event predictions for this channel - override in subclasses
	 */
	resolveEventPredictions() {
		throw new Error('Channel must implement resolveEventPredictions(events) method');
	}

	/**
	 * Resolve action inferences: select strongest for each output dimension
	 */
	resolveActionInferences() {
		throw new Error('Channel must implement resolveEventPredictions(actions) method');
	}

	/**
	 * Get output performance metrics - override in subclasses
	 * Returns channel-specific performance data (e.g., profit/loss, win rate, score, etc.)
	 * @returns {Object|null} - { value: number, label: string, format: string } or null
	 * Example: { value: 5.23, label: 'AAPL', format: 'currency' }
	 */
	getOutputPerformanceMetrics() {
		return null; // Default: no output performance tracking
	}
}