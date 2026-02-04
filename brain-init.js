import { Neuron } from './neurons/neuron.js';

/**
 * BrainInit - Handles initialization logic for Brain
 * Returns data structures that Brain uses to update its own state
 */
export class BrainInit {
	constructor(debug) {
		this.debug = debug;
	}

	/**
	 * Register a channel and return the channel instance
	 * @param {string} name - Channel name
	 * @param {Channel} channelClass - Channel class constructor
	 * @returns {object} - Channel instance
	 */
	registerChannel(name, channelClass) {
		const channel = new channelClass(name);
		if (this.debug) console.log(`Registered channel: ${name} (${channelClass.name})`);
		return channel;
	}

	/**
	 * Initialize channels and return mappings
	 * @param {Map} channels - Map of channel name to channel instance
	 * @returns {object} - { channelNameToId, channelIdToName }
	 */
	initializeChannels(channels) {
		const channelNameToId = {};
		const channelIdToName = {};

		for (const [channelName, channel] of channels) {
			channelNameToId[channelName] = channel.id;
			channelIdToName[channel.id] = channelName;
		}

		if (this.debug) console.log('Channels loaded:', channelNameToId);
		return { channelNameToId, channelIdToName };
	}

	/**
	 * Load dimensions and return mappings
	 * @param {Map} channels - Map of channel name to channel instance
	 * @returns {object} - { dimensionNameToId, dimensionIdToName }
	 */
	loadDimensions(channels) {
		const dimensionNameToId = {};
		const dimensionIdToName = {};

		for (const [, channel] of channels) {
			for (const dim of channel.getEventDimensions()) {
				dimensionNameToId[dim.name] = dim.id;
				dimensionIdToName[dim.id] = dim.name;
			}
			for (const dim of channel.getOutputDimensions()) {
				dimensionNameToId[dim.name] = dim.id;
				dimensionIdToName[dim.id] = dim.name;
			}
		}

		if (this.debug) console.log('Dimensions loaded:', dimensionNameToId);
		return { dimensionNameToId, dimensionIdToName };
	}

	/**
	 * Pre-create action neurons for all channels
	 * @param {Map} channels - Map of channel name to channel instance
	 * @param {object} channelNameToId - Channel name to ID mapping
	 * @param {Map} neurons - Neurons map (to add new neurons to)
	 * @param {Map} neuronsByValue - Neurons by value map (to add new neurons to)
	 * @returns {Map} - channelActions map (channel name -> Set of action neurons)
	 */
	initializeActionNeurons(channels, channelNameToId, neurons, neuronsByValue) {
		const channelActions = new Map();

		for (const [channelName, channel] of channels) {
			const actionCoords = channel.getActionNeurons();
			if (actionCoords.length === 0) continue;

			// Build frame points for action neurons
			const framePoints = actionCoords.map(coords => ({
				coordinates: coords,
				channel: channelName,
				channel_id: channelNameToId[channelName],
				type: 'action'
			}));

			// Create neurons (using same logic as getFrameNeurons)
			const actionNeurons = new Set();
			for (const point of framePoints) {
				const valueKey = Neuron.makeValueKey(point.coordinates);
				let neuron = neuronsByValue.get(valueKey);

				if (!neuron) {
					neuron = Neuron.createSensory(point.channel, point.type, point.coordinates);
					neurons.set(neuron.id, neuron);
					neuronsByValue.set(valueKey, neuron);
					if (this.debug) console.log(`Created new sensory neuron ${neuron.id} for ${valueKey}`);
				}

				actionNeurons.add(neuron);
			}

			channelActions.set(channelName, actionNeurons);
			if (this.debug) console.log(`Created ${actionCoords.length} action neurons for ${channelName}`);
		}

		return channelActions;
	}

	/**
	 * Initialize all channels (channel-specific setup)
	 * @param {Map} channels - Map of channel name to channel instance
	 */
	async initializeAllChannels(channels) {
		for (const [, channel] of channels)
			await channel.initialize();
	}
}

