import fs from 'node:fs';
import path from 'node:path';

/**
 * BrainDump - Handles dumping brain state to JSON files for debugging and comparison
 */
export class BrainDump {

	/**
	 * Create a dump file with brain state data from Thalamus
	 * @param {Array} neurons - All neurons from Thalamus
	 * @param {Map} channels - All channels from Thalamus
	 * @param {Object} channelNameToId - Channel name to ID mapping
	 * @param {Object} dimensionNameToId - Dimension name to ID mapping
	 */
	createDumpFile(neurons, channels, channelNameToId, dimensionNameToId) {
		const dumpDir = path.join(process.cwd(), 'data', 'brain-dumps');
		if (!fs.existsSync(dumpDir)) fs.mkdirSync(dumpDir, { recursive: true });

		const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
		const filename = `brain-dump-${timestamp}`;

		const channelsData = this.collectChannelsData(channels, channelNameToId);
		const dimensionsData = this.collectDimensionsData(dimensionNameToId);
		const neuronsData = this.collectNeuronsData(neurons);

		const dump = {
			timestamp: new Date().toISOString(),
			channels: channelsData,
			dimensions: dimensionsData,
			neurons: neuronsData,
			neuronCount: neuronsData.length,
			nextNeuronId: neurons.length > 0 ? Math.max(...Array.from(neurons, n => n.id)) + 1 : 1
		};

		const filepath = path.join(dumpDir, `${filename}.json`);
		fs.writeFileSync(filepath, JSON.stringify(dump, null, 2));
		console.log(`   💾 Brain dump saved: ${filename}.json (${dump.neuronCount} neurons)`);

		return filepath;
	}

	/**
	 * Collect and format channels data
	 */
	collectChannelsData(channels, channelNameToId) {
		const channelsData = [];
		for (const [channelName, channel] of channels) {
			channelsData.push({
				name: channelName,
				id: channelNameToId[channelName],
				class: channel.constructor.name
			});
		}
		channelsData.sort((a, b) => a.id - b.id);
		return channelsData;
	}

	/**
	 * Collect and format dimensions data
	 */
	collectDimensionsData(dimensionNameToId) {
		const dimensionsData = [];
		for (const [name, id] of Object.entries(dimensionNameToId)) {
			dimensionsData.push({ name, id });
		}
		dimensionsData.sort((a, b) => a.id - b.id);
		return dimensionsData;
	}

	/**
	 * Collect and format neurons data
	 */
	collectNeuronsData(neurons) {
		const neuronsData = [];
		for (const neuron of neurons) {
			const neuronData = {
				id: neuron.id,
				level: neuron.level
			};

			if (neuron.level === 0) {
				neuronData.channel = neuron.channel;
				neuronData.type = neuron.type;
				neuronData.coordinates = neuron.coordinates;
			}

			if (neuron.level > 0 && neuron.peak) neuronData.peak = neuron.peak.id;

			neuronData.connections = this.collectConnectionsData(neuron);
			neuronData.patterns = this.collectPatternsData(neuron);
			neuronData.contextRefs = this.collectContextRefsData(neuron);
			neuronData.activationStrength = neuron.level === 0 ? 0 : neuron.activationStrength;

			neuronsData.push(neuronData);
		}
		neuronsData.sort((a, b) => a.id - b.id);
		return neuronsData;
	}

	/**
	 * Collect and format connections data for a neuron
	 */
	collectConnectionsData(neuron) {
		const connections = [];
		for (const [distance, distanceMap] of neuron.connections) {
			for (const [toNeuron, conn] of distanceMap) {
				connections.push({
					distance,
					toNeuronId: toNeuron.id,
					strength: conn.strength,
					reward: conn.reward || 0
				});
			}
		}
		connections.sort((a, b) => {
			if (a.toNeuronId !== b.toNeuronId) return a.toNeuronId - b.toNeuronId;
			return a.distance - b.distance;
		});
		return connections;
	}

	/**
	 * Collect and format patterns data for a neuron
	 */
	collectPatternsData(neuron) {
		const patterns = [];
		for (const pattern of neuron.patterns) patterns.push(pattern.id);
		patterns.sort((a, b) => a - b);
		return patterns;
	}

	/**
	 * Collect and format context references data for a neuron
	 */
	collectContextRefsData(neuron) {
		const contextRefs = [];
		for (const [contextNeuron, distances] of neuron.contextRefs) {
			contextRefs.push({
				neuronId: contextNeuron.id,
				distances: Array.from(distances).sort((a, b) => a - b)
			});
		}
		contextRefs.sort((a, b) => a.neuronId - b.neuronId);
		return contextRefs;
	}
}