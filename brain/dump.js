import fs from 'node:fs';
import path from 'node:path';

/**
 * Handles dumping brain state to JSON files for debugging and comparison
 */
export class Dump {

	/**
	 * Save a brain snapshot to a JSON dump file.
	 * @param {Object} snapshot - Brain state snapshot from thalamus.getSnapshot()
	 */
	saveSnapshot(snapshot) {
		const dumpDir = path.join(process.cwd(), 'data', 'brain-dumps');
		if (!fs.existsSync(dumpDir)) fs.mkdirSync(dumpDir, { recursive: true });

		const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
		const filename = `brain-dump-${timestamp}`;

		const channelsData = this.collectChannelsData(snapshot.channels, snapshot.channelNameToId);
		const dimensionsData = this.collectDimensionsData(snapshot.dimensionNameToId);
		const neuronsData = this.collectNeuronsData(snapshot.neurons);

		const dump = {
			timestamp: new Date().toISOString(),
			channels: channelsData,
			dimensions: dimensionsData,
			neurons: neuronsData,
			neuronCount: neuronsData.length,
			nextNeuronId: snapshot.neurons.length > 0 ? Math.max(...snapshot.neurons.map(e => e.neuron.id)) + 1 : 1
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
		for (const { neuron, channel, type } of neurons) {
			const neuronData = {
				id: neuron.id,
				level: neuron.level
			};

			if (neuron.level === 0) {
				neuronData.channel = channel;
				neuronData.type = type;
				neuronData.coordinates = neuron.coordinates;
			}

			if (neuron.level > 0 && neuron.peak) neuronData.peak = neuron.peak.id;

			neuronData.connections = this.collectConnectionsData(neuron);
			neuronData.children = this.collectPatternsData(neuron);
			neuronData.contextRefs = this.collectContextRefsData(neuron);
			neuronData.patternContext = this.collectPatternContextData(neuron);
			neuronData.activationStrength = neuron.activationStrength;

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
			for (const [toNeuronId, conn] of distanceMap) {
				connections.push({
					distance,
					toNeuronId,
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
		for (const patternId of neuron.children) patterns.push(patternId);
		patterns.sort((a, b) => a - b);
		return patterns;
	}

	/**
	 * Collect and format context references data for a neuron
	 */
	collectContextRefsData(neuron) {
		const contextRefs = [];
		for (const [neuronId, distances] of neuron.contextRefs) {
			contextRefs.push({
				neuronId,
				distances: Array.from(distances).sort((a, b) => a - b)
			});
		}
		contextRefs.sort((a, b) => a.neuronId - b.neuronId);
		return contextRefs;
	}

	/**
	 * Collect and format pattern context data for a pattern neuron
	 */
	collectPatternContextData(neuron) {
		const patternContext = [];
		for (const { neuronId, distance, strength } of neuron.getPatternContext()) {
			patternContext.push({
				neuronId,
				distance,
				strength
			});
		}
		patternContext.sort((a, b) => {
			if (a.neuronId !== b.neuronId) return a.neuronId - b.neuronId;
			return a.distance - b.distance;
		});
		return patternContext;
	}
}