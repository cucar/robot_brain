/**
 * Neuron - Abstract base class for all neurons
 * Contains properties common to both sensory and pattern neurons
 */
export class Neuron {

	constructor(level = 0) {
		this.level = level;

		// Count of incoming references (connections, pattern_peaks, pattern_past, pattern_future)
		// Used for orphan detection - when this reaches 0 and neuron has no outgoing, it can be deleted
		this.incomingCount = 0;
	}
}
