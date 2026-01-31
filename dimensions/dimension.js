/**
 * Dimension represents a coordinate axis for neurons.
 * Each dimension has a unique ID (from static counter) and a name (for debugging).
 * Channels create and own their Dimension objects.
 */
export class Dimension {

	static nextId = 1; // Start at 1 to match typical DB conventions

	constructor(name) {
		this.id = Dimension.nextId++;
		this.name = name; // for debugging
	}
}