/**
 * channels handle conversion from different input mediums to a standard normalized input for the brain between 0 and 1.
 * this is the base class for all such channels. child classes implement different mediums like vision, audio, etc.
 * this is not currently implemented yet. the code regarding channels is just scratch code.
 */
export default class Channel {
	constructor(brain, dimensions) {
		this.brain = brain; // Reference to the Brain instance
		this.dimensions = dimensions; // Array of dimension names relevant to this channel, e.g., ['x', 'y', 'r', 'g', 'b']
		this.dimensionNameToId = {}; // Populated in init
	}

	/**
	 * create dimensions in the brain if not created before
	 */
	async init() {
		// Load dimension IDs from the brain for this channel's dimensions
		const dimIdMap = await this.brain.getDimensionIdMap();
		this.dimensions.forEach(dimName => {
			if (dimIdMap[dimName] === undefined) {
				console.warn(`Dimension '${dimName}' for channel not found in database.`);
				// In a real system, you might create it or throw an error.
			}
			this.dimensionNameToId[dimName] = dimIdMap[dimName];
		});
		console.log(`Channel initialized with dimensions: ${JSON.stringify(this.dimensionNameToId)}`);
	}

	/**
	 * Translates raw input data (e.g., pixel {x, y, r, g, b}) into
	 * an array of coordinate objects for activation.
	 * @param {Object} inputData - Raw input, e.g., { x: 10, y: 20, r: 255, g: 0, b: 0 }
	 * @returns {Array<Object>} - Array of { dimension_id, value } for the input.
	 */
	processInput(inputData) {
		const inputCoords = [];
		for (const dimName in inputData) {
			if (this.dimensionIds[dimName] !== undefined) {
				inputCoords.push({
					dimension_id: this.dimensionIds[dimName],
					value: inputData[dimName]
				});
			} else {
				console.warn(`Warning: Input dimension '${dimName}' not found in known dimensions.`);
			}
		}
		return inputCoords;
	}

	/**
	 * Abstract method: Processes raw input data into discretized coordinate pairs.
	 */
	async processRawInput(rawInput) {
		throw new Error("processRawInput must be implemented by child classes.");
	}
}