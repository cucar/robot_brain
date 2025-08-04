import Channel from './channel.js';

export default class VisionChannel extends Channel {
	constructor(brain) {
		super(brain, ['x', 'y', 'r', 'g', 'b']);
	}

	/**
	 * Processes a "pixel" input. In a real system, this would parse an image frame.
	 * For simplicity, we assume `pixelData` is already discretized.
	 * @param {Object} pixelData - Example: { x: 10, y: 20, r: 255, g: 0, b: 0 }
	 * @returns {Array<Object>} - Discretized coordinate pairs.
	 */
	async processRawInput(pixelData) {
		// Here, you'd implement logic to, e.g., sample pixels from a JPG/MPEG frame,
		// quantize color values if they are continuous, etc.
		// For this example, we directly use the provided pixelData as discretized coordinates.
		const coords = [];
		for (const dimName in pixelData) {
			if (this.dimensionNameToId[dimName] !== undefined) {
				coords.push({
					dimension_id: this.dimensionNameToId[dimName],
					value: pixelData[dimName]
				});
			}
		}
		return coords;
	}
}