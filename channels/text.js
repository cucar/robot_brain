import Channel from './channel.js';

export default class TextChannel extends Channel {

	constructor(brain) {
		super(brain, ['char_code', 'position']); // Example dimensions
	}

	async processRawInput(textString) {
		// Logic to convert characters/words to neuron activations
		return [ { dimension_id: this.dimensionNameToId['char_code'], value: textString.charCodeAt(0) } ];
	}
}
