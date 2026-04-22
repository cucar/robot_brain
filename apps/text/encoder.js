import { Dimension } from '../../channels/dimension.js';

/**
 * Text encoder — owns the single character-input dimension and the cursor over a training
 * string. No trader: this app is event-only, the brain just learns to predict the next
 * character. Mirrors the StockEncoder shape (channelId binding, getChannelSpec,
 * setData/nextFrame/resetFrames, encode) so the Job can use the same registration and
 * frame-streaming pattern as stocks.
 */
export class TextEncoder {

	constructor(name = 'text', dimensions = null) {
		this.name = name;

		// channelId is null until the brain registers the spec and hands back an ID via
		// bindChannelId(); the job uses that ID as the key into the inputs Map.
		this.channelId = null;

		// Restore-from-database path reuses the Dimension instance (preserves its ID);
		// fresh-construction path creates a new instance whose ID the Thalamus assigns.
		this.initializeDimensions(dimensions);

		// Per-episode text + cursor; setData() loads, nextFrame() advances, resetFrames()
		// rewinds without dropping the loaded text.
		this.text = null;
		this.index = 0;
	}

	initializeDimensions(dimensions) {
		if (dimensions && dimensions.length > 0) {
			this.charDim = dimensions.find(d => d.name === `${this.name}_char`);
			if (!this.charDim)
				throw new Error(`TextEncoder ${this.name}: Missing required dimensions in database`);
		}
		else this.charDim = new Dimension(`${this.name}_char`);
	}

	/**
	 * Load the training string for an episode and reset the cursor. Called once per
	 * episode (after any iteration-repeat the Job wants to apply).
	 */
	setData(text) {
		this.text = text;
		this.index = 0;
	}

	/**
	 * Rewind cursor without dropping the loaded text — used by the Job between episodes
	 * so the brain re-sees the same sequence while preserving learned patterns.
	 */
	resetFrames() {
		this.index = 0;
	}

	/**
	 * Pull the next character from the training string. Returns null when the stream is
	 * exhausted. Carries both the char and its ASCII code so the Job can log either one.
	 */
	nextFrame() {
		if (this.text === null || this.index >= this.text.length) return null;
		const char = this.text[this.index++];
		return { char, charCode: char.charCodeAt(0) };
	}

	/**
	 * Translate a frame into the dim → bucket-ID map the brain expects. Passthrough mode
	 * means the ASCII code IS the bucket ID — no scalar→bucket conversion needed since
	 * char codes are already discrete integers in [0, 255].
	 * @returns {Map<number, number>} dimId → bucket ID
	 */
	encode(frame) {
		const dimMap = new Map();
		dimMap.set(this.charDim.id, frame.charCode);
		return dimMap;
	}

	/**
	 * Channel spec for brain.registerChannelSpec(). Single input dim, no actions, no
	 * reward — text is pure event-prediction. Resolution 256 covers all single-byte ASCII.
	 */
	getChannelSpec() {
		return {
			name: this.name,
			emitsReward: false,
			learnActionSequences: false,
			dimensions: [
				{
					dim: this.charDim,
					kind: 'input',
					resolution: 256,
					mode: 'passthrough'
				}
			]
		};
	}

	bindChannelId(channelId) {
		this.channelId = channelId;
	}
}
