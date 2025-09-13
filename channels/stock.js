import Channel from './channel.js';
import { createReadStream, mkdirSync } from 'node:fs';
import { createInterface } from 'node:readline';
import path from 'node:path';

/**
 * Stock Channel Implementation - this channel is used for buying/selling stocks based on their values
 */
export default class StockChannel extends Channel {

	constructor(name) {
		super(name);

		// Extract symbol from name (e.g., "AAPL" from name)
		this.symbol = name;

		// State tracking
		this.owned = false; // Simple owned flag
		this.entryPrice = null; // Price when we bought
		this.holdingFrames = 0; // How long we've held the position
		this.previousPrice = null; // Track previous price for change calculation
		this.previousVolume = null; // Track previous volume for change calculation

		// CSV reading state
		this.csvPath = null;
		this.rl = null;
		this.lineIterator = null;
		this.currentPrice = null;
		this.currentVolume = null;

		// Exponential discretization buckets for percentage changes
		this.priceBuckets = [
			{ min: -Infinity, max: -50, value: -7 },  // -100%+ to -50%
			{ min: -50, max: -25, value: -6 },        // -50% to -25%
			{ min: -25, max: -12.5, value: -5 },      // -25% to -12.5%
			{ min: -12.5, max: -6.25, value: -4 },    // -12.5% to -6.25%
			{ min: -6.25, max: -3.125, value: -3 },   // -6.25% to -3.125%
			{ min: -3.125, max: -1.5625, value: -2 }, // -3.125% to -1.5625%
			{ min: -1.5625, max: -0.05, value: -1 },  // -1.5625% to -0.05%
			{ min: -0.05, max: 0.05, value: 0 },      // -0.05% to 0.05% (no change)
			{ min: 0.05, max: 1.5625, value: 1 },     // 0.05% to 1.5625%
			{ min: 1.5625, max: 3.125, value: 2 },    // 1.5625% to 3.125%
			{ min: 3.125, max: 6.25, value: 3 },      // 3.125% to 6.25%
			{ min: 6.25, max: 12.5, value: 4 },       // 6.25% to 12.5%
			{ min: 12.5, max: 25, value: 5 },         // 12.5% to 25%
			{ min: 25, max: 50, value: 6 },           // 25% to 50%
			{ min: 50, max: Infinity, value: 7 }      // 50%+ to 100%+
		];
	}

	/**
	 * initialize this stock channel: open CSV and prepare iterator
	 */
	async initialize() {
		const baseDir = path.resolve(process.cwd(), 'data', 'stock');
		try { mkdirSync(baseDir, { recursive: true }); } catch {}
		this.csvPath = path.resolve(baseDir, `${this.symbol}.csv`);
		this.rl = createInterface({ input: createReadStream(this.csvPath), crlfDelay: Infinity });
		this.lineIterator = this.rl[Symbol.asyncIterator]();
	}

	/**
	 * Reads the next non-empty CSV line, parses it as { price, volume }.
	 * Returns null on EOF. Throws on malformed lines.
	 */
	async readNextLine() {
		if (!this.lineIterator) return null;
		const { value, done } = await this.lineIterator.next();
        if (done || value === undefined) return null;
        const parts = String(value).trim().split(',');
        if (parts.length < 2) throw new Error(`${this.symbol}: Invalid CSV line: ${value}`);
        const price = parseFloat(parts[0]);
        const volume = parseFloat(parts[1]);
        if (Number.isNaN(price) || Number.isNaN(volume)) throw new Error(`${this.symbol}: Invalid CSV values: '${trimmed}'`);
        return { price, volume };
	}

	/**
	 * Compute discretized change inputs from previous → current and update state.
	 */
	computeChangeInputs() {
		const priceChange = ((this.currentPrice - this.previousPrice) / this.previousPrice) * 100;
		const volumeChange = ((this.currentVolume - this.previousVolume) / this.previousVolume) * 100;
		console.log(`${this.symbol}: Price: ${this.currentPrice} (${priceChange.toFixed(2)}%), Volume: ${this.currentVolume} (${volumeChange.toFixed(2)}%)`);
		this.previousPrice = this.currentPrice;
		this.previousVolume = this.currentVolume;
		return [
			{ [`${this.symbol}_price_change`]: this.discretizePercentageChange(priceChange) },
			{ [`${this.symbol}_volume_change`]: this.discretizePercentageChange(volumeChange) }
		];
	}

	/**
	 * returns the input dimensions for the channel
	 */
	getInputDimensions() {
		return [
			`${this.symbol}_price_change`, // Discretized percentage change in price
			`${this.symbol}_volume_change` // Discretized percentage change in volume
		];
	}

	/**
	 * returns the output dimensions for the channel
	 */
	getOutputDimensions() {
		return [
			`${this.symbol}_activity` // -1 for sell, +1 for buy
		];
	}

	/**
	 * Discretize percentage change into exponential buckets
	 */
	discretizePercentageChange(percentChange) {
		for (const bucket of this.priceBuckets)
			if (percentChange > bucket.min && percentChange <= bucket.max)
				return bucket.value;
		return 0; // Default to no change bucket
	}

	/**
	 * Get valid exploration actions based on current stock ownership
	 * Can't sell if not owned, can't buy if already owned
	 */
	getValidExplorationActions() {
		return [{ [`${this.symbol}_activity`]: (this.owned ? -1 : 1) }]
	}

	/**
	 * Get frame input data for this stock channel
	 */
	async getFrameInputs() {
		
		// Read next non-empty line; skip header if present
		const row = await this.readNextLine();
		if (!row) {
			console.log(`${this.symbol}: No more data available`);
			return [];
		}
		this.currentPrice = row.price;
		this.currentVolume = row.volume;

		// if this is the first frame, read another line so that we can start sending change stats
		if (this.previousPrice === null || this.previousVolume === null) {
			
			// seed baseline from the first observed line
			this.previousPrice = this.currentPrice;
			this.previousVolume = this.currentVolume;

			// read the next data line
			const nextRow = await this.readNextLine();
			this.currentPrice = nextRow.price;
			this.currentVolume = nextRow.volume;

			// compute and return discretized changes based on baseline → next
			return this.computeChangeInputs();
		}
		
		// Increment holding counter if we own the stock
		if (this.owned) this.holdingFrames++;

		// Compute and return discretized changes
		return this.computeChangeInputs();
	}

	/**
	 * Get feedback neurons based on position and price movement
	 */
	async getFeedbackNeurons() {

		// No feedback if we don't own stock
		if (!this.owned) return [];

		// Use last observed price
		const currentPrice = this.currentPrice;
		if (currentPrice === null || this.entryPrice === null) return [];
		const priceChange = currentPrice - this.entryPrice;
		const percentChange = (priceChange / this.entryPrice) * 100;

		let feedbackValue = 0;
		if (priceChange > 0) {
			feedbackValue = 1; // Joy - stock went up since we bought it
			console.log(`${this.symbol}: JOY! Stock up ${percentChange.toFixed(2)}% since purchase (${priceChange.toFixed(2)} profit)`);
		} 
		else if (priceChange < 0) {
			feedbackValue = -1; // Pain - stock went down since we bought it
			console.log(`${this.symbol}: PAIN! Stock down ${percentChange.toFixed(2)}% since purchase (${Math.abs(priceChange).toFixed(2)} loss)`);
		}

		return feedbackValue === 0 ? [] : [{ [`${this.symbol}_reward`]: feedbackValue }];
	}

	/**
	 * executes output neurons returned by the brain after processing the frame
	 */
	executeOutputs(predictions) {
		const outputs = {
			actions: new Map(),
			predictions: new Map()
		};

		if (!predictions || predictions.length === 0) return;

		// Extract activity predictions from all prediction frames
		predictions.forEach(frame => {
			frame.predictions.forEach(pred => {
				// Check for activity dimension (our new output format)
				if (pred.coordinates[`${this.symbol}_activity`] !== undefined) {
					const activityValue = pred.coordinates[`${this.symbol}_activity`];
					const confidence = pred.confidence;
					
					if (activityValue > 0) {
						// Positive activity = buy signal
						const current = outputs.actions.get('buy') || 0;
						outputs.actions.set('buy', current + confidence);
					}
					else if (activityValue < 0) {
						// Negative activity = sell signal
						const current = outputs.actions.get('sell') || 0;
						outputs.actions.set('sell', current + confidence);
					}
				}
			});
		});

		// Execute the strongest action (fake buying/selling)
		this.executeAction(outputs.actions);
	}

	executeAction(actions) {
		let strongestAction = null;
		let maxStrength = 0;

		for (const [action, strength] of actions) {
			if (strength > maxStrength) {
				maxStrength = strength;
				strongestAction = action;
			}
		}

		if (strongestAction && maxStrength > 0) {

			// Get current price for position tracking
			const currentPrice = this.currentPrice;

			if (strongestAction === 'buy') {
				if (!this.owned) {
					this.owned = true;
					this.entryPrice = currentPrice;
					this.holdingFrames = 0; // Reset holding counter
					console.log(`${this.symbol}: EXECUTED BUY at $${currentPrice} (strength: ${maxStrength.toFixed(3)})`);
				}
			} else if (strongestAction === 'sell') {
				if (this.owned) {
					// Sell owned stock
					const profit = currentPrice - this.entryPrice;
					const percentReturn = (profit / this.entryPrice) * 100;
					
					console.log(`${this.symbol}: EXECUTED SELL at $${currentPrice} (strength: ${maxStrength.toFixed(3)})`);
					console.log(`${this.symbol}: Profit/Loss: $${profit.toFixed(2)} (${percentReturn.toFixed(2)}%) over ${this.holdingFrames} frames`);
					
					this.owned = false;
					this.entryPrice = null;
					this.holdingFrames = 0; // Reset holding counter
				}
			}
		}
	}
}