import { Dimension } from '../../channels/dimension.js';

/**
 * Per-symbol stock encoder. Owns the dimensions, bucket boundaries, and change
 * discretization used to translate raw ticks into brain inputs.
 */
export class StockEncoder {

	constructor(symbol, dimensions = null) {
		this.symbol = symbol;
		this.activityDimName = `${symbol}_activity`;
		this.initializeDimensions(dimensions);
		this.initializeBuckets();
	}

	initializeDimensions(dimensions) {
		if (dimensions && dimensions.length > 0) {
			this.priceChangeDim = dimensions.find(d => d.name === `${this.symbol}_price_change`);
			this.volumeChangeDim = dimensions.find(d => d.name === `${this.symbol}_volume_change`);
			this.activityDim = dimensions.find(d => d.name === this.activityDimName);
			if (!this.priceChangeDim || !this.volumeChangeDim || !this.activityDim)
				throw new Error(`StockEncoder ${this.symbol}: Missing required dimensions in database`);
		}
		else {
			this.priceChangeDim = new Dimension(`${this.symbol}_price_change`);
			this.volumeChangeDim = new Dimension(`${this.symbol}_volume_change`);
			this.activityDim = new Dimension(this.activityDimName);
		}
	}

	initializeBuckets() {
		this.priceBoundaries = [0];
		this.volumeBoundaries = [0];
		this.bucketToPercent = this.buildBucketPercentMap();
	}

	buildBucketPercentMap() {
		const map = new Map();
		const categories = [
			{ dim: `${this.symbol}_price_change`,  boundaries: this.priceBoundaries },
			{ dim: `${this.symbol}_volume_change`, boundaries: this.volumeBoundaries }
		];
		for (const { dim, boundaries } of categories)
			for (let bucket = 1; bucket <= boundaries.length + 1; bucket++) {
				const { lo, hi } = this.getBucketRange(bucket, boundaries);
				map.set(`${dim}:${bucket}`, this.formatBucketRange(lo, hi));
			}
		return map;
	}

	getBucketRange(bucketValue, boundaries) {
		const idx = bucketValue - 1;
		const lo = idx === 0 ? -Infinity : boundaries[idx - 1];
		const hi = idx >= boundaries.length ? Infinity : boundaries[idx];
		return { lo, hi };
	}

	formatBucketRange(min, max) {
		if (min === -Infinity) return `<${max.toFixed(2)}%`;
		if (max === Infinity) return `>${min.toFixed(2)}%`;
		return `${min.toFixed(2)}%~${max.toFixed(2)}%`;
	}

	discretizeChange(value, boundaries) {
		for (let i = 0; i < boundaries.length; i++)
			if (value <= boundaries[i]) return i + 1;
		return boundaries.length + 1;
	}

	/**
	 * Compute discretized change inputs from previous → current tick data.
	 * @returns {{priceChange:number, volumeChange:number, inputs:Array<{dimension:string,value:number}>}}
	 */
	encode({ previousPrice, currentPrice, previousVolume, currentVolume }) {
		const priceChange = ((currentPrice - previousPrice) / previousPrice) * 100;
		const volumeChange = previousVolume === 0 ? 1000 : ((currentVolume - previousVolume) / previousVolume) * 100;
		const inputs = [
			{ dimension: `${this.symbol}_price_change`,  value: this.discretizeChange(priceChange,  this.priceBoundaries) },
			{ dimension: `${this.symbol}_volume_change`, value: this.discretizeChange(volumeChange, this.volumeBoundaries) }
		];
		return { priceChange, volumeChange, inputs };
	}

	/**
	 * Convert discretized bucket value back to approximate percentage change (midpoint of range).
	 */
	bucketValueToPercentage(bucketValue) {
		const { lo, hi } = this.getBucketRange(bucketValue, this.priceBoundaries);
		const loVal = lo === -Infinity ? hi - Math.abs(hi || 1) * 2 : lo;
		const hiVal = hi === Infinity ? lo + Math.abs(lo || 1) * 2 : hi;
		return (loVal + hiVal) / 2;
	}

	/**
	 * Describe the channel shape the brain should register for this symbol.
	 */
	getChannelSpec() {
		return {
			symbol: this.symbol,
			activityDimName: this.activityDimName,
			eventDims:  [ this.priceChangeDim, this.volumeChangeDim ],
			actionDims: [ this.activityDim ]
		};
	}
}
