import { Dimension } from '../../channels/dimension.js';

const POSITION_OWN = 1;
const POSITION_OUT = -1;

/**
 * Per-symbol stock encoder. Owns the channel and dimension names, the bucket boundaries,
 * and the raw-frame → scalar translation the brain's quantizer bucketizes. Channel and
 * dimension IDs are allocated by the Thalamus when the encoder's spec is registered —
 * the encoder reads them back via `bindChannelId()` and the Dimension instances it holds.
 */
export class StockEncoder {

	constructor(symbol, dimensions = null) {
		this.symbol = symbol;

		// channelId is null until the brain registers the spec and hands back an ID via
		// bindChannelId(); the trader then borrows the same ID so rewards, inputs, and
		// inferences all key off a single number per symbol.
		this.channelId = null;

		// Activity dim name is used both internally and by the trader for action coordinates.
		this.activityDimName = `${symbol}_activity`;

		// Dimensions are either reused from the database (restore path) or newly created.
		this.initializeDimensions(dimensions);

		// Bucket boundaries and the display map that turns bucket IDs into percent ranges.
		this.initializeBuckets();

		// Per-episode frame iteration state: the row slice being streamed, the cursor,
		// and the previous frame's reading (so each encode() call sees a full pair).
		this.rows = null;
		this.rowIndex = 0;
		this.previousPrice = null;
		this.previousVolume = null;
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
	 * Feed the encoder the chronological training rows for an episode and reset the
	 * frame iteration state. Called once per episode (after slicing for holdout/offset).
	 * @param {Array<{price:number, volume:number}>} rows
	 */
	setData(rows) {
		this.rows = rows;
		this.rowIndex = 0;
		this.previousPrice = null;
		this.previousVolume = null;
	}

	/**
	 * Pull the next frame's reading from the data source. Returns null when the stream
	 * is exhausted. The returned object carries both current and previous readings so
	 * encode() can be stateless — this lets the job consume the first frame as a warmup
	 * frame without entangling the encoder and the trader's step order.
	 * @returns {null | {price:number, volume:number, previousPrice:number|null, previousVolume:number|null}}
	 */
	nextFrame() {
		if (this.rows === null || this.rowIndex >= this.rows.length) return null;

		// Snapshot current (previous, current) pair before advancing the cursor.
		const row = this.rows[this.rowIndex++];
		const frame = {
			price: row.price,
			volume: row.volume,
			previousPrice: this.previousPrice,
			previousVolume: this.previousVolume
		};

		// Advance: the row we just emitted becomes the "previous" for the next call.
		this.previousPrice = row.price;
		this.previousVolume = row.volume;
		return frame;
	}

	/**
	 * Rewind frame iteration without dropping the loaded rows. Called once per episode
	 * so the brain re-sees the same sequence while preserving learned patterns.
	 */
	resetFrames() {
		this.rowIndex = 0;
		this.previousPrice = null;
		this.previousVolume = null;
	}

	/**
	 * Translate a frame into raw per-dimension scalars the brain's quantizer will
	 * bucketize. Returns null in two cases:
	 *   (1) first frame of a run — there is no previous reading yet, so no change
	 *       can be computed.
	 *   (2) the current frame has no trading data (price or volume is zero) — the
	 *       setup path backfills missing bars with zero volume; we skip those.
	 * @returns {Map<number, number>|null} dimId → raw scalar percentage change
	 */
	encode(frame) {
		if (frame.previousPrice === null || frame.previousVolume === null) return null;
		if (!frame.price || !frame.volume) return null;

		// Percent change since last frame; volume jumps from a zero-volume bar are
		// capped at 1000% to avoid infinities when the previous frame had no trades.
		const priceChange = ((frame.price - frame.previousPrice) / frame.previousPrice) * 100;
		const volumeChange = frame.previousVolume === 0 ? 1000 : ((frame.volume - frame.previousVolume) / frame.previousVolume) * 100;

		// Map of dimId → raw scalar; the brain's quantizer bucketizes these per-dim
		// according to the registered static boundaries.
		const dimMap = new Map();
		dimMap.set(this.priceChangeDim.id, priceChange);
		dimMap.set(this.volumeChangeDim.id, volumeChange);
		return dimMap;
	}

	/**
	 * Convert a (possibly fractional) bucket value back to an approximate percentage change.
	 * Midpoint of the bucket range, with open-ended outer buckets reflected.
	 */
	bucketValueToPercentage(bucketValue) {
		const { lo, hi } = this.getBucketRange(bucketValue, this.priceBoundaries);
		const loVal = lo === -Infinity ? hi - Math.abs(hi || 1) * 2 : lo;
		const hiVal = hi === Infinity ? lo + Math.abs(lo || 1) * 2 : hi;
		return (loVal + hiVal) / 2;
	}

	/**
	 * Channel name as far as the brain/thalamus is concerned. Used by the host-side
	 * vote-debug renderer to label per-channel sections; matches the spec name.
	 */
	get name() { return this.symbol; }

	/**
	 * Renderer hook: humanize an action coordinate into OWN/OUT for the vote-debug
	 * dump. Coordinate.value is the action bucket (POSITION_OWN/POSITION_OUT). Falls
	 * back to a JSON dump for unrecognized values so unknown actions still surface.
	 */
	formatActionLabel(coordinate) {
		if (coordinate.value === POSITION_OWN) return 'OWN';
		if (coordinate.value === POSITION_OUT) return 'OUT';
		return JSON.stringify(coordinate);
	}

	/**
	 * Renderer hook: turn a "dim=bucket, dim=bucket" coords string into one with
	 * the percent ranges appended (e.g. "AAPL_price_change=3(+0.50%~+1.00%)") so
	 * the vote dump shows what bucket numbers actually mean. Cross-channel voters
	 * are matched by dimension suffix since all StockEncoders share boundaries.
	 */
	formatCoordinates(coordsStr) {
		if (!coordsStr) return '(no coords)';
		if (!this.bucketToPercent) return coordsStr;
		return coordsStr.split(', ').map(part => {
			const [dimName, valStr] = part.split('=');
			const val = parseFloat(valStr);
			const key = `${dimName}:${val}`;
			let percentRange = this.bucketToPercent.get(key);
			// Fall back to matching by dimension suffix for cross-channel voters
			// (all StockEncoders share identical boundaries so this is accurate).
			if (!percentRange) {
				const underscoreIdx = dimName.indexOf('_');
				if (underscoreIdx >= 0) {
					const suffix = dimName.substring(underscoreIdx);
					percentRange = this.bucketToPercent.get(`${this.symbol}${suffix}:${val}`);
				}
			}
			if (percentRange) return `${dimName}=${val}(${percentRange})`;
			return part;
		}).join(', ');
	}

	/**
	 * Describe this encoder's channel for brain.registerChannelSpec(). Shape-only —
	 * no behavior. Each dim spec carries the Dimension instance; the Thalamus allocates
	 * an ID and writes it onto the instance in place. The brain stores the spec, registers
	 * dims with the quantizer, and pre-creates action neurons for the activity dim. The
	 * caller passes the returned channelId to bindChannelId().
	 */
	getChannelSpec() {
		return {
			name: this.symbol,
			emitsReward: true,
			learnActionSequences: false,
			dimensions: [
				{
					dim: this.priceChangeDim,
					kind: 'input',
					resolution: this.priceBoundaries.length + 1,
					mode: 'static',
					boundaries: [...this.priceBoundaries]
				},
				{
					dim: this.volumeChangeDim,
					kind: 'input',
					resolution: this.volumeBoundaries.length + 1,
					mode: 'static',
					boundaries: [...this.volumeBoundaries]
				},
				{
					dim: this.activityDim,
					kind: 'action',
					resolution: 2,
					mode: 'passthrough',
					actionBuckets: [ POSITION_OUT, POSITION_OWN ],
					defaultBucket: POSITION_OUT
				}
			]
		};
	}

	/**
	 * Called after brain.registerChannelSpec() returns the allocated channel ID. Dimension
	 * IDs are written onto the Dimension instances in place by the Thalamus, so the encoder
	 * just needs to record its channelId here.
	 */
	bindChannelId(channelId) {
		this.channelId = channelId;
	}
}
