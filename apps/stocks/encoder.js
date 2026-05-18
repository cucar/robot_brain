const POSITION_OWN = 1;
const POSITION_OUT = -1;

/**
 * Per-symbol stock encoder. Owns the channel and dimension names, the bucket boundaries,
 * and the raw-frame → scalar translation the brain's quantizer bucketizes. Channel and
 * dimension IDs are allocated by the Thalamus when the encoder's spec is registered —
 * the encoder reads them back via `bindIds()`.
 */
export class StockEncoder {

	// emulated transaction cost - not used in live trading - historical data
	// does not have bid/ask, so we use this to emulate the spread during training
	static transactionCost = 0;

	/**
	 * Create a new encoder for the given ticker symbol. Allocates dimension
	 * names and bucket boundaries but leaves IDs null until bindIds() is
	 * called after brain registration.
	 */
	constructor(symbol) {
		this.symbol = symbol;

		// IDs are null until the brain registers the spec and hands them back via
		// bindIds(); the trader then borrows the same channelId so rewards, inputs,
		// and inferences all key off a single number per symbol.
		this.channelId = null;
		this.priceChangeDimId = null;
		this.volumeChangeDimId = null;
		this.activityDimId = null;

		// Dim names: used in the spec and for cross-channel vote-debug matching.
		this.priceChangeDimName = `${symbol}_price_change`;
		this.volumeChangeDimName = `${symbol}_volume_change`;
		this.activityDimName = `${symbol}_activity`;

		// Bucket boundaries and the display map that turns bucket IDs into percent ranges.
		this.initializeBuckets();

		// Per-episode frame iteration state: the row slice being streamed, the cursor,
		// and the previous frame's reading (so each encode() call sees a full pair).
		this.rows = null;
		this.rowIndex = 0;
		this.previousPrice = null;
		this.previousVolume = null;
	}

	/**
	 * Set up the price and volume bucket boundaries and build the
	 * bucket-to-percent display map used by formatCoordinates().
	 */
	initializeBuckets() {
		this.priceBoundaries = [0];
		this.volumeBoundaries = [0];
		this.bucketToPercent = this.buildBucketPercentMap();
	}

	/**
	 * Build a Map keyed by "dimName:bucketId" whose values are human-readable
	 * percent-range strings. Used by formatCoordinates() to annotate vote-debug
	 * output with what each bucket number actually represents.
	 * @returns {Map<string, string>}
	 */
	buildBucketPercentMap() {
		const map = new Map();
		const categories = [
			{ dim: this.priceChangeDimName,  boundaries: this.priceBoundaries },
			{ dim: this.volumeChangeDimName, boundaries: this.volumeBoundaries }
		];
		for (const { dim, boundaries } of categories)
			for (let bucket = 1; bucket <= boundaries.length + 1; bucket++) {
				const { lo, hi } = this.getBucketRange(bucket, boundaries);
				map.set(`${dim}:${bucket}`, this.formatBucketRange(lo, hi));
			}
		return map;
	}

	/**
	 * Return the lower and upper bounds for a 1-based bucket index given an
	 * ordered array of split points. The first bucket extends to -Infinity
	 * and the last to +Infinity.
	 * @param {number} bucketValue
	 * @param {number[]} boundaries
	 * @returns {{ lo: number, hi: number }}
	 */
	getBucketRange(bucketValue, boundaries) {
		const idx = bucketValue - 1;
		const lo = idx === 0 ? -Infinity : boundaries[idx - 1];
		const hi = idx >= boundaries.length ? Infinity : boundaries[idx];
		return { lo, hi };
	}

	/**
	 * Format a bucket's lower/upper bounds into a compact percent-range string
	 * (e.g. "<0.00%", ">0.00%", or "0.00%~1.00%").
	 * @param {number} min
	 * @param {number} max
	 * @returns {string}
	 */
	formatBucketRange(min, max) {
		if (min === -Infinity) return `<${max.toFixed(2)}%`;
		if (max === Infinity) return `>${min.toFixed(2)}%`;
		return `${min.toFixed(2)}%~${max.toFixed(2)}%`;
	}

	/**
	 * Map a continuous change value into a 1-based bucket index by walking
	 * the ordered boundary array. Values beyond the last boundary fall into
	 * the final overflow bucket.
	 * @param {number} value
	 * @param {number[]} boundaries
	 * @returns {number}
	 */
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

		// Zero price or volume means no trading happened (gap-filled by setup.js) —
		// skip so the brain doesn't see a non-event as real market activity.
		if (!frame.price || !frame.volume) return null;

		// Spread-adjusted price change: "up" means current bid > previous ask,
		// i.e. the move exceeded transaction friction. When transactionCost is 0
		// this reduces to plain percent change.
		const costMul = StockEncoder.transactionCost / 100;
		const currentBid = frame.price * (1 - costMul);
		const previousAsk = frame.previousPrice * (1 + costMul);
		const priceChange = ((currentBid - previousAsk) / previousAsk) * 100;

		const volumeChange = frame.previousVolume === 0 ? 1000 : ((frame.volume - frame.previousVolume) / frame.previousVolume) * 100;

		// Map of dimId → raw scalar; the brain's quantizer bucketizes these per-dim
		// according to the registered static boundaries.
		const dimMap = new Map();
		dimMap.set(this.priceChangeDimId, priceChange);
		dimMap.set(this.volumeChangeDimId, volumeChange);
		return dimMap;
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
	 * no behavior. Each dim spec carries a plain name string; the Thalamus allocates
	 * an ID and hands it back via the return value. The brain stores the spec, registers
	 * dims with the quantizer, and pre-creates action neurons for the activity dim. The
	 * caller passes the returned { channelId, dimensionIds } to bindIds().
	 */
	getChannelSpec() {
		return {
			name: this.symbol,
			emitsReward: true,
			learnActionSequences: false,
			dimensions: [
				{
					name: this.priceChangeDimName,
					kind: 'input',
					resolution: this.priceBoundaries.length + 1,
					mode: 'static',
					boundaries: [...this.priceBoundaries]
				},
				{
					name: this.volumeChangeDimName,
					kind: 'input',
					resolution: this.volumeBoundaries.length + 1,
					mode: 'static',
					boundaries: [...this.volumeBoundaries]
				},
				{
					name: this.activityDimName,
					kind: 'action',
					resolution: 2,
					mode: 'passthrough',
					actions: [ POSITION_OUT, POSITION_OWN ],
					defaultAction: POSITION_OUT
				}
			]
		};
	}

	/**
	 * Called after brain.registerChannelSpec() returns the allocated IDs. Records the
	 * channelId and the per-dim IDs so encode() can key its output Map by dim ID.
	 */
	bindIds({ channelId, dimensionIds }) {
		this.channelId = channelId;
		this.priceChangeDimId = dimensionIds[this.priceChangeDimName];
		this.volumeChangeDimId = dimensionIds[this.volumeChangeDimName];
		this.activityDimId = dimensionIds[this.activityDimName];
	}
}