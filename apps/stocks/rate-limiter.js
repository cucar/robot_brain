/**
 * Sliding-window rate limiter.
 * Tracks timestamps of recent API calls and sleeps only when the window is full.
 */
export default class RateLimiter {

	/**
	 * @param maxRequests - maximum requests allowed within the window
	 * @param windowMs - sliding window duration in milliseconds
	 */
	constructor(maxRequests = 190, windowMs = 60_000) {
		this.maxRequests = maxRequests;
		this.windowMs = windowMs;
		this.timestamps = [];
	}

	/**
	 * Call before each API request.  Sleeps only if we've hit the limit.
	 */
	async wait() {
		const now = Date.now();

		// Discard timestamps outside the sliding window
		while (this.timestamps.length > 0 && this.timestamps[0] <= now - this.windowMs)
			this.timestamps.shift();

		// If we're at the limit, sleep until the oldest timestamp falls out of the window
		if (this.timestamps.length >= this.maxRequests) {
			const sleepMs = this.timestamps[0] - (now - this.windowMs) + 50; // +50ms safety margin
			console.log(`   ⏳ Rate limit: ${this.timestamps.length}/${this.maxRequests} calls in window, pausing ${Math.ceil(sleepMs / 1000)}s...`);
			await new Promise(resolve => setTimeout(resolve, sleepMs));
		}

		this.timestamps.push(Date.now());
	}
}
