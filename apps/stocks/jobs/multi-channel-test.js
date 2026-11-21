/**
 * Several stocks on a lap that repeats (lap-channels.js): by default KGC, GLD and SPY, each the first 12 daily opens
 * of its history, shown 20 times over.
 * Each lap prints the right share per stock; at the end, each stock's laps, the dictionary and the apex.
 *
 * usage: node apps/stocks/jobs/multi-channel-test.js [--mode three] [--laps 20] [--all4] [--volume] [--timeframe 1D]
 *        [--save-brain <label>] [--load-brain <label>] [--disable-learning]
 * The modes: one, two, three, three4, lap, lap65 (see lap-channels.js).
 */
import path from 'node:path';
import { runJob, readFlag, readNumber } from 'robot-brain';
import { LapChannelsJob, DATA_ROOT } from '../lap-channels.js';

/**
 * The settings of a lap run, the mode given; the trader trades nothing here, since a lap has no prices.
 */
export function readLapSettings(mode) {
	return {
		mode,
		cycles: readNumber('--laps', 20),
		folder: path.join(DATA_ROOT, readFlag('--timeframe', '1D')),
		symbols: [],
		every: 0,
		all4: process.argv.includes('--all4'),
		volume: process.argv.includes('--volume'),
		offset: 0,
		holdout: 0,
		maxFrames: null,
		trader: { maxPositions: 1, costPercent: 0, initialCapital: 15000, minPrice: 0, maxPrice: 5000 },
	};
}

class MultiChannelTestJob extends LapChannelsJob {

	constructor() {
		super(readLapSettings(readFlag('--mode', 'three')));
	}
}

await runJob(import.meta, MultiChannelTestJob);
