/**
 * One stock on a lap that repeats (lap-channels.js, mode one): KGC alone, the first 12 daily opens of its history,
 * shown 20 times over, so the brain learns its actions with no other channel to lean on.
 *
 * usage: node apps/stocks/jobs/synthetic-extended-test.js [--laps 20] [--volume] [--timeframe 1D]
 */
import { runJob } from 'robot-brain';
import { LapChannelsJob } from '../lap-channels.js';
import { readLapSettings } from './multi-channel-test.js';

class SyntheticExtendedTestJob extends LapChannelsJob {

	constructor() {
		super(readLapSettings('one'));
	}
}

await runJob(import.meta, SyntheticExtendedTestJob);
