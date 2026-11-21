/**
 * The six-bar lap (docs/algorithm-stocks.md §4): one instrument, `rise` and `fall`, the bars +1% −2% −3% +2% +3% −1%
 * repeated, and the action `own` or `disown` for the bar ahead.
 * Where the brain says nothing the world runs the right action.
 * Each frame's reward is the bar's change signed by the action that stood over it, scoped to the channel and that one frame.
 * Reports how often what the brain said was right, per bar of the lap, over the run and over the last ten laps,
 * and the dictionary as it stands at the end.
 *
 * usage: node apps/stocks/jobs/synthetic-cycle-test.js [laps=150] [history depth=256] [--regions R] [--columns C]
 */
import { Job, runJob, readPositionals, formatFixed } from 'robot-brain';

// the percent move of each bar of the lap
const LAP = [1, -2, -3, 2, 3, -1];

// the laps at the end whose score is kept apart
const LAST_LAPS = 10;

/**
 * The move a bar's percent stands for: 1 up, 0 down.
 */
function readMove(percent) {
	return percent > 0 ? 1 : 0;
}

/**
 * A share of a count as a percent with one decimal; nothing counted is 0%.
 */
function formatPercent(part, whole) {
	return formatFixed(100 * part / Math.max(whole, 1), 1);
}

class SixBarLapJob extends Job {

	constructor() {
		super();
		const [lapArg, depthArg] = readPositionals();
		this.laps = lapArg ? parseInt(lapArg) : 150;
		this.depth = depthArg ? parseInt(depthArg) : 256;
		// per bar of the lap: right and wrong over the run, and over the last ten laps
		this.right = Array(LAP.length).fill(0);
		this.wrong = Array(LAP.length).fill(0);
		this.rightLast = Array(LAP.length).fill(0);
		this.wrongLast = Array(LAP.length).fill(0);
		// the frames where the brain said nothing and the world ran the right action
		this.lessons = 0;
		// the first lap whose every bar the brain said right
		this.firstLapAllRight = null;
		// every inference the brain made, read against the bar it was made for
		this.inferenceRight = 0;
		this.inferences = 0;
		// the dictionary's fingerprint as it last stood, read when the run starts, and the frame it last changed
		this.lastDictionary = null;
		this.lastChange = 0;
	}

	getBrainOptions() {
		return { h: this.depth, policy: 'eventsToActions' };
	}

	declareChannels() {
		this.brain.declareChannel({ name: 'price', event: { name: 'move', resolution: 2 }, action: { name: 'position', resolution: 2 } });
	}

	/**
	 * Show every bar of every lap with the action that stands over it, and score what the brain said.
	 */
	async executeJob() {
		const price = this.channels.get('price');
		this.lastDictionary = this.brain.readDictionaryFingerprint();
		const total = this.laps * LAP.length;
		let said = null;
		let lapRight = 0;
		for (let k = 0; k < total; k++) {
			const bar = k % LAP.length;
			const move = readMove(LAP[bar]);
			const action = said ?? move;
			if (bar === 0) lapRight = 0;
			if (said === null) this.lessons++;
			else lapRight += this.scoreBar(bar, action === move, k + LAST_LAPS * LAP.length >= total);
			if (bar === LAP.length - 1 && lapRight === LAP.length && this.firstLapAllRight === null) this.firstLapAllRight = Math.floor(k / LAP.length) + 1;
			said = this.runBar(price, move, action, LAP[bar]);
			this.scoreInference(said, readMove(LAP[(k + 1) % LAP.length]));
			this.noteDictionary(k + 1);
		}
	}

	/**
	 * Count a bar the brain said, over the run and, near the end, over the last laps; 1 when it was right.
	 */
	scoreBar(bar, isRight, isInLastLaps) {
		if (isRight) this.right[bar]++; else this.wrong[bar]++;
		if (isInLastLaps) {
			if (isRight) this.rightLast[bar]++; else this.wrongLast[bar]++;
		}
		return isRight ? 1 : 0;
	}

	/**
	 * One frame: the move and the action that ran, rewarded by the bar's change; returns what the brain says next.
	 */
	runBar(price, move, action, percent) {
		const inputs = [{ dim: price.eventDim, bucket: move }, { dim: price.actionDim, bucket: action }];
		const reward = action === 1 ? percent : -percent;
		const winners = this.brain.processFrame(inputs, [{ reward, channels: [price.channelId], frames: 1 }]);
		return winners.find(w => w.dim === price.actionDim)?.bucket ?? null;
	}

	/**
	 * Count an inference against the move of the bar it was made for.
	 */
	scoreInference(said, nextMove) {
		if (said === null) return;
		this.inferences++;
		if (said === nextMove) this.inferenceRight++;
	}

	/**
	 * Note the frame when the dictionary differs from how it last stood.
	 */
	noteDictionary(frame) {
		const dictionary = this.brain.readDictionaryFingerprint();
		if (dictionary === this.lastDictionary) return;
		this.lastDictionary = dictionary;
		this.lastChange = frame;
	}

	async showResults() {
		console.log(`six-bar cycle: ${this.laps} laps, H=${this.depth}`);
		this.printBars();
		const descriptions = this.brain.getDictionary().map(f => f.description).sort();
		console.log(`neurons ${this.brain.countLivingNeurons()} | functions ${descriptions.length}`);
		if (descriptions.length <= 40) for (const description of descriptions) console.log(`  ${description}`);
		console.log(`stocks inference ${this.inferenceRight}/${this.inferences}`);
		console.log(`recognition: ${descriptions.length} functions at the end; the dictionary's functions and relations last changed at frame ${this.lastChange}`);
	}

	/**
	 * Each bar of the lap: right and wrong over the run and over the last laps; then the totals.
	 */
	printBars() {
		console.log('bar | right | wrong | last 10 laps');
		for (let i = 0; i < LAP.length; i++) {
			console.log(`${String(i + 1).padStart(3)} | ${String(this.right[i]).padStart(5)} | ${String(this.wrong[i]).padStart(5)} | ${this.rightLast[i]}/${this.rightLast[i] + this.wrongLast[i]}`);
		}
		const sum = counts => counts.reduce((a, b) => a + b, 0);
		const [right, wrong, rightLast, wrongLast] = [this.right, this.wrong, this.rightLast, this.wrongLast].map(sum);
		const firstLap = this.firstLapAllRight ?? 'none';
		console.log(`overall right ${right}/${right + wrong} = ${formatPercent(right, right + wrong)}% | last 10 laps ${rightLast}/${rightLast + wrongLast} = ${formatPercent(rightLast, rightLast + wrongLast)}% | lessons ${this.lessons} | first lap all right: ${firstLap}`);
	}
}

await runJob(import.meta, SixBarLapJob);
