/**
 * Tap while go, then done (docs/algorithm-loop.md): one channel over time alone, an event dimension `cue` (go, stop)
 * and an action dimension `act` (tap, done).
 * A stretch: go in frames 0, 1 and 2, stop in frame 3, no cue after.
 * Teaching: the world runs tap itself in frames 1, 2 and 3 and done in frame 4, each rewarded.
 * It then shows nothing and runs nothing for longer than any reach; six stretches.
 * Test: the world shows the cues, runs nothing itself, and executes what the brain outputs.
 * A tap in the frame after a go and done in the frame after a stop are rewarded 1, any other action −1.
 * The neighborhood policy is the stocks one: no action is anyone's neighbor, and only events and interneurons
 * connect, to actions alone.
 * Nothing runs where every candidate's estimate is negative.
 * Each frame prints the cue, what ran, what stands on the apex (a guessed activation marked ?), and the brain's output.
 *
 * usage: node apps/tap/jobs/tap.js [teaching stretches=6] [silence=8] [--regions R] [--columns C]
 */
import { Job, runJob, readPositionals } from 'robot-brain';

// the values of the cue and of the action
const GO = 0, STOP = 1;
const TAP = 0, DONE = 1;

// how many frames of go each test stretch shows
const TEST_STRETCHES = [3, 5, 2];

/**
 * An action value as a word, or "nothing".
 */
function nameAction(value) {
	return value === null ? 'nothing' : value === TAP ? 'tap' : 'done';
}

/**
 * A cue value as a word, or "-" for no cue.
 */
function nameCue(value) {
	return value === null ? '-' : value === GO ? 'go' : 'stop';
}

/**
 * The action the world expects in the frame after a cue: tap after go, done after stop, nothing after no cue.
 */
function expectAction(cue) {
	return cue === GO ? TAP : cue === STOP ? DONE : null;
}

/**
 * The guessed activations as Rust prints a list of pairs, as in `[(5, "tap")]`.
 */
function formatGuessed(guessed) {
	return `[${guessed.map(([frame, name]) => `(${frame}, "${name}")`).join(', ')}]`;
}

class TapJob extends Job {

	constructor() {
		super();
		// what stood guessed and not seen, as (frame, neuron's name)
		this.guessed = [];
		// the longest reach in time any open activation had
		this.longestReach = 0;
		// each test frame: the frame, the cue shown, what the brain output, what was expected
		this.table = [];
	}

	getBrainOptions() {
		return { h: 256, policy: 'eventsToActions', actionPolicy: 'nothing' };
	}

	declareChannels() {
		this.brain.declareChannel({ name: 'tapper', event: { name: 'cue', resolution: 2 }, action: { name: 'act', resolution: 2 } });
	}

	/**
	 * Teach the stretches, then test three.
	 */
	async executeJob() {
		const [stretchArg, silenceArg] = readPositionals();
		this.silence = silenceArg ? parseInt(silenceArg) : 8;
		const tapper = this.channels.get('tapper');
		this.channelId = tapper.channelId;
		this.cueDim = tapper.eventDim;
		this.actDim = tapper.actionDim;
		this.teach(stretchArg ? parseInt(stretchArg) : 6);
		this.test();
	}

	/**
	 * The cues of a stretch, frame by frame: go for `goFrames` frames, then stop, then no cue for the silence.
	 */
	makeCues(goFrames) {
		return [...Array(goFrames).fill(GO), STOP, ...Array(this.silence).fill(null)];
	}

	/**
	 * Each teaching stretch: three go, a stop; the world runs tap in frames 1 to 3 and done in frame 4, each rewarded.
	 */
	teach(stretches) {
		for (let r = 1; r <= stretches; r++) {
			this.makeCues(3).forEach((cue, k) => {
				const ran = k >= 1 && k <= 3 ? TAP : k === 4 ? DONE : null;
				this.runFrame(cue, ran, 1, `teach ${r}`);
			});
		}
	}

	/**
	 * Each test stretch: the world runs what the brain output the frame before, rewarded by whether it fit the cue before.
	 */
	test() {
		let said = null;
		let previousCue = null;
		TEST_STRETCHES.forEach((goFrames, i) => {
			for (const cue of this.makeCues(goFrames)) {
				const ran = said;
				said = this.runFrame(cue, ran, this.rewardAction(ran, previousCue), `test ${i + 1}`);
				this.table.push({ frame: this.brain.getFrameNumber(), cue: nameCue(cue), said, expected: expectAction(cue) });
				previousCue = cue;
			}
		});
	}

	/**
	 * 1 for a tap after go or done after stop, −1 for any other action.
	 */
	rewardAction(ran, previousCue) {
		return ran !== null && ran === expectAction(previousCue) ? 1 : -1;
	}

	/**
	 * One frame: the cue shown (if any), the action run (if any) and its reward.
	 * Prints the frame's line and returns the brain's output for the frame after.
	 */
	runFrame(cue, ran, reward, label) {
		const frame = this.brain.getFrameNumber() + 1;
		const inputs = [];
		if (cue !== null) inputs.push({ dim: this.cueDim, bucket: cue });
		if (ran !== null) inputs.push({ dim: this.actDim, bucket: ran });
		const rewards = ran !== null ? [{ reward, channels: [this.channelId], frames: 1 }] : [];
		const winners = this.brain.processFrame(inputs, rewards);
		const output = winners.find(w => w.dim === this.actDim)?.bucket ?? null;
		this.longestReach = Math.max(this.longestReach, this.brain.getLongestReach());
		const last = this.brain.getLastFrame();
		this.noteGuessed(frame, last);
		const apex = last.activations.filter(a => a.onApex).map(a => this.nameApexActivation(a));
		const ranWord = ran === null ? '-' : nameAction(ran);
		console.log(`f${String(frame).padStart(3)} ${label.padEnd(8)} cue ${nameCue(cue).padEnd(4)} ran ${ranWord.padEnd(4)} | apex [${apex.join(' ')}] | outputs ${nameAction(output)}`);
		return output;
	}

	/**
	 * Note each base activation of the frame that stood guessed, not seen.
	 */
	noteGuessed(frame, last) {
		for (const a of last.activations) {
			if (!a.strong && a.kind !== 'inter') this.guessed.push([frame, this.nameNeuron(a.neuron)]);
		}
	}

	/**
	 * An apex activation: a call by its pose, a base neuron by its name; a guessed one marked ?.
	 */
	nameApexActivation(a) {
		const name = a.kind === 'inter' ? this.brain.describePose(a.pose) : this.nameNeuron(a.neuron);
		return a.strong ? name : `${name}?`;
	}

	/**
	 * A base neuron by its value's word, any other by its id.
	 */
	nameNeuron(id) {
		const neuron = this.brain.readNeurons([id])[0];
		if (neuron.dim === this.cueDim) return nameCue(neuron.bucket);
		if (neuron.dim === this.actDim) return nameAction(neuron.bucket);
		return `#${id}`;
	}

	async showResults() {
		this.printTable();
		console.log(`longest reach in time seen: ${this.longestReach} (silence ${this.silence})`);
		console.log(`guessed activations that stood: ${formatGuessed(this.guessed)}`);
		this.printDictionary();
		this.printNeurons();
	}

	/**
	 * The test frames: the cue, what the brain output for the frame after, what was expected; then the count right.
	 */
	printTable() {
		console.log('\ntest: frame | cue shown | machine outputs (for the frame after) | expected');
		let right = 0;
		for (const { frame, cue, said, expected } of this.table) {
			const same = said === expected;
			if (same) right++;
			console.log(`  f${String(frame).padStart(3)} | ${cue.padEnd(4)} | ${nameAction(said).padEnd(7)} | ${nameAction(expected).padEnd(7)}${same ? '' : '   <- differs'}`);
		}
		console.log(`right ${right}/${this.table.length}`);
	}

	/**
	 * Every function of the dictionary, sorted by its description.
	 */
	printDictionary() {
		const dictionary = this.brain.getDictionary();
		console.log(`dictionary: ${dictionary.length} functions`);
		for (const description of dictionary.map(f => f.description).sort()) console.log(`  ${description}`);
	}

	/**
	 * Every living neuron with its connections.
	 */
	printNeurons() {
		console.log('neurons and their connections:');
		const ids = [...Array(this.brain.countNeurons()).keys()];
		const living = this.brain.readNeurons(ids).filter(n => n.alive).map(n => n.id);
		for (const text of this.brain.describeNeurons(living)) process.stdout.write(text);
	}
}

await runJob(import.meta, TapJob);
