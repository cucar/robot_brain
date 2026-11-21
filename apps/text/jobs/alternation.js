/**
 * The alternation (docs/algorithm-xy.md): a letter channel over time alone, a pair of letters alternating for a
 * stretch. Each frame prints the letter, what the brain had said for it, what stands on the apex, what is still open
 * from before, and the dictionary as it changes; at the end, how many it said right and the dictionary.
 *
 * usage: node apps/text/jobs/alternation.js [frames=30] [stream] [--regions R] [--columns C]
 * The stream is stretches separated by commas: two letters and a count of frames, or `_` and a count of silent
 * frames, as in `xy:12,_:4,ab:12`. Without one, `x y` alternates for every frame.
 */
import { Job, runJob, readPositionals } from 'robot-brain';

const LETTERS = 26;

/**
 * The letter a value of the letter dimension stands for.
 */
function nameLetter(value) {
	return String.fromCharCode('a'.charCodeAt(0) + value);
}

/**
 * The letter of each frame, or null for a silent frame, from a stream spec such as `xy:12,_:4`.
 */
function makeStream(spec) {
	const stream = [];
	for (const part of spec.split(',')) {
		const [what, count] = part.split(':');
		for (let i = 0; i < parseInt(count); i++) {
			stream.push(what[0] === '_' ? null : what.charCodeAt(i % 2) - 'a'.charCodeAt(0));
		}
	}
	return stream;
}

/**
 * The dictionary as one line: every function's description, sorted, joined.
 */
function describeDictionary(brain) {
	return brain.getDictionary().map(f => f.description).sort().join('; ');
}

class AlternationJob extends Job {

	constructor() {
		super();
		// each base neuron's letter, read once: the letters never change
		this.letterOf = new Map();
		// the count of frames asked, the letters said right, and the frames said wrong
		this.right = 0;
		this.asked = 0;
		this.wrongAt = [];
		// the dictionary as last printed, and the frame it last changed
		this.lastDictionary = '';
		this.lastChange = 0;
	}

	getBrainOptions() {
		return { h: 64, policy: 'allOn' };
	}

	declareChannels() {
		this.brain.declareChannel({ name: 'letters', event: { name: 'letter', resolution: LETTERS } });
	}

	/**
	 * Show the stream one letter a frame, and print each frame's line.
	 */
	async executeJob() {
		const [frameArg, streamArg] = readPositionals();
		const frames = frameArg ? parseInt(frameArg) : 30;
		const stream = makeStream(streamArg ?? `xy:${frames}`);
		const letterDim = this.channels.get('letters').eventDim;
		let said = null;
		for (let k = 0; k < Math.min(frames, stream.length); k++) {
			const letter = stream[k];
			this.scoreGuess(said, letter, k + 1);
			const inputs = letter === null ? [] : [{ dim: letterDim, bucket: letter }];
			const winners = this.brain.processFrame(inputs, []);
			const next = winners.find(w => w.dim === letterDim)?.bucket ?? null;
			this.printFrame(k + 1, letter, said, next);
			this.printDictionaryIfChanged(k + 1);
			said = next;
		}
	}

	/**
	 * Count a frame's letter against what the brain had said for it; a silent frame or nothing said is not counted.
	 */
	scoreGuess(said, letter, frame) {
		if (said === null || letter === null) return;
		this.asked++;
		if (said === letter) this.right++;
		else this.wrongAt.push(frame);
	}

	/**
	 * One frame's line: the letter, what was said for it, the board, and what the brain says next.
	 */
	printFrame(frame, letter, said, next) {
		const { apex, open } = this.readBoard();
		const mark = said === null || letter === null ? '  ' : said === letter ? 'ok' : 'NO';
		const shown = letter === null ? '_' : nameLetter(letter);
		const had = said === null ? '-' : nameLetter(said);
		const says = next === null ? '-' : nameLetter(next);
		console.log(`f${String(frame).padStart(3)} ${shown} had ${had} ${mark} | apex now [${apex.join(' ')}] open before [${open.join(' ')}] | says next ${says}`);
	}

	/**
	 * Print the dictionary when it differs from the last one printed, and note the frame.
	 */
	printDictionaryIfChanged(frame) {
		const dictionary = describeDictionary(this.brain);
		if (dictionary === this.lastDictionary) return;
		console.log(`      dictionary: ${dictionary === '' ? '(empty)' : dictionary}`);
		this.lastDictionary = dictionary;
		this.lastChange = frame;
	}

	/**
	 * What stands on the apex in the last frame, and what is still open from the frames before it:
	 * a strong apex activation whose reach in time spans to the last frame.
	 */
	readBoard() {
		const board = this.brain.getBoard();
		const last = board[board.length - 1];
		const apex = last.activations.filter(a => a.onApex).map(a => this.nameActivation(a));
		const open = [];
		for (const frame of board) {
			if (frame.frame >= last.frame) continue;
			for (const a of frame.activations) {
				if (a.onApex && a.strong && last.frame - a.time <= a.reach[0]) open.push(`${this.nameActivation(a)}@${frame.frame}`);
			}
		}
		return { apex, open };
	}

	/**
	 * How an activation is written: a call by its interneuron, a letter by itself.
	 */
	nameActivation(activation) {
		return activation.kind === 'inter' ? `#${activation.neuron}` : nameLetter(this.readLetter(activation.neuron));
	}

	/**
	 * The letter of a base neuron, read from the brain the first time it is asked.
	 */
	readLetter(neuron) {
		if (!this.letterOf.has(neuron)) this.letterOf.set(neuron, this.brain.readNeurons([neuron])[0].bucket);
		return this.letterOf.get(neuron);
	}

	async showResults() {
		const functions = this.brain.countFunctions();
		console.log(`right ${this.right}/${this.asked} | neurons ${this.brain.countLivingNeurons()} | functions ${functions}`);
		console.log(`alternation inference ${this.right}/${this.asked} | wrong at frames [${this.wrongAt.join(', ')}]`);
		console.log(`recognition: ${functions} function(s): ${this.lastDictionary} | the dictionary last changed at frame ${this.lastChange}`);
	}
}

await runJob(import.meta, AlternationJob);
