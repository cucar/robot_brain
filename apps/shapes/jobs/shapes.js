/**
 * Simple images, each shown again and again to a fresh brain: the image, then the digit 0 with a reward of +1, then the
 * context is reset.
 * Modes: hline, vline, square, hollow; and teach23, the 2 and the 3 in turn, each taught with its digit, then each
 * shown without it to a copy loaded with learning off.
 *
 * usage: node apps/shapes/jobs/shapes.js <mode=square> [showings=30] [--regions R] [--columns C]
 */
import { Job, runJob, readPositionals, formatFloat } from 'robot-brain';

/* ---------- The images ---------- */

// each image as rows of pixels: W white (value 1), B black (value 0), . reports nothing
const SHAPES = {
	hline: ['BBBBBBB', 'BBBBBBB', 'BWWWWWB', 'BBBBBBB', 'BBBBBBB'],
	vline: ['BBBBB', 'BBWBB', 'BBWBB', 'BBWBB', 'BBWBB', 'BBWBB', 'BBBBB'],
	square: ['BBBBB', 'BWWWB', 'BWWWB', 'BWWWB', 'BBBBB'],
	hollow: ['BBBBB', 'B...B', 'B...B', 'B...B', 'BBBBB'],
	two: ['BBBBBBB', 'BWWWWWB', 'BBBBBWB', 'BBBBBWB', 'BWWWWWB', 'BWBBBBB', 'BWBBBBB', 'BWWWWWB', 'BBBBBBB'],
	three: ['BBBBBBB', 'BWWWWWB', 'BBBBBWB', 'BBBBBWB', 'BWWWWWB', 'BBBBBWB', 'BBBBBWB', 'BWWWWWB', 'BBBBBBB'],
};

/**
 * The rows of a shape, by name.
 */
function makeShape(name) {
	if (!SHAPES[name]) throw new Error(`unknown shape ${name}`);
	return SHAPES[name];
}

/**
 * Print an image, one row a line.
 */
function printImage(rows) {
	for (const row of rows) console.log(`    ${row}`);
}

/**
 * A number with its sign, as Rust's `{:+}` writes it.
 */
function formatSigned(value) {
	return value >= 0 ? `+${value}` : `${value}`;
}

/**
 * An offset in words: the two places alone where time is 0, else time first.
 */
function describeOffset(offset) {
	const [t, x, y] = offset;
	if (t === 0) return `(${formatSigned(x)}, ${formatSigned(y)})`;
	return `(t ${formatSigned(t)}, ${formatSigned(x)}, ${formatSigned(y)})`;
}

/* ---------- Reading a brain ---------- */

/**
 * The words the report uses for what a brain holds: its pixels, its calls, its relations and its board.
 */
class BrainReader {

	constructor(brain) {
		this.brain = brain;
		this.neurons = new Map();
	}

	/**
	 * A neuron's fixed facts, read once.
	 */
	readNeuron(id) {
		if (!this.neurons.has(id)) this.neurons.set(id, this.brain.readNeurons([id])[0]);
		return this.neurons.get(id);
	}

	/**
	 * A pose in words: a pixel's colour, or a call with its functions and its phase.
	 */
	nameSym(sym) {
		const pose = this.brain.readPose(sym) ?? (this.readNeuron(sym).kind === 'inter' ? { neuron: sym } : null);
		if (!pose) return this.readNeuron(sym).bucket === 1 ? 'white' : 'black';
		if (pose.from === undefined) return `the call #${pose.neuron} of ${this.brain.listFunctionsOf(pose.neuron).map(f => `f${f}`).join(',') || 'no function now'}`;
		const functions = this.brain.listFunctionsOf(pose.neuron).map(f => `f${f}`);
		const owner = functions.length === 0 ? 'no function now' : functions.join(',');
		return `the call #${pose.neuron} of ${owner} (from ${this.nameEnd(pose.from)} to ${this.nameEnd(pose.to)}, ${pose.laps} laps)`;
	}

	/**
	 * An end of a phase: a call or an interneuron as the dictionary writes it, a pixel in words.
	 */
	nameEnd(sym) {
		if (this.brain.readPose(sym) || this.readNeuron(sym).kind === 'inter') return this.brain.describePose(sym);
		return this.nameSym(sym);
	}

	/**
	 * A relation in words: what sees what, at what offset.
	 */
	describeRelationInWords(relation) {
		return `${this.nameSym(relation.from)} sees ${this.nameSym(relation.to)} at ${describeOffset(relation.offset)}`;
	}

	/**
	 * A relation as the dictionary writes it: its two poses and its offset.
	 */
	describeRelation(relation) {
		return `${this.brain.describePose(relation.from)}→${this.brain.describePose(relation.to)}[${relation.offset.join(', ')}]`;
	}

	/**
	 * Every function's description, sorted: two fingerprints differ where the dictionary changed.
	 */
	readFingerprint() {
		return this.brain.getDictionary().map(f => f.description).sort().join('\n');
	}

	/**
	 * An activation with its place: a call by its pose, a pixel by its colour.
	 */
	nameActivation(a) {
		const [x, y] = a.position;
		if (a.kind === 'inter') return `${a.poseText}@${x},${y}`;
		return `${this.readNeuron(a.neuron).bucket === 1 ? 'W' : 'B'}@${x},${y}`;
	}
}

/**
 * The board as read once: each activation found by the frame it is recorded under and its index there.
 */
class BoardView {

	constructor(frames) {
		this.byFrame = new Map(frames.map(frame => [frame.frame, frame.activations]));
	}

	findActivation(ref) {
		return this.byFrame.get(ref.frame)?.[ref.index];
	}

	findFrame(frameNumber) {
		return this.byFrame.get(frameNumber);
	}

	/**
	 * How many pixels an activation stands for: itself if it is one, and every pixel under it, followed down.
	 */
	countPixels(activation) {
		const pixels = new Set();
		const stack = [...activation.covers];
		if (activation.kind === 'event') pixels.add(activation.position.join(','));
		while (stack.length > 0) {
			const covered = this.findActivation(stack.pop());
			if (!covered) continue;
			if (covered.kind === 'event') pixels.add(covered.position.join(','));
			stack.push(...covered.covers);
		}
		return pixels.size;
	}
}

/* ---------- The reports ---------- */

/**
 * What stands on the apex at the end of a showing, level by level, the relations among what stands, and the bare
 * pixels.
 * Called right after the image frame, before the digit frame.
 */
function reportApex(brain) {
	const reader = new BrainReader(brain);
	const board = new BoardView(brain.getBoard(true));
	const activations = board.findFrame(brain.getFrameNumber());
	if (!activations) return;
	const standing = activations.filter(a => a.onApex && a.kind !== 'action');
	const calls = standing.filter(a => a.kind === 'inter');
	const bare = standing.filter(a => a.kind === 'event').map(a => reader.nameActivation(a));
	console.log(`  on the apex: ${calls.length} calls, ${bare.length} bare pixels`);
	printCallsByLevel(calls, reader, board);
	if (bare.length > 0) console.log(`    bare pixels: ${bare.join(' ')}`);
	printApexRelations(standing, reader, board);
}

/**
 * The calls on the apex, grouped by level, lowest first.
 */
function printCallsByLevel(calls, reader, board) {
	const levels = [...new Set(calls.map(a => a.level))].sort((a, b) => a - b);
	for (const level of levels) {
		const atLevel = calls.filter(a => a.level === level);
		console.log(`    level ${level}: ${atLevel.length} calls`);
		for (const a of atLevel) console.log(`      ${describeCall(a, reader, board)}`);
	}
}

/**
 * A call on the apex: its pose, its place, its words, its function, what it covers, and how many pixels it stands for.
 */
function describeCall(a, reader, board) {
	const [x, y] = a.position;
	const covers = a.covers.map(ref => board.findActivation(ref)).filter(Boolean).map(c => reader.nameActivation(c)).join(' ');
	return `${a.poseText} at (${x}, ${y}) | ${reader.nameSym(a.pose)} | f${a.firedBy ?? 0} | covers [${covers}] | stands for ${board.countPixels(a)} pixels`;
}

/**
 * The relations the board holds between what stands on the apex, each pair once, with the function that holds it.
 */
function printApexRelations(standing, reader, board) {
	const seen = new Set();
	const lines = [];
	for (const a of standing) {
		for (const held of a.relations) {
			const b = board.findActivation(held.other);
			if (!b || !b.onApex || b.kind === 'action') continue;
			const pair = [reader.nameActivation(a), reader.nameActivation(b)];
			if (seen.has(pair.join('\u0000'))) continue;
			seen.add(pair.join('\u0000'));
			const relation = reader.brain.canonRelation(a.neuron, b.neuron, held.offset, a.numbers.map((v, i) => v - b.numbers[i]));
			const holder = reader.brain.findHolder(relation);
			lines.push(`      ${pair[0]} – ${pair[1]} : ${reader.describeRelationInWords(relation)} | as held: ${reader.describeRelation(relation)} | returned at strength ${formatFloat(held.strength)} | held by ${holder == null ? 'no function' : `f${holder}`}`);
		}
	}
	console.log(`    relations between what stands on the apex: ${lines.length}`);
	for (const line of lines) console.log(line);
}

/**
 * Every function of the dictionary, in id order: its origin, its interneuron, and its relations.
 */
function reportDictionary(brain) {
	const reader = new BrainReader(brain);
	const functions = brain.getDictionary();
	console.log(`  dictionary: ${functions.length} functions`);
	for (const f of functions) {
		const origin = f.origin == null ? 'a cycle, no origin' : `origin ${reader.nameSym(f.origin)}`;
		const interneuron = f.interneuron == null ? '' : `, interneuron #${f.interneuron}`;
		console.log(`    f${f.id}: ${origin}${interneuron}`);
		for (const r of f.relations) console.log(`      ${reader.describeRelationInWords(r)}   [${reader.describeRelation(r)}]`);
	}
}

/* ---------- The job ---------- */

class ShapesJob extends Job {

	constructor() {
		super();
		const [mode, showings] = readPositionals();
		this.mode = mode ?? 'square';
		this.showings = showings ? parseInt(showings) : 30;
	}

	getBrainOptions() {
		return { h: 2048, policy: 'eventsToActions' };
	}

	declareChannels() {
		const x = this.brain.declareAxis('x');
		const y = this.brain.declareAxis('y');
		this.brain.declareChannel({ name: 'image', event: { name: 'pixel', resolution: 2 }, axes: [x, y] });
		this.brain.declareChannel({ name: 'digit', action: { name: 'digit', resolution: 10 } });
	}

	async executeJob() {
		this.pixelDim = this.channels.get('image').eventDim;
		this.digitDim = this.channels.get('digit').actionDim;
		this.digitChannel = this.channels.get('digit').channelId;
		if (this.mode === 'teach23') this.teachTwoAndThree();
		else this.trainShape(this.mode);
	}

	/**
	 * Each pixel of an image that reports something, at its place.
	 */
	makeInputs(rows) {
		const inputs = [];
		rows.forEach((row, y) => [...row].forEach((pixel, x) => {
			if (pixel !== '.') inputs.push({ dim: this.pixelDim, bucket: pixel === 'W' ? 1 : 0, position: [x, y] });
		}));
		return inputs;
	}

	/**
	 * The digit the brain said in a frame's output, or null.
	 */
	readDigit(winners) {
		return winners.find(w => w.dim === this.digitDim)?.bucket ?? null;
	}

	/**
	 * Run the digit frame: the digit that ran, if any, and the reward, if any.
	 */
	runDigit(brain, digit, rewards) {
		brain.processFrame(digit === null ? [] : [{ dim: this.digitDim, bucket: digit }], rewards);
	}

	/**
	 * One shape shown again and again, each showing taught the digit 0; then the dictionary.
	 */
	trainShape(name) {
		const rows = makeShape(name);
		console.log(`${name}:`);
		printImage(rows);
		const reader = new BrainReader(this.brain);
		let previous = reader.readFingerprint();
		let lastChange = 0;
		for (let k = 1; k <= this.showings; k++) {
			const said = this.readDigit(this.brain.processFrame(this.makeInputs(rows), []));
			const fingerprint = reader.readFingerprint();
			const changed = fingerprint !== previous;
			if (changed) lastChange = k;
			previous = fingerprint;
			this.printShowing(name, k, said, changed);
			if (k === this.showings) reportApex(this.brain);
			this.runDigit(this.brain, 0, [{ reward: 1, channels: [this.digitChannel], frames: 1 }]);
			this.brain.resetContext();
		}
		console.log(`NEVERCALLED ${name} ${this.brain.countFunctionsNeverCalled()} of ${this.brain.countFunctions()}`);
		const settled = lastChange < this.showings ? ` (unchanged over the last ${this.showings - lastChange} showings)` : ' (still changing at the last showing)';
		console.log(`  ${name}: the dictionary last changed at showing ${lastChange}${settled}`);
		reportDictionary(this.brain);
	}

	/**
	 * A showing's line: what was said, the levels, what stands on the apex, the functions, and whether the dictionary changed.
	 */
	printShowing(name, k, said, changed) {
		const apex = this.brain.getLastFrame().activations.filter(a => a.onApex && a.kind !== 'action').length;
		console.log(`  ${name} showing ${String(k).padStart(2)}: said ${said ?? '-'} | levels ${this.brain.getStats().levels} | on the apex ${apex} | functions ${this.brain.countFunctions()} | dictionary ${changed ? 'changed' : 'same'}`);
	}

	/**
	 * The 2 and the 3 in turn, each taught with its digit; then each shown without it to a copy with learning off.
	 */
	teachTwoAndThree() {
		const images = [['2', makeShape('two'), 2], ['3', makeShape('three'), 3]];
		console.log('the 2:');
		printImage(images[0][1]);
		console.log('the 3:');
		printImage(images[1][1]);
		this.teachRounds(images);
		reportDictionary(this.brain);
		this.testWithoutDigits(images);
	}

	/**
	 * Each round shows the 2 and then the 3, each with its digit, and prints what was said.
	 */
	teachRounds(images) {
		const reader = new BrainReader(this.brain);
		let previous = reader.readFingerprint();
		let lastChange = 0;
		for (let k = 1; k <= this.showings; k++) {
			const said = images.map(([, rows, digit]) => this.showWithDigit(rows, digit));
			const fingerprint = reader.readFingerprint();
			const changed = fingerprint !== previous;
			if (changed) lastChange = k;
			previous = fingerprint;
			const dictionary = this.brain.getDictionary();
			const edges = dictionary.reduce((sum, f) => sum + f.relations.length, 0);
			console.log(`  round ${String(k).padStart(2)}: the 2 said ${said[0] ?? '-'} | the 3 said ${said[1] ?? '-'} | functions ${dictionary.length} edges ${edges} | dictionary ${changed ? 'changed' : 'same'}`);
		}
		console.log(`  the dictionary last changed at round ${lastChange}`);
	}

	/**
	 * One showing: the image, then its digit with a reward of +1; the context reset after. Returns what the brain said.
	 */
	showWithDigit(rows, digit) {
		const said = this.readDigit(this.brain.processFrame(this.makeInputs(rows), []));
		this.runDigit(this.brain, digit, [{ reward: 1, channels: [this.digitChannel], frames: 1 }]);
		this.brain.resetContext();
		return said;
	}

	/**
	 * On a copy with learning off, show each image without its digit; what the copy says runs, unrewarded.
	 */
	testWithoutDigits(images) {
		const copy = this.copyBrainWithoutLearning();
		for (const [name, rows] of images) {
			const said = this.readDigit(copy.processFrame(this.makeInputs(rows), []));
			console.log(`  test, the ${name} without its digit: says ${said ?? 'nothing'} | voters ${copy.getStats().voters}`);
			reportApex(copy);
			this.runDigit(copy, said, []);
			copy.resetContext();
		}
	}
}

await runJob(import.meta, ShapesJob);
