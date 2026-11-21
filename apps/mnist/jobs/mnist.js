/**
 * MNIST (docs/algorithm-mnist.md): an image channel over x, y and time with one event dimension of two values, white
 * and black, every pixel firing one of them in every frame; a digit channel over time alone with ten digit actions.
 * Two frames per image: the image, then the digit that ran and its reward, +1 where it named the image's digit and −1
 * where not, scoped to the digit channel and that one frame (R29, D35).
 * The world names the digit itself for the first `teach` training images (a lesson), and wherever the brain is
 * silent; after that what the brain says runs.
 * Then `test` held-out images on a copy of the brain loaded with learning off: its answer runs, with no reward and no
 * lesson, and nothing of it changes.
 * The data files are the raw (gunzipped) MNIST idx files in the data folder, taken in the file's order: the first
 * `train` images, or with `per-class` the first that many of each digit among them, shown `episodes` times over.
 * Progress every 100 images goes to stderr.
 *
 * usage: node apps/mnist/jobs/mnist.js <data folder> [train=all] [test=500] [h=2048] [size=14] [teach=all]
 *        [threshold=128] [trace-image=0] [per-class=all] [episodes=1] [--regions R] [--columns C]
 * The trace-image argument is read by nothing.
 */
import fs from 'node:fs';
import path from 'node:path';
import { Job, runJob, readPositionals, formatFixed } from 'robot-brain';

const SIDE = 28;
const DIGITS = 10;

// the episodes at the end whose every change to the dictionary is printed
const WATCHED_EPISODES = 5;

/* ---------- Reading the data ---------- */

/**
 * The images of an idx file, each its 784 pixel values.
 */
function readImages(file) {
	const bytes = fs.readFileSync(file);
	const count = bytes.readUInt32BE(4);
	return Array.from({ length: count }, (_, i) => bytes.subarray(16 + i * SIDE * SIDE, 16 + (i + 1) * SIDE * SIDE));
}

/**
 * The labels of an idx file.
 */
function readLabels(file) {
	return [...fs.readFileSync(file).subarray(8)];
}

/**
 * A positional argument as a count: null where it is absent or `all`.
 */
function readCount(value) {
	return value === undefined || value === 'all' ? null : parseInt(value);
}

/**
 * The training images shown each episode, in the file's order: the first `count`, or the first `perClass` of each digit.
 */
function chooseTrainingImages(labels, count, perClass) {
	const order = [];
	const picked = Array(DIGITS).fill(0);
	for (let i = 0; i < count; i++) {
		if (perClass !== null && picked[labels[i]] >= perClass) continue;
		picked[labels[i]]++;
		order.push(i);
	}
	return order;
}

/* ---------- Small helpers ---------- */

/**
 * A share as a percent with one decimal; nothing counted is 0%.
 */
function formatPercent(part, whole) {
	return formatFixed(100 * part / Math.max(whole, 1), 1);
}

/**
 * A ratio with one decimal; nothing counted is 0.
 */
function formatRatio(part, whole) {
	return formatFixed(part / Math.max(whole, 1), 1);
}

/**
 * Each function's stamp by id, from the brain's flat list: id, stamp, id, stamp.
 * A stamp changes where what the function is made of (its nodes, relations and interneuron) changes.
 */
function readStamps(brain) {
	const flat = brain.readFunctionStamps();
	const stamps = new Map();
	for (let i = 0; i < flat.length; i += 2) stamps.set(flat[i], flat[i + 1]);
	return stamps;
}

/* ---------- The dictionary's changes ---------- */

/**
 * The dictionary as last read, function by function: the stamp of what each was made of and how it read.
 * Only the functions whose stamp changed are read from the brain again, so a reading costs the changes, not the dictionary.
 */
class DictionaryWatch {

	constructor(brain) {
		this.brain = brain;
		// each function's stamp and description, by id
		this.known = new Map();
	}

	/**
	 * The stamps now, and the descriptions of the functions that are new or whose stamp changed, by id.
	 */
	readNewReadings() {
		const stamps = readStamps(this.brain);
		const moved = [...stamps.keys()].filter(id => this.known.get(id)?.stamp !== stamps.get(id));
		const descriptions = new Map(this.brain.describeFunctions(moved).map(f => [f.id, f.description]));
		return { stamps, descriptions };
	}

	/**
	 * Take out the functions the dictionary no longer holds, in id order; returns them with how they last read.
	 */
	removeGone(stamps) {
		const gone = [...this.known.keys()].filter(id => !stamps.has(id)).sort((a, b) => a - b);
		const left = gone.map(id => `f${id} ${this.known.get(id).description}`);
		for (const id of gone) this.known.delete(id);
		return left;
	}

	/**
	 * Bring what is known up to the dictionary as it stands, without reporting.
	 */
	sync() {
		const { stamps, descriptions } = this.readNewReadings();
		this.removeGone(stamps);
		for (const [id, description] of descriptions) this.known.set(id, { stamp: stamps.get(id), description });
	}

	/**
	 * What started, left and changed since the last reading, each as `f<id> <text>`, in id order; then bring what is
	 * known up to the dictionary.
	 * A function whose makeup changed but whose reading did not is not counted as changed.
	 */
	listChanges() {
		const { stamps, descriptions } = this.readNewReadings();
		const started = [], changed = [];
		for (const [id, description] of [...descriptions].sort((a, b) => a[0] - b[0])) {
			const known = this.known.get(id);
			if (!known) started.push(`f${id} ${description}`);
			else if (known.description !== description) changed.push(`f${id} ${description}`);
			this.known.set(id, { stamp: stamps.get(id), description });
		}
		const left = this.removeGone(stamps);
		return { started, left, changed };
	}
}

/* ---------- The job ---------- */

class MnistJob extends Job {

	constructor() {
		super();
		const args = readPositionals();
		if (!args[0]) throw new Error('usage: mnist <data folder> [train] [test] [h] [size] [teach] [threshold]');
		this.folder = args[0];
		this.trainCount = readCount(args[1]);
		this.testCount = readCount(args[2]) ?? 500;
		this.depth = readCount(args[3]) ?? 2048;
		this.size = readCount(args[4]) ?? 14;
		this.teach = readCount(args[5]) ?? Infinity;
		this.threshold = readCount(args[6]) ?? 128;
		this.perClass = readCount(args[8]);
		this.episodes = Math.max(readCount(args[9]) ?? 1, 1);
		if (SIDE % this.size !== 0) throw new Error('size must divide 28');
		this.block = SIDE / this.size;
	}

	getBrainOptions() {
		return { h: this.depth, policy: 'eventsToActions' };
	}

	declareChannels() {
		const x = this.brain.declareAxis('x');
		const y = this.brain.declareAxis('y');
		this.brain.declareChannel({ name: 'image', event: { name: 'pixel', resolution: 2 }, axes: [x, y] });
		this.brain.declareChannel({ name: 'digit', action: { name: 'digit', resolution: DIGITS } });
	}

	async executeJob() {
		this.pixelDim = this.channels.get('image').eventDim;
		this.digitDim = this.channels.get('digit').actionDim;
		this.digitChannel = this.channels.get('digit').channelId;
		this.readData();
		this.printHeader();
		this.train();
		this.reportDictionary();
		this.test();
	}

	/**
	 * Read the training and test images and labels, and choose the images each episode shows.
	 */
	readData() {
		const file = name => path.join(this.folder, name);
		this.trainImages = readImages(file('train-images-idx3-ubyte'));
		this.trainLabels = readLabels(file('train-labels-idx1-ubyte'));
		this.testImages = readImages(file('t10k-images-idx3-ubyte'));
		this.testLabels = readLabels(file('t10k-labels-idx1-ubyte'));
		const count = Math.min(this.trainCount ?? this.trainImages.length, this.trainImages.length);
		this.order = chooseTrainingImages(this.trainLabels, count, this.perClass);
		this.testCount = Math.min(this.testCount, this.testImages.length);
		this.shown = this.order.length * this.episodes;
	}

	printHeader() {
		const taught = this.teach >= this.shown ? 'all' : `first ${this.teach}`;
		console.log(`mnist: ${this.size}x${this.size}, black and white pixels, ${this.order.length} training images x ${this.episodes} episodes (${taught} taught), test ${this.testCount}, H=${this.depth}`);
	}

	/**
	 * The pixels of an image: 1 where the block's mean reaches the threshold, 0 elsewhere.
	 */
	makeInputs(pixels) {
		const inputs = [];
		for (let y = 0; y < this.size; y++) {
			for (let x = 0; x < this.size; x++) {
				inputs.push({ dim: this.pixelDim, bucket: this.readBlock(pixels, x, y) >= this.threshold ? 1 : 0, position: [x, y] });
			}
		}
		return inputs;
	}

	/**
	 * The mean of a block of pixels, rounded down.
	 */
	readBlock(pixels, x, y) {
		let sum = 0;
		for (let dy = 0; dy < this.block; dy++) {
			for (let dx = 0; dx < this.block; dx++) sum += pixels[(y * this.block + dy) * SIDE + x * this.block + dx];
		}
		return Math.floor(sum / (this.block * this.block));
	}

	/**
	 * The image frame: returns the digit the brain said, or null, and the milliseconds it took.
	 */
	showImage(brain, pixels) {
		const start = performance.now();
		const winners = brain.processFrame(this.makeInputs(pixels), []);
		const said = winners.find(w => w.dim === this.digitDim)?.bucket ?? null;
		return { said, ms: performance.now() - start };
	}

	/**
	 * The digit frame: the digit that ran, if any, and its reward, if any; returns the milliseconds it took.
	 */
	answer(brain, digit, reward) {
		const start = performance.now();
		const inputs = digit === null ? [] : [{ dim: this.digitDim, bucket: digit }];
		const rewards = reward === null ? [] : [{ reward, channels: [this.digitChannel], frames: 1 }];
		brain.processFrame(inputs, rewards);
		return performance.now() - start;
	}

	/* ---------- Training ---------- */

	/**
	 * Show every training image, episode after episode; the context is reset after each, so no image sees another.
	 */
	train() {
		this.trainStart = Date.now();
		this.totals = { spoke: 0, right: 0, lessons: 0, ms: 0, apex: 0 };
		this.window = { spoke: 0, right: 0, ms: 0, added: 0, retired: 0 };
		this.imagesShown = 0;
		this.lastApex = [];
		this.watch = new DictionaryWatch(this.brain);
		let lastDictionary = '';
		for (let episode = 0; episode < this.episodes; episode++) {
			const counts = this.runEpisode(episode);
			const dictionary = this.brain.getDictionary().map(f => `${f.id}\u0000${f.description}`).join('\n');
			const dictionaryChanged = dictionary !== lastDictionary;
			lastDictionary = dictionary;
			this.printEpisode(episode, counts, dictionaryChanged);
		}
		this.printTraining();
	}

	/**
	 * One episode: every training image once. Returns the episode's counts.
	 */
	runEpisode(episode) {
		const counts = { right: 0, spoke: 0, levels: 0, ms: 0, added: 0, retired: 0, apex: 0, said: [], apexNote: '' };
		const isWatched = episode + WATCHED_EPISODES >= this.episodes;
		const isLast = episode + 1 === this.episodes;
		for (const i of this.order) {
			this.imagesShown++;
			const label = this.trainLabels[i];
			if (isWatched) this.watch.sync();
			const { said, ms } = this.showImage(this.brain, this.trainImages[i]);
			this.noteImageFrame(episode, i, counts, isLast);
			this.totals.ms += ms;
			counts.ms += ms;
			counts.said.push(`${label}:${said ?? '-'}`);
			this.scoreImage(said, label, counts);
			this.totals.ms += this.runAnswer(said, label);
			if (isWatched) this.printChanges(episode, i);
			this.brain.resetContext();
			if (this.imagesShown % 100 === 0) this.printProgress();
		}
		return counts;
	}

	/**
	 * After an image frame: the calls on the apex, the functions added and retired, the levels, and in the last
	 * episode the image's line.
	 */
	noteImageFrame(episode, i, counts, isLast) {
		const stats = this.brain.getStats();
		const apex = this.brain.countApex();
		counts.apex += apex.calls;
		if (isLast) console.log(`LASTEP image ${i} (digit ${this.trainLabels[i]}) | calls on the apex ${apex.calls} | bare pixels on the apex ${apex.events} | levels ${stats.levels}`);
		this.window.added += stats.functionsAdded;
		counts.added += stats.functionsAdded;
		counts.retired += stats.functionsRetired;
		this.window.retired += stats.functionsRetired;
		counts.levels = Math.max(counts.levels, stats.levels);
		this.totals.apex += stats.apexActivations;
		if (this.order.length === 1) counts.apexNote = this.noteSingleImageApex(episode, stats);
	}

	/**
	 * With one training image: what stands on the apex after it, each neuron with its place, and how much of it stood
	 * the time before.
	 */
	noteSingleImageApex(episode, stats) {
		const apex = this.brain.getLastFrame().activations.filter(a => a.onApex);
		const standing = apex.filter(a => a.kind !== 'action').map(a => {
			const [x, y] = a.position;
			if (a.kind === 'inter') return `#${a.neuron}@${x},${y}`;
			return `${this.brain.readNeurons([a.neuron])[0].bucket === 1 ? 'B' : 'W'}@${x},${y}`;
		}).sort();
		const kept = standing.filter(s => this.lastApex.includes(s)).length;
		let note = ` | apex ${standing.length} (${kept} kept, ${standing.length - kept} new, ${this.lastApex.length - kept} gone) levels ${stats.levels} added ${stats.functionsAdded}`;
		if (episode + 1 === this.episodes || episode < 3) note += ` [${standing.join(' ')}]`;
		this.lastApex = standing;
		return note;
	}

	/**
	 * Count what the brain said against the label.
	 */
	scoreImage(said, label, counts) {
		if (said === null) return;
		this.totals.spoke++;
		this.window.spoke++;
		counts.spoke++;
		if (said !== label) return;
		this.totals.right++;
		this.window.right++;
		counts.right++;
	}

	/**
	 * Run the digit frame of a training image: the label as a lesson for the first images taught and wherever the brain
	 * was silent, else what it said; +1 where the digit that ran is the label, −1 where not.
	 */
	runAnswer(said, label) {
		const isLesson = this.imagesShown <= this.teach || said === null;
		if (isLesson) this.totals.lessons++;
		const ran = isLesson ? label : said;
		return this.answer(this.brain, ran, ran === label ? 1 : -1);
	}

	/**
	 * Every function that started, left or changed with this image.
	 */
	printChanges(episode, i) {
		const { started, left, changed } = this.watch.listChanges();
		if (started.length + left.length + changed.length === 0) return;
		console.log(`CHANGE episode ${episode + 1} image ${i} | started ${started.length} | left ${left.length} | changed ${changed.length}`);
		for (const s of started) console.log(`    + ${s}`);
		for (const s of left) console.log(`    - ${s}`);
		for (const s of changed) console.log(`    ~ ${s}`);
	}

	/**
	 * Every 100 images, to stderr: the last 100 images' answers, speed and functions, and the totals so far.
	 */
	printProgress() {
		const w = this.window;
		const k = this.imagesShown;
		console.error(`  image ${k}/${this.shown} | last 100: spoke ${w.spoke} right ${w.right} (${formatPercent(w.right, w.spoke)}% of spoken) ${formatFixed((this.totals.ms - w.ms) / 100, 0)} ms/image, functions added ${w.added} retired ${w.retired} | neurons ${this.brain.countLivingNeurons()} functions ${this.brain.countFunctions()} | apex/image ${formatFixed(this.totals.apex / k, 1)} | ${formatFixed(this.totals.ms / k, 1)} ms/image so far`);
		this.window = { spoke: 0, right: 0, ms: this.totals.ms, added: 0, retired: 0 };
	}

	/**
	 * The episode's line, and with more than one episode its details.
	 */
	printEpisode(episode, counts, dictionaryChanged) {
		const images = this.order.length;
		console.log(`EPISODE ${episode + 1} | dictionary ${dictionaryChanged ? 'changed' : 'same'} | functions ${this.brain.countFunctions()} | right ${counts.right}/${images} | calls on the apex per image ${formatRatio(counts.apex, images)}`);
		if (this.episodes <= 1) return;
		console.log(`  episode ${String(episode + 1).padStart(2)}: added ${counts.added} retired ${counts.retired} | right ${counts.right}/${images} (spoke ${counts.spoke}) | max levels ${counts.levels} | neurons ${this.brain.countLivingNeurons()} functions ${this.brain.countFunctions()} | ${formatFixed(counts.ms / Math.max(images, 1), 0)} ms/image | digit:said ${counts.said.join(' ')}${counts.apexNote}`);
	}

	printTraining() {
		const t = this.totals;
		const seconds = formatFixed((Date.now() - this.trainStart) / 1000, 0);
		console.log(`NEVERCALLED ${this.brain.countFunctionsNeverCalled()} of ${this.brain.countFunctions()}`);
		console.log(`  train: ${this.shown} images | spoke ${t.spoke} | right ${t.right} (${formatPercent(t.right, this.shown)}% of all, ${formatPercent(t.right, t.spoke)}% of spoken) | lessons ${t.lessons} | apex/image ${formatRatio(t.apex, this.shown)} | neurons ${this.brain.countLivingNeurons()} functions ${this.brain.countFunctions()} | ${seconds}s`);
	}

	/**
	 * What the dictionary holds: functions counted by relations, nodes, cycle and interneuron; then the twelve largest.
	 */
	reportDictionary() {
		const dictionary = this.brain.getDictionary();
		const groups = new Map();
		for (const f of dictionary) {
			const key = [f.relations.length, f.nodes.length, f.isCycle, f.interneuron != null];
			const id = JSON.stringify(key);
			groups.set(id, { key, count: (groups.get(id)?.count ?? 0) + 1 });
		}
		const compareKeys = (a, b) => a.key[0] - b.key[0] || a.key[1] - b.key[1] || Number(a.key[2]) - Number(b.key[2]) || Number(a.key[3]) - Number(b.key[3]);
		const entries = [...groups.values()].sort(compareKeys).map(({ key, count }) => `(${key.join(', ')}): ${count}`);
		console.log(`  dictionary by (relations, nodes, cycle, has interneuron): {${entries.join(', ')}}`);
		const largest = dictionary.filter(f => f.relations.length > 1).sort((a, b) => b.relations.length - a.relations.length);
		for (const f of largest.slice(0, 12)) console.log(`    ${f.description}`);
	}

	/* ---------- Testing ---------- */

	/**
	 * Show each test image to a copy with learning off: its answer runs, unrewarded, and nothing is taught.
	 */
	test() {
		const brain = this.testCount > 0 ? this.copyBrainWithoutLearning() : this.brain;
		const start = Date.now();
		let spoke = 0, right = 0, apex = 0;
		const confusion = Array.from({ length: DIGITS }, () => Array(DIGITS).fill(0));
		for (let t = 0; t < this.testCount; t++) {
			const label = this.testLabels[t];
			const { said } = this.showImage(brain, this.testImages[t]);
			apex += brain.getStats().apexActivations;
			if (said !== null) {
				spoke++;
				confusion[label][said]++;
				if (said === label) right++;
			}
			this.answer(brain, said, null);
			brain.resetContext();
			if ((t + 1) % 100 === 0) console.error(`  test ${t + 1}/${this.testCount} | spoke ${spoke} right ${right} (${formatPercent(right, t + 1)}%)`);
		}
		const seconds = formatFixed((Date.now() - start) / 1000, 0);
		console.log(`  test: ${this.testCount} images | spoke ${spoke} | right ${right} (${formatPercent(right, this.testCount)}%) | apex/image ${formatRatio(apex, this.testCount)} | neurons ${brain.countLivingNeurons()} functions ${brain.countFunctions()} | ${seconds}s`);
		console.log('  confusion (row = digit, column = said):');
		confusion.forEach((row, digit) => console.log(`    ${digit} [${row.join(', ')}]`));
	}
}

await runJob(import.meta, MnistJob);
