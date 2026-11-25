/**
 * MNIST (docs/algorithm-mnist.md): an image channel over x, y and time with one event dimension, a pixel's value in
 * `buckets` steps, every pixel firing in every frame; a digit channel over time alone with ten digit actions.
 *
 * A training run shows two frames per image: the image, then the image's digit, run as a lesson and rewarded +1,
 * scoped to the digit channel and that one frame (R29, D35).
 * The training images are the first `per-class` of each digit in the file's order (with 0, as many as the smallest
 * digit has), or the whole set with `--no-balance`, or the first N with `--train-images N`; shown `episodes` times.
 *
 * An evaluation run (`--load-brain <label> --disable-learning`) trains nothing: it shows the held-out test images
 * (`--test-data`) or the training images to the loaded brain, whose answer runs with no reward, and nothing changes.
 *
 * A Split-MNIST run (`--split`) trains five tasks in turn, the digits {0,1} {2,3} {4,5} {6,7} {8,9}, each on the first
 * `per-class` images of its two digits, once, with no task given and all ten digits as the answers; after each task a
 * copy of the brain with learning off is tested on the first `--max-test-images` test images. It prints the retention
 * matrix (the share right on each task's test digits after each task), the average after the last task, and the
 * forgetting of each task (its best share less its last).
 * Progress every 100 images goes to stderr.
 *
 * usage: node apps/mnist/jobs/test.js [--image-size 7] [--buckets 2] [--per-class 0] [--no-balance]
 *        [--train-images N] [--episodes 1] [--max-test-images 0] [--test-data] [--data <folder>]
 *        [--split] [--silent-black] [--save-brain <label>] [--load-brain <label>] [--disable-learning]
 * With --silent-black a pixel of the lowest value reports nothing, so an image is its white pixels alone.
 * The data folder holds the four raw (gunzipped) MNIST idx files; apps/mnist/data by default.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Job, runJob, formatFixed } from 'robot-brain';

const SIDE = 28;
const DIGITS = 10;

// every neuron's history depth
const DEPTH = 2048;

// the tasks of Split-MNIST, two digits each, trained in this order
const SPLIT_TASKS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]];

// the episodes at the end whose every change to the dictionary is printed
const WATCHED_EPISODES = 5;

const DATA_FOLDER = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'data');

/* ---------- Reading the command line and the data ---------- */

/**
 * The value after a flag on the command line, or null where the flag is absent.
 */
function readFlag(flag) {
	const at = process.argv.indexOf(flag);
	return at !== -1 && process.argv[at + 1] !== undefined ? process.argv[at + 1] : null;
}

/**
 * The value after a flag as a whole number, or the fallback where the flag is absent.
 */
function readNumber(flag, fallback) {
	const value = readFlag(flag);
	return value === null ? fallback : parseInt(value);
}

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
 * The first `perClass` images of each digit, in the file's order; with 0, as many as the smallest digit has.
 */
function chooseBalanced(labels, perClass) {
	const counts = Array(DIGITS).fill(0);
	for (const label of labels) counts[label]++;
	const cap = perClass > 0 ? perClass : Math.min(...counts);
	const order = [];
	const picked = Array(DIGITS).fill(0);
	labels.forEach((label, i) => {
		if (picked[label] >= cap) return;
		picked[label]++;
		order.push(i);
	});
	return order;
}

/**
 * The first `count` images, in the file's order.
 */
function chooseFirst(labels, count) {
	return [...Array(Math.min(count, labels.length)).keys()];
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
		this.folder = readFlag('--data') ?? DATA_FOLDER;
		this.size = readNumber('--image-size', 7);
		this.buckets = readNumber('--buckets', 2);
		this.perClass = readNumber('--per-class', 0);
		this.isBalanced = !process.argv.includes('--no-balance');
		this.trainCount = readNumber('--train-images', null);
		this.episodes = Math.max(readNumber('--episodes', 1), 1);
		this.maxTestImages = readNumber('--max-test-images', 0);
		this.isOnTestData = process.argv.includes('--test-data');
		this.isSplit = process.argv.includes('--split');
		this.isBlackSilent = process.argv.includes('--silent-black');
		if (SIDE % this.size !== 0) throw new Error('the image size must divide 28');
		this.block = SIDE / this.size;
	}

	getBrainOptions() {
		return { h: DEPTH, policy: 'eventsToActions' };
	}

	declareChannels() {
		const x = this.brain.declareAxis('x');
		const y = this.brain.declareAxis('y');
		this.brain.declareChannel({ name: 'image', event: { name: 'pixel', resolution: this.buckets }, axes: [x, y] });
		this.brain.declareChannel({ name: 'digit', action: { name: 'digit', resolution: DIGITS } });
	}

	/**
	 * A training run trains and reports; an evaluation run, with learning off, only evaluates.
	 */
	async executeJob() {
		this.pixelDim = this.channels.get('image').eventDim;
		this.digitDim = this.channels.get('digit').actionDim;
		this.digitChannel = this.channels.get('digit').channelId;
		this.readData();
		if (this.options.learning === false) return this.evaluate();
		if (this.isSplit) return this.runSplit();
		this.printHeader();
		this.train();
		this.reportDictionary();
		this.printEmptyTest();
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
		this.order = this.chooseTrainingImages();
		this.shown = this.order.length * this.episodes;
	}

	/**
	 * The training images each episode shows: the first N, the whole set, or the balanced set.
	 */
	chooseTrainingImages() {
		if (this.trainCount !== null) return chooseFirst(this.trainLabels, this.trainCount);
		if (!this.isBalanced) return chooseFirst(this.trainLabels, this.trainLabels.length);
		return chooseBalanced(this.trainLabels, this.perClass);
	}

	printHeader() {
		console.log(`mnist: ${this.size}x${this.size}, ${this.describePixels()}, ${this.order.length} training images x ${this.episodes} episodes (all taught), test 0, H=${DEPTH}`);
	}

	/**
	 * The pixels' values in words.
	 */
	describePixels() {
		const values = this.buckets === 2 ? 'black and white pixels' : `pixels in ${this.buckets} values`;
		return this.isBlackSilent ? `${values}, the black ones silent` : values;
	}

	/**
	 * The pixels of an image: each block's mean, in `buckets` steps from 0 to 255.
	 */
	makeInputs(pixels) {
		const inputs = [];
		for (let y = 0; y < this.size; y++) {
			for (let x = 0; x < this.size; x++) {
				const bucket = this.readBucket(pixels, x, y);
				// with the black pixels silent, a pixel of the lowest value reports nothing
				if (this.isBlackSilent && bucket === 0) continue;
				inputs.push({ dim: this.pixelDim, bucket, position: [x, y] });
			}
		}
		return inputs;
	}

	/**
	 * A block's mean value in `buckets` steps: the mean times the buckets over 256, rounded down.
	 */
	readBucket(pixels, x, y) {
		let sum = 0;
		for (let dy = 0; dy < this.block; dy++) {
			for (let dx = 0; dx < this.block; dx++) sum += pixels[(y * this.block + dy) * SIDE + x * this.block + dx];
		}
		const area = this.block * this.block;
		return Math.min(Math.floor(sum * this.buckets / (area * 256)), this.buckets - 1);
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
			this.totals.ms += this.teachDigit(label);
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
	 * Run the digit frame of a training image: every training image is a lesson, its digit run and rewarded +1.
	 */
	teachDigit(label) {
		this.totals.lessons++;
		return this.answer(this.brain, label, 1);
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

	/* ---------- Evaluation ---------- */

	/**
	 * A training run evaluates nothing; it prints the evaluation over no images, as the benchmarks' saved output holds.
	 */
	printEmptyTest() {
		this.printEvaluation('test', { count: 0, spoke: 0, right: 0, apex: 0, seconds: 0, confusion: this.makeConfusion() });
	}

	/**
	 * An empty confusion matrix: a row per digit shown, a column per digit said.
	 */
	makeConfusion() {
		return Array.from({ length: DIGITS }, () => Array(DIGITS).fill(0));
	}

	/**
	 * Show each image to the loaded brain, with learning off: its answer runs, unrewarded, and nothing changes.
	 * The images are the held-out test set with `--test-data`, else the training images; at most `--max-test-images`.
	 */
	evaluate() {
		const images = this.isOnTestData ? this.testImages : this.order.map(i => this.trainImages[i]);
		const labels = this.isOnTestData ? this.testLabels : this.order.map(i => this.trainLabels[i]);
		const count = this.maxTestImages > 0 ? Math.min(this.maxTestImages, images.length) : images.length;
		const tag = this.isOnTestData ? 'test' : 'train';
		console.log(`mnist: ${this.size}x${this.size}, ${this.describePixels()}, the brain loaded from ${this.options.loadBrain} on ${count} ${tag} images, learning off`);
		const result = { count, spoke: 0, right: 0, apex: 0, confusion: this.makeConfusion() };
		const start = Date.now();
		for (let t = 0; t < count; t++) {
			this.evaluateImage(this.brain, images[t], labels[t], result);
			if ((t + 1) % 100 === 0) console.error(`  ${tag} ${t + 1}/${count} | spoke ${result.spoke} right ${result.right} (${formatPercent(result.right, t + 1)}%)`);
		}
		result.seconds = (Date.now() - start) / 1000;
		this.printEvaluation(tag, result);
	}

	/**
	 * One image to a frozen brain: count what it said against the label; its answer runs, unrewarded.
	 */
	evaluateImage(brain, pixels, label, result) {
		const { said } = this.showImage(brain, pixels);
		result.apex += brain.getStats().apexActivations;
		if (said !== null) {
			result.spoke++;
			result.confusion[label][said]++;
			if (said === label) result.right++;
		}
		this.answer(brain, said, null);
		brain.resetContext();
	}

	/* ---------- Split-MNIST ---------- */

	/**
	 * Train the five tasks in turn; after each, test a frozen copy on the test images; then print the retention matrix.
	 */
	runSplit() {
		const testCount = this.maxTestImages > 0 ? Math.min(this.maxTestImages, this.testImages.length) : this.testImages.length;
		const cap = this.findPerClassCap();
		console.log(`split mnist: ${this.size}x${this.size}, ${this.describePixels()}, ${cap} images of each digit, ${SPLIT_TASKS.length} tasks of two digits, ${testCount} test images after each task, H=${DEPTH}`);
		this.totals = { spoke: 0, right: 0, lessons: 0, ms: 0, apex: 0 };
		const matrix = [];
		SPLIT_TASKS.forEach((digits, t) => {
			this.trainTask(t, digits, cap);
			matrix.push(this.testTasks(t, testCount));
		});
		this.printRetention(matrix);
	}

	/**
	 * The images of each digit a task takes: `--per-class`, or with 0, as many as the smallest digit has.
	 */
	findPerClassCap() {
		const counts = Array(DIGITS).fill(0);
		for (const label of this.trainLabels) counts[label]++;
		return this.perClass > 0 ? this.perClass : Math.min(...counts);
	}

	/**
	 * Train one task: the first `cap` images of each of its digits, in the file's order, each a lesson.
	 */
	trainTask(t, digits, cap) {
		const order = chooseBalanced(this.trainLabels, cap).filter(i => digits.includes(this.trainLabels[i]));
		const start = Date.now();
		let right = 0;
		order.forEach((i, n) => {
			const { said } = this.showImage(this.brain, this.trainImages[i]);
			if (said === this.trainLabels[i]) right++;
			this.teachDigit(this.trainLabels[i]);
			this.brain.resetContext();
			if ((n + 1) % 100 === 0) console.error(`  task ${t} image ${n + 1}/${order.length} | right so far ${right} | functions ${this.brain.countFunctions()}`);
		});
		console.log(`TASK ${t} (digits ${digits.join(' and ')}) | ${order.length} images | right ${right} (${formatPercent(right, order.length)}%) | neurons ${this.brain.countLivingNeurons()} functions ${this.brain.countFunctions()} | ${formatFixed((Date.now() - start) / 1000, 0)}s`);
	}

	/**
	 * Test a copy of the brain with learning off on the test images; returns the share right on each task's digits.
	 */
	testTasks(t, testCount) {
		const copy = this.copyBrainWithoutLearning();
		const result = { count: testCount, spoke: 0, right: 0, apex: 0, confusion: this.makeConfusion() };
		const start = Date.now();
		for (let n = 0; n < testCount; n++) {
			this.evaluateImage(copy, this.testImages[n], this.testLabels[n], result);
			if ((n + 1) % 1000 === 0) console.error(`  after task ${t}: test ${n + 1}/${testCount} | right ${result.right}`);
		}
		const shares = SPLIT_TASKS.map(digits => {
			const shown = digits.reduce((sum, d) => sum + result.confusion[d].reduce((a, b) => a + b, 0), 0);
			const total = this.testLabels.slice(0, testCount).filter(label => digits.includes(label)).length;
			const right = digits.reduce((sum, d) => sum + result.confusion[d][d], 0);
			return { right, total, shown };
		});
		console.log(`TEST after task ${t} | ${testCount} images | right ${result.right} (${formatPercent(result.right, testCount)}%) | ${formatFixed((Date.now() - start) / 1000, 0)}s`);
		return shares.map(({ right, total }) => right / Math.max(total, 1));
	}

	/**
	 * The retention matrix, the average share after the last task, and each task's forgetting.
	 */
	printRetention(matrix) {
		const label = digits => `T${SPLIT_TASKS.indexOf(digits)}(${digits.join('')})`;
		console.log(`Retention matrix  ${SPLIT_TASKS.map(d => label(d).padEnd(9)).join('')}`);
		matrix.forEach((row, t) => console.log(`  after T${t}        ${row.map(share => `${formatFixed(100 * share, 1)}%`.padEnd(9)).join('')}`));
		const last = matrix[matrix.length - 1];
		const average = last.reduce((a, b) => a + b, 0) / last.length;
		console.log(`  Average accuracy after the last task: ${formatFixed(100 * average, 2)}%`);
		const forgetting = SPLIT_TASKS.map((_, j) => Math.max(...matrix.slice(j).map(row => row[j])) - last[j]);
		console.log(`  Forgetting per task (best less last): ${forgetting.map((f, j) => `T${j}:${formatFixed(100 * f, 1)}pp`).join(' ')}`);
		console.log(`  Average forgetting: ${formatFixed(100 * forgetting.reduce((a, b) => a + b, 0) / forgetting.length, 2)}pp`);
	}

	/**
	 * The evaluation's line and its confusion matrix.
	 */
	printEvaluation(tag, { count, spoke, right, apex, seconds, confusion }) {
		console.log(`  ${tag}: ${count} images | spoke ${spoke} | right ${right} (${formatPercent(right, count)}%) | apex/image ${formatRatio(apex, count)} | neurons ${this.brain.countLivingNeurons()} functions ${this.brain.countFunctions()} | ${formatFixed(seconds, 0)}s`);
		console.log('  confusion (row = digit, column = said):');
		confusion.forEach((row, digit) => console.log(`    ${digit} [${row.join(', ')}]`));
	}
}

await runJob(import.meta, MnistJob);
