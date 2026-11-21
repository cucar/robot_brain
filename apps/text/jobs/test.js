/**
 * Stream a text through one letter channel, one byte a frame, and count how often the brain said the next byte.
 * Each episode shows the text `iterations` times over; the context is reset between episodes, the dictionary kept.
 * With more than one episode, each prints its line; at the end, the last episode's count, the neurons and the functions.
 * With `--describe <letters>`, the dictionary and the base neurons of those letters are printed too.
 *
 * usage: node apps/text/jobs/test.js ["<text>"] [--file <name>] [--episodes 1] [--iterations 1] [--describe <letters>]
 *        [--save-brain <label>] [--load-brain <label>] [--disable-learning]
 * A file named without a folder is read from apps/text/data.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Job, runJob, readPositionals, readFlag, readNumber, formatFixed } from 'robot-brain';

const DATA_FOLDER = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'data');

// every neuron's history depth
const DEPTH = 64;

/**
 * The text to stream: the file named, or the text given on the command line.
 * A bare file name is read from the data folder; a name with a folder, from where it points.
 */
function readText() {
	const file = readFlag('--file');
	if (file === null) return readPositionals()[0] ?? '';
	const where = path.basename(file) === file ? path.join(DATA_FOLDER, file) : path.resolve(file);
	return fs.readFileSync(where, 'utf8');
}

class TextTestJob extends Job {

	constructor() {
		super();
		this.bytes = [...Buffer.from(readText().repeat(readNumber('--iterations', 1)), 'utf8')];
		this.episodes = readNumber('--episodes', 1);
		this.lettersToDescribe = readFlag('--describe');
		this.right = 0;
	}

	getBrainOptions() {
		return { h: DEPTH, policy: 'allOn' };
	}

	declareChannels() {
		this.brain.declareChannel({ name: 'text', event: { name: 'letter', resolution: 256 } });
	}

	/**
	 * Run the episodes; the context is reset before each one after the first.
	 */
	async executeJob() {
		for (let episode = 1; episode <= this.episodes; episode++) {
			if (episode > 1) this.brain.resetContext();
			this.right = this.runEpisode();
			if (this.episodes > 1) this.printEpisode(episode);
		}
	}

	/**
	 * Show each byte in turn; returns how many the brain said in the frame before.
	 */
	runEpisode() {
		const letterDim = this.channels.get('text').eventDim;
		let right = 0;
		this.bytes.forEach((byte, i) => {
			const winners = this.brain.processFrame([{ dim: letterDim, bucket: byte }], []);
			const said = winners.find(w => w.dim === letterDim)?.bucket;
			if (i + 1 < this.bytes.length && said === this.bytes[i + 1]) right++;
		});
		return right;
	}

	/**
	 * An episode's line: how many bytes it said right, of those that had a next byte.
	 */
	printEpisode(episode) {
		const asked = this.bytes.length - 1;
		console.log(`EPISODE ${episode} | right ${this.right}/${asked} (${formatFixed(100 * this.right / Math.max(asked, 1), 2)}%) | functions ${this.brain.countFunctions()}`);
	}

	async showResults() {
		const stats = this.brain.getStats();
		const heights = this.brain.countHeights().join(', ');
		console.log(`right ${this.right}/${this.bytes.length - 1} | neurons ${stats.neurons} alive of ${this.brain.countNeurons()} | patterns ${stats.functions} | heights [${heights}]`);
		if (this.lettersToDescribe === null) return;
		this.printDictionary();
		this.describeLetters();
	}

	/**
	 * Every function of the dictionary, in id order.
	 */
	printDictionary() {
		for (const f of this.brain.getDictionary()) console.log(`DICTF f${f.id} | ${f.description}`);
	}

	/**
	 * The base neurons of the letters named, with their rows and connections.
	 */
	describeLetters() {
		const letterDim = this.channels.get('text').eventDim;
		const neurons = [...Buffer.from(this.lettersToDescribe, 'utf8')].map(byte => this.brain.findBaseNeuron(letterDim, byte));
		for (const text of this.brain.describeNeurons(neurons)) process.stdout.write(text);
	}
}

await runJob(import.meta, TextTestJob);
