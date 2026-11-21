/**
 * Stream a text through one letter channel, one byte a frame, and count how often the brain said the next byte.
 * At the end, print the dictionary and describe the base neurons of the letters named.
 *
 * usage: node apps/text/jobs/text.js "<text>" [unused] [history depth=64] [letters to describe=h] [--regions R] [--columns C]
 * The second argument is read by nothing: a base neuron sees one position in every dimension.
 */
import { Job, runJob, readPositionals } from 'robot-brain';

const DEFAULT_TEXT = 'th ch sh wh th ch sh wh th ch sh wh th ch sh wh ';

class TextJob extends Job {

	constructor() {
		super();
		const [text, , depth, letters] = readPositionals();
		this.bytes = [...Buffer.from(text ?? DEFAULT_TEXT, 'utf8')];
		this.depth = depth ? parseInt(depth) : 64;
		this.lettersToDescribe = letters ?? 'h';
		this.right = 0;
	}

	getBrainOptions() {
		return { h: this.depth, policy: 'allOn' };
	}

	declareChannels() {
		this.brain.declareChannel({ name: 'text', event: { name: 'letter', resolution: 256 } });
	}

	/**
	 * Show each byte in turn, and count each one the brain said in the frame before.
	 */
	async executeJob() {
		const letterDim = this.channels.get('text').eventDim;
		this.bytes.forEach((byte, i) => {
			const winners = this.brain.processFrame([{ dim: letterDim, bucket: byte }], []);
			const said = winners.find(w => w.dim === letterDim)?.bucket;
			if (i + 1 < this.bytes.length && said === this.bytes[i + 1]) this.right++;
		});
	}

	async showResults() {
		const stats = this.brain.getStats();
		const heights = this.brain.countHeights().join(', ');
		console.log(`right ${this.right}/${this.bytes.length - 1} | neurons ${stats.neurons} alive of ${this.brain.countNeurons()} | patterns ${stats.functions} | heights [${heights}]`);
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

await runJob(import.meta, TextJob);
