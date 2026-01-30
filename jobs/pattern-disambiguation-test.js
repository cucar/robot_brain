import Job from './job.js';
import TextChannel from '../channels/text.js';

/**
 * Pattern Disambiguation Test - Stock-like Multi-Dimensional
 *
 * Replicates the stock trading scenario with multiple dimensions:
 * - 3 "stocks" with price_change and volume_change (6 dimensions total)
 * - Sequence A: specific price/volume pattern → predicts outcome D
 * - Sequence B: different price/volume pattern with overlapping middle → predicts outcome Z
 *
 * Tests if patterns can disambiguate when same middle state appears in different contexts
 */
export default class PatternDisambiguationTest extends Job {

	constructor() {
		super();
		this.hardReset = true;

		this.config = {
			trainingEpisodes: 500,
			testEpisodes: 2
		};

		this.currentEpisode = 0;
	}

	getChannels() {
		return [{
			name: 'text',
			channelClass: TextChannel
		}];
	}

	async showStartupInfo() {
		console.log(`🧪 Pattern Disambiguation Test`);
		console.log(`   Sequence A: a b c d`);
		console.log(`   Sequence B: x y c z`);
		console.log(`   Training episodes: ${this.config.trainingEpisodes}`);
		console.log('');
	}

	async configureChannels() {
		const textChannel = this.brain.channels.get('text');
		textChannel.debug = false;
		textChannel.maxIterations = 1;
	}

	async executeJob() {
		for (this.currentEpisode = 1; this.currentEpisode <= this.config.trainingEpisodes + this.config.testEpisodes; this.currentEpisode++) {
			await this.runEpisode();
		}
	}

	async runEpisode() {
		const isTraining = this.currentEpisode <= this.config.trainingEpisodes;
		const isTest = !isTraining;

		if (isTest) {
			console.log(`\n${'='.repeat(60)}`);
			console.log(`TEST EPISODE ${this.currentEpisode - this.config.trainingEpisodes}/${this.config.testEpisodes}`);
			console.log('='.repeat(60));
			this.brain.debug = true;
		} else {
			if (this.currentEpisode % 20 === 0) console.log(`Training episode ${this.currentEpisode}/${this.config.trainingEpisodes}`);
			this.brain.debug = false;
		}

		this.brain.debugPatterns = true; // Always show pattern debug

		await this.brain.resetContext();
		this.brain.resetAccuracyStats();

		// Sequence A: frame1 → frame2 → frame3 (middle) → frameD
		// Sequence B: frameX → frameY → frame3 (same middle) → frameZ

		if (isTraining) {
			// Run both sequences - use different values for each dimension to create peaks
			await this.processFrame({ s1_price: 1, s1_vol: 10, s2_price: 2, s2_vol: 20, s3_price: 3, s3_vol: 30 }); // A1
			await this.processFrame({ s1_price: 4, s1_vol: 40, s2_price: 5, s2_vol: 50, s3_price: 6, s3_vol: 60 }); // A2
			await this.processFrame({ s1_price: 7, s1_vol: 70, s2_price: 8, s2_vol: 80, s3_price: 9, s3_vol: 90 }); // Middle (same in both)
			await this.processFrame({ s1_price: 11, s1_vol: 110, s2_price: 12, s2_vol: 120, s3_price: 13, s3_vol: 130 }); // D

			await this.processFrame({ s1_price: 14, s1_vol: 140, s2_price: 15, s2_vol: 150, s3_price: 16, s3_vol: 160 }); // B1
			await this.processFrame({ s1_price: 17, s1_vol: 170, s2_price: 18, s2_vol: 180, s3_price: 19, s3_vol: 190 }); // B2
			await this.processFrame({ s1_price: 7, s1_vol: 70, s2_price: 8, s2_vol: 80, s3_price: 9, s3_vol: 90 }); // Middle (same!)
			await this.processFrame({ s1_price: 21, s1_vol: 210, s2_price: 22, s2_vol: 220, s3_price: 23, s3_vol: 230 }); // Z
		} else {
			// Test: can it disambiguate?
			console.log('\nTest A: After A1→A2→Middle, should predict D');
			await this.processFrame({ s1_price: 1, s1_vol: 10, s2_price: 2, s2_vol: 20, s3_price: 3, s3_vol: 30 });
			await this.processFrame({ s1_price: 4, s1_vol: 40, s2_price: 5, s2_vol: 50, s3_price: 6, s3_vol: 60 });
			await this.processFrame({ s1_price: 7, s1_vol: 70, s2_price: 8, s2_vol: 80, s3_price: 9, s3_vol: 90 });

			console.log('\nTest B: After B1→B2→Middle, should predict Z');
			await this.processFrame({ s1_price: 14, s1_vol: 140, s2_price: 15, s2_vol: 150, s3_price: 16, s3_vol: 160 });
			await this.processFrame({ s1_price: 17, s1_vol: 170, s2_price: 18, s2_vol: 180, s3_price: 19, s3_vol: 190 });
			await this.processFrame({ s1_price: 7, s1_vol: 70, s2_price: 8, s2_vol: 80, s3_price: 9, s3_vol: 90 });
		}

		if (isTest) {
			console.log(`\nLearned: ${this.brain.neurons.size()} neurons, ${this.brain.connections.size()} connections, ${this.brain.patterns.size()} patterns`);
		}
	}

	async processFrame() {
		// NOTE: This test needs refactoring to work with channel-based frame processing
		// The manual frame construction was already being ignored by the brain
		await this.brain.processFrame();
	}

	resetChannelStates() {
		for (const [_, channel] of this.brain.channels) {
			channel.currentLetterIndex = 0;
			channel.patternIterations = 0;
			channel.lastPredictedChar = null;
		}
	}

	async showFinalResults() {
		console.log(`\n${'='.repeat(60)}`);
		console.log('ANALYSIS');
		console.log('='.repeat(60));

		// Find neurons for d and z
		const allNeurons = Array.from(this.brain.neurons.byCoordinates.values());
		const neuronD = allNeurons.find(nid => {
			const coords = this.brain.neurons.getCoordinates(nid);
			const charCoord = coords.find(c => this.brain.dimensions.get(c.dimension_id)?.name === 'char_input');
			return charCoord && charCoord.val === 'd'.charCodeAt(0);
		});
		const neuronZ = allNeurons.find(nid => {
			const coords = this.brain.neurons.getCoordinates(nid);
			const charCoord = coords.find(c => this.brain.dimensions.get(c.dimension_id)?.name === 'char_input');
			return charCoord && charCoord.val === 'z'.charCodeAt(0);
		});

		console.log('\n🔍 Peak status:');
		if (neuronD) {
			const patternsForD = this.brain.patternPeaks.getPatterns(neuronD);
			console.log(`  'd' (N${neuronD}): ${patternsForD ? patternsForD.size : 0} patterns have this as peak`);
		} else console.log(`  'd': NOT FOUND`);

		if (neuronZ) {
			const patternsForZ = this.brain.patternPeaks.getPatterns(neuronZ);
			console.log(`  'z' (N${neuronZ}): ${patternsForZ ? patternsForZ.size : 0} patterns have this as peak`);
		} else console.log(`  'z': NOT FOUND`);

		console.log('\n✅ Key question: Are d and z identified as peaks?');
		console.log('   If not, pattern inference CANNOT predict them!');
		console.log('='.repeat(60));
	}
}

