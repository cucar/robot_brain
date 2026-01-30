import Job from './job.js';
import TextChannel from '../channels/text.js';

/**
 * Text Training Job - Trains the brain on repeating text patterns
 * Tests pattern learning and prediction accuracy
 */
export default class TextTrainingJob extends Job {

	constructor() {
		super();
		this.hardReset = true; // Hard reset before first episode

		// Simple configuration - edit these values as needed
		this.config = {
			patterns: ['abcabcabc', 'abdabdabd'],  // Patterns to learn (will switch between them)
			maxEpisodes: 2,         // Number of training episodes (one per pattern)
			iterationsPerEpisode: 5 // How many times to repeat pattern per episode
		};

		// Training metrics
		this.episodeResults = [];
		this.currentEpisode = 0;
	}

	/**
	 * Returns the channels for the job
	 */
	getChannels() {
		return [{
			name: 'text',
			channelClass: TextChannel
		}];
	}

	/**
	 * Hook: Show startup information
	 */
	async showStartupInfo() {
		console.log(`🚀 Starting Text Training Job`);
		console.log(`📝 Patterns: ${this.config.patterns.map(p => `"${p}"`).join(' → ')}`);
		console.log(`🔄 Max Episodes: ${this.config.maxEpisodes}`);
		console.log(`🔁 Iterations per Episode: ${this.config.iterationsPerEpisode}`);
		console.log('');
	}

	/**
	 * Hook: Configure text channel after brain initialization
	 */
	async configureChannels() {
		const textChannel = this.brain.channels.get('text'); // the single text channel we set up
		// Pattern will be set per episode in runEpisode()
		textChannel.maxIterations = this.config.iterationsPerEpisode;
	}

	/**
	 * Hook: Execute main job logic - multi-episode training
	 */
	async executeJob() {
		// Run training episodes
		for (this.currentEpisode = 1; this.currentEpisode <= this.config.maxEpisodes; this.currentEpisode++) {
			await this.runEpisode();
		}
		process.exit();
	}

	/**
	 * Hook: Show results
	 */
	async showResults() {
		this.showFinalResults();
	}

	/**
	 * Run a single training episode through the pattern
	 */
	async runEpisode() {
		const startTime = Date.now();

		// Get pattern for this episode (cycle through patterns)
		const patternIndex = (this.currentEpisode - 1) % this.config.patterns.length;
		const currentPattern = this.config.patterns[patternIndex];

		process.stdout.write(`📝 Episode ${this.currentEpisode}/${this.config.maxEpisodes} (pattern: "${currentPattern}")... `);

		// Reset context but keep learned patterns
		await this.brain.resetContext();

		// Reset channel state for new episode and set pattern
		this.resetChannelStates();
		const textChannel = this.brain.channels.get('text');
		textChannel.pattern = currentPattern;
		
		// Initialize episode metrics
		const episodeMetrics = {
			episode: this.currentEpisode,
			frames: 0,
			accuracy: 0
		};

		// Process all frames until channel is exhausted
		let frameCount = 0;
		while (await this.brain.processFrame())
			frameCount++;

		// Collect accuracy stats (base level only)
		if (this.brain.accuracyStats.total > 0)
			episodeMetrics.accuracy = (this.brain.accuracyStats.correct / this.brain.accuracyStats.total * 100);

		const duration = Date.now() - startTime;
		episodeMetrics.duration = duration;
		episodeMetrics.frames = frameCount;

		this.episodeResults.push(episodeMetrics);
		
		// Show episode summary
		const acc = episodeMetrics.accuracy.toFixed(1);
		console.log(`Accuracy: ${acc}% (${frameCount} frames, ${duration}ms)`);
	}

	/**
	 * Reset all channel states for a new episode
	 */
	resetChannelStates() {
		for (const [_, channel] of this.brain.channels) {
			channel.currentLetterIndex = 0;
			channel.patternIterations = 0;
			channel.currentPosition = 0;
			channel.lastPredictedChar = null;
		}
	}

	/**
	 * Show final training results
	 */
	showFinalResults() {
		console.log(`\n🎯 Final Training Results (${this.config.maxEpisodes} episodes):`);
		console.log('='.repeat(80));

		// Calculate average accuracy
		const avgAccuracy = this.episodeResults.reduce((sum, ep) => sum + ep.accuracy, 0) / this.episodeResults.length;

		console.log(`📊 Average Prediction Accuracy: ${avgAccuracy.toFixed(1)}%`);

		// Show improvement trend
		if (this.episodeResults.length >= 4) {
			const first2 = this.episodeResults.slice(0, 2);
			const last2 = this.episodeResults.slice(-2);

			const firstAvg = first2.reduce((sum, ep) => sum + ep.accuracy, 0) / first2.length;
			const lastAvg = last2.reduce((sum, ep) => sum + ep.accuracy, 0) / last2.length;
			const improvement = lastAvg - firstAvg;

			console.log(`\n📈 Learning Progress:`);
			console.log(`   First 2 avg: ${firstAvg.toFixed(1)}%, Last 2 avg: ${lastAvg.toFixed(1)}%, Improvement: ${improvement.toFixed(1)}%`);
		}

		// Show best episodes
		console.log('\n🏆 Best Episodes:');
		const sortedByAccuracy = [...this.episodeResults].sort((a, b) => b.accuracy - a.accuracy);
		for (let i = 0; i < Math.min(3, sortedByAccuracy.length); i++) {
			const ep = sortedByAccuracy[i];
			console.log(`   #${ep.episode}: ${ep.accuracy.toFixed(1)}%`);
		}

		console.log('='.repeat(80));

		// Validation
		const finalAcc = this.episodeResults[this.episodeResults.length - 1].accuracy;

		console.log('\n✅ VALIDATION:');
		if (finalAcc >= 90) {
			console.log(`   ✓ Prediction accuracy: ${finalAcc.toFixed(1)}% (>= 90%)`);
		} else {
			console.log(`   ✗ Prediction accuracy: ${finalAcc.toFixed(1)}% (< 90%)`);
		}
	}
}

