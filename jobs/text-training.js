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
			pattern: 'abcabcabc',  // Pattern to learn
			maxEpisodes: 1,         // Number of training episodes
			iterationsPerEpisode: 3 // How many times to repeat pattern per episode
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
		console.log(`📝 Pattern: "${this.config.pattern}"`);
		console.log(`🔄 Max Episodes: ${this.config.maxEpisodes}`);
		console.log(`🔁 Iterations per Episode: ${this.config.iterationsPerEpisode}`);
		console.log('');
	}

	/**
	 * Hook: Configure text channel after brain initialization
	 */
	async configureChannels() {
		const textChannel = this.brain.channels.get('text'); // the single text channel we set up
		textChannel.pattern = this.config.pattern;
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
		process.stdout.write(`📝 Episode ${this.currentEpisode}/${this.config.maxEpisodes}... `);

		// Reset context but keep learned patterns
		await this.brain.resetContext();
		
		// Reset channel state for new episode
		this.resetChannelStates();
		
		// Initialize episode metrics
		const episodeMetrics = {
			episode: this.currentEpisode,
			frames: 0,
			level0Accuracy: { connection: 0, pattern: 0, resolved: 0 },
			level1Accuracy: { connection: 0, pattern: 0, resolved: 0 },
			level2Accuracy: { connection: 0, pattern: 0, resolved: 0 }
		};

		// Process all frames until channel is exhausted
		let frameCount = 0;
		while (true) {
			// Get combined frame from all channels
			const frame = await this.brain.getFrame();

			// If no input data from any channel, episode is complete
			if (!frame || frame.length === 0) break;

			frameCount++;

			// Get feedback and process frame
			const feedback = await this.brain.getFeedback();
			await this.brain.processFrame(frame, feedback);
		}

		// Collect accuracy stats
		for (let level = 0; level <= 2; level++) {
			if (this.brain.accuracyStats.has(level)) {
				const stats = this.brain.accuracyStats.get(level);
				const levelKey = `level${level}Accuracy`;
				
				if (stats.connection.total > 0)
					episodeMetrics[levelKey].connection = (stats.connection.correct / stats.connection.total * 100);
				
				if (stats.pattern.total > 0)
					episodeMetrics[levelKey].pattern = (stats.pattern.correct / stats.pattern.total * 100);
				
				if (stats.resolved.total > 0)
					episodeMetrics[levelKey].resolved = (stats.resolved.correct / stats.resolved.total * 100);
			}
		}

		const duration = Date.now() - startTime;
		episodeMetrics.duration = duration;
		episodeMetrics.frames = frameCount;

		this.episodeResults.push(episodeMetrics);
		
		// Show episode summary
		const l0Acc = episodeMetrics.level0Accuracy.resolved.toFixed(1);
		const l1Acc = episodeMetrics.level1Accuracy.resolved.toFixed(1);
		console.log(`L0: ${l0Acc}%, L1: ${l1Acc}% (${frameCount} frames, ${duration}ms)`);
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
		
		// Calculate averages
		const avgL0Conn = this.episodeResults.reduce((sum, ep) => sum + ep.level0Accuracy.connection, 0) / this.episodeResults.length;
		const avgL0Pattern = this.episodeResults.reduce((sum, ep) => sum + ep.level0Accuracy.pattern, 0) / this.episodeResults.length;
		const avgL0Resolved = this.episodeResults.reduce((sum, ep) => sum + ep.level0Accuracy.resolved, 0) / this.episodeResults.length;
		
		const avgL1Conn = this.episodeResults.reduce((sum, ep) => sum + ep.level1Accuracy.connection, 0) / this.episodeResults.length;
		const avgL1Pattern = this.episodeResults.reduce((sum, ep) => sum + ep.level1Accuracy.pattern, 0) / this.episodeResults.length;
		const avgL1Resolved = this.episodeResults.reduce((sum, ep) => sum + ep.level1Accuracy.resolved, 0) / this.episodeResults.length;
		
		console.log(`📊 Average Prediction Accuracy:`);
		console.log(`   Level 0: Conn=${avgL0Conn.toFixed(1)}%, Pattern=${avgL0Pattern.toFixed(1)}%, Resolved=${avgL0Resolved.toFixed(1)}%`);
		console.log(`   Level 1: Conn=${avgL1Conn.toFixed(1)}%, Pattern=${avgL1Pattern.toFixed(1)}%, Resolved=${avgL1Resolved.toFixed(1)}%`);
		
		// Show improvement trend
		if (this.episodeResults.length >= 4) {
			const first2 = this.episodeResults.slice(0, 2);
			const last2 = this.episodeResults.slice(-2);
			
			const firstAvgL0 = first2.reduce((sum, ep) => sum + ep.level0Accuracy.resolved, 0) / first2.length;
			const lastAvgL0 = last2.reduce((sum, ep) => sum + ep.level0Accuracy.resolved, 0) / last2.length;
			const improvementL0 = lastAvgL0 - firstAvgL0;
			
			const firstAvgL1 = first2.reduce((sum, ep) => sum + ep.level1Accuracy.resolved, 0) / first2.length;
			const lastAvgL1 = last2.reduce((sum, ep) => sum + ep.level1Accuracy.resolved, 0) / last2.length;
			const improvementL1 = lastAvgL1 - firstAvgL1;
			
			console.log(`\n📈 Learning Progress:`);
			console.log(`   Level 0: First 2 avg: ${firstAvgL0.toFixed(1)}%, Last 2 avg: ${lastAvgL0.toFixed(1)}%, Improvement: ${improvementL0.toFixed(1)}%`);
			console.log(`   Level 1: First 2 avg: ${firstAvgL1.toFixed(1)}%, Last 2 avg: ${lastAvgL1.toFixed(1)}%, Improvement: ${improvementL1.toFixed(1)}%`);
		}
		
		// Show best episodes
		console.log('\n🏆 Best Episodes (Level 0 Accuracy):');
		const sortedByL0 = [...this.episodeResults].sort((a, b) => b.level0Accuracy.resolved - a.level0Accuracy.resolved);
		for (let i = 0; i < Math.min(3, sortedByL0.length); i++) {
			const ep = sortedByL0[i];
			console.log(`   #${ep.episode}: L0=${ep.level0Accuracy.resolved.toFixed(1)}%, L1=${ep.level1Accuracy.resolved.toFixed(1)}%`);
		}
		
		console.log('='.repeat(80));
		
		// Validation
		const finalL0Acc = this.episodeResults[this.episodeResults.length - 1].level0Accuracy.resolved;
		const finalL1Acc = this.episodeResults[this.episodeResults.length - 1].level1Accuracy.resolved;
		
		console.log('\n✅ VALIDATION:');
		if (finalL0Acc >= 90) {
			console.log(`   ✓ Level 0 prediction accuracy: ${finalL0Acc.toFixed(1)}% (>= 90%)`);
		} else {
			console.log(`   ✗ Level 0 prediction accuracy: ${finalL0Acc.toFixed(1)}% (< 90%)`);
		}
		
		if (finalL1Acc > 0) {
			console.log(`   ✓ Level 1 patterns created with ${finalL1Acc.toFixed(1)}% accuracy`);
		} else {
			console.log(`   ✗ No level 1 pattern predictions`);
		}
	}
}

