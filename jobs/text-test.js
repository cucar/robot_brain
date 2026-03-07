import { Job } from './job.js';
import { TextChannel } from '../channels/text.js';

/**
 * Text Test Job - Trains the brain on repeating text patterns
 * Tests pattern learning and event prediction accuracy
 * Goal: Validate if brain can memorize and predict character sequences
 */
export default class TextTestJob extends Job {

	constructor() {
		super();

		// Simple configuration - edit these values as needed
		this.config = {
			pattern: 'i don\'t like green eggs and ham. i do not like them, sam-i-am.',              // Pattern to learn
			maxEpisodes: 10,              // Number of training episodes
			iterationsPerEpisode: 10      // How many times to repeat pattern per episode
		};

		// Training metrics
		this.episodeResults = [];
		this.currentEpisode = 0;
	}

	/**
	 * Apply command line options to config
	 */
	applyOptions(options) {
		if (options.episodes !== null && options.episodes !== undefined)
			this.config.maxEpisodes = options.episodes;
		if (options.pattern !== null && options.pattern !== undefined)
			this.config.pattern = options.pattern;
		if (options.iterations !== null && options.iterations !== undefined)
			this.config.iterationsPerEpisode = options.iterations;
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
		console.log(`🚀 Starting Text Test Job`);
		console.log(`📝 Pattern: "${this.config.pattern}"`);
		console.log(`🔄 Max Episodes: ${this.config.maxEpisodes}`);
		console.log(`🔁 Iterations per Episode: ${this.config.iterationsPerEpisode}`);
		console.log('');
	}

	/**
	 * Hook: Configure text channel after brain initialization
	 */
	async configureChannels() {
		const textChannel = this.brain.getChannel('text');
		textChannel.setTraining(this.config.pattern, this.config.iterationsPerEpisode);
	}

	/**
	 * Hook: Execute main job logic - multi-episode training
	 */
	async executeJob() {
		for (this.currentEpisode = 1; this.currentEpisode <= this.config.maxEpisodes; this.currentEpisode++) {

			// Run episode
			await this.runEpisode();

			// If interrupt is received, stop processing
			if (this.isShuttingDown) return;

			// Show progress every 10 episodes or on last episode
			if (this.currentEpisode % 10 === 0 || this.currentEpisode === this.config.maxEpisodes)
				this.showProgress();
		}
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
		console.log(`📝 Episode ${this.currentEpisode}/${this.config.maxEpisodes}...`);

		// Reset context but keep learned patterns
		this.brain.resetContext();

		// Reset channel training data for new episode
		const textChannel = this.brain.getChannel('text');
		textChannel.setTraining(this.config.pattern, this.config.iterationsPerEpisode);

		// Initialize episode metrics
		const episodeMetrics = {
			episode: this.currentEpisode,
			pattern: this.config.pattern,
			baseAccuracy: null
		};

		// Calculate expected number of frames
		const expectedFrames = this.config.pattern.length * this.config.iterationsPerEpisode;

		// Process all frames for the episode duration
		let frameCount = 0;
		while (frameCount < expectedFrames) {
			await this.brain.processFrame();
			frameCount++;

			// Show progress every 100 frames
			if (frameCount % 100 === 0)
				process.stdout.write(`\r📝 Episode ${this.currentEpisode}/${this.config.maxEpisodes} - Frame ${frameCount}/${expectedFrames}...`);

			// If interrupt is received, stop processing
			if (this.isShuttingDown) return;
		}

		// Clear progress line (only if stdout is a TTY)
		if (!process.stdout.isTTY) process.stdout.write('\n');
		else {
			process.stdout.write('\r');
			process.stdout.clearLine?.(0);
		}

		// Set frame count and duration
		const duration = Date.now() - startTime;
		episodeMetrics.duration = duration;
		episodeMetrics.frames = frameCount;

		// Capture base level accuracy stats
		const summary = this.brain.getEpisodeSummary();
		if (summary.accuracy.total > 0)
			episodeMetrics.baseAccuracy = (summary.accuracy.correct / summary.accuracy.total * 100);

		this.episodeResults.push(episodeMetrics);

		// Show episode summary
		const accStr = episodeMetrics.baseAccuracy !== null
			? `${episodeMetrics.baseAccuracy.toFixed(2)}%`
			: 'N/A';
		console.log(`✅ Accuracy: ${accStr} (${frameCount} frames, ${duration}ms)`);
	}

	/**
	 * Show training progress
	 */
	showProgress() {
		console.log(`\n📊 Training Progress (Episode ${this.currentEpisode}/${this.config.maxEpisodes}):`);

		if (this.episodeResults.length >= 10) {
			const recent10 = this.episodeResults.slice(-10);
			const avgAcc = recent10.reduce((sum, ep) => sum + (ep.baseAccuracy || 0), 0) / recent10.length;
			console.log(`   Last 10 episodes avg accuracy: ${avgAcc.toFixed(2)}%`);
		}

		const validResults = this.episodeResults.filter(ep => ep.baseAccuracy !== null);
		if (validResults.length > 0) {
			const bestEpisode = validResults.reduce((best, ep) =>
				(ep.baseAccuracy || 0) > (best.baseAccuracy || 0) ? ep : best);
			const worstEpisode = validResults.reduce((worst, ep) =>
				(ep.baseAccuracy || 0) < (worst.baseAccuracy || 0) ? ep : worst);

			console.log(`   Best episode: #${bestEpisode.episode} (${bestEpisode.baseAccuracy?.toFixed(2)}%)`);
			console.log(`   Worst episode: #${worstEpisode.episode} (${worstEpisode.baseAccuracy?.toFixed(2)}%)`);
		}
		console.log('');
	}

	/**
	 * Show final training results
	 */
	showFinalResults() {
		console.log(`\n🎯 Final Training Results (${this.config.maxEpisodes} episodes):`);
		console.log('='.repeat(60));

		const validResults = this.episodeResults.filter(ep => ep.baseAccuracy !== null);
		if (validResults.length === 0) {
			console.log('No valid accuracy data collected.');
			console.log('='.repeat(60));
			return;
		}

		// Calculate average accuracy
		const avgAccuracy = validResults.reduce((sum, ep) => sum + ep.baseAccuracy, 0) / validResults.length;

		console.log(`📈 Overall Performance:`);
		console.log(`   Pattern: "${this.config.pattern}"`);
		console.log(`   Iterations per Episode: ${this.config.iterationsPerEpisode}`);
		console.log(`   Average Accuracy: ${avgAccuracy.toFixed(2)}%`);

		// Show accuracy per episode
		console.log(`\n📊 Accuracy by Episode:`);
		for (const ep of this.episodeResults) {
			const accStr = ep.baseAccuracy !== null ? `${ep.baseAccuracy.toFixed(2)}%` : 'N/A';
			console.log(`   Episode ${ep.episode}: ${accStr} (${ep.frames} frames)`);
		}

		// Show improvement trend
		if (validResults.length >= 4) {
			const firstHalf = validResults.slice(0, Math.floor(validResults.length / 2));
			const secondHalf = validResults.slice(Math.floor(validResults.length / 2));

			const firstAvg = firstHalf.reduce((sum, ep) => sum + ep.baseAccuracy, 0) / firstHalf.length;
			const secondAvg = secondHalf.reduce((sum, ep) => sum + ep.baseAccuracy, 0) / secondHalf.length;
			const improvement = secondAvg - firstAvg;

			console.log(`\n📈 Learning Progress:`);
			console.log(`   First half avg: ${firstAvg.toFixed(2)}%`);
			console.log(`   Second half avg: ${secondAvg.toFixed(2)}%`);
			console.log(`   Improvement: ${improvement >= 0 ? '+' : ''}${improvement.toFixed(2)}pp ${improvement >= 0 ? '📈' : '📉'}`);
		}

		// Show best episodes
		console.log('\n🏆 Best Episodes (by Accuracy):');
		const sortedByAccuracy = [...validResults].sort((a, b) => b.baseAccuracy - a.baseAccuracy);
		for (let i = 0; i < Math.min(5, sortedByAccuracy.length); i++) {
			const ep = sortedByAccuracy[i];
			console.log(`   #${ep.episode}: ${ep.baseAccuracy.toFixed(2)}%`);
		}

		console.log('='.repeat(60));
	}
}

