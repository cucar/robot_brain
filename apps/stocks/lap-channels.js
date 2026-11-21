/**
 * Several channels in one brain, no reset (docs/algorithm.md D34, D24): the job the stock jobs run, each with its own
 * settings (multi-channel-test.js, synthetic-extended-test.js, test.js).
 * One channel per stock: the price's move up or down as the event, own or out as the action, the reward scoped to that
 * stock's channel and the frame; every stock fires in every frame.
 * With volume on, each stock with volume data has a second channel, event only: the volume's move up or down.
 * Neighbors by kind (the eventsToActions policy): an event reads every event and call of any channel in its frame and
 * the one before; no action is anyone's neighbor.
 *
 * Modes:
 *   one    KGC alone, its first 12 daily opens (apps/stocks/data/1D), cycled
 *   two    KGC and GLD, the same way
 *   three  KGC, GLD and SPY, the same way
 *   three4 KGC, GLD and SPY, the same way, and Q, made up, moving one percent a bar
 *   lap    two stocks on six-bar laps, each with its action: A +1 −2 −3 +2 +3 −1, and B +2 +1 −1 −3 +3 −2
 *   lap65  A on its six-bar lap and B on a five-bar lap +2 +1 −1 +3 −2: together they repeat every 30 frames
 *   seq    stocks over their whole daily history as the CSVs hold it, one lap the whole sequence, with a trader
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Job, formatFixed } from 'robot-brain';
import { LapTrader } from './lap-trader.js';

// the folder of each timeframe's data
export const DATA_ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), 'data');

/* ---------- Reading the series ---------- */

/**
 * The percent change from each value to the next; with `wraps`, the last to the first.
 * A change from nothing counts as a rise of 1000%.
 */
function readChanges(values, wraps) {
	const count = wraps ? values.length : values.length - 1;
	return Array.from({ length: count }, (_, i) => {
		const next = values[(i + 1) % values.length];
		return values[i] === 0 ? 1000 : (next - values[i]) / values[i] * 100;
	});
}

/**
 * A stock's 12-bar lap: the percent move of each of its first 12 daily opens and of their volumes, the last wrapping
 * to the first.
 */
function readTwelveBarLap(folder, symbol) {
	const bars = JSON.parse(fs.readFileSync(path.join(folder, `${symbol}.json`), 'utf8')).slice(0, 12);
	return { name: symbol, moves: readChanges(bars.map(bar => bar.OpenPrice), true), volumes: readChanges(bars.map(bar => bar.Volume), true) };
}

/**
 * A symbol's daily history: its prices, the percent move from each to the next, the same for its volume, and which
 * moves are filled.
 * A move is filled where the row it moves to has no price or no volume: the data fills the days before a stock was
 * listed, and the days it did not trade, with the price held and a volume of 0. No trading happened on such a day.
 * The rows kept are those after the first `offset` and before the last `holdout`, at most `maxFrames` moves of them.
 */
function readHistory(folder, symbol, { offset, holdout, maxFrames }) {
	const text = fs.readFileSync(path.join(folder, `${symbol}.csv`), 'utf8');
	let rows = text.split('\n').filter(line => line.trim() !== '').map(line => line.split(',').map(parseFloat));
	rows = rows.slice(offset, rows.length - holdout);
	if (maxFrames !== null) rows = rows.slice(0, maxFrames + 1);
	const prices = rows.map(row => row[0]);
	const filled = rows.slice(1).map(([price, volume]) => !price || !volume);
	return { prices, moves: readChanges(prices, false), volumes: readChanges(rows.map(row => row[1]), false), filled };
}

/**
 * A lap of ±1% moves from a string of U and D.
 */
function makeLap(letters) {
	return [...letters].map(letter => letter === 'U' ? 1 : -1);
}

/**
 * The series of a mode: each channel's name, the percent move of each bar of its lap and, where the data has it, of
 * its volume; the length of the cycle; and in seq mode each stock's prices for the trader.
 * Made-up stocks have no volume.
 */
function readSeries(settings) {
	const lapOf = symbol => readTwelveBarLap(settings.folder, symbol);
	switch (settings.mode) {
		case 'one': return { series: ['KGC'].map(lapOf), cycle: 12 };
		case 'two': return { series: ['KGC', 'GLD'].map(lapOf), cycle: 12 };
		case 'three': return { series: ['KGC', 'GLD', 'SPY'].map(lapOf), cycle: 12 };
		case 'three4': return { series: [...['KGC', 'GLD', 'SPY'].map(lapOf), { name: 'Q', moves: makeLap('UDDUUDUDDUDD') }], cycle: 12 };
		case 'lap': return { series: [{ name: 'A', moves: [1, -2, -3, 2, 3, -1] }, { name: 'B', moves: [2, 1, -1, -3, 3, -2] }], cycle: 6 };
		case 'lap65': return { series: [{ name: 'A', moves: [1, -2, -3, 2, 3, -1] }, { name: 'B', moves: [2, 1, -1, 3, -2] }], cycle: 30 };
		case 'seq': return readSequence(settings);
		default: throw new Error('mode: one, two, three, three4, lap, lap65 or seq');
	}
}

/**
 * The seq mode's series: each stock's history, one cycle the whole sequence.
 */
function readSequence(settings) {
	const histories = settings.symbols.map(symbol => readHistory(settings.folder, symbol, settings));
	const series = settings.symbols.map((symbol, i) => ({ name: symbol, moves: histories[i].moves, volumes: histories[i].volumes, filled: histories[i].filled }));
	return { series, prices: histories.map(h => h.prices), cycle: series[0].moves.length };
}

/* ---------- Small helpers ---------- */

/**
 * The move a bar's percent stands for: 1 up, 0 down.
 */
function readMove(percent) {
	return percent > 0 ? 1 : 0;
}

/**
 * A source of random numbers in [0, 1) from a seed, the same numbers for the same seed (mulberry32).
 */
function makeRandom(seed) {
	let state = seed >>> 0;
	return () => {
		state = (state + 0x6D2B79F5) >>> 0;
		let t = state;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/**
 * Whether a stock's bar is filled: no trading happened on it. Only real histories have filled bars.
 */
function isFilled(stock, bar) {
	return stock.filled?.[bar % stock.filled.length] ?? false;
}

/**
 * Right and wrong in a row of the grid ('R', 'W', '-').
 */
function countMarks(row) {
	return { right: row.filter(c => c === 'R').length, wrong: row.filter(c => c === 'W').length };
}

/**
 * A share of a count as a percent with two decimals; nothing counted is 0%.
 */
function formatPercent(part, whole) {
	return formatFixed(100 * part / Math.max(whole, 1), 2);
}

/**
 * A set of numbers as Rust prints a sorted set, as in `{3, 4, 5}`.
 */
function formatSet(numbers) {
	return `{${[...numbers].sort((a, b) => a - b).join(', ')}}`;
}

/**
 * Compare two strings by their characters, as Rust orders them.
 */
function compareText(a, b) {
	return a < b ? -1 : a > b ? 1 : 0;
}

/* ---------- The board ---------- */

/**
 * The board as read once: each activation found by the frame it is recorded under and its index there.
 */
class BoardView {

	constructor(frames, registry) {
		this.frames = frames;
		this.registry = registry;
		this.byFrame = new Map(frames.map(frame => [frame.frame, frame.activations]));
	}

	/**
	 * Every activation of the board, oldest frame first.
	 */
	listActivations() {
		return this.frames.flatMap(frame => frame.activations);
	}

	/**
	 * The activation a reference names, or undefined when its frame has left the board.
	 */
	findActivation(ref) {
		return this.byFrame.get(ref.frame)?.[ref.index];
	}

	/**
	 * The events a call stands for, at any depth, each as its event dimension and frame.
	 * An event reached by two paths is listed once.
	 */
	listUnder(activation) {
		const events = new Map();
		const stack = [...activation.covers];
		while (stack.length > 0) {
			const ref = stack.pop();
			const covered = this.findActivation(ref);
			if (!covered) continue;
			if (covered.covers.length > 0) stack.push(...covered.covers);
			else if (covered.kind === 'event') {
				const dim = this.registry.readDim(covered.neuron);
				events.set(`${dim}:${ref.frame}`, { dim, frame: ref.frame });
			}
		}
		return [...events.values()];
	}
}

/**
 * The fixed facts of the neurons, read from the brain once each.
 */
class NeuronFacts {

	constructor(brain) {
		this.brain = brain;
		this.facts = new Map();
	}

	read(id) {
		if (!this.facts.has(id)) this.facts.set(id, this.brain.readNeurons([id])[0]);
		return this.facts.get(id);
	}

	readDim(id) {
		return this.read(id).dim;
	}
}

/* ---------- The job ---------- */

/**
 * The job: its settings come from the job file that runs it.
 * The settings: `mode`, `cycles` (the laps), `folder` (the data), `symbols` (seq mode), `every` (a progress line every
 * that many frames, 0 for none), `all4` (the ALL4 lines), `volume` (the volume channels), `offset`, `holdout` and
 * `maxFrames` (the rows of seq mode), `trader` (the LapTrader's settings), and `randomBaseline` with its `seed`.
 * The random baseline leaves the brain out: at each bar a coin, from the seed, says all out half the time, and the
 * other half own one stock picked at random; the trader trades on that as it would on the brain's actions.
 */
export class LapChannelsJob extends Job {

	constructor(settings) {
		super();
		this.mode = settings.mode;
		this.isLapMode = this.mode.startsWith('lap');
		this.showAllFour = settings.all4;
		const { series, prices, cycle } = readSeries(settings);
		this.series = series;
		this.prices = prices ?? [];
		this.cycle = cycle;
		this.cycles = settings.cycles;
		this.reportEvery = settings.every;
		// the stocks given a volume channel: with volume on, those whose data has volume
		this.volumeSeries = settings.volume ? series.filter(s => s.volumes) : [];
		this.trader = new LapTrader(series.map(s => s.name), settings.trader);
		this.random = settings.randomBaseline ? makeRandom(settings.seed ?? 1) : null;
		this.prepareCounts();
	}

	/**
	 * The counts kept over the run.
	 */
	prepareCounts() {
		const count = this.series.length;
		// what the brain said for each stock, for the bar ahead; its estimate; and the last action it said
		this.pending = Array(count).fill(null);
		this.estimates = Array(count).fill(0);
		this.lastActions = Array(count).fill(null);
		// per stock, per cycle, per bar: R right, W wrong, - said nothing
		this.grid = this.series.map(() => Array.from({ length: this.cycles }, () => Array(this.cycle).fill('-')));
		// per cycle, per frame: the combination calls, level-1 calls over two of the frame's bars
		this.combos = Array.from({ length: this.cycles }, () => []);
		// per channel set (each channel alone, then all): the laps at whose end one apex call stands for its six bars
		this.sixLaps = Array.from({ length: count + 1 }, () => []);
		// the functions that had a call in the last cycle
		this.calledLast = new Set();
		// per channel: the frame its bar 6 of the second-last cycle first came under a level-3 call
		this.underLevelThree = Array(count).fill(null);
		// the dictionary's fingerprint as it last stood, read when the run starts, and the frame it last changed
		this.lastDictionary = null;
		this.lastChange = 0;
		// when the run started, for the seconds the progress lines print
		this.clockStart = Date.now();
	}

	getBrainOptions() {
		return { h: 256, policy: 'eventsToActions' };
	}

	declareChannels() {
		for (const { name } of this.series) {
			this.brain.declareChannel({ name, event: { name: 'move', resolution: 2 }, action: { name: 'position', resolution: 2 } });
		}
		for (const { name } of this.volumeSeries) this.brain.declareChannel({ name: `${name} volume`, event: { name: 'volume', resolution: 2 } });
	}

	/**
	 * Run every frame of every cycle, and print the cycle lines as they end.
	 */
	async executeJob() {
		this.channelList = this.series.map(s => this.channels.get(s.name));
		this.volumeChannels = this.volumeSeries.map(s => this.channels.get(`${s.name} volume`));
		this.neurons = new NeuronFacts(this.brain);
		this.lastDictionary = this.brain.readDictionaryFingerprint();
		this.printHeader();
		const total = this.cycle * this.cycles;
		for (let k = 0; k < total; k++) {
			if (k % this.cycle === 0) this.trader.startLap();
			if (this.random) {
				this.runRandomBar(k);
				continue;
			}
			const winners = this.runFrame(k);
			const frameNumber = this.brain.getFrameNumber();
			this.readWinners(k, winners);
			if (this.prices.length > 0) this.runTrader(k);
			this.noteDictionary(frameNumber);
			this.readBoardAfterFrame(k, frameNumber);
		}
	}

	/**
	 * One bar of the random baseline: the brain sees nothing; the coin's actions are graded and traded on.
	 */
	runRandomBar(k) {
		this.chooseRandomActions();
		this.series.forEach((stock, i) => {
			if (isFilled(stock, k + 1)) return;
			const nextMove = readMove(stock.moves[(k + 1) % stock.moves.length]);
			this.grid[i][Math.floor(k / this.cycle)][k % this.cycle] = this.pending[i] === nextMove ? 'R' : 'W';
		});
		if (this.prices.length > 0) this.runTrader(k);
		if (k % this.cycle === this.cycle - 1) this.printCycle(k, k + 1, new BoardView([], this.neurons));
	}

	/**
	 * The coin's actions: half the time every stock out, the other half one stock, picked at random, owned.
	 */
	chooseRandomActions() {
		const owned = this.random() < 0.5 ? -1 : Math.floor(this.random() * this.series.length);
		this.series.forEach((_, i) => {
			this.pending[i] = i === owned ? 1 : 0;
			this.estimates[i] = 0;
		});
	}

	printHeader() {
		console.log(`${this.mode}: ${this.cycles} cycles of ${this.cycle} bars`);
		for (const { name, moves } of this.series) {
			if (moves.length === this.cycle) console.log(`  ${name} moves per cycle: ${moves.map(p => p > 0 ? 'U' : 'D').join('')}`);
		}
		if (this.volumeSeries.length > 0) console.log(`  volume channels: ${this.volumeSeries.map(s => s.name).join(', ')}`);
	}

	/**
	 * One frame: each stock's move, and the action that runs for it, rewarded by the move.
	 * What the brain said for the bar runs; where it said nothing, the move itself runs as a lesson.
	 */
	runFrame(k) {
		const inputs = [];
		const rewards = [];
		this.series.forEach((stock, i) => {
			// a filled bar is no trading: the stock reports nothing, runs nothing and earns nothing this frame
			if (isFilled(stock, k)) return;
			const percent = stock.moves[k % stock.moves.length];
			const move = readMove(percent);
			const channel = this.channelList[i];
			const action = this.pending[i] ?? move;
			inputs.push({ dim: channel.eventDim, bucket: move });
			inputs.push({ dim: channel.actionDim, bucket: action });
			rewards.push({ reward: action === 1 ? percent : -percent, channels: [channel.channelId], frames: 1 });
		});
		this.volumeSeries.forEach((stock, i) => {
			if (isFilled(stock, k)) return;
			inputs.push({ dim: this.volumeChannels[i].eventDim, bucket: readMove(stock.volumes[k % stock.volumes.length]) });
		});
		return this.brain.processFrame(inputs, rewards);
	}

	/**
	 * Read what the brain said for each stock, and mark it right or wrong against the next bar's move.
	 * A filled next bar is not graded: there was no move to say.
	 */
	readWinners(k, winners) {
		this.series.forEach((stock, i) => {
			const winner = winners.find(w => w.dim === this.channelList[i].actionDim);
			this.pending[i] = winner?.bucket ?? null;
			this.estimates[i] = winner?.estimate ?? 0;
			if (this.pending[i] === null || isFilled(stock, k + 1)) return;
			const nextMove = readMove(stock.moves[(k + 1) % stock.moves.length]);
			this.grid[i][Math.floor(k / this.cycle)][k % this.cycle] = this.pending[i] === nextMove ? 'R' : 'W';
		});
	}

	/**
	 * The trader, at the price this frame's bar ended at.
	 * A stock keeps its last action where the brain says nothing new; where none has an action yet, nothing trades.
	 */
	runTrader(k) {
		this.pending.forEach((action, i) => { if (action !== null) this.lastActions[i] = action; });
		const at = (k % this.cycle) + 1;
		const prices = this.prices.map(p => p[at]);
		if (this.lastActions.some(a => a !== null)) this.trader.trade(this.lastActions, this.estimates, prices);
		this.trader.recordValue(prices);
		if (this.reportEvery > 0 && at % this.reportEvery === 0) this.printProgress(k, prices);
		if (at === this.cycle) this.printTraderLap(k, prices);
	}

	/**
	 * Every N frames: neurons, functions, the share right over the last N frames and so far, and the profit.
	 */
	printProgress(k, prices) {
		const lap = Math.floor(k / this.cycle);
		const seen = (k % this.cycle) + 1;
		const from = seen - this.reportEvery;
		let rightRecent = 0, recent = 0, rightSoFar = 0, soFar = 0;
		for (const rows of this.grid) {
			rows[lap].slice(0, seen).forEach((mark, bar) => {
				if (mark === '-') return;
				soFar++;
				if (mark === 'R') rightSoFar++;
				if (bar >= from) {
					recent++;
					if (mark === 'R') rightRecent++;
				}
			});
		}
		const seconds = formatFixed((Date.now() - this.clockStart) / 1000, 0);
		console.log(`EVERY100 frame ${k + 1} | neurons ${this.brain.countLivingNeurons()} | functions ${this.brain.countFunctions()} | right last 100 frames ${formatPercent(rightRecent, recent)}% | right so far ${formatPercent(rightSoFar, soFar)}% | net profit $${formatFixed(this.trader.readProfit(prices), 2)} | trades ${this.trader.trades} | ${seconds}s`);
	}

	/**
	 * At the end of a lap, the trader's lap: profit, ROI, the per-frame return, the Sharpe ratio, the cost paid and the trades.
	 */
	printTraderLap(k, prices) {
		const profit = this.trader.readProfit(prices);
		const roi = profit / this.trader.initialCapital * 100;
		const perFrame = this.trader.readPerBarReturn() * 100;
		const sharpe = this.trader.readSharpe();
		const signed = (value, decimals) => `${value >= 0 ? '+' : ''}${formatFixed(value, decimals)}`;
		console.log(`TRADER lap ${Math.floor(k / this.cycle) + 1} | net profit $${formatFixed(profit, 2)} | ROI ${signed(roi, 2)}% | ${signed(perFrame, 6)}%/frame | Sharpe ${sharpe === null ? '-' : formatFixed(sharpe, 2)} | transaction cost $${formatFixed(this.trader.costPaid, 2)} | trades ${this.trader.trades}`);
	}

	/**
	 * Note the frame when the dictionary differs from how it last stood.
	 */
	noteDictionary(frameNumber) {
		const dictionary = this.brain.readDictionaryFingerprint();
		if (dictionary === this.lastDictionary) return;
		this.lastDictionary = dictionary;
		this.lastChange = frameNumber;
	}

	/**
	 * What the board holds after a frame: the cycle line at a cycle's end, the calls over every bar of the frame,
	 * the combination calls, the laps an apex call stands for, and what came under a level-3 call.
	 * The whole board is read only where one of them needs more than the last frame.
	 */
	readBoardAfterFrame(k, frameNumber) {
		const isCycleEnd = k % this.cycle === this.cycle - 1;
		const needsBoard = isCycleEnd || this.showAllFour || this.isLapMode;
		const frames = needsBoard ? this.brain.getBoard() : [this.brain.getLastFrame()];
		const board = new BoardView(frames, this.neurons);
		const last = frames[frames.length - 1];
		if (isCycleEnd) this.printCycle(k, frameNumber, board);
		if (this.showAllFour) this.printAllFour(frameNumber, board);
		this.combos[Math.floor(k / this.cycle)].push(this.countCombinationCalls(last, frameNumber, board));
		if (this.isLapMode && isCycleEnd) this.noteSixBarLaps(k, frameNumber, board);
		if (Math.floor(k / this.cycle) === this.cycles - 1) this.noteCalledFunctions(last);
		if (this.isLapMode && this.cycles >= 2) this.noteLevelThree(frameNumber, board);
	}

	/**
	 * The cycle line: neurons, functions, the highest call level in the cycle, the dictionary's last change, each
	 * stock's share right in the cycle, and in seq mode the profit.
	 */
	printCycle(k, frameNumber, board) {
		const lap = Math.floor(k / this.cycle);
		const inCycle = board.frames.filter(f => f.frame > frameNumber - this.cycle).flatMap(f => f.activations);
		const top = inCycle.reduce((highest, a) => Math.max(highest, a.level), 0);
		const shares = this.series.map(({ name }, i) => {
			const { right, wrong } = countMarks(this.grid[i][lap]);
			return `${name} ${formatPercent(right, right + wrong)}%`;
		});
		let profit = '';
		if (this.prices.length > 0) {
			const value = this.trader.readProfit(this.prices.map(p => p[(k % this.cycle) + 1]));
			profit = ` | net profit $${formatFixed(value, 2)}, ROI ${formatFixed(value / this.trader.initialCapital * 100, 2)}% (${this.trader.trades} trades)`;
		}
		console.log(`CYCLE ${lap + 1} | neurons ${this.brain.countLivingNeurons()} | functions ${this.brain.countFunctions()} | highest call level in the cycle ${top} | the dictionary last changed at frame ${this.lastChange} | right ${shares.join(', ')}${profit}`);
	}

	/**
	 * The calls standing in this frame over every bar of this frame.
	 */
	printAllFour(frameNumber, board) {
		const frame = board.frames.find(f => f.frame === frameNumber);
		if (!frame) return;
		const bars = frame.activations.filter(a => a.kind === 'event').length;
		const calls = frame.activations
			.filter(a => a.covers.length > 0 && !a.gone)
			.filter(a => board.listUnder(a).filter(e => e.frame === frameNumber).length === bars)
			.map(a => `L${a.level} f${a.firedBy ?? 0}`);
		console.log(`ALL4 ${frameNumber} | ${calls.join('; ')}`);
	}

	/**
	 * The level-1 calls of the frame over two or more of the frame's bars.
	 */
	countCombinationCalls(last, frameNumber, board) {
		const isBarOfFrame = ref => ref.frame === frameNumber && board.findActivation(ref)?.kind === 'event';
		return last.activations.filter(a => a.level === 1 && a.firedBy != null && a.covers.filter(isBarOfFrame).length >= 2).length;
	}

	/**
	 * Note, per channel and for all of them, whether one call on the apex stands for the six bars of this lap.
	 */
	noteSixBarLaps(k, frameNumber, board) {
		const lapFrames = Array.from({ length: 6 }, (_, i) => frameNumber - 5 + i);
		const eventDims = this.channelList.map(c => c.eventDim);
		const sets = [...eventDims.map(d => [d]), eventDims];
		const apexCalls = board.listActivations().filter(a => a.onApex && a.covers.length > 0);
		const under = apexCalls.map(a => new Set(board.listUnder(a).map(e => `${e.dim}:${e.frame}`)));
		sets.forEach((set, j) => {
			const stands = under.some(u => set.every(d => lapFrames.every(f => u.has(`${d}:${f}`))));
			if (stands) this.sixLaps[j].push(Math.floor(k / this.cycle) + 1);
		});
	}

	/**
	 * Note the functions that fired an activation of the last frame.
	 */
	noteCalledFunctions(last) {
		for (const a of last.activations) if (a.firedBy != null) this.calledLast.add(a.firedBy);
	}

	/**
	 * Note the frame at which bar 6 of the second-last cycle first comes under a level-3 call, per channel.
	 */
	noteLevelThree(frameNumber, board) {
		const target = this.cycle * (this.cycles - 1);
		if (frameNumber < target) return;
		const levelThree = board.listActivations().filter(a => a.level === 3);
		this.channelList.forEach((channel, i) => {
			if (this.underLevelThree[i] !== null) return;
			if (levelThree.some(a => board.listUnder(a).some(e => e.dim === channel.eventDim && e.frame === target))) this.underLevelThree[i] = frameNumber;
		});
	}

	/* ---------- The results ---------- */

	async showResults() {
		this.printInference();
		const functions = this.brain.countFunctions();
		console.log(`recognition: ${functions} functions | the dictionary last changed at frame ${this.lastChange}`);
		console.log(`NEVERCALLED ${this.brain.countFunctionsNeverCalled()} of ${functions}`);
		console.log(`  combination calls per frame, per cycle: ${this.combos.map((counts, c) => `${c + 1}:${counts.join('')}`).join(' ')}`);
		if (this.isLapMode) this.printLaps();
		new DictionaryReport(this.brain, this.neurons, this.series, this.channelList, this.volumeSeries, this.volumeChannels, this.calledLast).print();
		this.printApex();
		console.log(`  ${formatFixed((Date.now() - this.clockStart) / 1000, 1)} s`);
	}

	/**
	 * Per stock: right of all it said, the cycle from which it said every bar right, and each cycle or lap.
	 */
	printInference() {
		this.series.forEach(({ name }, i) => {
			const { right, wrong } = countMarks(this.grid[i].flat());
			let from = null;
			for (let c = this.cycles - 1; c >= 0 && this.grid[i][c].every(x => x === 'R'); c--) from = c;
			console.log(`inference ${name}: right ${right} of ${right + wrong} | every bar right from cycle ${from === null ? 'never' : from + 1}`);
			this.grid[i].forEach((row, c) => {
				const cycleLabel = String(c + 1).padStart(2);
				if (this.cycle > 60) {
					const counts = countMarks(row);
					console.log(`  ${name} lap ${cycleLabel}: right ${counts.right} of ${counts.right + counts.wrong} (${formatPercent(counts.right, counts.right + counts.wrong)}%)`);
				}
				else console.log(`  ${name} cycle ${cycleLabel}: ${row.join('')}`);
			});
		});
	}

	/**
	 * The laps an apex call stood for, per channel and for all, and when bar 6 came under a level-3 call.
	 */
	printLaps() {
		const show = laps => laps.length === 0 ? 'never' : `from lap ${laps[0]} (${laps.length} of ${this.cycles} laps)`;
		this.series.forEach(({ name }, j) => console.log(`  one call on the apex stands for ${name}'s six bars of a lap: ${show(this.sixLaps[j])}`));
		if (this.series.length > 1) console.log(`  one call on the apex stands for every channel's six bars of a lap: ${show(this.sixLaps[this.series.length])}`);
		this.series.forEach(({ name }, j) => {
			const at = this.underLevelThree[j] ?? 'never';
			console.log(`  ${name}: bar 6 of cycle ${this.cycles - 1} (frame ${this.cycle * (this.cycles - 1)}) first comes under a level-3 call at frame ${at}`);
		});
	}

	/**
	 * The calls on the apex at the last frame, those reaching its last six frames, and each one's level and frames.
	 */
	printApex() {
		const last = this.brain.getFrameNumber();
		const board = new BoardView(this.brain.getBoard(), this.neurons);
		const calls = board.listActivations().filter(a => a.onApex && a.covers.length > 0);
		const reaching = calls.filter(a => board.listUnder(a).some(e => e.frame >= last - 5)).length;
		console.log(`  calls on the apex at the last frame: ${calls.length} | reaching frames ${last - 5}-${last}: ${reaching}`);
		for (const a of calls) {
			const under = board.listUnder(a);
			const frames = new Set(under.map(e => e.frame));
			const channels = new Set(under.map(e => e.dim));
			console.log(`  apex: call of f${a.firedBy ?? 0} | level ${a.level} | frames ${formatSet(frames)} | channels ${channels.size}`);
		}
	}
}

/* ---------- The dictionary at the end ---------- */

/**
 * The dictionary at the end, by level and kind, with how many had a call in the last cycle; then every function.
 */
class DictionaryReport {

	constructor(brain, neurons, series, channelList, volumeSeries, volumeChannels, calledLast) {
		this.brain = brain;
		this.neurons = neurons;
		this.calledLast = calledLast;
		this.functions = new Map(brain.getDictionary().map(f => [f.id, f]));
		this.poses = new Map();
		// each channel's name by its event and action dimensions
		this.channelNames = new Map();
		channelList.forEach((channel, i) => {
			this.channelNames.set(channel.eventDim, series[i].name);
			this.channelNames.set(channel.actionDim, series[i].name);
		});
		volumeChannels.forEach((channel, i) => this.channelNames.set(channel.eventDim, `${volumeSeries[i].name} volume`));
		// each function's level and channels, read once
		this.info = new Map();
	}

	print() {
		const ids = [...this.functions.keys()].sort((a, b) => a - b);
		for (const id of ids) this.readInfo(id, new Set());
		this.printTable(ids);
		ids.sort((a, b) => this.info.get(a).level - this.info.get(b).level || a - b);
		for (const id of ids) this.printFunction(id);
	}

	/**
	 * The call a pose names, or null where it names a base neuron.
	 */
	readPose(sym) {
		if (!this.poses.has(sym)) this.poses.set(sym, this.brain.readPose(sym));
		return this.poses.get(sym);
	}

	/**
	 * The function a call's pose belongs to: of its interneuron's functions, the one holding the pose's first node,
	 * else its first.
	 */
	findFunctionOf(sym) {
		const pose = this.readPose(sym);
		if (!pose) return null;
		const owned = this.brain.listFunctionsOf(pose.neuron);
		return owned.find(id => this.functions.get(id)?.nodes.includes(pose.from)) ?? owned[0] ?? null;
	}

	/**
	 * Whether a function relates two bars of one frame: a relation at time offset 0 between two base neurons.
	 */
	isAssociationOfBars(id) {
		const f = this.functions.get(id);
		return !!f && f.relations.some(r => r.offset[0] === 0 && !this.readPose(r.from) && !this.readPose(r.to));
	}

	/**
	 * A function's level (one above its highest call node) and the dimensions under it.
	 * A function whose nodes reach back to itself counts that path as level 0 with no dimensions.
	 */
	readInfo(id, open) {
		if (this.info.has(id)) return this.info.get(id);
		if (open.has(id)) return { level: 0, dims: new Set() };
		open.add(id);
		let level = 0;
		const dims = new Set();
		for (const node of this.functions.get(id)?.nodes ?? []) {
			if (this.readPose(node)) {
				const inner = this.findFunctionOf(node);
				if (inner === null) continue;
				const info = this.readInfo(inner, open);
				level = Math.max(level, info.level);
				for (const d of info.dims) dims.add(d);
			}
			else {
				const dim = this.neurons.readDim(node);
				if (dim != null) dims.add(dim);
			}
		}
		open.delete(id);
		const info = { level: level + 1, dims };
		this.info.set(id, info);
		return info;
	}

	/**
	 * What a function relates: bars of one frame, the combination calls, one channel, or channels across time.
	 */
	describeKind(id) {
		const f = this.functions.get(id);
		if (this.isAssociationOfBars(id)) return 'association between bars';
		const overCombinations = f.nodes.some(n => {
			if (!this.readPose(n)) return false;
			const inner = this.findFunctionOf(n);
			return inner !== null && this.isAssociationOfBars(inner);
		});
		if (overCombinations) return 'over the combination calls';
		const { dims } = this.info.get(id);
		if (dims.size === 1) return `${this.channelNames.get([...dims][0]) ?? ''} only`;
		return 'across the channels in time';
	}

	/**
	 * How many functions stand at each level and kind, and how many of them had a call in the last cycle.
	 */
	printTable(ids) {
		const table = new Map();
		for (const id of ids) {
			const key = `${this.info.get(id).level}\u0000${this.describeKind(id)}`;
			const entry = table.get(key) ?? { level: this.info.get(id).level, kind: this.describeKind(id), count: 0, called: 0 };
			entry.count++;
			if (this.calledLast.has(id)) entry.called++;
			table.set(key, entry);
		}
		const rows = [...table.values()].sort((a, b) => a.level - b.level || compareText(a.kind, b.kind));
		for (const { level, kind, count, called } of rows) console.log(`  dictionary: level ${level} | ${kind} | ${count} | with a call in the last cycle ${called}`);
	}

	/**
	 * A node as the report writes it: a call by its function, a bar by its stock and move, an action by its stock and side.
	 */
	nameNode(sym) {
		if (this.readPose(sym)) return `call(f${this.findFunctionOf(sym) ?? -1})`;
		const neuron = this.neurons.read(sym);
		const name = neuron.dim != null ? this.channelNames.get(neuron.dim) ?? '' : '';
		if (neuron.kind === 'event') return `${name}${neuron.bucket === 1 ? '↑' : '↓'}`;
		return `${name}:${neuron.bucket === 1 ? 'own' : 'out'}`;
	}

	printFunction(id) {
		const f = this.functions.get(id);
		const nodes = f.nodes.map(n => this.nameNode(n)).join(', ');
		const relations = f.relations.map(r => `${this.nameNode(r.from)} → ${this.nameNode(r.to)} at t${r.offset[0]}`).join('; ');
		const cycle = f.isCycle ? ' | a cycle' : '';
		const called = this.calledLast.has(id) ? ' | a call in the last cycle' : '';
		console.log(`  function f${id} | level ${this.info.get(id).level} | nodes ${nodes} | relations ${relations}${cycle}${called}`);
	}
}

