/**
 * Stock trading on real daily history: one channel per stock, the brain saying own or out for each bar, the trader
 * acting on what it said (lap-channels.js, seq mode).
 * One episode is one lap of the whole history; the trader starts each lap with its starting capital.
 * Each lap prints the right share per stock and a TRADER line: profit, ROI, per-frame return, Sharpe ratio, the
 * transaction cost paid and the trades.
 *
 * usage: node apps/stocks/jobs/test.js [--symbols A,B,...] [--episodes 1] [--max-positions 1]
 *        [--transaction-cost 0] [--initial-capital 15000] [--min-price 0] [--max-price 5000] [--holdout 0]
 *        [--offset 0] [--max-frames N] [--timeframe 1D] [--volume] [--every N] [--random-baseline] [--seed 1]
 *        [--save-brain <label>] [--load-brain <label>] [--disable-learning]
 */
import path from 'node:path';
import { runJob, readFlag, readNumber } from 'robot-brain';
import { LapChannelsJob, DATA_ROOT } from '../lap-channels.js';

// the stocks traded where --symbols names none
const SYMBOLS = [
	'SO', 'VALE', 'STLD', 'GOOGL', 'MU', 'PLTR', 'UUUU', 'PFE', 'CRM', 'HAL',
	'AWR', 'GM', 'EQIX', 'RTX', 'KGC', 'ALB', 'AAPL', 'CVX', 'HD',
	'WPM', 'BEP', 'AREC', 'JNJ', 'SLB', 'PLD', 'EXK', 'NVDA', 'CAT', 'WFC',
	'RGLD', 'WEAT', 'OXY', 'CEG', 'LOW', 'PAAS', 'MP', 'LMT', 'GS', 'COST',
	'AG', 'TECK', 'MRK', 'INTC', 'BIP', 'PSA', 'DVN', 'AVAV', 'PEP', 'CDE',
	'TSM', 'FCX', 'PM', 'NUE', 'LEU', 'AMT', 'WMT', 'MRVL', 'F',
	'RIO', 'NOC', 'V', 'ENB', 'BTU', 'AEM', 'AMZN', 'KLAC', 'CLF', 'O',
	'NEM', 'GD', 'BAC', 'NEE', 'SQM', 'ABBV', 'AMAT', 'KMI', 'PG', 'UEC',
	'GOLD', 'BHP', 'CRML', 'LLY', 'AVGO', 'FNV', 'JPM', 'DE', 'TM', 'WM',
	'HL', 'CCJ', 'COP', 'USAR', 'XOM', 'AMD', 'LAC', 'MSFT', 'MUX', 'SPY',
];

class StockTestJob extends LapChannelsJob {

	constructor() {
		super({
			mode: 'seq',
			cycles: readNumber('--episodes', 1),
			folder: path.join(DATA_ROOT, readFlag('--timeframe', '1D')),
			symbols: readFlag('--symbols')?.split(',') ?? SYMBOLS,
			every: readNumber('--every', 0),
			all4: false,
			volume: process.argv.includes('--volume'),
			offset: readNumber('--offset', 0),
			holdout: readNumber('--holdout', 0),
			maxFrames: readNumber('--max-frames', null),
			randomBaseline: process.argv.includes('--random-baseline'),
			seed: readNumber('--seed', 1),
			trader: {
				maxPositions: readNumber('--max-positions', 1),
				costPercent: readNumber('--transaction-cost', 0),
				initialCapital: readNumber('--initial-capital', 15000),
				minPrice: readNumber('--min-price', 0),
				maxPrice: readNumber('--max-price', 5000),
			},
		});
	}
}

await runJob(import.meta, StockTestJob);
