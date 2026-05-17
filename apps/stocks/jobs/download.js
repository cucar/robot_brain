import dotenv from 'dotenv';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import Alpaca from '../alpaca.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load .env from the stocks app folder so the script works no matter which cwd it's run from.
dotenv.config({ path: path.join(__dirname, '..', '.env') });

/**
 * Stock Download Script - Downloads historical stock data from Alpaca and saves as JSON
 * Run with: node stock-download.js --timeframe 1Min
 */

// Configuration - edit these values as needed
const config = {
	timeframe: '5Min',
	symbols: [
		// 100 stocks - expected to be good
		'SO', 'VALE', 'STLD', 'GOOGL', 'MU', 'PLTR', 'UUUU', 'PFE', 'CRM', 'HAL',
		'AWR', 'SAND', 'GM', 'EQIX', 'RTX', 'KGC', 'ALB', 'AAPL', 'CVX', 'HD',
		'WPM', 'BEP', 'AREC', 'JNJ', 'SLB', 'PLD', 'EXK', 'NVDA', 'CAT', 'WFC',
		'RGLD', 'WEAT', 'OXY', 'CEG', 'LOW', 'PAAS', 'MP', 'LMT', 'GS', 'COST',
		'AG', 'TECK', 'MRK', 'INTC', 'BIP', 'PSA', 'DVN', 'AVAV', 'PEP', 'CDE',
		'TSM', 'FCX', 'PM', 'NUE', 'LEU', 'AMT', 'WMT', 'MRVL', 'F', 'SILV',
		'RIO', 'NOC', 'V', 'ENB', 'BTU', 'AEM', 'AMZN', 'KLAC', 'CLF', 'O',
		'NEM', 'GD', 'BAC', 'NEE', 'SQM', 'ABBV', 'AMAT', 'KMI', 'PG', 'UEC',
		'GOLD', 'BHP', 'CRML', 'LLY', 'AVGO', 'FNV', 'JPM', 'DE', 'TM', 'WM',
		'HL', 'CCJ', 'COP', 'USAR', 'XOM', 'AMD', 'LAC', 'MSFT', 'MUX', 'SPY'

		// loser batch 1 - it's very bad, but alive
		// 'PTON', 'RIVN', 'BABA', 'MRNA', 'PARA', 'SNAP', 'PYPL', 'INTC', 'KSS', 'PLUG'

		// loser batch 2 - catastrophic losses - going to zero - de-listings
		// 'HOOD', 'COIN', 'RKT', 'TDOC', 'NKLA', 'LCID', 'ZM', 'DOCU', 'AFRM', 'UPST'
	],
	startDate: '2021-05-13',
	endDate: '2026-05-13'
};

/**
 * Parse command line arguments
 */
function parseArgs() {
	let timeframe = config.timeframe;
	const timeframeIndex = process.argv.indexOf('--timeframe');
	if (timeframeIndex !== -1 && process.argv[timeframeIndex + 1]) timeframe = process.argv[timeframeIndex + 1];

	let startDate = config.startDate; // default
	const startIndex = process.argv.indexOf('--start');
	if (startIndex !== -1 && process.argv[startIndex + 1]) startDate = process.argv[startIndex + 1];

	let endDate = config.endDate; // default
	const endIndex = process.argv.indexOf('--end');
	if (endIndex !== -1 && process.argv[endIndex + 1]) endDate = process.argv[endIndex + 1];

	return { timeframe, startDate, endDate };
}

/**
 * Download historical data for a single symbol from Alpaca
 */
async function downloadSymbol(alpacaClient, symbol, timeframe, startDate, endDate, dataDir) {
	console.log(`📊 Downloading ${symbol} (${timeframe})...`);

	const bars = await alpacaClient.getBars(symbol, startDate, endDate, timeframe);

	// Save raw bars to JSON
	const filePath = path.join(dataDir, `${symbol}.json`);
	fs.writeFileSync(filePath, JSON.stringify(bars, null, 2));

	console.log(`   ✅ ${symbol}.json: ${bars.length} bars`);
}

/**
 * Main download function
 */
async function main() {
	const { timeframe, startDate, endDate } = parseArgs();

	console.log('📥 Stock Data Download');
	console.log(`📊 Symbols: ${config.symbols.join(', ')}`);
	console.log(`⏱️  Timeframe: ${timeframe}`);
	console.log(`📅 Date Range: ${startDate} to ${endDate}`);
	console.log('');

	// Create data directory
	const dataDir = path.join(__dirname, '..', 'data', timeframe);
	if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true });

	// Initialize Alpaca client
	const alpacaClient = new Alpaca();

	// Download each symbol
	for (const symbol of config.symbols)
		await downloadSymbol(alpacaClient, symbol, timeframe, startDate, endDate, dataDir);

	console.log('');
	console.log('✅ All data downloaded successfully!');
}

// Run the script
main().catch(error => {
	console.error('❌ Error:', error.message);
	process.exit(1);
});