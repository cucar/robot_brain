import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import Alpaca from './channels/alpaca.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Stock Download Script - Downloads historical stock data from Alpaca and saves as JSON
 * Run with: node stock-download.js --timeframe=1Min
 */

// Configuration - edit these values as needed
const config = {
	symbols: ['KGC', 'GLD', 'SPY', 'AAPL', 'NEM', 'GDX'],
	startDate: '2020-01-01',
	endDate: '2026-01-01'
};

/**
 * Parse command line arguments
 */
function parseArgs() {
	const args = process.argv.slice(2);
	let timeframe = '1Min'; // default
	for (const arg of args)
		if (arg.startsWith('--timeframe=')) timeframe = arg.split('=')[1];
	return { timeframe };
}

/**
 * Download historical data for a single symbol from Alpaca
 */
async function downloadSymbol(alpacaClient, symbol, timeframe, dataDir) {
	console.log(`📊 Downloading ${symbol} (${timeframe})...`);
	
	const bars = await alpacaClient.getBars(symbol, config.startDate, config.endDate, timeframe);
	
	// Save raw bars to JSON
	const filePath = path.join(dataDir, `${symbol}.json`);
	fs.writeFileSync(filePath, JSON.stringify(bars, null, 2));
	
	console.log(`   ✅ ${symbol}.json: ${bars.length} bars`);
}

/**
 * Main download function
 */
async function main() {
	const { timeframe } = parseArgs();

	console.log('📥 Stock Data Download');
	console.log(`📊 Symbols: ${config.symbols.join(', ')}`);
	console.log(`⏱️  Timeframe: ${timeframe}`);
	console.log(`📅 Date Range: ${config.startDate} to ${config.endDate}`);
	console.log('');

	// Create data directory
	const dataDir = path.join(__dirname, 'data', 'stock', timeframe);
	if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true });

	// Initialize Alpaca client
	const alpacaClient = new Alpaca();

	// Download each symbol
	for (const symbol of config.symbols)
		await downloadSymbol(alpacaClient, symbol, timeframe, dataDir);

	console.log('');
	console.log('✅ All data downloaded successfully!');
}

// Run the script
main().catch(error => {
	console.error('❌ Error:', error.message);
	process.exit(1);
});