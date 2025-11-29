import StockChannel from './channels/stock.js';

/**
 * Test to verify that feedback is properly delayed by one frame
 * This ensures rewards are applied based on the outcome of executed decisions,
 * not the current observation.
 */

async function testFeedbackTiming() {
	console.log('Testing feedback timing fix...\n');

	// Create a stock channel
	const channel = new StockChannel('TEST');
	channel.debug = true;

	// Create test data with clear price movements
	channel.allRows = [
		{ price: 100, volume: 1000 },  // Day 0
		{ price: 100, volume: 1000 },  // Day 1 (baseline)
		{ price: 105, volume: 1100 },  // Day 2 (+5% price increase)
		{ price: 110, volume: 1200 },  // Day 3 (+4.76% price increase)
		{ price: 108, volume: 1150 },  // Day 4 (-1.82% price decrease)
	];
	channel.prepareDataIterator();

	console.log('=== Frame 1: Initial frame (reads 2 rows) ===');
	const frame1 = await channel.getFrameEvents();
	console.log('Frame 1 inputs:', frame1);
	console.log('currentPrice:', channel.currentPrice);
	console.log('priceForFeedback:', channel.priceForFeedback);
	
	const feedback1 = await channel.getRewards();
	console.log('Feedback 1:', feedback1, '(should be 1.0 - no trade yet)\n');

	// Simulate a buy decision
	console.log('=== Executing BUY decision ===');
	channel.hasTraded = true;
	channel.owned = true;
	channel.entryPrice = channel.currentPrice;
	console.log('Bought at:', channel.entryPrice, '\n');

	console.log('=== Frame 2: After buy decision ===');
	const frame2 = await channel.getFrameEvents();
	console.log('Frame 2 inputs:', frame2);
	console.log('currentPrice:', channel.currentPrice);
	console.log('priceForFeedback:', channel.priceForFeedback);
	
	const feedback2 = await channel.getRewards();
	console.log('Feedback 2:', feedback2);
	console.log('Expected: 105/100 = 1.05 (5% gain)\n');

	console.log('=== Frame 3: Continue holding ===');
	const frame3 = await channel.getFrameEvents();
	console.log('Frame 3 inputs:', frame3);
	console.log('currentPrice:', channel.currentPrice);
	console.log('priceForFeedback:', channel.priceForFeedback);
	
	const feedback3 = await channel.getRewards();
	console.log('Feedback 3:', feedback3);
	console.log('Expected: 110/105 = 1.0476 (4.76% gain)\n');

	console.log('=== Frame 4: Continue holding ===');
	const frame4 = await channel.getFrameEvents();
	console.log('Frame 4 inputs:', frame4);
	console.log('currentPrice:', channel.currentPrice);
	console.log('priceForFeedback:', channel.priceForFeedback);
	
	const feedback4 = await channel.getRewards();
	console.log('Feedback 4:', feedback4);
	console.log('Expected: 108/110 = 0.9818 (1.82% loss)\n');

	// Verify the fix
	console.log('=== Verification ===');
	if (Math.abs(feedback1 - 1.0) < 0.001) console.log('✓ Frame 1: Correct (no trade yet)');
	else console.log('✗ Frame 1: FAILED - expected 1.0, got', feedback1);

	if (Math.abs(feedback2 - 1.05) < 0.001) console.log('✓ Frame 2: Correct (5% gain)');
	else console.log('✗ Frame 2: FAILED - expected 1.05, got', feedback2);

	if (Math.abs(feedback3 - 1.0476) < 0.001) console.log('✓ Frame 3: Correct (4.76% gain)');
	else console.log('✗ Frame 3: FAILED - expected 1.0476, got', feedback3);

	if (Math.abs(feedback4 - 0.9818) < 0.001) console.log('✓ Frame 4: Correct (1.82% loss)');
	else console.log('✗ Frame 4: FAILED - expected 0.9818, got', feedback4);

	console.log('\nTest complete!');
}

testFeedbackTiming().catch(console.error);

