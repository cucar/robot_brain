import 'dotenv/config'; // Load environment variables from .env file
import Alpaca from './alpaca.js';

it('should connect to alpaca and get bars', async () => {
	const alpaca = new Alpaca();
	const bars = await alpaca.getBars('AAPL', '2020-01-01', '2020-02-01', '1D');
	console.log(bars);
});