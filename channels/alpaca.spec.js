import path from 'node:path';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';
import Alpaca from './alpaca.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(__dirname, '..', 'apps', 'stocks', '.env') });

it('should connect to alpaca and get bars', async () => {
	const alpaca = new Alpaca();
	const bars = await alpaca.getBars('AAPL', '2020-01-01', '2020-02-01', '1D');
	console.log(bars);
});