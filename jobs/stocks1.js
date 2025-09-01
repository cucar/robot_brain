import Job from './job.js';
import StockChannel from '../channels/stock.js';

export default class StocksJob1 extends Job {

	/**
	 * returns the channels for the job - we will have many stocks/stats here - each stock will be a separate channel
	 */
	getChannels() {
		return [ { name: 'AAPL', channelClass: StockChannel } ];
	}

}