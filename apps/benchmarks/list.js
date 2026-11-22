/**
 * The 23 benchmarks: each one's name (its saved output is docs/benchmarks/output/<name>.out), its job, the job's
 * arguments (`<mnist>` stands for the MNIST data folder), and what of its output is no part of its results.
 * `drop` names the lines of diagnostics the build no longer prints, and the TRADER lines the stock jobs print since;
 * `edit` takes such a diagnostic out of a result line that also carried one.
 */

const STOCK_DROP = /^(CLOSING|UNPAID|LIFTED TO THE STANDING CALL|LIFT PASSED OVER|CYCLELAPS|JOINSMOVES|TRADER) /;
const TEXT_DROP = /^(CLOSING|UNPAID) /;
const SHAPES_DROP = /^(BARE|CLOSING|UNPAID|LIFTED TO THE STANDING CALL|LIFT PASSED OVER|TWICE|CYCLELAPS) |^dictionary: [0-9]+ functions, [0-9]+ edges/;
const MNIST_DROP = /^(CLOSING|UNPAID|LIFTED TO THE STANDING CALL|LIFT PASSED OVER|TWICE|CYCLELAPS|JOINSMOVES) /;
const MNIST_EDIT = { pattern: /^(CHANGE .*) \| joins [0-9]+ \| moves [0-9]+$/, replacement: '$1' };

const TEN_STOCKS = 'SO,VALE,STLD,GOOGL,MU,PLTR,UUUU,PFE,CRM,HAL';

/**
 * An MNIST benchmark: 14×14 images, the first `images` training images, shown `episodes` times.
 */
function makeMnist(name, images, episodes) {
	const args = ['--data', '<mnist>', '--image-size', '14', '--train-images', String(images), '--episodes', String(episodes)];
	return { name, job: 'apps/mnist/jobs/test.js', args, drop: MNIST_DROP, edit: MNIST_EDIT };
}

export const BENCHMARKS = [
	// the alternation
	{ name: 'alternation', job: 'apps/text/jobs/alternation.js', args: ['33'] },
	// stocks on laps
	{ name: 'six-bar-lap', job: 'apps/stocks/jobs/synthetic-cycle-test.js', args: ['30', '256'], drop: /^(LIFTED TO THE STANDING CALL|LIFT PASSED OVER) / },
	{ name: 'kgc-gld', job: 'apps/stocks/jobs/multi-channel-test.js', args: ['--mode', 'two', '--laps', '10'], drop: STOCK_DROP },
	{ name: 'two-six-bar-laps', job: 'apps/stocks/jobs/multi-channel-test.js', args: ['--mode', 'lap', '--laps', '30'], drop: STOCK_DROP },
	{ name: 'six-and-five-bar-laps', job: 'apps/stocks/jobs/multi-channel-test.js', args: ['--mode', 'lap65', '--laps', '12'], drop: STOCK_DROP },
	{ name: 'three-stocks', job: 'apps/stocks/jobs/multi-channel-test.js', args: ['--mode', 'three', '--laps', '100'], drop: STOCK_DROP },
	{ name: 'four-stocks', job: 'apps/stocks/jobs/multi-channel-test.js', args: ['--mode', 'three4', '--laps', '100', '--all4'], drop: STOCK_DROP },
	// real stocks
	{ name: 'three-stocks-5-laps', job: 'apps/stocks/jobs/test.js', args: ['--symbols', 'KGC,GOLD,SPY', '--episodes', '5', '--volume'], drop: STOCK_DROP },
	{ name: 'ten-stocks', job: 'apps/stocks/jobs/test.js', args: ['--symbols', TEN_STOCKS, '--episodes', '1', '--max-positions', '3', '--transaction-cost', '0.02', '--every', '100', '--volume'], drop: STOCK_DROP },
	// tap and text
	{ name: 'tap', job: 'apps/tap/jobs/tap.js', args: ['6', '8'] },
	{ name: 'text', job: 'apps/text/jobs/test.js', args: ['the the the the the the the the the ', '--describe', 'h'], drop: TEXT_DROP },
	{ name: 'text-without-space', job: 'apps/text/jobs/test.js', args: ['theathebthecthedthefthegtheithejthek', '--describe', 'h'], drop: TEXT_DROP },
	// lines and squares
	{ name: 'hline', job: 'apps/shapes/jobs/shapes.js', args: ['hline', '30'], drop: SHAPES_DROP },
	{ name: 'vline', job: 'apps/shapes/jobs/shapes.js', args: ['vline', '30'], drop: SHAPES_DROP },
	{ name: 'filled-square', job: 'apps/shapes/jobs/shapes.js', args: ['square', '30'], drop: SHAPES_DROP },
	{ name: 'hollow-square', job: 'apps/shapes/jobs/shapes.js', args: ['hollow', '30'], drop: SHAPES_DROP },
	{ name: 'two-and-three', job: 'apps/shapes/jobs/shapes.js', args: ['teach23', '30'], drop: SHAPES_DROP },
	// MNIST
	makeMnist('mnist-1x60', 1, 60),
	makeMnist('mnist-10x20', 10, 20),
	makeMnist('mnist-10x300', 10, 300),
	makeMnist('mnist-255', 255, 1),
	makeMnist('mnist-100', 100, 1),
	makeMnist('mnist-1000', 1000, 1),
];
