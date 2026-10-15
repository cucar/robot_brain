/**
 * Stocks on UCAR (docs/algorithm-stocks.md): a price channel per symbol over time alone, one event
 * dimension (the bar's return, quantized), one action dimension (own / disown), the neighborhood
 * policy where no action is anyone's neighbor and only events and interneurons connect, to actions
 * alone; a world that runs the right action wherever the machine is silent; and a reward scoped to the
 * channel and the frame.
 *
 *   node apps/stocks/jobs/ucar.js [--symbols AAPL,MSFT] [--timeframe 1D] [--episodes 1] [--h 256]
 *                                 [--radius 1] [--holdout 0] [--max-frames 0] [--buckets 7]
 *
 * Reports, per episode: how often the machine spoke, how often what it said was right (the bar moved
 * the way it said), lessons taken, and the return of an equal-weight portfolio that holds a symbol
 * while the action over it is `own`, against buy-and-hold over the same bars.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Ucar } from '../../../brain/src/ucar.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function arg(flag, def, parse = x => x) {
	const i = process.argv.indexOf(flag);
	return i !== -1 && process.argv[i + 1] !== undefined ? parse(process.argv[i + 1]) : def;
}

const symbols = arg('--symbols', 'AAPL,MSFT,GOOGL,AMZN,NVDA,JPM,XOM,PG,CAT,PFE').split(',');
const timeframe = arg('--timeframe', '1D');
const episodes = arg('--episodes', 1, Number);
const h = arg('--h', 256, Number);
const radius = arg('--radius', 1, Number);
const holdout = arg('--holdout', 0, Number);
const maxFrames = arg('--max-frames', 0, Number);
const nBuckets = arg('--buckets', 7, Number);

// bucket boundaries on the percent return: symmetric, finer near zero
const BOUNDS = { 3: [-0.25, 0.25], 5: [-1, -0.25, 0.25, 1], 7: [-2, -0.75, -0.2, 0.2, 0.75, 2], 9: [-3, -1.5, -0.6, -0.15, 0.15, 0.6, 1.5, 3] }[nBuckets];
if (!BOUNDS) throw new Error('--buckets must be 3, 5, 7 or 9');
const bucketOf = pct => { let i = 0; while (i < BOUNDS.length && pct > BOUNDS[i]) i++; return i; };
const DISOWN = 0, OWN = 1;

function loadRows(symbol) {
	const csv = fs.readFileSync(path.join(__dirname, '..', 'data', timeframe, `${symbol}.csv`), 'utf-8');
	return csv.split('\n').filter(l => l.trim()).map(l => { const p = l.trim().split(','); return { price: parseFloat(p[0]), volume: parseFloat(p[1]) }; });
}

const data = symbols.map(loadRows);
const rowCount = Math.min(...data.map(r => r.length));
const frames = maxFrames > 0 ? Math.min(maxFrames, rowCount) : rowCount;

const m = new Ucar({ h, timeRadius: radius, policy: 'eventsToActions', actionPolicy: 'leastBad' });
const channels = symbols.map(s => m.declareChannel({ name: s, event: { name: `${s}_move`, resolution: nBuckets }, action: { name: `${s}_own`, resolution: 2 } }));

console.log(`UCAR stocks: ${symbols.length} symbols, ${timeframe}, ${frames} bars (${holdout} held out at the end), H=${h} R=${radius} buckets=${nBuckets}`);

for (let ep = 1; ep <= episodes; ep++) {
	m.resetContext();
	const t0 = Date.now();
	// per-symbol state: the action in effect over the coming bar, and whether the machine chose it
	const pending = symbols.map(() => null);
	const seg = { train: newSeg(), test: newSeg() };
	let brainMs = 0;
	const inputs = new Int32Array(symbols.length * 2 * 4);
	for (let f = 1; f < frames; f++) {
		const S = f >= frames - holdout ? seg.test : seg.train;
		const rewards = [];
		let k = 0;
		let barReturn = 0, machineReturn = 0;
		for (let i = 0; i < symbols.length; i++) {
			const prev = data[i][f - 1].price, cur = data[i][f].price;
			if (!(prev > 0 && cur > 0)) { continue; }
			const pct = (cur - prev) / prev * 100;
			const right = pct > 0 ? OWN : DISOWN;
			let action, lesson;
			if (pending[i] === null) { action = right; lesson = true; } else { action = pending[i]; lesson = false; }
			const reward = action === OWN ? pct : -pct;
			if (lesson) S.lessons++; else { S.spoke++; if (reward > 0) S.right++; else if (reward < 0) S.wrong++; }
			// equal-weight portfolio: a symbol earns the bar's return while owned
			barReturn += pct / symbols.length;
			machineReturn += (action === OWN ? pct : 0) / symbols.length;
			inputs[k++] = channels[i].eventDim; inputs[k++] = bucketOf(pct); inputs[k++] = 0; inputs[k++] = 0;
			inputs[k++] = channels[i].actionDim; inputs[k++] = action; inputs[k++] = 0; inputs[k++] = 0;
			rewards.push({ reward, channels: [channels[i].channelId], frames: 1 });
		}
		S.frames++;
		S.buyHold *= 1 + barReturn / 100;
		S.machine *= 1 + machineReturn / 100;
		const r = m.processFrame(inputs.subarray(0, k), rewards);
		brainMs += r.stats.elapsedMs;
		for (let i = 0; i < symbols.length; i++) {
			const w = r.winners.find(w => w.dim === channels[i].actionDim);
			pending[i] = w ? w.bucket : null;
		}
		if (f % 500 === 0) process.stdout.write(`\r  bar ${f}/${frames} spoke=${S.spoke} right=${(100 * S.right / Math.max(1, S.right + S.wrong)).toFixed(1)}% neurons=${r.stats.neurons} patterns=${r.stats.patterns} ${r.stats.elapsedMs.toFixed(1)}ms/frame   `);
	}
	process.stdout.write('\n');
	const s = m.stats();
	console.log(`episode ${ep} (${Date.now() - t0}ms, brain ${brainMs.toFixed(0)}ms): neurons ${s.neurons} patterns ${s.patterns} connections ${s.connections} levels ${JSON.stringify(m.levelCounts())}`);
	report('train', seg.train);
	if (holdout > 0) report('holdout', seg.test);
}

function newSeg() { return { frames: 0, spoke: 0, right: 0, wrong: 0, lessons: 0, buyHold: 1, machine: 1 }; }
function report(name, S) {
	const decided = S.right + S.wrong;
	console.log(`  ${name}: ${S.frames} bars | machine spoke ${S.spoke} of ${S.spoke + S.lessons} symbol-bars (${S.lessons} lessons) | right ${S.right}/${decided} (${(100 * S.right / Math.max(1, decided)).toFixed(1)}%) | portfolio return: machine ${((S.machine - 1) * 100).toFixed(1)}% vs buy&hold ${((S.buyHold - 1) * 100).toFixed(1)}%`);
}
if (process.argv.includes('--dump')) console.log(m.dumpTables(30));
