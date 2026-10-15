/**
 * The synthetic extended cycle test on UCAR (the twin of synthetic-extended-test.js): the first 12 rows
 * of KGC repeated, binary price and volume buckets, own/out, the world running the right action where the
 * machine is silent, and a reward scoped to the price channel and the frame. Reports optimality per cycle
 * frame, overall and over the last ten cycles.
 *
 *   node apps/stocks/jobs/ucar-extended.js [--cycles 100] [--h 256] [--radius 12] [--gap 0] [--no-variables]
 *                                          [--no-volume] [--trace]
 *
 * Defaults are the spec's configuration: no silent frames and variables on. `--gap N` is a diagnostic probe
 * that inserts N silent frames after each cycle; `--no-variables` is a probe that skips the pick's first
 * stage. Neither is part of the design, and results from them are not the spec's numbers.
 */
import { Ucar } from '../../../brain/src/ucar.js';

function arg(flag, def, parse = x => x) {
	const i = process.argv.indexOf(flag);
	return i !== -1 && process.argv[i + 1] !== undefined ? parse(process.argv[i + 1]) : def;
}
const cycles = arg('--cycles', 100, Number);
const h = arg('--h', 256, Number);
const radius = arg('--radius', 4, Number);
const gap = arg('--gap', 0, Number);
const variables = !process.argv.includes('--no-variables');
const noVolume = process.argv.includes('--no-volume');
const trace = process.argv.includes('--trace');

// the first 12 rows of KGC.csv, as in synthetic-extended-test.js
const ROWS = [
	[8.10, 1447100], [8.11, 2112900], [8.35, 1411400], [8.29, 2091100], [8.20, 1247200], [8.15, 770000],
	[8.19, 1948400], [8.05, 2701100], [7.94, 1280800], [7.86, 3083500], [7.22, 2742800], [7.51, 1510600],
];
const n = ROWS.length;

const m = new Ucar({ h, timeRadius: radius, policy: 'eventsToActions', actionPolicy: 'leastBad', variables });
const price = m.declareChannel({ name: 'price', event: { name: 'price_change', resolution: 2 }, action: { name: 'activity', resolution: 2 } });
const volume = noVolume ? null : m.declareChannel({ name: 'volume', event: { name: 'volume_change', resolution: 2 } });

console.log(`UCAR extended cycle: ${cycles} cycles of 12 bars, H=${h} R=${radius} gap=${gap} variables=${variables} volume=${!noVolume}`);

const total = cycles * n;
let pending = null;
const stats = Array.from({ length: n }, () => ({ optimal: 0, suboptimal: 0 }));
const last10 = Array.from({ length: n }, () => ({ optimal: 0, suboptimal: 0 }));
let lessons = 0, pnl = 0, added = 0, retired = 0, minted = 0;
const t0 = Date.now();
for (let k = 1; k <= total; k++) {
	const prev = ROWS[(k - 1) % n], cur = ROWS[k % n];
	const pct = (cur[0] - prev[0]) / prev[0] * 100;
	const vol = (cur[1] - prev[1]) / prev[1] * 100;
	const up = pct > 0 ? 1 : 0;
	const lesson = pending === null;
	const action = lesson ? up : pending;
	if (lesson) lessons++;
	const reward = action === 1 ? pct : -pct;
	if (action === 1) pnl += pct;
	const cf = (k - 1) % n;
	if (!lesson) {
		const S = stats[cf];
		if (action === up) S.optimal++; else S.suboptimal++;
		if (k > total - 10 * n) { const L = last10[cf]; if (action === up) L.optimal++; else L.suboptimal++; }
	}
	const inputs = noVolume
		? new Int32Array([price.eventDim, up, 0, 0, price.actionDim, action, 0, 0])
		: new Int32Array([price.eventDim, up, 0, 0, price.actionDim, action, 0, 0, volume.eventDim, vol > 0 ? 1 : 0, 0, 0]);
	let r = m.processFrame(inputs, [{ reward, channels: [price.channelId], frames: 1 }]);
	added += r.stats.added; retired += r.stats.retired; minted += r.stats.minted;
	if (gap > 0 && k % n === 0) for (let g = 0; g < gap; g++) r = m.processFrame(new Int32Array(0), []);
	const w = r.winners.find(w => w.dim === price.actionDim);
	pending = w ? w.bucket : null;
	if (trace && k > total - 2 * n)
		console.log(`  bar ${k} cf${cf + 1} price=${up} ran=${action}${lesson ? 'L' : ''} r=${reward.toFixed(2)} | says ${pending} (est ${w ? w.estimate.toFixed(2) : '-'}, voters ${w ? w.voters : 0}) | apex ${r.stats.apexActs} minted ${r.stats.minted}`);
}

console.log('cycle frame | optimal | suboptimal | last 10 cycles');
let to = 0, ts = 0, lo = 0, ls = 0;
for (let i = 0; i < n; i++) {
	const S = stats[i], L = last10[i];
	console.log(`${String(i + 1).padStart(11)} | ${String(S.optimal).padStart(7)} | ${String(S.suboptimal).padStart(10)} | ${L.optimal}/${L.optimal + L.suboptimal}`);
	to += S.optimal; ts += S.suboptimal; lo += L.optimal; ls += L.suboptimal;
}
const s = m.stats();
console.log(`overall optimal ${to}/${to + ts} = ${(100 * to / Math.max(1, to + ts)).toFixed(1)}% | last 10 cycles ${lo}/${lo + ls} = ${(100 * lo / Math.max(1, lo + ls)).toFixed(1)}% | lessons ${lessons} | pnl ${pnl >= 0 ? '+' : ''}${pnl.toFixed(1)}%`);
console.log(`neurons ${s.neurons} alive (${minted} minted) | patterns ${s.patterns} alive (${added} added, ${retired} retired) | levels ${JSON.stringify(m.levelCounts())} | ${Date.now() - t0}ms`);
