/**
 * The simple synthetic cycle test on UCAR (the twin of synthetic-cycle-test.js): one channel, a binary
 * price bucket and own/out, the six-bar pattern +1% -2% -3% +2% +3% -1% repeated. The world runs the right
 * action where the machine is silent; the reward is the bar's return signed by the action, scoped to the
 * channel and the frame. Spec configuration: nothing switched off.
 *
 *   node apps/stocks/jobs/ucar-cycle.js [--cycles 100] [--h 256] [--radius 6] [--trace]
 */
import { Ucar } from '../../../brain/src/ucar.js';

function arg(flag, def, parse = x => x) {
	const i = process.argv.indexOf(flag);
	return i !== -1 && process.argv[i + 1] !== undefined ? parse(process.argv[i + 1]) : def;
}
const cycles = arg('--cycles', 100, Number);
const h = arg('--h', 256, Number);
const radius = arg('--radius', 6, Number);
const trace = process.argv.includes('--trace');
const PATTERN = [0.009, -0.019, -0.029, 0.019, 0.029, -0.009];
const n = PATTERN.length;

const m = new Ucar({ h, timeRadius: radius, policy: 'eventsToActions', actionPolicy: 'leastBad' });
const ch = m.declareChannel({ name: 'TEST', event: { name: 'price_change', resolution: 2 }, action: { name: 'activity', resolution: 2 } });
console.log(`UCAR simple cycle: ${cycles} cycles of ${n} bars, H=${h} R=${radius}`);

const total = cycles * n;
let pending = null, lessons = 0, minted = 0, added = 0, retired = 0;
const stats = Array.from({ length: n }, () => ({ optimal: 0, suboptimal: 0 }));
const last10 = Array.from({ length: n }, () => ({ optimal: 0, suboptimal: 0 }));
const t0 = Date.now();
for (let k = 0; k < total; k++) {
	const pct = PATTERN[k % n] * 100;
	const up = pct > 0 ? 1 : 0;
	const lesson = pending === null;
	const action = lesson ? up : pending;
	if (lesson) lessons++;
	const reward = action === 1 ? pct : -pct;
	const cf = k % n;
	if (!lesson) {
		if (action === up) stats[cf].optimal++; else stats[cf].suboptimal++;
		if (k >= total - 10 * n) { if (action === up) last10[cf].optimal++; else last10[cf].suboptimal++; }
	}
	const r = m.processFrame(new Int32Array([ch.eventDim, up, 0, 0, ch.actionDim, action, 0, 0]), [{ reward, channels: [ch.channelId], frames: 1 }]);
	minted += r.stats.minted; added += r.stats.added; retired += r.stats.retired;
	const w = r.winners.find(w => w.dim === ch.actionDim);
	pending = w ? w.bucket : null;
	if (trace && k + 2 * n >= total) {
		const nextUp = PATTERN[(k + 1) % n] > 0 ? 1 : 0;
		console.log(`  f${k + 1} cf${cf + 1} price=${up} ran=${action}${lesson ? 'L' : ''} r=${reward.toFixed(1)} | apex ${r.stats.apexActs} levels ${r.stats.levels} minted ${r.stats.minted} voters ${r.stats.votersAboveBase}/${r.stats.voters} | says ${pending} est ${w ? w.estimate.toFixed(2) : '-'} -> ${pending === nextUp ? 'ok' : 'WRONG'}`);
	}
}
console.log('cycle frame | optimal | suboptimal | last 10 cycles');
let to = 0, ts = 0, lo = 0, ls = 0;
for (let i = 0; i < n; i++) {
	const S = stats[i], L = last10[i];
	console.log(`${String(i + 1).padStart(11)} | ${String(S.optimal).padStart(7)} | ${String(S.suboptimal).padStart(10)} | ${L.optimal}/${L.optimal + L.suboptimal}`);
	to += S.optimal; ts += S.suboptimal; lo += L.optimal; ls += L.suboptimal;
}
const s = m.stats();
console.log(`overall optimal ${to}/${to + ts} = ${(100 * to / Math.max(1, to + ts)).toFixed(1)}% | last 10 cycles ${lo}/${lo + ls} = ${(100 * lo / Math.max(1, lo + ls)).toFixed(1)}% | lessons ${lessons}`);
console.log(`neurons ${s.neurons} alive (${minted} minted) | patterns ${s.patterns} alive (${added} added, ${retired} retired) | levels ${JSON.stringify(m.levelCounts())} | ${Date.now() - t0}ms`);
