/**
 * Text on UCAR (docs/algorithm-text.md): one channel of letters over time alone, no actions.
 * Streams a file one byte per frame and scores the event the machine infers for the frame ahead
 * against the byte that arrives. Standalone: imports the binding by path, no workspace install needed.
 *
 *   node apps/text/jobs/ucar.js [--file test.txt] [--episodes 5] [--h 256] [--radius 1] [--trace N]
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

const fileArg = arg('--file', 'test.txt');
const file = path.basename(fileArg) === fileArg ? path.join(__dirname, '..', 'data', fileArg) : path.resolve(fileArg);
const episodes = arg('--episodes', 5, Number);
const h = arg('--h', 256, Number);
const radius = arg('--radius', 1, Number);
const trace = arg('--trace', 0, Number);

const bytes = fs.readFileSync(file);
const m = new Ucar({ h, timeRadius: radius, policy: 'all' });
const ch = m.declareChannel({ name: 'text', event: { name: 'letter', resolution: 256 } });
const dim = ch.eventDim;

console.log(`UCAR text: ${file} (${bytes.length} bytes), episodes=${episodes} H=${h} R=${radius}`);
const results = [];
for (let ep = 1; ep <= episodes; ep++) {
	m.resetContext();
	const t0 = Date.now();
	let pred = null, correct = 0, predicted = 0, apexCost = 0, childVoters = 0, brainMs = 0;
	const frame = new Int32Array(4);
	for (let i = 0; i < bytes.length; i++) {
		const b = bytes[i];
		if (pred !== null) { predicted++; if (pred === b) correct++; }
		frame[0] = dim; frame[1] = b; frame[2] = 0; frame[3] = 0;
		const r = m.processFrame(frame, []);
		const w = r.winners.find(w => w.dim === dim);
		pred = w ? w.bucket : null;
		apexCost += r.stats.apexCost;
		brainMs += r.stats.elapsedMs;
		if (r.stats.votersAboveBase > 0) childVoters++;
		if (trace && ep === episodes && i >= bytes.length - trace)
			console.log(`  '${String.fromCharCode(b)}' apex=${r.stats.apexCost.toFixed(1)} levels=${r.stats.levels} bids=${r.stats.bids} acc=${r.stats.accepted} minted=${r.stats.minted} added=${r.stats.added} retired=${r.stats.retired} voters=${r.stats.votersAboveBase}/${r.stats.voters} -> '${pred === null ? '' : String.fromCharCode(pred)}'`);
	}
	const s = m.stats();
	const acc = 100 * correct / Math.max(1, predicted);
	results.push({ ep, acc });
	console.log(`episode ${ep}: accuracy ${acc.toFixed(1)}% (${correct}/${predicted}) | apex ${(apexCost / bytes.length).toFixed(2)} bits/frame | child voters ${(100 * childVoters / bytes.length).toFixed(0)}% of frames | neurons ${s.neurons} patterns ${s.patterns} connections ${s.connections} levels ${JSON.stringify(m.levelCounts())} | ${Date.now() - t0}ms (brain ${brainMs.toFixed(0)}ms)`);
}
if (process.argv.includes('--dump')) console.log(m.dumpTables(40));
