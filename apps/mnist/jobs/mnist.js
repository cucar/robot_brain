/**
 * MNIST (docs/algorithm-mnist.md): an image channel over x, y and time (radius 1 each) with
 * one event dimension (ink; a pixel with no ink reports nothing), and a digit channel over time alone
 * with one action dimension of ten digits. Two frames per image: the image, then the digit action
 * that ran with its reward, scoped to the digit channel and that frame (R29).
 *
 * The world names the digit itself for the first `--teach` images (a lesson; every training image
 * unless told otherwise); after that the machine's output runs and is rewarded +1 / -1. Then `--test`
 * held-out images are shown with the machine frozen: it answers, its answer runs with no reward and
 * no lesson, and nothing of the machine changes.
 *
 * Training takes every training image in the file's order, or with `--per-class` at most that many of each digit.
 *
 *   node apps/mnist/jobs/mnist.js [--image-size 14] [--per-class all] [--teach all] [--test 500] [--h 2048] [--episodes 1]
 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Brain } from '../../../brain/src/index.js';
import { loadImages, loadLabels } from '../loader.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
function arg(flag, def, parse = x => x) {
	const i = process.argv.indexOf(flag);
	return i !== -1 && process.argv[i + 1] !== undefined ? parse(process.argv[i + 1]) : def;
}
const imageSize = arg('--image-size', 14, Number);
const perClass = arg('--per-class', Infinity, Number);
const teachArg = arg('--teach', Infinity, Number);
const testCount = arg('--test', 500, Number);
const h = arg('--h', 2048, Number);
const episodes = arg('--episodes', 1, Number);
const threshold = arg('--threshold', 128, Number);
const silent = process.argv.includes('--silent-frame');

const dataDir = path.join(__dirname, '..', 'data');
const trainImages = loadImages(path.join(dataDir, 'train-images-idx3-ubyte.gz'));
const trainLabels = loadLabels(path.join(dataDir, 'train-labels-idx1-ubyte.gz'));
const testImages = loadImages(path.join(dataDir, 't10k-images-idx3-ubyte.gz'));
const testLabels = loadLabels(path.join(dataDir, 't10k-labels-idx1-ubyte.gz'));

// the training images in the file's order: every one, or at most `--per-class` of each digit
const picked = new Array(10).fill(0);
const train = [];
for (let i = 0; i < trainImages.length; i++) {
	const d = trainLabels[i];
	if (picked[d] < perClass) { picked[d]++; train.push(i); }
}
const teach = Math.min(teachArg, train.length);

const f = 28 / imageSize;
function inked(px) {
	const out = [];
	for (let y = 0; y < imageSize; y++) for (let x = 0; x < imageSize; x++) {
		let sum = 0;
		for (let dy = 0; dy < f; dy++) for (let dx = 0; dx < f; dx++) sum += px[(y * f + dy) * 28 + x * f + dx];
		if (sum / (f * f) >= threshold) out.push([x, y]);
	}
	return out;
}

const m = new Brain({ h, timeRadius: 1, policy: 'eventsToActions', actionPolicy: 'leastBad' });
const X = m.declareAxis('x', 1), Y = m.declareAxis('y', 1);
const image = m.declareChannel({ name: 'image', event: { name: 'ink', resolution: 2 }, axes: [X, Y] });
const digit = m.declareChannel({ name: 'digit', action: { name: 'digit', resolution: 10 } });

console.log(`UCAR mnist: ${imageSize}x${imageSize}, train ${train.length} (${perClass === Infinity ? 'all' : perClass + '/class'}, first ${teach} taught), test ${testCount}, H=${h}`);

function showImage(px, label, teachIt, S, testing = false) {
	const pts = inked(px);
	const inputs = new Int32Array(pts.length * 4);
	for (let i = 0; i < pts.length; i++) { inputs[i * 4] = image.eventDim; inputs[i * 4 + 1] = 1; inputs[i * 4 + 2] = pts[i][0]; inputs[i * 4 + 3] = pts[i][1]; }
	const r = m.processFrame(inputs, []);
	S.ms += r.stats.elapsedMs;
	S.levels = Math.max(S.levels, r.stats.levels);
	S.apex += r.stats.apexActs;
	const w = r.winners.find(w => w.dim === digit.actionDim);
	const said = w ? w.bucket : -1;
	if (said >= 0) { S.spoke++; if (said === label) S.correct++; }
	S.n++;
	if (testing) {
		// frozen: the machine's own answer runs, unrewarded, and nothing is taught
		const r2 = m.processFrame(new Int32Array(said >= 0 ? [digit.actionDim, said, 0, 0] : []), []);
		S.ms += r2.stats.elapsedMs;
		if (silent) m.processFrame(new Int32Array(0), []);
		return said;
	}
	const ran = teachIt || said < 0 ? label : said;
	if (teachIt || said < 0) S.lessons++;
	const reward = ran === label ? 1 : -1;
	const r2 = m.processFrame(new Int32Array([digit.actionDim, ran, 0, 0]), [{ reward, channels: [digit.channelId], frames: 1 }]);
	S.ms += r2.stats.elapsedMs;
	if (silent) m.processFrame(new Int32Array(0), []);
	return said;
}
function newSeg() { return { n: 0, spoke: 0, correct: 0, lessons: 0, ms: 0, levels: 0, apex: 0 }; }
function report(name, S, t0) {
	const s = m.stats();
	console.log(`  ${name}: ${S.n} images | spoke ${S.spoke} | correct ${S.correct}/${S.n} (${(100 * S.correct / Math.max(1, S.n)).toFixed(1)}%; of spoken ${(100 * S.correct / Math.max(1, S.spoke)).toFixed(1)}%) | lessons ${S.lessons} | apex/image ${(S.apex / Math.max(1, S.n)).toFixed(1)} max levels ${S.levels} | neurons ${s.neurons} patterns ${s.patterns} conns ${s.connections} levels ${JSON.stringify(m.heightCounts())} | ${Date.now() - t0}ms (brain ${S.ms.toFixed(0)}ms)`);
}

for (let ep = 1; ep <= episodes; ep++) {
	const t0 = Date.now();
	const S = newSeg();
	for (let i = 0; i < train.length; i++) {
		const idx = train[i];
		showImage(trainImages[idx], trainLabels[idx], i < teach, S);
		if ((i + 1) % 100 === 0) process.stdout.write(`\r  image ${i + 1}/${train.length} spoke=${S.spoke} correct=${S.correct} (${(100 * S.correct / Math.max(1, S.spoke)).toFixed(1)}% of spoken) neurons=${m.stats().neurons} ${(S.ms / (i + 1)).toFixed(1)}ms/image   `);
	}
	process.stdout.write('\n');
	report(`train episode ${ep}`, S, t0);
}
{
	const t0 = Date.now();
	const S = newSeg();
	m.setFrozen(true);
	for (let i = 0; i < Math.min(testCount, testImages.length); i++) {
		showImage(testImages[i], testLabels[i], false, S, true);
		if ((i + 1) % 100 === 0) process.stdout.write(`\r  test ${i + 1}/${testCount} correct=${S.correct} (${(100 * S.correct / (i + 1)).toFixed(1)}%)   `);
	}
	process.stdout.write('\n');
	report('test', S, t0);
}
if (process.argv.includes('--dump')) console.log(m.dumpTables(30));
