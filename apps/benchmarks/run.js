/**
 * The benchmarks: run each benchmark's JS job, strip the timings from what it printed, keep its result lines, and
 * compare them with the result lines of the output saved in docs/benchmarks/output.
 * Prints "same" or "different" per benchmark with its run time, and exits non-zero where any benchmark differs.
 * Only stdout is compared: stderr holds progress and traces, which are no part of the results.
 *
 * usage: node apps/benchmarks/run.js [--only <name>] [--record] [--regions R] [--columns C] [--flag <flag>]...
 *        [--out <folder>] [--data <mnist folder>]
 *   --record    save each run's stripped output as the expected output instead of comparing
 *   --only      run that benchmark alone
 *   --flag      pass that flag to every job, as `--flag --volume`
 *   --out       write each run's output there, instead of apps/benchmarks/got
 *   --data      the folder of the four raw MNIST idx files; apps/mnist/data by default, downloaded there if missing
 */
import fs from 'node:fs';
import path from 'node:path';
import https from 'node:https';
import zlib from 'node:zlib';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { BENCHMARKS } from './list.js';

const REPOSITORY = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const SAVED = path.join(REPOSITORY, 'docs', 'benchmarks', 'output');
const MNIST_FILES = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'];
const MNIST_MIRROR = 'https://storage.googleapis.com/cvdf-datasets/mnist';

/* ---------- The command line ---------- */

/**
 * The runner's settings from the command line.
 */
function readSettings(argv) {
	const settings = { only: null, record: false, extra: [], out: path.join(REPOSITORY, 'apps', 'benchmarks', 'got'), data: path.join(REPOSITORY, 'apps', 'mnist', 'data') };
	for (let i = 0; i < argv.length; i++) {
		const [flag, value] = [argv[i], argv[i + 1]];
		if (flag === '--record') settings.record = true;
		else if (flag === '--only') { settings.only = value; i++; }
		else if (flag === '--regions' || flag === '--columns') { settings.extra.push(flag, value); i++; }
		else if (flag === '--flag') { settings.extra.push(value); i++; }
		else if (flag === '--out') { settings.out = path.resolve(value); i++; }
		else if (flag === '--data') { settings.data = path.resolve(value); i++; }
		else throw new Error(`unknown flag ${flag}`);
	}
	return settings;
}

/* ---------- Getting ready ---------- */

/**
 * Make sure the four raw MNIST files are in the data folder: unzip a .gz beside one that is missing, or download it.
 */
async function prepareMnist(folder) {
	fs.mkdirSync(folder, { recursive: true });
	for (const name of MNIST_FILES) {
		const file = path.join(folder, name);
		if (fs.existsSync(file) && fs.statSync(file).size > 0) continue;
		const zipped = fs.existsSync(`${file}.gz`) ? fs.readFileSync(`${file}.gz`) : await download(`${MNIST_MIRROR}/${name}.gz`);
		fs.writeFileSync(file, zlib.gunzipSync(zipped));
		console.log(`unpacked ${name} into ${folder}`);
	}
}

/**
 * The bytes at a URL.
 */
function download(url) {
	console.log(`downloading ${url}`);
	return new Promise((resolve, reject) => {
		https.get(url, response => {
			if (response.statusCode !== 200) return reject(new Error(`${url}: ${response.statusCode}`));
			const parts = [];
			response.on('data', part => parts.push(part));
			response.on('end', () => resolve(Buffer.concat(parts)));
		}).on('error', reject);
	});
}

/**
 * Build the Node addon and copy the library cargo made to brain-napi.node, where the brain package loads it.
 */
function buildAddon() {
	console.log('building the Node addon');
	const brain = path.join(REPOSITORY, 'brain');
	const build = spawnSync('cargo', ['build', '--release', '-p', 'brain-napi'], { cwd: brain, encoding: 'utf8', shell: process.platform === 'win32' });
	if (build.status !== 0) {
		console.error(build.stderr);
		process.exit(2);
	}
	const library = { win32: 'brain_napi.dll', darwin: 'libbrain_napi.dylib' }[process.platform] ?? 'libbrain_napi.so';
	fs.copyFileSync(path.join(brain, 'target', 'release', library), path.join(brain, 'brain-napi', 'brain-napi.node'));
}

/* ---------- Reading the output ---------- */

/**
 * An output with its timings taken out: any number followed by `ms` or `ms/image`, a number of seconds at the end of
 * a line, and a line that is a number of seconds alone.
 */
function stripTimings(text) {
	return text.replace(/\r/g, '').split('\n')
		.map(line => line.replace(/[0-9]+(\.[0-9]+)? ?ms(\/image)?/g, '').replace(/ ?\|? [0-9]+(\.[0-9]+)? ?s$/, ''))
		.filter(line => !/^ *[0-9]+(\.[0-9]+)? ?s$/.test(line))
		.join('\n');
}

/**
 * The result lines of an output: every line but those the benchmark drops, each with its edit made.
 */
function readResults(text, benchmark) {
	return text.split('\n')
		.filter(line => !benchmark.drop || !benchmark.drop.test(line))
		.map(line => benchmark.edit ? line.replace(benchmark.edit.pattern, benchmark.edit.replacement) : line)
		.join('\n');
}

/* ---------- Running ---------- */

/**
 * The arguments of a benchmark's job, the MNIST folder put in where the list names it.
 */
function makeArguments(benchmark, settings) {
	return benchmark.args.map(arg => arg === '<mnist>' ? settings.data : arg);
}

/**
 * Run one benchmark: its job, its output written to the out folder, its result lines compared or recorded.
 * Returns whether it gave the same results.
 */
function runBenchmark(benchmark, settings) {
	const start = Date.now();
	const run = spawnSync(process.execPath, [path.join(REPOSITORY, benchmark.job), ...makeArguments(benchmark, settings), ...settings.extra], { cwd: REPOSITORY, encoding: 'utf8', maxBuffer: 1 << 30 });
	const seconds = (Date.now() - start) / 1000;
	const out = stripTimings(run.stdout ?? '');
	fs.writeFileSync(path.join(settings.out, `${benchmark.name}.raw.out`), run.stdout ?? '');
	fs.writeFileSync(path.join(settings.out, `${benchmark.name}.raw.err`), run.stderr ?? '');
	fs.writeFileSync(path.join(settings.out, `${benchmark.name}.out`), out);
	if (settings.record) {
		fs.writeFileSync(path.join(SAVED, `${benchmark.name}.out`), out);
		fs.writeFileSync(path.join(SAVED, `${benchmark.name}.err`), stripTimings(run.stderr ?? ''));
	}
	const expected = stripTimings(fs.readFileSync(path.join(SAVED, `${benchmark.name}.out`), 'utf8'));
	const isSame = run.status === 0 && readResults(expected, benchmark) === readResults(out, benchmark);
	const verdict = run.status !== 0 ? `failed(${run.status})` : isSame ? 'same' : 'different';
	console.log(`${benchmark.name.padEnd(26)} ${verdict.padEnd(10)} ${seconds.toFixed(1)}s`);
	return isSame;
}

async function main() {
	const settings = readSettings(process.argv.slice(2));
	const chosen = BENCHMARKS.filter(b => settings.only === null || b.name === settings.only);
	if (chosen.length === 0) throw new Error(`no benchmark named ${settings.only}`);
	if (chosen.some(b => b.args.includes('<mnist>'))) await prepareMnist(settings.data);
	buildAddon();
	fs.mkdirSync(settings.out, { recursive: true });
	console.log(`${'benchmark'.padEnd(26)} ${'output'.padEnd(10)} time`);
	let different = 0;
	for (const benchmark of chosen) if (!runBenchmark(benchmark, settings)) different++;
	console.log(`different: ${different}`);
	process.exit(different === 0 ? 0 : 1);
}

await main();
