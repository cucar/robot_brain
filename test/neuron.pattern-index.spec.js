import assert from 'node:assert/strict';
import { Context } from '../brain/context.js';
import { Neuron } from '../brain/neuron.js';

const makeContext = entries => {
	const context = new Context();
	for (const [neuronId, distance, strength = 1] of entries) context.addNeuron(neuronId, distance, strength);
	return context;
};

const findBestLinearMatch = (parent, observed, age, currentFrame = 0) => {
	let best = null;
	for (const [patternId, entry] of parent.routingTable) {
		if (parent.getChildEffectiveActivationStrength(patternId, currentFrame) <= 0) continue;
		const match = entry.context.match(observed, age, parent.mergeThreshold);
		if (match && (!best || match.score > best.score)) best = { ...match, patternId, age };
	}
	return best;
};

describe('Neuron pattern index', () => {
	it('finds age-adjusted candidate patterns from the inverted index', () => {
		const parent = new Neuron(0.01, 0.5, new Map(), 100);
		parent.addPattern(201, [{ neuronId: 11, distance: 1 }, { neuronId: 12, distance: 2 }]);
		parent.addPattern(202, [{ neuronId: 11, distance: 2 }, { neuronId: 13, distance: 3 }]);

		const observed = makeContext([[11, 3], [99, 4]]);
		assert.deepStrictEqual([...parent.getPatternCandidatesAtAge(observed, 2)].sort((a, b) => a - b), [201]);
		assert.deepStrictEqual([...parent.getPatternCandidatesAtAge(observed, 1)].sort((a, b) => a - b), [202]);
	});

	it('matches the same best pattern as a full linear scan when pruning candidates', () => {
		const parent = new Neuron(0.01, 0.5, new Map(), 101);
		parent.addPattern(301, [{ neuronId: 7, distance: 1 }, { neuronId: 8, distance: 2 }]);
		parent.addPattern(302, [{ neuronId: 50, distance: 1 }, { neuronId: 51, distance: 2 }]);
		parent.addPattern(303, [{ neuronId: 7, distance: 1 }, { neuronId: 11, distance: 2 }, { neuronId: 12, distance: 3 }]);

		const observed = makeContext([[7, 1], [11, 2], [12, 3], [99, 4]]);
		assert.deepStrictEqual([...parent.getPatternCandidatesAtAge(observed, 0)].sort((a, b) => a - b), [301, 303]);

		const actual = parent.findBestPatternMatchAtAge(observed, 0, null, 0);
		const expected = findBestLinearMatch(parent, observed, 0, 0);
		assert.deepStrictEqual(actual, expected);
	});

	it('preserves routing-table tie-breaking when the index yields a different candidate order', () => {
		const parent = new Neuron(0.01, 0.5, new Map(), 105);
		parent.addPattern(601, [{ neuronId: 11, distance: 1 }]);
		parent.addPattern(602, [{ neuronId: 22, distance: 1 }]);

		const observed = makeContext([[22, 1], [11, 1]]);
		assert.deepStrictEqual([...parent.getPatternCandidatesAtAge(observed, 0)], [602, 601]);

		const actual = parent.findBestPatternMatchAtAge(observed, 0, null, 0);
		const expected = findBestLinearMatch(parent, observed, 0, 0);
		assert.deepStrictEqual(actual, expected);
		assert.equal(actual.patternId, 601);
	});

	it('returns null when the index produces no candidates', () => {
		const parent = new Neuron(0.01, 0.5, new Map(), 102);
		parent.addPattern(311, [{ neuronId: 40, distance: 1 }, { neuronId: 41, distance: 2 }]);
		parent.addPattern(312, [{ neuronId: 42, distance: 1 }, { neuronId: 43, distance: 2 }]);

		const observed = makeContext([[99, 1], [100, 2]]);
		assert.deepStrictEqual([...parent.getPatternCandidatesAtAge(observed, 0)], []);
		assert.equal(parent.findBestPatternMatchAtAge(observed, 0, null, 0), null);
	});

	it('keeps the inverted index in sync when patterns are removed', () => {
		const parent = new Neuron(0.01, 0.5, new Map(), 103);
		parent.addPattern(401, [{ neuronId: 21, distance: 1 }]);
		parent.addPattern(402, [{ neuronId: 21, distance: 1 }, { neuronId: 22, distance: 2 }]);

		const observed = makeContext([[21, 1], [22, 2]]);
		assert.deepStrictEqual([...parent.getPatternCandidatesAtAge(observed, 0)].sort((a, b) => a - b), [401, 402]);

		parent.removeChild(401);
		assert.deepStrictEqual([...parent.getPatternCandidatesAtAge(observed, 0)].sort((a, b) => a - b), [402]);

		parent.removeChild(402);
		assert.deepStrictEqual([...parent.getPatternCandidatesAtAge(observed, 0)], []);
		assert.equal(parent.contextIndex.size, 0);
	});

	it('throws when the index points to a pattern missing from the routing table', () => {
		const parent = new Neuron(0.01, 0.5, new Map(), 104);
		parent.addPattern(501, [{ neuronId: 31, distance: 1 }]);

		const observed = makeContext([[31, 1]]);
		parent.routingTable.delete(501);

		assert.throws(
			() => parent.findBestPatternMatchAtAge(observed, 0, null, 0),
			/Cannot find context for pattern: 501/
		);
	});
});