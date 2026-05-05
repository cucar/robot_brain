import assert from 'node:assert/strict';
import { Thalamus } from '../src/thalamus.js';

const make = (regions, columns) => new Thalamus(false, 0.01, 0.5, 10, 'static', 0.5, regions, columns);

describe('Thalamus routeNeuron', () => {

	it('routes everything to (0, 0) at R=1, C=1 (default)', () => {
		const thalamus = make(1, 1);
		for (const id of [1, 2, 5, 17, 1000]) {
			assert.deepStrictEqual(thalamus.routeNeuron(id), { regionIdx: 0, columnIdx: 0 });
		}
	});

	it('interleaves region assignment by id mod R when R>1', () => {
		const thalamus = make(4, 1);
		// regionIdx = id % 4; columnIdx = floor(id / 4) % 1 = 0
		assert.deepStrictEqual(thalamus.routeNeuron(0), { regionIdx: 0, columnIdx: 0 });
		assert.deepStrictEqual(thalamus.routeNeuron(1), { regionIdx: 1, columnIdx: 0 });
		assert.deepStrictEqual(thalamus.routeNeuron(2), { regionIdx: 2, columnIdx: 0 });
		assert.deepStrictEqual(thalamus.routeNeuron(3), { regionIdx: 3, columnIdx: 0 });
		assert.deepStrictEqual(thalamus.routeNeuron(4), { regionIdx: 0, columnIdx: 0 });
		assert.deepStrictEqual(thalamus.routeNeuron(5), { regionIdx: 1, columnIdx: 0 });
	});

	it('interleaves column assignment by floor(id / R) mod C when C>1', () => {
		const thalamus = make(1, 4);
		// regionIdx = id % 1 = 0; columnIdx = floor(id / 1) % 4 = id % 4
		assert.deepStrictEqual(thalamus.routeNeuron(0), { regionIdx: 0, columnIdx: 0 });
		assert.deepStrictEqual(thalamus.routeNeuron(1), { regionIdx: 0, columnIdx: 1 });
		assert.deepStrictEqual(thalamus.routeNeuron(2), { regionIdx: 0, columnIdx: 2 });
		assert.deepStrictEqual(thalamus.routeNeuron(3), { regionIdx: 0, columnIdx: 3 });
		assert.deepStrictEqual(thalamus.routeNeuron(4), { regionIdx: 0, columnIdx: 0 });
	});

	it('combines region and column interleaving when both R>1 and C>1', () => {
		const thalamus = make(2, 3);
		// regionIdx = id % 2; columnIdx = floor(id / 2) % 3
		assert.deepStrictEqual(thalamus.routeNeuron(0), { regionIdx: 0, columnIdx: 0 }); // id=0: r=0, c=floor(0/2)%3=0
		assert.deepStrictEqual(thalamus.routeNeuron(1), { regionIdx: 1, columnIdx: 0 }); // id=1: r=1, c=floor(1/2)%3=0
		assert.deepStrictEqual(thalamus.routeNeuron(2), { regionIdx: 0, columnIdx: 1 }); // id=2: r=0, c=floor(2/2)%3=1
		assert.deepStrictEqual(thalamus.routeNeuron(3), { regionIdx: 1, columnIdx: 1 }); // id=3: r=1, c=floor(3/2)%3=1
		assert.deepStrictEqual(thalamus.routeNeuron(4), { regionIdx: 0, columnIdx: 2 }); // id=4: r=0, c=floor(4/2)%3=2
		assert.deepStrictEqual(thalamus.routeNeuron(5), { regionIdx: 1, columnIdx: 2 }); // id=5: r=1, c=floor(5/2)%3=2
		assert.deepStrictEqual(thalamus.routeNeuron(6), { regionIdx: 0, columnIdx: 0 }); // id=6: wraps back
	});

	it('is pure — same id always returns the same route', () => {
		const thalamus = make(3, 5);
		const first = thalamus.routeNeuron(42);
		for (let i = 0; i < 100; i++) {
			assert.deepStrictEqual(thalamus.routeNeuron(42), first);
		}
	});

	it('spreads a contiguous id burst across all (R*C) buckets before repeating', () => {
		const R = 3, C = 4;
		const thalamus = make(R, C);
		const seen = new Set();
		for (let id = 0; id < R * C; id++) {
			const { regionIdx, columnIdx } = thalamus.routeNeuron(id);
			seen.add(`${regionIdx},${columnIdx}`);
		}
		assert.equal(seen.size, R * C, 'a burst of R*C ids should hit every bucket exactly once');
	});
});
