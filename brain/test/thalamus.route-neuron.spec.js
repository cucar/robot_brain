import assert from 'node:assert/strict';
import { Thalamus } from '../src/thalamus.js';

const make = (regions, columns) => new Thalamus(false, 0.01, 0.5, 10, 'static', 0.5, regions, columns);

describe('Thalamus routeNeuron', () => {

	it('routes everything to region 0 when R=1', () => {
		const thalamus = make(1, 1);
		for (const id of [0, 1, 2, 5, 17, 1000])
			assert.equal(thalamus.routeNeuron(id), 0);
	});

	it('interleaves region assignment by id mod R when R>1', () => {
		const thalamus = make(4, 1);
		assert.equal(thalamus.routeNeuron(0), 0);
		assert.equal(thalamus.routeNeuron(1), 1);
		assert.equal(thalamus.routeNeuron(2), 2);
		assert.equal(thalamus.routeNeuron(3), 3);
		assert.equal(thalamus.routeNeuron(4), 0);
		assert.equal(thalamus.routeNeuron(5), 1);
	});

	it('is independent of C — column count does not affect region routing', () => {
		const t1 = make(3, 1);
		const t2 = make(3, 17);
		for (let id = 0; id < 30; id++)
			assert.equal(t1.routeNeuron(id), t2.routeNeuron(id));
	});

	it('is pure — same id always returns the same region', () => {
		const thalamus = make(3, 5);
		const first = thalamus.routeNeuron(42);
		for (let i = 0; i < 100; i++)
			assert.equal(thalamus.routeNeuron(42), first);
	});

	it('spreads a contiguous id burst across all R regions before repeating', () => {
		const R = 5;
		const thalamus = make(R, 1);
		const seen = new Set();
		for (let id = 0; id < R; id++) seen.add(thalamus.routeNeuron(id));
		assert.equal(seen.size, R, 'a burst of R ids should hit every region exactly once');
	});
});
