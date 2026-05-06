import assert from 'node:assert/strict';
import { Region } from '../src/region.js';

const make = (C) => new Region(C, new Map(), new Set(), new Map());

describe('Region routeNeuron', () => {

	it('routes everything to column 0 when C=1', () => {
		const region = make(1);
		for (const id of [0, 1, 2, 5, 17, 1000])
			assert.equal(region.routeNeuron(id), 0);
	});

	it('distributes ids across columns with id % C', () => {
		const region = make(4);
		assert.equal(region.routeNeuron(0), 0);
		assert.equal(region.routeNeuron(1), 1);
		assert.equal(region.routeNeuron(2), 2);
		assert.equal(region.routeNeuron(3), 3);
		assert.equal(region.routeNeuron(4), 0);
	});

	it('is pure — same id always returns the same column', () => {
		const region = make(5);
		const first = region.routeNeuron(42);
		for (let i = 0; i < 100; i++)
			assert.equal(region.routeNeuron(42), first);
	});

	it('spreads a contiguous id burst across all C columns before repeating', () => {
		const C = 5;
		const region = make(C);
		const seen = new Set();
		for (let id = 0; id < C; id++) seen.add(region.routeNeuron(id));
		assert.equal(seen.size, C, 'a burst of C ids should hit every column exactly once');
	});
});
