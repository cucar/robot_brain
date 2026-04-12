import assert from 'node:assert/strict';
import Brain from '../brain/brain.js';

describe('Brain cleanupDeadPatterns', () => {
	it('reaps the previous frame bucket after getFrame increments frameNumber', () => {
		const calls = [];
		const deletedPatterns = [{ id: 9001 }];
		const fakeBrain = {
			frameNumber: 10,
			debug: false,
			thalamus: {
				reapDeadNeurons(frame) {
					calls.push(['reap', frame]);
					return ['dead-pattern'];
				},
				deletePatterns(patterns, frame) {
					calls.push(['delete', patterns, frame]);
					return deletedPatterns;
				}
			},
			memory: {
				assertNotActive(patterns) {
					calls.push(['assertNotActive', patterns]);
				}
			}
		};

		Brain.prototype.cleanupDeadPatterns.call(fakeBrain);

		assert.deepStrictEqual(calls, [
			['reap', 9],
			['delete', ['dead-pattern'], 9],
			['assertNotActive', deletedPatterns]
		]);
	});
});