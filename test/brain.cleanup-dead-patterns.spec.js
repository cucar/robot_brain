import assert from 'node:assert/strict';
import Brain from '../brain/brain.js';

describe('Brain cleanupDeadPatterns', () => {
	it('reaps the previous frame bucket after getFrame increments frameNumber', () => {
		const calls = [];
		const deletedPatternIds = [9001];
		const fakeBrain = {
			frameNumber: 10,
			debug: false,
			thalamus: {
				reapDeadNeurons(frame) {
					calls.push(['reap', frame]);
					return [8001];
				},
				deletePatterns(patternIds, frame) {
					calls.push(['delete', patternIds, frame]);
					return deletedPatternIds;
				}
			},
			memory: {
				assertNotActive(patternIds) {
					calls.push(['assertNotActive', patternIds]);
				}
			}
		};

		Brain.prototype.cleanupDeadPatterns.call(fakeBrain);

		assert.deepStrictEqual(calls, [
			['reap', 10],
			['delete', [8001], 10],
			['assertNotActive', deletedPatternIds]
		]);
	});
});