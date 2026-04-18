import assert from 'node:assert/strict';
import { Thalamus } from '../brain/thalamus.js';

describe('Thalamus reapDeadNeurons', () => {
	it('reaps only neurons scheduled for the current frame', () => {
		const thalamus = new Thalamus(false, 0.01, 0.5);
		const dueNow = thalamus.addSensoryNeuron(1001);
		const dueLater = thalamus.addSensoryNeuron(1002);

		thalamus.registerDeath(dueNow.id, 10);
		thalamus.registerDeath(dueLater.id, 11);

		assert.deepStrictEqual(thalamus.reapDeadNeurons(10), [dueNow.id]);
		assert.equal(thalamus.neuronDeathFrame.has(dueNow.id), false);
		assert.equal(thalamus.neuronDeathFrame.get(dueLater.id), 11);
		assert.deepStrictEqual(thalamus.reapDeadNeurons(10), []);
		assert.deepStrictEqual(thalamus.reapDeadNeurons(11), [dueLater.id]);
	});
});