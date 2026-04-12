import assert from 'node:assert/strict';
import { Thalamus } from '../brain/thalamus.js';
import { Neuron } from '../brain/neuron.js';

describe('Thalamus reapDeadNeurons', () => {
	it('reaps only neurons scheduled for the current frame', () => {
		const thalamus = new Thalamus(false, 0.01, 0.5);
		const dueNow = new Neuron(0.01, 0.5, 1001);
		const dueLater = new Neuron(0.01, 0.5, 1002);

		thalamus.addNeuron(dueNow, 1, undefined, 999);
		thalamus.addNeuron(dueLater, 1, undefined, 999);
		thalamus.registerDeath(dueNow, 10);
		thalamus.registerDeath(dueLater, 11);

		assert.deepStrictEqual(thalamus.reapDeadNeurons(10), [dueNow]);
		assert.equal(thalamus.neuronDeathFrame.has(dueNow), false);
		assert.equal(thalamus.neuronDeathFrame.get(dueLater), 11);
		assert.deepStrictEqual(thalamus.reapDeadNeurons(10), []);
		assert.deepStrictEqual(thalamus.reapDeadNeurons(11), [dueLater]);
	});
});