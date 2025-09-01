import Channel from './channel.js';

/**
 * Arm Channel - Handles touch/proprioception input and motor output
 * Input: touch, proprioception (joint positions, muscle tension)
 * Output: motor commands (muscle activations)
 * Feedback: movement success rewards
 */
export default class ArmChannel extends Channel {
	
	constructor(name) {
		super(name);
		
		// Arm state tracking
		this.currentPosition = { shoulder: 0.0, elbow: 0.0, wrist: 0.0 }; // Current joint positions
		this.targetPosition = null;
		this.lastMovement = null;
		
		// Sample reaching sequence data from main.js motor control example
		this.reachSequence = [
			{ shoulder: 0.0, elbow: 0.0, wrist: 0.0 }, // Rest position
			{ shoulder: 0.3, elbow: 0.0, wrist: 0.0 }, // Shoulder extension
			{ shoulder: 0.3, elbow: 0.4, wrist: 0.0 }, // Add elbow flexion
			{ shoulder: 0.3, elbow: 0.4, wrist: 0.2 }, // Add wrist extension
			{ shoulder: 0.3, elbow: 0.4, wrist: 0.1 }  // Adjust for grasp
		];
		this.currentSequenceIndex = 0;
		this.sequenceTrials = 0;
		this.maxTrials = 5;
	}

	getInputDimensions() {
		return [
			'touch_shoulder', 'touch_elbow', 'touch_wrist', // Touch/proprioception inputs
			'joint_shoulder', 'joint_elbow', 'joint_wrist'  // Current joint positions
		];
	}

	getOutputDimensions() {
		return [
			'motor_shoulder', 'motor_elbow', 'motor_wrist' // Motor command outputs
		];
	}

	getFeedbackDimensions() {
		return [
			'movement_reward' // +1 for successful reach, -1 for failed movement
		];
	}

	/**
	 * Get proprioceptive and touch input data
	 */
	async getFrameInputs() {
		if (this.sequenceTrials >= this.maxTrials) {
			console.log(`${this.name}: Completed all reaching trials`);
			return [];
		}

		// Start new trial if needed
		if (this.currentSequenceIndex >= this.reachSequence.length) {
			this.currentSequenceIndex = 0;
			this.sequenceTrials++;
			this.currentPosition = { shoulder: 0.0, elbow: 0.0, wrist: 0.0 }; // Reset to rest
			
			if (this.sequenceTrials >= this.maxTrials) {
				return [];
			}
			
			console.log(`${this.name}: Starting reaching trial ${this.sequenceTrials + 1}`);
		}

		// Get target position for this step
		this.targetPosition = this.reachSequence[this.currentSequenceIndex];
		this.currentSequenceIndex++;
		this.frameNumber++;
		
		console.log(`${this.name}: Target position:`, this.targetPosition);
		console.log(`${this.name}: Current position:`, this.currentPosition);
		
		// Calculate touch/proprioception inputs (difference between current and target)
		const shoulderDiff = this.targetPosition.shoulder - this.currentPosition.shoulder;
		const elbowDiff = this.targetPosition.elbow - this.currentPosition.elbow;
		const wristDiff = this.targetPosition.wrist - this.currentPosition.wrist;
		
		// Return proprioceptive input neurons
		return [
			{ touch_shoulder: shoulderDiff }, // Touch feedback showing needed movement
			{ touch_elbow: elbowDiff },
			{ touch_wrist: wristDiff },
			{ joint_shoulder: this.currentPosition.shoulder }, // Current joint position
			{ joint_elbow: this.currentPosition.elbow },
			{ joint_wrist: this.currentPosition.wrist }
		];
	}

	/**
	 * Get feedback based on movement success
	 */
	async getFeedbackNeurons() {
		if (!this.lastMovement || !this.targetPosition) {
			return [];
		}

		// Calculate how close we got to the target
		const shoulderError = Math.abs(this.currentPosition.shoulder - this.targetPosition.shoulder);
		const elbowError = Math.abs(this.currentPosition.elbow - this.targetPosition.elbow);
		const wristError = Math.abs(this.currentPosition.wrist - this.targetPosition.wrist);
		
		const totalError = shoulderError + elbowError + wristError;
		const threshold = 0.05; // Acceptable error threshold

		let feedbackValue = 0;

		if (totalError < threshold) {
			feedbackValue = 1; // Reward for successful movement
			console.log(`${this.name}: REWARD! Successful reach (error: ${totalError.toFixed(3)})`);
		} else {
			feedbackValue = -1; // Penalty for inaccurate movement
			console.log(`${this.name}: PENALTY! Missed target (error: ${totalError.toFixed(3)})`);
		}

		// Extra reward for completing the full sequence
		if (this.currentSequenceIndex === this.reachSequence.length) {
			feedbackValue = Math.max(feedbackValue, 1); // Ensure positive reward for completion
			console.log(`${this.name}: SEQUENCE COMPLETED! Extra reward.`);
		}

		return [{ movement_reward: feedbackValue }];
	}

	/**
	 * Execute motor commands based on brain output
	 */
	async executeOutputs(predictions, frameNumber) {
		const outputs = {
			actions: new Map(),
			predictions: new Map()
		};

		if (!predictions || predictions.length === 0) {
			return outputs;
		}

		// Extract motor command predictions
		let motorShoulder = 0, motorElbow = 0, motorWrist = 0;
		let totalConfidence = 0;

		predictions.forEach(frame => {
			frame.predictions.forEach(pred => {
				if (pred.coordinates.motor_shoulder !== undefined) {
					motorShoulder += pred.coordinates.motor_shoulder * pred.confidence;
					totalConfidence += pred.confidence;
				}
				if (pred.coordinates.motor_elbow !== undefined) {
					motorElbow += pred.coordinates.motor_elbow * pred.confidence;
					totalConfidence += pred.confidence;
				}
				if (pred.coordinates.motor_wrist !== undefined) {
					motorWrist += pred.coordinates.motor_wrist * pred.confidence;
					totalConfidence += pred.confidence;
				}
			});
		});

		if (totalConfidence > 0) {
			motorShoulder /= totalConfidence;
			motorElbow /= totalConfidence;
			motorWrist /= totalConfidence;
			
			// Execute the motor commands (move arm joints)
			const movementScale = 0.1; // Scale movement speed
			this.currentPosition.shoulder += motorShoulder * movementScale;
			this.currentPosition.elbow += motorElbow * movementScale;
			this.currentPosition.wrist += motorWrist * movementScale;
			
			// Keep within joint limits
			this.currentPosition.shoulder = Math.max(0, Math.min(1, this.currentPosition.shoulder));
			this.currentPosition.elbow = Math.max(0, Math.min(1, this.currentPosition.elbow));
			this.currentPosition.wrist = Math.max(0, Math.min(1, this.currentPosition.wrist));
			
			this.lastMovement = { shoulder: motorShoulder, elbow: motorElbow, wrist: motorWrist };
			this.lastOutputFrame = frameNumber;
			
			console.log(`${this.name}: EXECUTED MOVEMENT - Shoulder: ${motorShoulder.toFixed(3)}, Elbow: ${motorElbow.toFixed(3)}, Wrist: ${motorWrist.toFixed(3)}`);
			console.log(`${this.name}: New position:`, {
				shoulder: this.currentPosition.shoulder.toFixed(3),
				elbow: this.currentPosition.elbow.toFixed(3),
				wrist: this.currentPosition.wrist.toFixed(3)
			});
			
			outputs.actions.set('movement', { 
				shoulder: motorShoulder, 
				elbow: motorElbow, 
				wrist: motorWrist, 
				confidence: totalConfidence 
			});
		}

		return outputs;
	}
}
