/**
 * Digit action constants shared by every MNIST encoder. 10 buckets, one
 * per digit 0–9, registered as the action dimension on whichever channel
 * the encoder owns the digit action on.
 */
export const DIGITS = 10;
export const DIGIT_ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
