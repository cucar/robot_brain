/**
 * Number formatting that prints what the Rust examples print, so a job's output compares line by line with theirs.
 */

/**
 * A number with a fixed count of decimals, rounded as Rust's `{:.N}` rounds it.
 * Both round to the nearest; they differ only where a number lies exactly halfway, as 0.25 at one decimal.
 * There Rust keeps the even last digit ("0.2") where `toFixed` rounds away from zero ("0.3").
 * A number lies exactly halfway at `decimals` decimals when it is an odd multiple of 2^-(decimals+1).
 */
export function formatFixed(value, decimals) {
	const halves = value * 2 ** (decimals + 1);
	const isTie = Number.isInteger(halves) && halves % 2 !== 0;
	if (!isTie) return value.toFixed(decimals);
	const scaled = Math.abs(value) * 10 ** decimals;
	const below = Math.floor(scaled);
	const even = below % 2 === 0 ? below : below + 1;
	const sign = value < 0 && even !== 0 ? '-' : '';
	return sign + (even / 10 ** decimals).toFixed(decimals);
}

/**
 * A number as Rust's `{}` prints it: the shortest digits that read back as the same number, never with an exponent.
 * Both pick the same digits; JS writes very small and very large numbers with an exponent (1e-7), Rust writes them out.
 */
export function formatFloat(value) {
	if (Number.isNaN(value)) return 'NaN';
	if (!Number.isFinite(value)) return value > 0 ? 'inf' : '-inf';
	if (Object.is(value, -0)) return '-0';
	const text = String(value);
	const exponentAt = text.indexOf('e');
	if (exponentAt === -1) return text;
	return expandExponent(text.slice(0, exponentAt), parseInt(text.slice(exponentAt + 1)));
}

/**
 * Write out `mantissa` × 10^`exponent` in plain digits, as 1.5 and -7 give 0.00000015.
 */
function expandExponent(mantissa, exponent) {
	const sign = mantissa.startsWith('-') ? '-' : '';
	const [whole, fraction = ''] = mantissa.replace('-', '').split('.');
	const digits = whole + fraction;
	const point = whole.length + exponent;
	if (point <= 0) return `${sign}0.${'0'.repeat(-point)}${digits}`;
	if (point >= digits.length) return `${sign}${digits}${'0'.repeat(point - digits.length)}`;
	return `${sign}${digits.slice(0, point)}.${digits.slice(point)}`;
}
