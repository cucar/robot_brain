//! D6 offsets: rounding to one significant digit in base R + 1, and the offset alphabets.

use crate::types::{log2, Offset, Pos, MAX_DIMS};

/// Round a signed distance to one significant digit in base `r + 1` (D6).
pub fn round_offset(x: i32, r: i32) -> i32 {
    let a = x.abs();
    if a <= r { return x; }
    let base = (r + 1) as i64;
    let mut p: i64 = 1;
    while p * base <= a as i64 { p *= base; }
    let rounded = (a as i64 / p) * p;
    (rounded as i32) * x.signum()
}

/// How many distinct offsets the distances 1..=reach name (D6): `R·k + floor(reach / (R+1)^k)`.
pub fn offsets_count(reach: i32, r: i32) -> i32 {
    if reach <= 0 { return 0; }
    let base = (r + 1) as i64;
    let mut k = 0;
    let mut p: i64 = 1;
    while p * base <= reach as i64 { p *= base; k += 1; }
    r * k + (reach as i64 / p) as i32
}

/// Bits of one offset field over an owner's reach, summed over its axes (D13): `offsets(r) + 1`
/// buckets in time and `2·offsets(r) + 1` in any other axis.
pub fn offset_bits(reach: &Pos, radii: &[i32], axes: u8) -> f64 {
    let mut bits = 0.0;
    for d in 0..MAX_DIMS.min(radii.len()) {
        if axes & (1 << d) == 0 { continue; }
        let n = offsets_count(reach[d], radii[d]);
        let buckets = if d == 0 { n + 1 } else { 2 * n + 1 };
        bits += log2(buckets as f64);
    }
    bits
}

/// Round a raw coordinate difference per axis (D6). Axes not shared are zeroed.
pub fn round_delta(delta: &Pos, radii: &[i32], axes: u8) -> Offset {
    let mut out = [0; MAX_DIMS];
    for d in 0..MAX_DIMS.min(radii.len()) {
        if axes & (1 << d) == 0 { continue; }
        out[d] = round_offset(delta[d], radii[d]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_as_d6_says() {
        assert_eq!(round_offset(5, 1), 4);
        assert_eq!(round_offset(7, 1), 4);
        assert_eq!(round_offset(-13, 1), -8);
        assert_eq!(round_offset(7, 10), 7);
        assert_eq!(round_offset(15, 10), 11);
        assert_eq!(round_offset(95, 10), 88);
        assert_eq!(round_offset(-250, 10), -242);
        assert_eq!(round_offset(0, 1), 0);
        assert_eq!(round_offset(1, 1), 1);
        assert_eq!(round_offset(2, 1), 2);
        assert_eq!(round_offset(3, 1), 2);
    }

    #[test]
    fn counts_offsets() {
        assert_eq!(offsets_count(38, 1), 6);
        assert_eq!(offsets_count(1, 1), 1);
        assert_eq!(offsets_count(250, 10), 22);
        assert_eq!(offsets_count(4, 4), 4);
    }
}
