//! A small, dependency-free PRNG for `noisegen`'s random draws. Deliberately
//! not a Mersenne Twister/ziggurat pair -- see `noisegen`'s module docs:
//! output matches the reference noise statistics, not an exact bit sequence
//! for a given seed.

use std::f64::consts::PI;

pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform on `(0, 1)`, strictly excluding both endpoints (needed for
    /// `-ln(u)` draws in `stabrnd`).
    pub fn next_f64(&mut self) -> f64 {
        let v = (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        v.clamp(2.0 * f64::EPSILON, 1.0 - 2.0 * f64::EPSILON)
    }

    /// Standard normal, via Box-Muller.
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_draws_stay_in_range() {
        let mut rng = SplitMix64::new(42);
        for _ in 0..10_000 {
            let v = rng.next_f64();
            assert!(v > 0.0 && v < 1.0);
        }
    }

    #[test]
    fn normal_draws_have_roughly_unit_variance() {
        let mut rng = SplitMix64::new(7);
        let n = 50_000;
        let samples: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let var: f64 = samples.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        assert!((var - 1.0).abs() < 0.1, "variance {var} too far from 1.0");
    }
}
