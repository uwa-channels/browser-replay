//! Polyphase rational resampling, ported to match the reference
//! `resample(x, p, q)` bit-for-bit as closely as f64 arithmetic allows: same
//! Kaiser-windowed-sinc anti-aliasing filter design, same `upfirdn` polyphase
//! evaluation, and the same group-delay compensation formula
//! (`nz`/`delay`/`Ly`).

use num_complex::Complex64;
use num_traits::Zero;
use std::ops::{Add, Mul};

/// Rational approximation `p/q` of `x`, via the nearest-integer continued
/// fraction — matches the reference `rat(x)` (default tolerance `1e-6*|x| + 1e-12`).
pub fn rat(x: f64, tol: Option<f64>) -> (i64, i64) {
    let tol = tol.unwrap_or(1e-6 * x.abs() + 1e-12);

    let mut a = x.round();
    let (mut h1, mut h2) = (1.0_f64, a);
    let (mut k1, mut k2) = (0.0_f64, 1.0_f64);
    let mut r = x - a;

    while r != 0.0 {
        if (x - h2 / k2).abs() <= tol {
            break;
        }
        r = 1.0 / r;
        a = r.round();
        r -= a;
        let h0 = a * h2 + h1;
        h1 = h2;
        h2 = h0;
        let k0 = a * k2 + k1;
        k1 = k2;
        k2 = k0;
    }

    let (mut n, mut d) = (h2 as i64, k2 as i64);
    if d < 0 {
        n = -n;
        d = -d;
    }
    (n, d)
}

/// Normalized sinc: `sin(pi x) / (pi x)`, with `sinc(0) = 1`.
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// Modified Bessel function of the first kind, order 0, via its power series.
/// Matches the reference `besseli0` used by the Kaiser window design.
fn bessel_i0(x: f64) -> f64 {
    let mut y = 1.0_f64;
    let mut term = 1.0_f64;
    let mut k = 0.0_f64;
    while term > 1e-12 * y {
        k += 1.0;
        term *= (x / (2.0 * k)).powi(2);
        y += term;
    }
    y
}

/// Kaiser window of length `l` with shape parameter `beta`, matching the
/// reference `kaiser(l, beta)`.
pub fn kaiser(l: usize, beta: f64) -> Vec<f64> {
    if l <= 1 {
        return vec![1.0; l];
    }
    let alpha = (l as f64 - 1.0) / 2.0;
    let denom = bessel_i0(beta);
    (0..l)
        .map(|n| {
            let r = (n as f64 - alpha) / alpha;
            bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / denom
        })
        .collect()
}

/// Anti-aliasing lowpass filter for resampling by `p/q`: a Kaiser-windowed
/// sinc, cutoff at half the lower of the two rates, unit DC gain scaled by
/// `p` to preserve amplitude through the zero-stuffing upsample step.
fn design_filter(p: u64, q: u64, n: usize, beta: f64) -> Vec<f64> {
    let pqmax = p.max(q) as f64;
    let fc = 1.0 / (2.0 * pqmax);
    let l = 2 * n * (pqmax as usize) + 1;
    let mid = (l as f64 - 1.0) / 2.0;
    let win = kaiser(l, beta);

    let mut h: Vec<f64> = (0..l)
        .map(|i| {
            let t = i as f64 - mid;
            2.0 * fc * sinc(2.0 * fc * t) * win[i]
        })
        .collect();

    let sum: f64 = h.iter().sum();
    let scale = p as f64 / sum;
    for v in h.iter_mut() {
        *v *= scale;
    }
    h
}

/// `yy[n] = w[n*q]` for `n in 0..n_out`, where `w = conv(upsample(x, p), h)`,
/// computed via polyphase gathers instead of materializing the p-fold
/// upsampled signal.
fn upfirdn_poly<T>(x: &[T], h: &[f64], p: u64, q: u64, n_out: usize) -> Vec<T>
where
    T: Copy + Zero + Add<Output = T> + Mul<f64, Output = T>,
{
    let lx = x.len() as i64;
    let lh = h.len();
    let ntap = lh.div_ceil(p as usize);

    let kmaxmax = if n_out == 0 {
        0
    } else {
        ((n_out as i64 - 1) * q as i64) / p as i64
    };
    let pad = ntap as i64;
    let k_trail = (kmaxmax - lx + 1).max(0) + 1;

    let mut xpad = vec![T::zero(); (pad + lx + k_trail) as usize];
    for (i, &v) in x.iter().enumerate() {
        xpad[(pad as usize) + i] = v;
    }

    let mut hpad = vec![0.0_f64; ntap * p as usize];
    hpad[..lh].copy_from_slice(h);

    let mut yy = vec![T::zero(); n_out];
    for (n, slot) in yy.iter_mut().enumerate() {
        let nq = (n as i64) * (q as i64);
        let kmax = nq / (p as i64);
        let r = nq - kmax * (p as i64);
        let mut acc = T::zero();
        for t in 0..ntap {
            let xi = kmax - t as i64 + pad;
            let hi = r as usize + t * p as usize;
            acc = acc + xpad[xi as usize] * hpad[hi];
        }
        *slot = acc;
    }
    yy
}

/// Resample a single (real or complex) sequence by the ratio `p/q` (already
/// reduced to lowest terms by the caller), with filter half-length factor `n`
/// (reference default 10) and Kaiser shape `beta` (reference default 5).
/// Mirrors the reference `resample1` group-delay compensation exactly: the filter length
/// `2*n*max(p,q)+1` is always odd, so `half` is an exact integer throughout.
pub fn resample_with_params<T>(x: &[T], p: u64, q: u64, n: usize, beta: f64) -> Vec<T>
where
    T: Copy + Zero + Add<Output = T> + Mul<f64, Output = T>,
{
    let lx = x.len() as u64;
    let h = design_filter(p, q, n, beta);
    let lh = h.len() as u64;

    let half = (lh - 1) / 2;
    let nz = q - (half % q);
    let mut h_shifted = vec![0.0_f64; nz as usize];
    h_shifted.extend_from_slice(&h);
    let half = half + nz;
    let delay = half / q;
    let ly = (lx * p).div_ceil(q);

    let yy = upfirdn_poly(x, &h_shifted, p, q, (delay + ly) as usize);
    yy[delay as usize..(delay + ly) as usize].to_vec()
}

/// `resample_with_params` with the reference defaults (`N=10`, `beta=5`).
pub fn resample<T>(x: &[T], p: u64, q: u64) -> Vec<T>
where
    T: Copy + Zero + Add<Output = T> + Mul<f64, Output = T>,
{
    resample_with_params(x, p, q, 10, 5.0)
}

/// Resample real-valued `x` from `source_fs` to `target_fs`, finding `p/q` via
/// `rat()` the same way `replay.m`/`replay.py` do before every polyphase
/// resample call.
pub fn resample_to_rate_f64(x: &[f64], source_fs: f64, target_fs: f64) -> Vec<f64> {
    let (p, q) = rat(target_fs / source_fs, None);
    resample(x, p as u64, q as u64)
}

/// Complex-valued counterpart of [`resample_to_rate_f64`], used for the
/// baseband up/downsampling steps in `replay()`.
pub fn resample_to_rate_c64(x: &[Complex64], source_fs: f64, target_fs: f64) -> Vec<Complex64> {
    let (p, q) = rat(target_fs / source_fs, None);
    resample(x, p as u64, q as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rat_reduces_simple_ratios() {
        assert_eq!(rat(0.5, None), (1, 2));
        assert_eq!(rat(2.0, None), (2, 1));
        assert_eq!(rat(48000.0 / 44100.0, None), (160, 147));
    }

    #[test]
    fn kaiser_window_is_symmetric_and_unit_peak() {
        let w = kaiser(11, 5.0);
        assert_eq!(w.len(), 11);
        for i in 0..w.len() {
            assert!((w[i] - w[w.len() - 1 - i]).abs() < 1e-12);
        }
        assert!((w[5] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn resample_identity_when_p_eq_q() {
        let x: Vec<f64> = (0..16).map(|i| (i as f64).sin()).collect();
        let y = resample(&x, 1, 1);
        assert_eq!(y.len(), x.len());
        for (a, b) in x.iter().zip(y.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn resample_upsample_preserves_low_freq_tone() {
        // A pure tone well inside the passband should survive up/downsampling
        // with only a small amplitude/phase perturbation from the FIR filter.
        let fs = 1000.0_f64;
        let f0 = 50.0_f64;
        let n = 2000;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * f0 * i as f64 / fs).sin())
            .collect();
        let y = resample(&x, 3, 2);
        let y_back = resample(&y, 2, 3);
        // Compare in the overlapping middle region to avoid edge transients.
        let m = (n - 400).min(y_back.len().saturating_sub(400));
        let err: f64 = (0..m).map(|i| (x[200 + i] - y_back[200 + i]).abs()).sum::<f64>() / m as f64;
        assert!(err < 0.05, "mean abs error too high: {err}");
    }
}
