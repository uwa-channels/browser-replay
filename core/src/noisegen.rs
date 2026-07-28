//! Underwater acoustic noise generation, ported from the reference
//! `noisegen` implementation: either "textbook" pink noise (a fixed
//! -17 dB/decade Kaiser-free FIR filter applied to white Gaussian noise) or
//! a measured spatial "mixing" model (a `beta[M,M,K]` tensor applied to a
//! symmetric alpha-stable driver, per Chambers-Mallows-Stuck / McCulloch's
//! `stabrnd`).
//!
//! The deterministic pieces (filter design, FIR filtering, the mixing
//! contraction, the CMS stable-variate formula given uniform/exponential
//! draws) are ported to match the reference exactly. The draws themselves
//! come from a caller-supplied RNG rather than the reference's Mersenne
//! Twister: noise output matches the reference's *statistics* (spectral
//! slope, spatial correlation, tail weight), not its bit-for-bit sequence
//! for a given seed (see README).

use crate::fft::{fft, fftshift, ifft};
use crate::resample;
use num_complex::Complex64;
use std::f64::consts::PI;

/// The pink-noise coloring filter's impulse response, matching `noise_pink`'s
/// `h` in `noisegen.m` exactly (a deterministic, closed-form design — no
/// randomness involved). `nfft = 4096` and the hardcoded `fmin=0, fmax=fs/2`
/// band are baked in, matching `noisegen.m`, which does not expose them.
pub fn pink_filter(fs: f64) -> Vec<f64> {
    let nfft = 4096usize;
    let a = fs / 2.0 / nfft as f64;
    let b = fs / 2.0;
    let f: Vec<f64> = (0..nfft).map(|k| a + k as f64 * (b - a) / (nfft as f64 - 1.0)).collect();

    let mut h_oneside: Vec<f64> = f
        .iter()
        .map(|&fi| {
            let h_db = -17.0 * (fi / 1e3).log10();
            10f64.powf(h_db / 10.0)
        })
        .collect();
    // fmax = fs/2 exactly hits bin `nfft` (1-based), so noisegen.m's
    // `H_oneside(ceil(fmax/binwidth):end) = 0` zeroes just the last bin.
    *h_oneside.last_mut().unwrap() = 0.0;

    let mut h_two_sided: Vec<f64> = h_oneside.clone();
    h_two_sided.extend(h_oneside[1..].iter().rev());

    let spec: Vec<Complex64> = h_two_sided.iter().map(|&v| Complex64::new(v.sqrt(), 0.0)).collect();
    let time = ifft(&spec);
    let shifted = fftshift(&time);
    let mut h: Vec<f64> = shifted.iter().map(|c| c.re).collect();

    let energy: f64 = h.iter().map(|v| v * v).sum();
    let scale = 1.0 / energy.sqrt();
    for v in h.iter_mut() {
        *v *= scale;
    }
    h
}

/// Causal FIR filtering (`y[n] = sum_k h[k]*x[n-k]`, zero initial state),
/// truncated to `x`'s length — the same operation `fftfilt(h, x)` performs,
/// just computed as one FFT-based linear convolution rather than
/// block-overlap-add (identical result, since both implement the same
/// causal convolution).
fn causal_convolve(h: &[f64], x: &[f64]) -> Vec<f64> {
    let n = h.len() + x.len() - 1;
    let mut hc: Vec<Complex64> = h.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    hc.resize(n, Complex64::new(0.0, 0.0));
    let mut xc: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    xc.resize(n, Complex64::new(0.0, 0.0));

    let hf = fft(&hc);
    let xf = fft(&xc);
    let yf: Vec<Complex64> = hf.iter().zip(xf.iter()).map(|(a, b)| a * b).collect();
    let y = ifft(&yf);
    y[..x.len()].iter().map(|c| c.re).collect()
}

/// Applies the pink-noise filter to already-generated white noise `w` (one
/// `Vec` per channel), matching `noise_pink`'s centered/"same"-style
/// filtering (pad by the filter's half-length, run the causal filter, then
/// crop out the centered window to compensate the FIR's group delay).
pub fn apply_pink_filter(w: &[Vec<f64>], fs: f64) -> Vec<Vec<f64>> {
    let h = pink_filter(fs);
    let offset = h.len() / 2;
    w.iter()
        .map(|col| {
            let mut padded = col.clone();
            padded.extend(std::iter::repeat_n(0.0, offset));
            let filtered = causal_convolve(&h, &padded);
            filtered[offset..offset + col.len()].to_vec()
        })
        .collect()
}

/// Generates independent pink Gaussian noise across `cols` channels of
/// `rows` samples each, using `randn` for the underlying white noise.
pub fn noise_pink(rows: usize, cols: usize, fs: f64, mut randn: impl FnMut() -> f64) -> Vec<Vec<f64>> {
    let w: Vec<Vec<f64>> = (0..cols).map(|_| (0..rows).map(|_| randn()).collect()).collect();
    apply_pink_filter(&w, fs)
}

/// Symmetric/general Chambers-Mallows-Stuck stable-variate transform, given
/// an exponential draw `w = -ln(u1)` and a uniform-on-`(-pi/2,pi/2)` draw
/// `phi`, matching McCulloch's `stabrnd` core formula (the part downstream of
/// its two `rand()` calls) exactly.
fn stabrnd_core(alpha: f64, beta: f64, c: f64, delta: f64, w: f64, phi: f64) -> f64 {
    if alpha == 2.0 {
        let x = 2.0 * w.sqrt() * phi.sin();
        return delta + c * x;
    }

    let x = if beta == 0.0 {
        if alpha == 1.0 {
            phi.tan()
        } else {
            (((1.0 - alpha) * phi).cos() / w).powf(1.0 / alpha - 1.0) * (alpha * phi).sin() / phi.cos().powf(1.0 / alpha)
        }
    } else {
        let cosphi = phi.cos();
        if (alpha - 1.0).abs() > 1e-8 {
            let zeta = beta * (PI * alpha / 2.0).tan();
            let aphi = alpha * phi;
            let a1phi = (1.0 - alpha) * phi;
            ((aphi.sin() + zeta * aphi.cos()) / cosphi)
                * (((a1phi.cos() + zeta * a1phi.sin()) / (w * cosphi)).powf((1.0 - alpha) / alpha))
        } else {
            let bphi = PI / 2.0 + beta * phi;
            let mut xx = (2.0 / PI) * (bphi * phi.tan() - beta * ((PI / 2.0) * w * cosphi / bphi).ln());
            if alpha != 1.0 {
                xx += beta * (PI * alpha / 2.0).tan();
            }
            xx
        }
    };
    delta + c * x
}

/// `m*n` iid stable random numbers (flat, column-major: `data[i + m*j]`),
/// matching `stabrnd(alpha, beta, c, delta, m, n)`'s output layout. Draws
/// come from the caller-supplied `rand01` (uniform on `(0,1)`), not
/// the reference's RNG — see the module docs.
pub fn stabrnd(alpha: f64, beta: f64, c: f64, delta: f64, m: usize, n: usize, mut rand01: impl FnMut() -> f64) -> Vec<f64> {
    (0..m * n)
        .map(|_| {
            let w = -rand01().ln();
            let phi = (rand01() - 0.5) * PI;
            stabrnd_core(alpha, beta, c, delta, w, phi)
        })
        .collect()
}

/// The `beta[M,M,K]` spatial/spectral mixing tensor, flat and column-major
/// (`data[i + M*(j + M*k)]`, matching the reference `beta(:)`).
pub struct Beta3 {
    pub m: usize,
    pub k: usize,
    pub data: Vec<f64>,
}

impl Beta3 {
    pub fn new(m: usize, k: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), m * m * k, "Beta3 data length must be m*m*k");
        Beta3 { m, k, data }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize, k: usize) -> f64 {
        self.data[i + self.m * (j + self.m * k)]
    }
}

/// `w(n, i) = sum_j sum_k beta(i, j, k) * z(n+k, j)`, matching `noise_mixing`'s
/// documented contraction (implemented there as a `conv2`/`rot90`, but that's
/// just an efficient way to compute this same double sum).
fn mix(beta: &Beta3, z: &[f64], z_len: usize, out_len: usize, recv_idx: usize) -> Vec<f64> {
    let m = beta.m;
    let k_mix = beta.k;
    (0..out_len)
        .map(|n| {
            let mut acc = 0.0;
            for j in 0..m {
                for k in 0..k_mix {
                    acc += beta.get(recv_idx, j, k) * z[(n + k) + z_len * j];
                }
            }
            acc
        })
        .collect()
}

/// Noise struct fields required by the mixing model (`noise.alpha`,
/// `noise.Fs`, `noise.beta`).
pub struct NoiseStruct {
    pub alpha: f64,
    pub measurement_fs: f64,
    pub beta: Beta3,
}

/// Generates mixing-coefficient noise for the receivers in `array_index`
/// (0-based), at output rate `fs`, `input_rows` samples per channel.
pub fn noise_mixing(
    input_rows: usize,
    array_index: &[usize],
    fs: f64,
    noise: &NoiseStruct,
    rand01: impl FnMut() -> f64,
) -> Vec<Vec<f64>> {
    let (p, q) = resample::rat(fs / noise.measurement_fs, None);
    let k_total = ((input_rows as i64 * q + p - 1) / p) as usize; // ceil(input_rows*q/p)
    let m_full = noise.beta.m;
    let k_mix = noise.beta.k;
    let z_len = k_total + k_mix;

    let z = stabrnd(noise.alpha, 0.0, 1.0 / 2f64.sqrt(), 0.0, z_len, m_full, rand01);

    array_index
        .iter()
        .map(|&i| {
            let w = mix(&noise.beta, &z, z_len, k_total, i);
            let mut resampled = resample::resample(&w, p as u64, q as u64);
            resampled.truncate(input_rows);
            resampled
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pink_filter_has_unit_energy() {
        let h = pink_filter(48000.0);
        let energy: f64 = h.iter().map(|v| v * v).sum();
        assert!((energy - 1.0).abs() < 1e-9);
    }

    #[test]
    fn causal_convolve_matches_naive_convolution() {
        let h = vec![1.0, 0.5, -0.25];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let got = causal_convolve(&h, &x);
        let want: Vec<f64> = (0..x.len())
            .map(|n| {
                (0..h.len())
                    .filter(|&k| n >= k)
                    .map(|k| h[k] * x[n - k])
                    .sum()
            })
            .collect();
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-12);
        }
    }

    #[test]
    fn stabrnd_alpha2_is_box_muller_gaussian() {
        // alpha=2 must exactly match the direct Box-Muller formula, since
        // stabrnd's Gaussian branch is Box-Muller by construction.
        let w = 0.7_f64;
        let phi = 0.3_f64;
        let got = stabrnd_core(2.0, 0.0, 1.0, 0.0, w, phi);
        let want = 2.0 * w.sqrt() * phi.sin();
        assert!((got - want).abs() < 1e-12);
    }

    #[test]
    fn mix_matches_direct_double_sum() {
        let beta = Beta3::new(2, 3, vec![
            // i + 2*(j + 2*k), m=2,k=3 -> 12 entries
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
        ]);
        let z_len = 5;
        let z: Vec<f64> = (0..z_len * beta.m).map(|i| i as f64 * 0.1).collect();
        let out_len = z_len - beta.k + 1;
        let got = mix(&beta, &z, z_len, out_len, 1);
        for (n, &g) in got.iter().enumerate() {
            let mut want = 0.0;
            for j in 0..beta.m {
                for k in 0..beta.k {
                    want += beta.get(1, j, k) * z[(n + k) + z_len * j];
                }
            }
            assert!((g - want).abs() < 1e-12);
        }
    }
}
