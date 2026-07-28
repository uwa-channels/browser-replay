//! Thin wrapper around `rustfft`, plus a couple of helpers (`ifft` with
//! the conventional `1/N` normalization, and centered `fftshift`) used by
//! `noisegen`'s pink-noise filter design and by `spectrogram`.

use num_complex::Complex64;
use rustfft::FftPlanner;
use std::sync::Arc;

pub fn fft(x: &[Complex64]) -> Vec<Complex64> {
    let mut buf = x.to_vec();
    planner_fft(x.len(), false).process(&mut buf);
    buf
}

/// Inverse FFT with the conventional `1/N` normalization (`rustfft`'s raw
/// inverse transform is unnormalized).
pub fn ifft(x: &[Complex64]) -> Vec<Complex64> {
    let mut buf = x.to_vec();
    planner_fft(x.len(), true).process(&mut buf);
    let n = buf.len() as f64;
    for v in buf.iter_mut() {
        *v /= n;
    }
    buf
}

fn planner_fft(len: usize, inverse: bool) -> Arc<dyn rustfft::Fft<f64>> {
    let mut planner = FftPlanner::new();
    if inverse {
        planner.plan_fft_inverse(len)
    } else {
        planner.plan_fft_forward(len)
    }
}

/// Swaps the left and right halves of `x`, i.e. a centered `fftshift` for
/// a 1-D vector (for even length this is an exact half-length rotation; for
/// odd length the extra sample stays with the second half, using the
/// conventional `floor(n/2)` as the split point).
pub fn fftshift<T: Copy>(x: &[T]) -> Vec<T> {
    let n = x.len();
    let split = n - n / 2; // fftshift = circshift(x, floor(n/2))
    let mut out = Vec::with_capacity(n);
    out.extend_from_slice(&x[split..]);
    out.extend_from_slice(&x[..split]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ifft_of_fft_is_identity() {
        let x: Vec<Complex64> = (0..16)
            .map(|i| Complex64::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let y = ifft(&fft(&x));
        for (a, b) in x.iter().zip(y.iter()) {
            assert!((a - b).norm() < 1e-9);
        }
    }

    #[test]
    fn fftshift_handles_even_odd() {
        assert_eq!(fftshift(&[1, 2, 3, 4]), vec![3, 4, 1, 2]);
        assert_eq!(fftshift(&[1, 2, 3, 4, 5]), vec![4, 5, 1, 2, 3]);
    }
}
