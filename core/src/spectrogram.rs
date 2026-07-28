//! Short-time Fourier transform for the input/output spectrogram displays.
//! This is a visualization aid, not part of the accuracy-critical replay/
//! noisegen pipeline, so it isn't ported to bit-match any reference
//! implementation -- it's a standard Hann-windowed STFT with a Welch-style
//! PSD scaling.

use crate::fft::fft;
use num_complex::Complex64;
use std::f64::consts::PI;

pub fn hann_window(n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos())
        .collect()
}

/// A spectrogram: `n_time` frames × `n_freq` one-sided frequency bins, flat
/// and row-major by time (`power_db[t*n_freq + f]`).
pub struct Spectrogram {
    pub times: Vec<f64>,
    pub freqs: Vec<f64>,
    pub power_db: Vec<f64>,
    pub n_time: usize,
    pub n_freq: usize,
}

/// Computes a power spectrogram of `x` (sampled at `fs`) using a Hann window
/// of `window_len` samples and hop size `hop`.
pub fn spectrogram(x: &[f64], fs: f64, window_len: usize, hop: usize) -> Spectrogram {
    let win = hann_window(window_len);
    let win_energy: f64 = win.iter().map(|w| w * w).sum();
    let n_freq = window_len / 2 + 1;

    let starts: Vec<usize> = if x.len() < window_len {
        vec![]
    } else {
        (0..=(x.len() - window_len)).step_by(hop).collect()
    };
    let n_time = starts.len();

    let mut power_db = vec![0.0_f64; n_time * n_freq];
    for (t_idx, &s) in starts.iter().enumerate() {
        let frame: Vec<Complex64> = (0..window_len).map(|i| Complex64::new(x[s + i] * win[i], 0.0)).collect();
        let spec = fft(&frame);
        for (f_idx, slot) in power_db[t_idx * n_freq..t_idx * n_freq + n_freq].iter_mut().enumerate() {
            let psd = spec[f_idx].norm_sqr() / (fs * win_energy);
            *slot = 10.0 * psd.max(1e-300).log10();
        }
    }

    let times: Vec<f64> = starts.iter().map(|&s| (s as f64 + window_len as f64 / 2.0) / fs).collect();
    let freqs: Vec<f64> = (0..n_freq).map(|i| i as f64 * fs / window_len as f64).collect();

    Spectrogram {
        times,
        freqs,
        power_db,
        n_time,
        n_freq,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_tone_peaks_at_its_own_frequency() {
        let fs = 8000.0;
        let f0 = 1000.0;
        let n = 4000;
        let x: Vec<f64> = (0..n).map(|i| (2.0 * PI * f0 * i as f64 / fs).sin()).collect();
        let spec = spectrogram(&x, fs, 512, 256);
        assert!(spec.n_time > 0);

        let mid_frame = spec.n_time / 2;
        let row = &spec.power_db[mid_frame * spec.n_freq..(mid_frame + 1) * spec.n_freq];
        let (peak_idx, _) = row.iter().enumerate().fold((0, f64::MIN), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) });
        let peak_freq = spec.freqs[peak_idx];
        assert!((peak_freq - f0).abs() < spec.freqs[1], "peak at {peak_freq}, expected near {f0}");
    }
}
