//! Passes a passband signal through a measured, time-varying underwater
//! acoustic channel. Ported line-for-line from the reference channel-replay
//! implementation (uwa-channels toolbox): downconvert to baseband, resample to the
//! channel's delay-domain rate, run a time-varying convolution against a
//! spline-interpolated impulse response, reinsert tracked phase/delay drift,
//! resample back and upconvert to passband.

use crate::resample;
use crate::spline::CubicSpline;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Time-varying complex-baseband impulse response cube, `h_hat` in the
/// channel MAT-file format: `l` delay taps × `m` receivers × `t` time
/// snapshots. Stored flat in the same column-major order as the reference
/// `h_hat(:)` (delay index fastest, then receiver, then time).
pub struct Cir {
    pub l: usize,
    pub m: usize,
    pub t: usize,
    pub data: Vec<Complex64>,
}

impl Cir {
    pub fn new(l: usize, m: usize, t: usize, data: Vec<Complex64>) -> Self {
        assert_eq!(data.len(), l * m * t, "Cir data length must be l*m*t");
        Cir { l, m, t, data }
    }

    #[inline]
    pub fn get(&self, l_idx: usize, m_idx: usize, t_idx: usize) -> Complex64 {
        self.data[l_idx + self.l * (m_idx + self.m * t_idx)]
    }

    /// `abs(h_hat(:, m_idx, :))`, squeezed to `l x t` and flattened
    /// column-major (`data[l_idx + l*t_idx]`) -- the time-varying CIR for one
    /// receiver, for plotting.
    pub fn magnitude_slice(&self, m_idx: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.l * self.t);
        for t_idx in 0..self.t {
            for l_idx in 0..self.l {
                out.push(self.get(l_idx, m_idx, t_idx).norm());
            }
        }
        out
    }
}

/// `theta_hat`/`phi_hat`: a real-valued phase (or delay-drift) field, one row
/// per receiver, sampled at `fs_delay`. Stored flat, column-major
/// (`data[m_idx + m*n_idx]`), matching the reference `field(:)`.
pub struct PhaseField {
    pub m: usize,
    pub n: usize,
    pub data: Vec<f64>,
}

impl PhaseField {
    pub fn new(m: usize, n: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), m * n, "PhaseField data length must be m*n");
        PhaseField { m, n, data }
    }

    #[inline]
    pub fn get(&self, m_idx: usize, n_idx: usize) -> f64 {
        self.data[m_idx + self.m * n_idx]
    }
}

pub struct ChannelParams {
    pub fs_delay: f64,
    pub fs_time: f64,
    pub fc: f64,
}

/// Which optional tracking field the channel provides, mirroring the
/// mutually-exclusive `theta_hat`/`phi_hat` fields in the MAT-file format:
/// `phi_hat` carries phase + delay drift (`h_hat` itself is static), while
/// `theta_hat` carries phase only (the drift is already baked into `h_hat`).
pub enum Tracking {
    None,
    Theta(PhaseField),
    Phi(PhaseField),
}

pub struct Channel {
    pub h_hat: Cir,
    pub params: ChannelParams,
    pub tracking: Tracking,
    pub f_resamp: Option<f64>,
    pub version: f64,
}

/// Ported from replay.m's `validate_inputs`: the channel MAT-file version
/// gate, the input-signal-duration-vs-channel-recording-length check, and the
/// receiver index sanity checks. `array_index` is 0-based here (unlike
/// the reference implementation's 1-based `array_index`), so the bounds
/// message is phrased in 1-based terms to match what a user actually typed
/// into the UI.
fn validate_inputs(input_len: usize, fs: f64, array_index: &[usize], channel: &Channel) -> Result<(), String> {
    if channel.version < 1.0 {
        return Err(format!(
            "The minimum version of the channel matrix is 1.0, and you have {:.1}.",
            channel.version
        ));
    }

    let t = input_len as f64 / fs;
    let t_max = channel.h_hat.t as f64 / channel.params.fs_time;
    if !(t < t_max) {
        return Err(format!(
            "Duration of the input signal, {:.2}ms, should be no larger than {:.2}ms.",
            t * 1e3,
            t_max * 1e3
        ));
    }

    let n = channel.h_hat.m;
    let mut sorted = array_index.to_vec();
    sorted.sort_unstable();
    if sorted.windows(2).any(|w| w[0] == w[1]) {
        return Err("array_index must not contain duplicate indices.".to_string());
    }
    if array_index.iter().any(|&i| i >= n) {
        return Err(format!("array_index must be positive integers and cannot exceed {n}."));
    }

    Ok(())
}

/// Replays `input` (a real passband signal at rate `fs`) through `channel`,
/// returning one real passband output column per requested receiver in
/// `array_index` (0-based). `start` is the raw sample offset into the
/// channel's delay-domain timeline (matching the reference `start` argument
/// exactly — see `valid_start_range` for how to pick one).
pub fn replay(input: &[f64], fs: f64, array_index: &[usize], channel: &Channel, start: i64) -> Result<Vec<Vec<f64>>, String> {
    validate_inputs(input.len(), fs, array_index, channel)?;

    let fs_delay = channel.params.fs_delay;
    let fc = channel.params.fc;
    let l = channel.h_hat.l;

    // Downconvert to baseband, then resample fs -> fs_delay. The exact (p, q)
    // from this single `rat` call is reused (as (q, p)) for the inverse
    // resample at the end, exactly as replay.m does -- recomputing `rat`
    // independently on the reciprocal ratio can pick a different fraction.
    let (p, q) = resample::rat(fs_delay / fs, None);
    let baseband_fs: Vec<Complex64> = input
        .iter()
        .enumerate()
        .map(|(n, &x)| {
            let theta = -2.0 * PI * fc * n as f64 / fs;
            Complex64::new(x * theta.cos(), x * theta.sin())
        })
        .collect();
    let baseband = resample::resample(&baseband_fs, p as u64, q as u64);
    let t_len = baseband.len() as i64;

    let buffer: i64 = 20;
    let n_out_len = (t_len + l as i64 - 1).max(0) as usize; // N
    let total_len = (t_len + buffer + l as i64).max(0) as usize; // T + buffer + L

    let mut baseband_padded = vec![Complex64::new(0.0, 0.0); l - 1];
    baseband_padded.extend_from_slice(&baseband);
    baseband_padded.extend(std::iter::repeat_n(Complex64::new(0.0, 0.0), l - 1));

    let channel_time: Vec<f64> = (0..channel.h_hat.t)
        .map(|k| k as f64 / channel.params.fs_time)
        .collect();
    let signal_time: Vec<f64> = (0..total_len)
        .map(|i| (i as f64 + start as f64) / fs_delay)
        .collect();

    let mut outputs: Vec<Vec<f64>> = Vec::with_capacity(array_index.len());

    for &recv_idx in array_index {
        // Time-varying convolution: for each of the L delay taps (accessed in
        // reverse order, matching replay.m's `flip(..., 2)`), spline-
        // interpolate that tap's slowly-sampled trajectory onto the signal's
        // fine time grid, then accumulate its contribution to every output
        // sample.
        let mut conv_out = vec![Complex64::new(0.0, 0.0); n_out_len];
        for l_tap in 0..l {
            let orig_l = l - 1 - l_tap;
            let col: Vec<Complex64> = (0..channel.h_hat.t)
                .map(|t_idx| channel.h_hat.get(orig_l, recv_idx, t_idx))
                .collect();
            let sp = CubicSpline::new(&channel_time, &col);
            let ir_col = sp.eval_many_fill(&signal_time, Complex64::new(0.0, 0.0));
            for (n, c) in conv_out.iter_mut().enumerate() {
                *c += ir_col[n] * baseband_padded[l_tap + n];
            }
        }

        // Phase / delay-drift reinsertion.
        let mut out_col = vec![Complex64::new(0.0, 0.0); total_len];
        match &channel.tracking {
            Tracking::Phi(phi) => {
                for n in 0..n_out_len {
                    let ph = phi.get(recv_idx, (start - 1 + n as i64) as usize);
                    out_col[n] = conv_out[n] * Complex64::new(0.0, ph).exp();
                }
                let drift: Vec<f64> = (0..total_len)
                    .map(|i| phi.get(recv_idx, (start - 1 + i as i64) as usize) / (2.0 * PI * fc))
                    .collect();
                let sp = CubicSpline::new(&signal_time, &out_col);
                let query: Vec<f64> = (0..total_len).map(|i| signal_time[i] + drift[i]).collect();
                out_col = sp.eval_many_fill(&query, Complex64::new(0.0, 0.0));
            }
            Tracking::Theta(theta) => {
                for n in 0..n_out_len {
                    let ph = theta.get(recv_idx, (start - 1 + n as i64) as usize);
                    out_col[n] = conv_out[n] * Complex64::new(0.0, ph).exp();
                }
            }
            Tracking::None => {
                out_col[..n_out_len].copy_from_slice(&conv_out);
            }
        }

        // Resample back to fs (reusing (q, p), the exact reciprocal of the
        // forward ratio) and upconvert to passband.
        let back = resample::resample(&out_col, q as u64, p as u64);
        let mut passband: Vec<f64> = back
            .iter()
            .enumerate()
            .map(|(n, &v)| {
                let theta = 2.0 * PI * fc * n as f64 / fs;
                2.0 * (v * Complex64::new(theta.cos(), theta.sin())).re
            })
            .collect();

        if let Some(f_resamp) = channel.f_resamp {
            let (p2, q2) = resample::rat(f_resamp, None);
            passband = resample::resample(&passband, p2 as u64, q2 as u64);
        }

        outputs.push(passband);
    }

    Ok(outputs)
}

/// The span of the channel's own time axis (seconds, matching `channel_time`
/// above) that a `replay()` call with these arguments actually reads from
/// `h_hat`'s T axis -- e.g. for drawing which slice of a time-varying CIR
/// plot produced a given replay. `t_len` (the resampled baseband length) is
/// computed the same way `resample::resample`'s `ly` is (`ceil(input_len *
/// p / q)`), so this matches `replay()`'s actual `total_len` exactly without
/// re-running the resample.
pub fn replay_time_range(channel: &Channel, fs: f64, input_len: usize, start: i64) -> (f64, f64) {
    let fs_delay = channel.params.fs_delay;
    let l = channel.h_hat.l as i64;
    let (p, q) = resample::rat(fs_delay / fs, None);
    let t_len = (input_len as i64 * p + q - 1) / q;
    let buffer: i64 = 20;
    let total_len = (t_len + buffer + l).max(0);
    (start as f64 / fs_delay, (start + total_len) as f64 / fs_delay)
}

/// Inclusive valid range for `start`, matching the reference
/// `randi([1, T_max - T - L - buffer - 1])`: `T` is the input's length after
/// resampling to `fs_delay`, and `T_max` is the channel recording's total
/// duration in delay-domain samples. Callers pick a `start` in this range
/// with whatever RNG they like -- `replay()` itself takes no randomness.
pub fn valid_start_range(channel: &Channel, fs: f64, input_len: usize) -> (i64, i64) {
    let fs_delay = channel.params.fs_delay;
    let (p, q) = resample::rat(fs_delay / fs, None);
    let numerator = input_len as i64 * p;
    let t_estimate = (numerator + q - 1) / q;
    let l = channel.h_hat.l as i64;
    let buffer = 20i64;
    let t_max = (channel.h_hat.t as f64 / channel.params.fs_time * fs_delay).round() as i64;
    (1, (t_max - t_estimate - l - buffer - 1).max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_channel(l: usize, m: usize, t: usize, fs_delay: f64, fs_time: f64) -> Channel {
        let data: Vec<Complex64> = (0..l * m * t).map(|i| Complex64::new(i as f64, -(i as f64))).collect();
        Channel {
            h_hat: Cir::new(l, m, t, data),
            params: ChannelParams { fs_delay, fs_time, fc: 0.0 },
            tracking: Tracking::None,
            f_resamp: None,
            version: 1.0,
        }
    }

    #[test]
    fn magnitude_slice_matches_direct_get() {
        let ch = tiny_channel(3, 2, 4, 1000.0, 10.0);
        let mag = ch.h_hat.magnitude_slice(1);
        for t_idx in 0..4 {
            for l_idx in 0..3 {
                assert_eq!(mag[l_idx + 3 * t_idx], ch.h_hat.get(l_idx, 1, t_idx).norm());
            }
        }
    }

    #[test]
    fn replay_time_range_matches_replay_total_len() {
        let ch = tiny_channel(4, 1, 50, 1000.0, 5.0);
        let fs = 800.0;
        let input_len = 200;
        let start = 10i64;

        let (seg_start, seg_end) = replay_time_range(&ch, fs, input_len, start);
        assert_eq!(seg_start, start as f64 / ch.params.fs_delay);

        let implied_total_len = ((seg_end - seg_start) * ch.params.fs_delay).round() as i64;
        let (p, q) = resample::rat(ch.params.fs_delay / fs, None);
        let t_len = (input_len as i64 * p + q - 1) / q;
        let expected_total_len = t_len + 20 + ch.h_hat.l as i64;
        assert_eq!(implied_total_len, expected_total_len);
    }
}
