pub mod fft;
pub mod noisegen;
pub mod replay;
pub mod resample;
pub mod rng;
pub mod spectrogram;
pub mod spline;

use noisegen::{Beta3, NoiseStruct};
use num_complex::Complex64;
use replay::{Channel, ChannelParams, Cir, PhaseField, Tracking};
use rng::SplitMix64;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// JS-facing wrapper around a [`Channel`]. `h_hat_re`/`h_hat_im` must be flat
/// and column-major (delay index fastest, then receiver, then time) --
/// exactly `h_hat(:)`'s byte layout in the MAT-file, which is also exactly
/// what reading the raw HDF5 dataset gives you (once you reverse the
/// HDF5-reported shape to recover `[l, m, t]`; see `web/src/lib/hdf5file.ts`).
#[wasm_bindgen]
pub struct WasmChannel {
    inner: Channel,
}

#[wasm_bindgen]
impl WasmChannel {
    #[wasm_bindgen(constructor)]
    pub fn new(l: usize, m: usize, t: usize, h_hat_re: Vec<f64>, h_hat_im: Vec<f64>, fs_delay: f64, fs_time: f64, fc: f64, version: f64) -> WasmChannel {
        let data: Vec<Complex64> = h_hat_re.into_iter().zip(h_hat_im).map(|(re, im)| Complex64::new(re, im)).collect();
        WasmChannel {
            inner: Channel {
                h_hat: Cir::new(l, m, t, data),
                params: ChannelParams { fs_delay, fs_time, fc },
                tracking: Tracking::None,
                f_resamp: None,
                version,
            },
        }
    }

    /// `data` flat, column-major (`[m, n]`, receiver fastest).
    pub fn set_theta_hat(&mut self, m: usize, n: usize, data: Vec<f64>) {
        self.inner.tracking = Tracking::Theta(PhaseField::new(m, n, data));
    }

    /// `data` flat, column-major (`[m, n]`, receiver fastest).
    pub fn set_phi_hat(&mut self, m: usize, n: usize, data: Vec<f64>) {
        self.inner.tracking = Tracking::Phi(PhaseField::new(m, n, data));
    }

    pub fn set_f_resamp(&mut self, f_resamp: f64) {
        self.inner.f_resamp = Some(f_resamp);
    }

    /// Inclusive `[min, max]` valid range for `start`, given the input's
    /// sample rate and length.
    pub fn valid_start_range(&self, fs: f64, input_len: usize) -> Vec<i64> {
        let (lo, hi) = replay::valid_start_range(&self.inner, fs, input_len);
        vec![lo, hi]
    }

    /// `abs(h_hat(:, m_idx, :))` (0-based receiver index), flattened
    /// column-major (`[l, t]`, delay index fastest) -- the time-varying CIR
    /// magnitude for one receiver, for plotting.
    pub fn cir_magnitude(&self, m_idx: usize) -> Vec<f64> {
        self.inner.h_hat.magnitude_slice(m_idx)
    }

    /// `[seg_start_sec, seg_end_sec]`: the span of the channel's own time
    /// axis that a `replay_js` call with these same arguments reads from
    /// `h_hat`'s T axis, e.g. to highlight which slice of a time-varying CIR
    /// plot produced a given replay.
    pub fn replay_time_range(&self, fs: f64, input_len: usize, start: i64) -> Vec<f64> {
        let (s, e) = replay::replay_time_range(&self.inner, fs, input_len, start);
        vec![s, e]
    }
}

/// Replays `input` through `channel`, returning the per-receiver outputs
/// flattened column-major (`[rows, array_index.len()]`, so
/// `output[row + rows*col]`); `rows` is `output.len() / array_index.len()`.
///
/// `fc_override` (pass `undefined`/`null` for the channel's own `fc`) swaps in
/// a different carrier for the replay -- see [`replay::replay`].
#[wasm_bindgen]
pub fn replay_js(
    input: Vec<f64>,
    fs: f64,
    array_index: Vec<u32>,
    channel: &WasmChannel,
    start: i64,
    fc_override: Option<f64>,
) -> Result<Vec<f64>, JsValue> {
    let idx: Vec<usize> = array_index.into_iter().map(|v| v as usize).collect();
    let outputs = replay::replay(&input, fs, &idx, &channel.inner, start, fc_override).map_err(|e| JsValue::from_str(&e))?;
    let mut flat = Vec::with_capacity(outputs.iter().map(|c| c.len()).sum());
    for col in &outputs {
        flat.extend_from_slice(col);
    }
    Ok(flat)
}

/// Generates independent pink Gaussian noise, `rows` samples across `cols`
/// channels, flattened column-major.
#[wasm_bindgen]
pub fn noise_pink_js(rows: usize, cols: usize, fs: f64, seed: u64) -> Vec<f64> {
    let mut rng = SplitMix64::new(seed);
    let outputs = noisegen::noise_pink(rows, cols, fs, || rng.next_normal());
    let mut flat = Vec::with_capacity(rows * cols);
    for col in &outputs {
        flat.extend_from_slice(col);
    }
    flat
}

/// JS-facing wrapper around a [`NoiseStruct`] (the mixing-coefficient noise
/// model). `beta` flat, column-major (`[m, m, k]`, matching `beta(:)`).
#[wasm_bindgen]
pub struct WasmNoise {
    inner: NoiseStruct,
}

#[wasm_bindgen]
impl WasmNoise {
    #[wasm_bindgen(constructor)]
    pub fn new(m: usize, k: usize, beta: Vec<f64>, alpha: f64, measurement_fs: f64) -> WasmNoise {
        WasmNoise {
            inner: NoiseStruct {
                alpha,
                measurement_fs,
                beta: Beta3::new(m, k, beta),
            },
        }
    }
}

/// Generates mixing-coefficient noise for `array_index` (0-based), flattened
/// column-major (`[input_rows, array_index.len()]`).
#[wasm_bindgen]
pub fn noise_mixing_js(input_rows: usize, array_index: Vec<u32>, fs: f64, noise: &WasmNoise, seed: u64) -> Vec<f64> {
    let idx: Vec<usize> = array_index.into_iter().map(|v| v as usize).collect();
    let mut rng = SplitMix64::new(seed);
    let outputs = noisegen::noise_mixing(input_rows, &idx, fs, &noise.inner, || rng.next_f64());
    let mut flat = Vec::with_capacity(input_rows * idx.len());
    for col in &outputs {
        flat.extend_from_slice(col);
    }
    flat
}

/// A computed spectrogram, exposed to JS via getter methods (each clones its
/// backing `Vec` into a fresh typed array, same as any other Rust->JS
/// boundary crossing).
#[wasm_bindgen]
pub struct WasmSpectrogram {
    times: Vec<f64>,
    freqs: Vec<f64>,
    power_db: Vec<f64>,
    n_time: usize,
    n_freq: usize,
}

#[wasm_bindgen]
impl WasmSpectrogram {
    pub fn times(&self) -> Vec<f64> {
        self.times.clone()
    }
    pub fn freqs(&self) -> Vec<f64> {
        self.freqs.clone()
    }
    /// Flat, row-major by time frame (`power_db[t*n_freq + f]`).
    pub fn power_db(&self) -> Vec<f64> {
        self.power_db.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn n_time(&self) -> usize {
        self.n_time
    }
    #[wasm_bindgen(getter)]
    pub fn n_freq(&self) -> usize {
        self.n_freq
    }
}

#[wasm_bindgen]
pub fn compute_spectrogram(x: Vec<f64>, fs: f64, window_len: usize, hop: usize) -> WasmSpectrogram {
    let s = spectrogram::spectrogram(&x, fs, window_len, hop);
    WasmSpectrogram {
        times: s.times,
        freqs: s.freqs,
        power_db: s.power_db,
        n_time: s.n_time,
        n_freq: s.n_freq,
    }
}
