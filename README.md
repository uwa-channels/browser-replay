# [browser-replay](https://uwa-channels.github.io/browser-replay/)

A browser-based, WASM-accelerated implementation of underwater acoustic
**channel replay** and **noise generation**, ported from the
[uwa-channels](https://github.com/uwa-channels) reference toolboxes.

Pick a measured channel from the
[Zenodo dataset](https://zenodo.org/records/21287414), upload a passband
signal, and the app convolves it with the real, time-varying channel impulse
response, adds measured or synthetic ambient noise, and lets you download the
result — all client-side, no server round-trip.

## Status

Functional end to end: pick a channel, upload a WAV, replay it through the
real channel (with phase/delay-drift tracking and optional bulk Doppler
correction), add measured or pink noise, view input/output spectrograms, and
download the result. Deploys to GitHub Pages via `.github/workflows/deploy.yml`.

## Running locally

```sh
cd web
npm install
npm run dev
```

`npm run dev`/`npm run build` both build the WASM bindings first (via a
`predev`/`prebuild` hook running `web/scripts/build-wasm.mjs`), which requires
a Rust toolchain with the `wasm32-unknown-unknown` target and a
`wasm-bindgen` CLI matching the version pinned in `core/Cargo.toml` exactly
(currently `0.2.125`) -- install with:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.125
```

## Layout

- `core/` — Rust crate compiled to WebAssembly. Implements the numerical
  pipeline: polyphase resampling, spline interpolation, FFT/STFT, the
  time-varying convolution + Doppler/phase-drift reinsertion from `replay.m`,
  and the pink-noise / spatial-mixing noise models from `noisegen.m`.
- `web/` — React + Vite + TypeScript frontend: channel picker (fetched live
  from the Zenodo API), signal upload/spectrogram/download, Web Worker
  offload for the heavy compute.

## Accuracy

The resampling, spline interpolation, FFT, and convolution paths are ported
to match the reference toolboxes' algorithms as closely as possible. Random
noise generation (`noisegen`) matches the reference implementation's
*statistical* properties (spectral slope, spatial correlation, tail behavior)
but does not reproduce its exact RNG bit stream — a given seed will not
replay the identical noise realization as the reference implementation.

## License

MIT — see `LICENSE`. Ports algorithms from the MIT-licensed
[uwa-channels](https://github.com/uwa-channels) toolbox.
