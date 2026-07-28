// Runs the heavy compute (HDF5 parsing, replay, noise generation,
// spectrograms) off the main thread so the UI stays responsive. Holds the
// currently-loaded channel/noise as WASM objects in worker memory rather
// than shuttling the (potentially 100MB+) h_hat array back and forth.

import init, {
  WasmChannel,
  WasmNoise,
  replay_js,
  noise_pink_js,
  noise_mixing_js,
  compute_spectrogram,
} from "./wasm/replay_core.js";
import wasmUrl from "./wasm/replay_core_bg.wasm?url";
import { parseChannelMat, parseNoiseMat } from "./lib/hdf5file";

let channel: WasmChannel | null = null;
let noise: WasmNoise | null = null;
let initPromise: Promise<void> | null = null;

function ensureInit(): Promise<void> {
  if (!initPromise) initPromise = init({ module_or_path: wasmUrl }).then(() => undefined);
  return initPromise;
}

type Request =
  | { id: number; type: "loadChannel"; buffer: ArrayBuffer }
  | { id: number; type: "loadNoise"; buffer: ArrayBuffer }
  | { id: number; type: "validStartRange"; fs: number; inputLen: number }
  | { id: number; type: "cirMagnitude"; mIdx: number }
  | { id: number; type: "replayTimeRange"; fs: number; inputLen: number; start: string }
  | { id: number; type: "runReplay"; input: number[]; fs: number; arrayIndex: number[]; start: string }
  | { id: number; type: "runNoisePink"; rows: number; cols: number; fs: number; seed: string }
  | { id: number; type: "runNoiseMixing"; rows: number; arrayIndex: number[]; fs: number; seed: string }
  | { id: number; type: "spectrogram"; x: number[]; fs: number; windowLen: number; hop: number };

function post(msg: unknown) {
  (self as unknown as Worker).postMessage(msg);
}

self.onmessage = async (ev: MessageEvent<Request>) => {
  const req = ev.data;
  try {
    await ensureInit();
    switch (req.type) {
      case "loadChannel": {
        const data = await parseChannelMat(req.buffer, (fraction) => post({ id: req.id, type: "progress", fraction }));
        channel = new WasmChannel(data.l, data.m, data.t, data.hHatRe, data.hHatIm, data.fsDelay, data.fsTime, data.fc, data.version);
        if (data.tracking?.kind === "phi") channel.set_phi_hat(data.tracking.m, data.tracking.n, data.tracking.data);
        if (data.tracking?.kind === "theta") channel.set_theta_hat(data.tracking.m, data.tracking.n, data.tracking.data);
        if (data.fResamp != null) channel.set_f_resamp(data.fResamp);
        post({
          id: req.id,
          type: "result",
          payload: {
            l: data.l,
            m: data.m,
            t: data.t,
            fsDelay: data.fsDelay,
            fsTime: data.fsTime,
            fc: data.fc,
            version: data.version,
            trackingKind: data.tracking?.kind ?? null,
            hasFResamp: data.fResamp != null,
          },
        });
        break;
      }
      case "loadNoise": {
        const data = await parseNoiseMat(req.buffer);
        noise = new WasmNoise(data.m, data.k, data.beta, data.alpha, data.measurementFs);
        post({ id: req.id, type: "result", payload: { m: data.m, k: data.k, alpha: data.alpha, measurementFs: data.measurementFs } });
        break;
      }
      case "validStartRange": {
        if (!channel) throw new Error("No channel loaded");
        const [lo, hi] = channel.valid_start_range(req.fs, req.inputLen);
        post({ id: req.id, type: "result", payload: [lo.toString(), hi.toString()] });
        break;
      }
      case "cirMagnitude": {
        if (!channel) throw new Error("No channel loaded");
        const magnitude = channel.cir_magnitude(req.mIdx);
        post({ id: req.id, type: "result", payload: { magnitude: Array.from(magnitude) } });
        break;
      }
      case "replayTimeRange": {
        if (!channel) throw new Error("No channel loaded");
        const [segStart, segEnd] = channel.replay_time_range(req.fs, req.inputLen, BigInt(req.start));
        post({ id: req.id, type: "result", payload: [segStart, segEnd] });
        break;
      }
      case "runReplay": {
        if (!channel) throw new Error("No channel loaded");
        const flat = replay_js(Float64Array.from(req.input), req.fs, Uint32Array.from(req.arrayIndex), channel, BigInt(req.start));
        const cols = req.arrayIndex.length;
        const rows = cols > 0 ? flat.length / cols : 0;
        post({ id: req.id, type: "result", payload: { flat: Array.from(flat), rows, cols } });
        break;
      }
      case "runNoisePink": {
        const flat = noise_pink_js(req.rows, req.cols, req.fs, BigInt(req.seed));
        post({ id: req.id, type: "result", payload: { flat: Array.from(flat), rows: req.rows, cols: req.cols } });
        break;
      }
      case "runNoiseMixing": {
        if (!noise) throw new Error("No noise struct loaded");
        const flat = noise_mixing_js(req.rows, Uint32Array.from(req.arrayIndex), req.fs, noise, BigInt(req.seed));
        post({ id: req.id, type: "result", payload: { flat: Array.from(flat), rows: req.rows, cols: req.arrayIndex.length } });
        break;
      }
      case "spectrogram": {
        const spec = compute_spectrogram(Float64Array.from(req.x), req.fs, req.windowLen, req.hop);
        post({
          id: req.id,
          type: "result",
          payload: {
            times: Array.from(spec.times()),
            freqs: Array.from(spec.freqs()),
            powerDb: Array.from(spec.power_db()),
            nTime: spec.n_time,
            nFreq: spec.n_freq,
          },
        });
        break;
      }
    }
  } catch (err) {
    post({ id: req.id, type: "error", message: err instanceof Error ? err.message : String(err) });
  }
};
