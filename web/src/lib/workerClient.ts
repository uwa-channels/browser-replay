// Wraps the raw postMessage protocol in worker.ts as Promise-returning calls,
// with an optional progress callback for the slow channel-parsing step.

export interface ChannelInfo {
  l: number;
  m: number;
  t: number;
  fsDelay: number;
  fsTime: number;
  fc: number;
  version: number;
  trackingKind: "theta" | "phi" | null;
  hasFResamp: boolean;
}

export interface NoiseInfo {
  m: number;
  k: number;
  alpha: number;
  measurementFs: number;
}

export interface FlatMatrix {
  flat: number[];
  rows: number;
  cols: number;
}

export interface SpectrogramResult {
  times: number[];
  freqs: number[];
  powerDb: number[];
  nTime: number;
  nFreq: number;
}

type PendingEntry = {
  resolve: (v: any) => void;
  reject: (e: Error) => void;
  onProgress?: (fraction: number) => void;
};

export class ReplayWorkerClient {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, PendingEntry>();

  constructor() {
    this.worker = new Worker(new URL("../worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (ev: MessageEvent) => {
      const msg = ev.data;
      const entry = this.pending.get(msg.id);
      if (!entry) return;
      if (msg.type === "progress") {
        entry.onProgress?.(msg.fraction);
        return;
      }
      this.pending.delete(msg.id);
      if (msg.type === "error") entry.reject(new Error(msg.message));
      else entry.resolve(msg.payload);
    };
  }

  private call<T>(req: Record<string, unknown>, onProgress?: (fraction: number) => void): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      this.worker.postMessage({ id, ...req });
    });
  }

  loadChannel(buffer: ArrayBuffer, onProgress?: (fraction: number) => void): Promise<ChannelInfo> {
    return this.call({ type: "loadChannel", buffer }, onProgress);
  }

  loadNoise(buffer: ArrayBuffer): Promise<NoiseInfo> {
    return this.call({ type: "loadNoise", buffer });
  }

  validStartRange(fs: number, inputLen: number): Promise<[string, string]> {
    return this.call({ type: "validStartRange", fs, inputLen });
  }

  /** `abs(h_hat(:, mIdx, :))` (0-based receiver index), flat column-major
   * (`[l, t]`, delay index fastest). */
  cirMagnitude(mIdx: number): Promise<{ magnitude: number[] }> {
    return this.call({ type: "cirMagnitude", mIdx });
  }

  /** `[segStartSec, segEndSec]` on the channel's own time axis: the span of
   * `h_hat`'s T axis a replay with these arguments actually reads from. */
  replayTimeRange(fs: number, inputLen: number, start: string): Promise<[number, number]> {
    return this.call({ type: "replayTimeRange", fs, inputLen, start });
  }

  runReplay(input: number[], fs: number, arrayIndex: number[], start: string): Promise<FlatMatrix> {
    return this.call({ type: "runReplay", input, fs, arrayIndex, start });
  }

  runNoisePink(rows: number, cols: number, fs: number, seed: string): Promise<FlatMatrix> {
    return this.call({ type: "runNoisePink", rows, cols, fs, seed });
  }

  runNoiseMixing(rows: number, arrayIndex: number[], fs: number, seed: string): Promise<FlatMatrix> {
    return this.call({ type: "runNoiseMixing", rows, arrayIndex, fs, seed });
  }

  spectrogram(x: number[], fs: number, windowLen: number, hop: number): Promise<SpectrogramResult> {
    return this.call({ type: "spectrogram", x, fs, windowLen, hop });
  }

  terminate() {
    this.worker.terminate();
  }
}
