// Parses uwa-channels' v7.3 (HDF5) channel/noise .mat files via h5wasm.
//
// Key fact this whole module leans on (verified against a real blue_1.mat
// both through h5wasm and through the reference toolchain's own partial-load
// API): the writer that produced these files stores an N-D array with its
// dimension *tuple* reversed relative to its logical shape, but leaves the
// underlying byte sequence untouched (column-major storage of shape
// [d1,d2,...,dn] has the same linear byte order as row-major storage of
// shape [dn,...,d2,d1]). So:
//   - h5wasm's reported `.shape` is the logical shape *reversed*.
//   - h5wasm's flat `.value`/`.slice()` output, read in the natural order it
//     comes back in, is already exactly the column-major flatten of
//     the *logical* (unreversed) shape -- no manual reordering needed, only
//     `reverse(shape)` to label the axes correctly.
// This was confirmed for both a 3-D compound dataset (h_hat) and a 2-D real
// dataset (phi_hat) against concrete values from a reference session.
//
// h_hat is stored as an HDF5 compound type ({real, imag} fields). h5wasm's
// public slice()/value API decodes compound datasets by allocating a nested
// JS array-of-[re,im] per element (see its process_data()) -- fine for small
// attributes, but for a full channel (tens of millions of taps) that's tens
// of millions of small object/array allocations, measured at ~17s alone for
// a real 200MB channel file. Since {real: f64, imag: f64} has no padding
// (verified against real files' metadata: offsets 0/8, size 16), the raw
// hyperslab bytes are already exactly an interleaved Float64Array -- so we
// read those bytes directly via h5wasm's lower-level Module.get_dataset_data
// (the same call slice() makes internally) and de-interleave with one flat
// loop, skipping the expensive per-element decode entirely. Measured ~0.2s
// for the same file. Correctness verified against the slow path's output.
import * as hdf5 from "h5wasm";

let readyPromise: Promise<void> | null = null;
function ensureReady(): Promise<void> {
  if (!readyPromise) {
    readyPromise = hdf5.ready.then(() => undefined);
  }
  return readyPromise;
}

function reversed(shape: number[]): number[] {
  return [...shape].reverse();
}

function scalar(ds: any): number {
  const v = ds.value;
  return Array.isArray(v) ? v[0] : v[0] ?? v;
}

/** Writes an ArrayBuffer into h5wasm's virtual FS under a unique path and
 * opens it, so concurrent parses (e.g. channel + noise file) don't collide. */
function openBuffer(buf: ArrayBuffer): InstanceType<typeof hdf5.File> {
  const path = `/f_${Math.random().toString(36).slice(2)}.mat`;
  // Non-null: callers always await ensureReady() first, which resolves once
  // h5wasm has initialized its virtual filesystem.
  hdf5.FS!.writeFile(path, new Uint8Array(buf));
  return new hdf5.File(path, "r");
}

export interface TrackingField {
  kind: "theta" | "phi";
  /** Flat, column-major: `data[m + M*n]`. */
  m: number;
  n: number;
  data: Float64Array;
}

export interface ChannelData {
  l: number;
  m: number;
  t: number;
  /** Flat, column-major: `data[l + L*(m + M*t)]` (matches `h_hat(:)`). */
  hHatRe: Float64Array;
  hHatIm: Float64Array;
  fsDelay: number;
  fsTime: number;
  fc: number;
  version: number;
  tracking: TrackingField | null;
  fResamp: number | null;
}

const T_CHUNK = 64;

/** Reads a `[t0,t1) x [0,m) x [0,l)` hyperslab of a `{real,imag} f64`
 * compound dataset (h5wasm's on-disk shape order, i.e. reversed from the
 * logical `[l,m,t]`) straight from the WASM heap, bypassing h5wasm's
 * per-element compound decode. `real`/`imag` field offsets are read from the
 * dataset's own metadata (rather than assumed 0/8) so this stays correct if
 * a file ever orders the fields the other way round; both must be 8-byte
 * floats with no padding for the fast interleaved-view path to apply. */
function readHHatChunk(ds: any, t0: number, t1: number, m: number, l: number): { re: Float64Array; im: Float64Array } {
  const meta = ds.metadata;
  const members: { name: string; offset: number; size: number }[] = meta.compound_type.members;
  const realMember = members.find((mm) => mm.name === "real");
  const imagMember = members.find((mm) => mm.name === "imag");
  if (!realMember || !imagMember || meta.size !== 16 || realMember.size !== 8 || imagMember.size !== 8) {
    throw new Error("h_hat compound layout is not the expected {real: f64, imag: f64} pair");
  }
  const reFirst = realMember.offset < imagMember.offset;

  const Module = hdf5.Module;
  const n = (t1 - t0) * m * l;
  const nbytes = 16 * n;
  const ptr = Module._malloc(nbytes);
  try {
    Module.get_dataset_data(
      ds.file_id,
      ds.path,
      [BigInt(t1 - t0), BigInt(m), BigInt(l)],
      [BigInt(t0), 0n, 0n],
      [1n, 1n, 1n],
      BigInt(ptr),
    );
    // Copy out of the WASM heap before decoding: `_malloc`/growth elsewhere
    // can move/invalidate the underlying buffer.
    const bytes = Module.HEAPU8.slice(ptr, ptr + nbytes);
    const interleaved = new Float64Array(bytes.buffer);
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      re[i] = interleaved[2 * i + (reFirst ? 0 : 1)];
      im[i] = interleaved[2 * i + (reFirst ? 1 : 0)];
    }
    return { re, im };
  } finally {
    Module._free(ptr);
  }
}

export async function parseChannelMat(buf: ArrayBuffer, onProgress?: (fraction: number) => void): Promise<ChannelData> {
  await ensureReady();
  const f = openBuffer(buf);
  try {
    const hHatDs: any = f.get("h_hat");
    const [l, m, t] = reversed(hHatDs.shape as number[]);

    const hHatRe = new Float64Array(l * m * t);
    const hHatIm = new Float64Array(l * m * t);
    for (let t0 = 0; t0 < t; t0 += T_CHUNK) {
      const t1 = Math.min(t0 + T_CHUNK, t);
      const { re, im } = readHHatChunk(hHatDs, t0, t1, m, l);
      const base = l * m * t0;
      hHatRe.set(re, base);
      hHatIm.set(im, base);
      onProgress?.(t1 / t);
    }

    const params: any = f.get("params");
    const fsDelay = scalar(params.get("fs_delay"));
    const fsTime = scalar(params.get("fs_time"));
    const fc = scalar(params.get("fc"));
    const version = scalar(f.get("version"));

    const keys: string[] = f.keys();
    let tracking: TrackingField | null = null;
    for (const kind of ["phi", "theta"] as const) {
      const key = `${kind}_hat`;
      if (keys.includes(key)) {
        const ds: any = f.get(key);
        const [mm, nn] = reversed(ds.shape as number[]);
        tracking = { kind, m: mm, n: nn, data: ds.value as Float64Array };
        break;
      }
    }

    const fResamp = keys.includes("f_resamp") ? scalar(f.get("f_resamp")) : null;

    return { l, m, t, hHatRe, hHatIm, fsDelay, fsTime, fc, version, tracking, fResamp };
  } finally {
    f.close();
  }
}

export interface NoiseData {
  m: number;
  k: number;
  /** Flat, column-major: `data[i + M*(j + M*k)]` (matches `beta(:)`). */
  beta: Float64Array;
  alpha: number;
  measurementFs: number;
  version: number;
}

export async function parseNoiseMat(buf: ArrayBuffer): Promise<NoiseData> {
  await ensureReady();
  const f = openBuffer(buf);
  try {
    const betaDs: any = f.get("beta");
    const [m, m2, k] = reversed(betaDs.shape as number[]);
    if (m !== m2) {
      throw new Error(`noise beta tensor is not square in its receiver dims: [${m}, ${m2}, ${k}]`);
    }
    const beta = betaDs.value as Float64Array;

    const alpha = scalar(f.get("alpha"));
    const measurementFs = scalar(f.get("Fs"));
    const version = scalar(f.get("version"));

    return { m, k, beta, alpha, measurementFs, version };
  } finally {
    f.close();
  }
}
