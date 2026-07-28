// Fetches the channel/noise catalog from the Zenodo record and downloads
// individual files with progress, caching them in IndexedDB (via
// idb-keyval) so picking the same channel twice doesn't re-download ~200MB.

import { get, set } from "idb-keyval";

const RECORD_ID = "21287414";
const API_URL = `https://zenodo.org/api/records/${RECORD_ID}`;

export interface ZenodoFile {
  key: string;
  size: number;
  downloadUrl: string;
}

export interface ChannelEntry {
  /** e.g. "blue_1", "brown" (no trailing number for single-file colors). */
  name: string;
  color: string;
  index: number | null;
  file: ZenodoFile;
  /** The paired noise file for this color, if the catalog has one. */
  noiseFile: ZenodoFile | null;
}

let catalogPromise: Promise<ZenodoFile[]> | null = null;

async function fetchRawCatalog(): Promise<ZenodoFile[]> {
  const res = await fetch(API_URL);
  if (!res.ok) throw new Error(`Failed to fetch Zenodo catalog: ${res.status} ${res.statusText}`);
  const json = await res.json();
  const files = json.files as Array<{ key: string; size: number; links: { self: string } }>;
  return files.map((f) => ({ key: f.key, size: f.size, downloadUrl: f.links.self }));
}

export function fetchCatalog(): Promise<ZenodoFile[]> {
  if (!catalogPromise) catalogPromise = fetchRawCatalog();
  return catalogPromise;
}

const CHANNEL_RE = /^([a-z]+)(?:_(\d+))?\.mat$/;
const NOISE_RE = /^([a-z]+)_noise(?:_(\d+))?\.mat$/;

/** Channel-index -> noise-index remaps for colors where the pairing isn't
 * 1:1 by filename index, per
 * https://uwa-channels.github.io/content/noises: several channels can share
 * one noise recording. Colors not listed here (blue, red, green, black,
 * pink, brown) pair up correctly from the filename index alone (falling
 * back to a single unindexed noise file where there's no per-index one). */
const NOISE_INDEX_OVERRIDES: Record<string, (channelIndex: number | null) => number | null> = {
  // yellow_1..3 -> yellow_noise_1, yellow_4 -> _2, yellow_5 -> _3, yellow_6 -> _4.
  yellow: (idx) => (idx == null ? null : idx <= 3 ? 1 : idx - 2),
  // purple_1..15 cycle through only 5 noise recordings (_1..5, repeating every 5 channels).
  purple: (idx) => (idx == null ? null : ((idx - 1) % 5) + 1),
};

/** Groups the raw file catalog into selectable channels, each paired with
 * its color's noise file (preferring an index-matched noise file, e.g.
 * `purple_2` -> `purple_noise_2`, falling back to the base `<color>_noise`). */
export function buildChannelList(files: ZenodoFile[]): ChannelEntry[] {
  const noiseByColor = new Map<string, Map<number | null, ZenodoFile>>();
  for (const f of files) {
    const m = NOISE_RE.exec(f.key);
    if (!m) continue;
    const color = m[1];
    const index = m[2] ? parseInt(m[2], 10) : null;
    if (!noiseByColor.has(color)) noiseByColor.set(color, new Map());
    noiseByColor.get(color)!.set(index, f);
  }

  const entries: ChannelEntry[] = [];
  for (const f of files) {
    const m = CHANNEL_RE.exec(f.key);
    if (!m || f.key.includes("_noise")) continue;
    const color = m[1];
    const index = m[2] ? parseInt(m[2], 10) : null;
    const byIndex = noiseByColor.get(color);
    const remap = NOISE_INDEX_OVERRIDES[color];
    const noiseIndex = remap ? remap(index) : index;
    const noiseFile = byIndex ? (byIndex.get(noiseIndex) ?? byIndex.get(null) ?? null) : null;
    entries.push({ name: f.key.replace(/\.mat$/, ""), color, index, file: f, noiseFile });
  }

  entries.sort((a, b) => a.color.localeCompare(b.color) || (a.index ?? 0) - (b.index ?? 0));
  return entries;
}

export type ProgressCallback = (loaded: number, total: number) => void;

async function fetchWithProgress(url: string, onProgress?: ProgressCallback, signal?: AbortSignal): Promise<ArrayBuffer> {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`Download failed: ${res.status} ${res.statusText}`);
  const total = parseInt(res.headers.get("content-length") ?? "0", 10);

  if (!res.body || !onProgress) {
    return res.arrayBuffer();
  }

  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }

  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out.buffer;
}

const cacheKey = (name: string) => `zenodo-file:${name}`;

/** Downloads `file`, using an IndexedDB-cached copy if present. */
export async function downloadFile(file: ZenodoFile, onProgress?: ProgressCallback, signal?: AbortSignal): Promise<ArrayBuffer> {
  const cached = await get<ArrayBuffer>(cacheKey(file.key));
  if (cached && cached.byteLength === file.size) {
    onProgress?.(file.size, file.size);
    return cached;
  }

  const buf = await fetchWithProgress(file.downloadUrl, onProgress, signal);
  try {
    await set(cacheKey(file.key), buf);
  } catch {
    // IndexedDB quota exceeded or unavailable -- caching is a nice-to-have,
    // not a correctness requirement, so just proceed without it.
  }
  return buf;
}
