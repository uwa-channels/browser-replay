// Minimal WAV reader/writer preserving the exact sample rate in the file
// header. Deliberately not using the Web Audio API's decodeAudioData: it
// silently resamples to the AudioContext's own rate, which would corrupt the
// fs_delay/fs_time-relative math the channel replay depends on.

export interface WavData {
  sampleRate: number;
  /** Mono only -- multi-channel WAVs are down-mixed to channel 0. */
  samples: Float64Array;
}

function readString(view: DataView, offset: number, length: number): string {
  let s = "";
  for (let i = 0; i < length; i++) s += String.fromCharCode(view.getUint8(offset + i));
  return s;
}

export function parseWav(buf: ArrayBuffer): WavData {
  const view = new DataView(buf);
  if (readString(view, 0, 4) !== "RIFF" || readString(view, 8, 4) !== "WAVE") {
    throw new Error("Not a valid WAV file (missing RIFF/WAVE header)");
  }

  let pos = 12;
  let fmt: { audioFormat: number; numChannels: number; sampleRate: number; bitsPerSample: number } | null = null;
  let dataOffset = -1;
  let dataLength = 0;

  while (pos + 8 <= view.byteLength) {
    const chunkId = readString(view, pos, 4);
    const chunkSize = view.getUint32(pos + 4, true);
    const bodyStart = pos + 8;

    if (chunkId === "fmt ") {
      fmt = {
        audioFormat: view.getUint16(bodyStart, true),
        numChannels: view.getUint16(bodyStart + 2, true),
        sampleRate: view.getUint32(bodyStart + 4, true),
        bitsPerSample: view.getUint16(bodyStart + 14, true),
      };
    } else if (chunkId === "data") {
      dataOffset = bodyStart;
      dataLength = chunkSize;
    }

    pos = bodyStart + chunkSize + (chunkSize % 2); // chunks are word-aligned
  }

  if (!fmt) throw new Error("WAV file has no fmt chunk");
  if (dataOffset < 0) throw new Error("WAV file has no data chunk");

  const { numChannels, sampleRate, bitsPerSample, audioFormat } = fmt;
  const bytesPerSample = bitsPerSample / 8;
  const frameCount = Math.floor(dataLength / (bytesPerSample * numChannels));
  const samples = new Float64Array(frameCount);

  const readSample: (byteOffset: number) => number = (() => {
    if (audioFormat === 3 && bitsPerSample === 32) return (o: number) => view.getFloat32(o, true);
    if (audioFormat === 1 && bitsPerSample === 16) return (o: number) => view.getInt16(o, true) / 32768;
    if (audioFormat === 1 && bitsPerSample === 8) return (o: number) => (view.getUint8(o) - 128) / 128;
    if (audioFormat === 1 && bitsPerSample === 32) return (o: number) => view.getInt32(o, true) / 2147483648;
    throw new Error(`Unsupported WAV format: audioFormat=${audioFormat}, bitsPerSample=${bitsPerSample}`);
  })();

  for (let i = 0; i < frameCount; i++) {
    let sum = 0;
    for (let c = 0; c < numChannels; c++) {
      sum += readSample(dataOffset + (i * numChannels + c) * bytesPerSample);
    }
    samples[i] = sum / numChannels;
  }

  return { sampleRate, samples };
}

/** Writes an N-channel, interleaved 32-bit float PCM WAV file. `channels` is
 * one sample array per channel (all the same length) -- e.g. one per
 * replayed hydrophone, so the output round-trips as a single file instead of
 * one mono WAV per channel. */
export function writeWavMultiChannel(sampleRate: number, channels: ArrayLike<number>[]): ArrayBuffer {
  const numChannels = channels.length;
  const n = numChannels > 0 ? channels[0].length : 0;
  const dataSize = n * numChannels * 4;
  const buf = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buf);

  function writeString(offset: number, s: string) {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 3, true); // IEEE float
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * 4, true); // byte rate
  view.setUint16(32, numChannels * 4, true); // block align
  view.setUint16(34, 32, true); // bits per sample
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < numChannels; c++) {
      view.setFloat32(offset, channels[c][i], true);
      offset += 4;
    }
  }

  return buf;
}

/** Writes mono 32-bit float PCM, so the exact output values round-trip
 * without the quantization a 16-bit integer format would introduce. */
export function writeWav(sampleRate: number, samples: ArrayLike<number>): ArrayBuffer {
  return writeWavMultiChannel(sampleRate, [samples]);
}
