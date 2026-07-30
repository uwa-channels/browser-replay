import { useEffect, useRef, useState } from "react";

interface Props {
  /** Ready-to-play samples, or null to compute them via `prepare` on click. */
  samples?: ArrayLike<number> | null;
  /** Produces the samples on click when `samples` is null -- for output that's
   * only worth computing if you actually ask to hear it (the audible-band
   * replay). Resolving to null means "nothing to play", e.g. after an error. */
  prepare?: () => Promise<ArrayLike<number> | null>;
  /** Rate the samples were produced at, not the AudioContext's rate. */
  sampleRate: number;
  label?: string;
  disabled?: boolean;
  /** Called once playback actually starts, so the page can follow along (e.g.
   * switch the displayed spectrogram to the variant being heard). */
  onPlay?: () => void;
  /** Changing this stops playback: whatever is sounding has been superseded. */
  invalidateKey?: unknown;
}

/** Peak-normalizes to ±1 so the replayed output -- whose absolute scale is
 * arbitrary, and often far below or above unity after the channel gain and
 * added noise -- is audible without clipping. */
function peakNormalized(samples: ArrayLike<number>): Float32Array<ArrayBuffer> {
  const out = new Float32Array(new ArrayBuffer(samples.length * 4));
  let peak = 0;
  for (let i = 0; i < samples.length; i++) {
    const a = Math.abs(samples[i]);
    if (a > peak) peak = a;
  }
  const gain = peak > 0 ? 1 / peak : 1;
  for (let i = 0; i < samples.length; i++) out[i] = samples[i] * gain;
  return out;
}

export function PlayButton({ samples, prepare, sampleRate, label = "Play", disabled, onPlay, invalidateKey }: Props) {
  const ctxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<AudioBufferSourceNode | null>(null);
  const [playing, setPlaying] = useState(false);
  const [preparing, setPreparing] = useState(false);

  function stop() {
    const source = sourceRef.current;
    if (source) {
      source.onended = null; // don't let our own stop() re-enter through onended
      source.stop();
      source.disconnect();
      sourceRef.current = null;
    }
    setPlaying(false);
  }

  // A new run replaces the samples under us, so whatever is playing is stale.
  useEffect(() => stop, [samples, invalidateKey]);

  // Contexts are a limited per-page resource, so release ours on unmount.
  useEffect(() => {
    return () => {
      void ctxRef.current?.close();
      ctxRef.current = null;
    };
  }, []);

  async function play() {
    let data = samples ?? null;
    if (!data && prepare) {
      setPreparing(true);
      try {
        data = await prepare();
      } finally {
        setPreparing(false);
      }
    }
    if (!data || data.length === 0) return;
    stop();

    let ctx = ctxRef.current;
    if (!ctx || ctx.state === "closed") {
      try {
        // Prefer a context at the signal's own rate: no resampling, so what
        // you hear is exactly the samples that get written to the WAV.
        ctx = new AudioContext({ sampleRate });
      } catch {
        // Rate outside what the platform's audio output accepts -- fall back
        // to the default rate and let the source node resample.
        ctx = new AudioContext();
      }
      ctxRef.current = ctx;
    }
    if (ctx.state === "suspended") await ctx.resume();

    const normalized = peakNormalized(data);
    // Tagged with the signal's rate regardless of the context's, so playback
    // keeps its real duration and pitch even on the fallback path.
    const buffer = ctx.createBuffer(1, normalized.length, sampleRate);
    buffer.copyToChannel(normalized, 0);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.onended = () => {
      if (sourceRef.current === source) sourceRef.current = null;
      setPlaying(false);
    };
    sourceRef.current = source;
    source.start();
    setPlaying(true);
    onPlay?.();
  }

  const nothingToPlay = (!samples || samples.length === 0) && !prepare;
  return (
    <button
      type="button"
      onClick={() => (playing ? stop() : void play())}
      disabled={disabled || nothingToPlay || preparing}
      title="Playback is peak-normalized to ±1"
    >
      {preparing ? "Preparing…" : playing ? "Stop" : `▶ ${label}`}
    </button>
  );
}
