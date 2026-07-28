import { useEffect, useMemo, useRef } from "react";
import { colorFor, colorbarGradientCss, peakOf } from "../lib/colormap";

// CIR taps decay fast relative to a spectrogram's noise floor, so a smaller
// dynamic range than the spectrogram's (70 dB) keeps the multipath structure
// visible instead of washed out at the top of the scale.
const DYNAMIC_RANGE_DB = 40;

export interface CirData {
  /** Delay taps (rows). */
  l: number;
  /** Time snapshots (columns). */
  t: number;
  /** `abs(h_hat(:, m_idx, :))`, flat column-major: `magnitude[l_idx + l*t_idx]`. */
  magnitude: number[];
}

function magnitudeToDb(magnitude: number[]): Float64Array {
  const db = new Float64Array(magnitude.length);
  for (let i = 0; i < magnitude.length; i++) {
    db[i] = 20 * Math.log10(Math.max(magnitude[i], 1e-12));
  }
  return db;
}

/** `imshow(squeeze(abs(h_hat(:, m_idx, :))))`: the time-varying CIR for one
 * receiver, delay taps on the y-axis (top = tap 1) and time snapshots on the
 * x-axis. `highlightRangeSec`, if given, draws a red rectangle over the
 * `[start, end]` span (seconds, on the channel's own time axis) that a
 * replay actually read from this CIR. */
export function CirCanvas({
  data,
  fsDelay,
  fsTime,
  label,
  highlightRangeSec,
}: {
  data: CirData | null;
  fsDelay: number;
  fsTime: number;
  label: string;
  highlightRangeSec?: [number, number] | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const db = useMemo(() => (data ? magnitudeToDb(data.magnitude) : null), [data]);
  const max = useMemo(() => (db ? peakOf(db) : 0), [db]);
  const floor = max - DYNAMIC_RANGE_DB;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data || !db || data.l === 0 || data.t === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = data.t;
    canvas.height = data.l;

    const img = ctx.createImageData(data.t, data.l);
    for (let tIdx = 0; tIdx < data.t; tIdx++) {
      for (let lIdx = 0; lIdx < data.l; lIdx++) {
        const v = db[lIdx + data.l * tIdx];
        const frac = (Math.max(v, floor) - floor) / DYNAMIC_RANGE_DB;
        const [r, g, b] = colorFor(frac);
        const idx = (lIdx * data.t + tIdx) * 4;
        img.data[idx] = r;
        img.data[idx + 1] = g;
        img.data[idx + 2] = b;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [data, db, floor]);

  const durationSec = data && fsTime > 0 ? data.t / fsTime : 0;
  const maxDelayMs = data && fsDelay > 0 ? ((data.l - 1) / fsDelay) * 1000 : 0;

  let highlightStyle: { left: string; width: string } | null = null;
  if (highlightRangeSec && durationSec > 0) {
    const [segStart, segEnd] = highlightRangeSec;
    const leftPct = Math.max(0, Math.min(1, segStart / durationSec)) * 100;
    const rightPct = Math.max(0, Math.min(1, segEnd / durationSec)) * 100;
    highlightStyle = { left: `${leftPct}%`, width: `${Math.max(0, rightPct - leftPct)}%` };
  }

  return (
    <div className="spectrogram">
      <h4>{label}</h4>
      {data && data.t > 0 && data.l > 0 ? (
        <div className="spectrogram-row">
          <div className="spectrogram-plot">
            <div className="plot-row">
              <div className="y-axis-labels">
                <span>0 ms</span>
                <span>{(maxDelayMs / 2).toFixed(1)} ms</span>
                <span>{maxDelayMs.toFixed(1)} ms</span>
              </div>
              <div className="plot-col">
                <div className="cir-canvas-wrap">
                  <canvas ref={canvasRef} className="spectrogram-canvas" style={{ imageRendering: "pixelated" }} />
                  {highlightStyle && <div className="cir-highlight" style={highlightStyle} title="Segment replayed" />}
                </div>
                <div className="time-axis">
                  <span>0s</span>
                  <span>{(durationSec / 2).toFixed(2)}s</span>
                  <span>{durationSec.toFixed(2)}s</span>
                </div>
              </div>
            </div>
          </div>
          <div className="colorbar">
            <div className="colorbar-gradient" style={{ background: colorbarGradientCss() }} />
            <div className="colorbar-labels">
              <span>{max.toFixed(0)} dB</span>
              <span>{(max - DYNAMIC_RANGE_DB / 2).toFixed(0)} dB</span>
              <span>{floor.toFixed(0)} dB</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="spectrogram-empty">No channel loaded</div>
      )}
    </div>
  );
}
