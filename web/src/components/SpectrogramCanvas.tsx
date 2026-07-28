import { useEffect, useMemo, useRef, useState } from "react";
import type { SpectrogramResult } from "../lib/workerClient";
import { colorFor, colorbarGradientCss, peakOf } from "../lib/colormap";

// Fixed dynamic range below the peak, not the raw min/max: near-silent
// frames (e.g. zero-padding at the signal's edges) can sit 100+ dB below the
// signal, and normalizing against that true min squashes all the actual
// signal content up near the top of the scale (all yellow).
const DYNAMIC_RANGE_DB = 70;

export function SpectrogramCanvas({
  spec,
  label,
  fc,
  fsDelay,
}: {
  spec: SpectrogramResult | null;
  label: string;
  /** Channel carrier frequency (Hz), for the "zoom to bandwidth" button. */
  fc?: number;
  /** Channel fs_delay (Hz): the effective bandwidth is fs_delay / 2. */
  fsDelay?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [zoomed, setZoomed] = useState(false);
  const max = useMemo(() => (spec ? peakOf(spec.powerDb) : 0), [spec]);
  const floor = max - DYNAMIC_RANGE_DB;

  const zoomRange: [number, number] | null =
    fc != null && fsDelay != null && fsDelay > 0 ? [fc - fsDelay / 4, fc + fsDelay / 4] : null;

  // Index bounds (inclusive) into spec.freqs for the currently displayed band.
  const { loIdx, hiIdx } = useMemo(() => {
    if (!spec || spec.nFreq === 0) return { loIdx: 0, hiIdx: 0 };
    if (!zoomed || !zoomRange) return { loIdx: 0, hiIdx: spec.nFreq - 1 };
    const step = spec.freqs.length > 1 ? spec.freqs[1] - spec.freqs[0] : 1;
    const lo = Math.min(Math.max(Math.round(zoomRange[0] / step), 0), spec.nFreq - 1);
    const hi = Math.min(Math.max(Math.round(zoomRange[1] / step), 0), spec.nFreq - 1);
    return lo <= hi ? { loIdx: lo, hiIdx: hi } : { loIdx: 0, hiIdx: spec.nFreq - 1 };
  }, [spec, zoomed, zoomRange]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !spec || spec.nTime === 0 || spec.nFreq === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const bandFreq = hiIdx - loIdx + 1;
    canvas.width = spec.nTime;
    canvas.height = bandFreq;

    const img = ctx.createImageData(spec.nTime, bandFreq);
    for (let ti = 0; ti < spec.nTime; ti++) {
      for (let fi = loIdx; fi <= hiIdx; fi++) {
        const v = spec.powerDb[ti * spec.nFreq + fi];
        const t = (Math.max(v, floor) - floor) / DYNAMIC_RANGE_DB;
        const [r, g, b] = colorFor(t);
        // Flip vertically so low frequencies are at the bottom.
        const row = hiIdx - fi;
        const idx = (row * spec.nTime + ti) * 4;
        img.data[idx] = r;
        img.data[idx + 1] = g;
        img.data[idx + 2] = b;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [spec, floor, loIdx, hiIdx]);

  const freqLoKhz = spec ? spec.freqs[loIdx] / 1000 : 0;
  const freqHiKhz = spec ? spec.freqs[hiIdx] / 1000 : 0;

  return (
    <div className="spectrogram">
      <div className="spectrogram-header">
        <h4>{label}</h4>
        {zoomRange && (
          <button className="zoom-button" onClick={() => setZoomed((z) => !z)} disabled={!spec || spec.nTime === 0}>
            {zoomed ? "Zoom out" : "Zoom to bandwidth"}
          </button>
        )}
      </div>
      {spec && spec.nTime > 0 ? (
        <div className="spectrogram-row">
          <div className="spectrogram-plot">
            <div className="plot-row">
              <div className="y-axis-labels">
                <span>{freqHiKhz.toFixed(1)} kHz</span>
                <span>{((freqHiKhz + freqLoKhz) / 2).toFixed(1)} kHz</span>
                <span>{freqLoKhz.toFixed(1)} kHz</span>
              </div>
              <div className="plot-col">
                <canvas ref={canvasRef} className="spectrogram-canvas" style={{ imageRendering: "pixelated" }} />
                <div className="time-axis">
                  <span>{spec.times[0].toFixed(2)}s</span>
                  <span>{spec.times[Math.floor(spec.times.length / 2)].toFixed(2)}s</span>
                  <span>{spec.times[spec.times.length - 1].toFixed(2)}s</span>
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
        <div className="spectrogram-empty">No signal yet</div>
      )}
    </div>
  );
}
