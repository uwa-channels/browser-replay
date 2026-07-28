// A small viridis-like perceptually-ordered colormap (dark purple -> teal ->
// yellow), interpolated between a handful of control points -- avoids
// pulling in a colormap library for a single gradient. Shared by the
// spectrogram and CIR canvases so their color scales read as one system.
const STOPS: [number, number, number][] = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
];

/** `t` in `[0, 1]` -> an `[r, g, b]` triple (0-255). */
export function colorFor(t: number): [number, number, number] {
  const clamped = Math.min(1, Math.max(0, t));
  const scaled = clamped * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(scaled));
  const frac = scaled - i;
  const [r0, g0, b0] = STOPS[i];
  const [r1, g1, b1] = STOPS[i + 1];
  return [r0 + (r1 - r0) * frac, g0 + (g1 - g0) * frac, b0 + (b1 - b0) * frac];
}

export function colorbarGradientCss(): string {
  const stops = STOPS.map(([r, g, b], i) => {
    const pct = (i / (STOPS.length - 1)) * 100;
    return `rgb(${r}, ${g}, ${b}) ${pct}%`;
  });
  return `linear-gradient(to top, ${stops.join(", ")})`;
}

export function peakOf(values: ArrayLike<number>): number {
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > max) max = values[i];
  }
  return max;
}
