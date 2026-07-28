// Static per-channel frequency parameters, so the picker can show a
// channel's supported band before it's downloaded (the .mat files carry the
// same numbers in params/fc, but only after downloading the whole file).
// Sourced from the channel repository paper (Table "Parameters of the
// experiments" + per-channel description text). Blue, Red, and Purple state
// their in-water acoustic band explicitly (10.5-15.5 kHz, 20-30 kHz, and
// 9.2-15.8 kHz respectively); for those, `bandExplicit` is true and the band
// is exactly fc +/- symbolRateKHz/2. The other channels only state fc and
// symbol rate, but that same fc +/- symbolRateKHz/2 relationship holds for
// all three explicit channels, so it's used to approximate their bands too.
export interface ChannelBand {
  fcKHz: number;
  symbolRateKHz: number;
  bandExplicit: boolean;
}

export const CHANNEL_BANDS: Record<string, ChannelBand> = {
  blue: { fcKHz: 13, symbolRateKHz: 5, bandExplicit: true },
  red: { fcKHz: 25, symbolRateKHz: 9.6, bandExplicit: true },
  purple: { fcKHz: 12.5, symbolRateKHz: 6.51, bandExplicit: true },
  yellow: { fcKHz: 13, symbolRateKHz: 6.25, bandExplicit: false },
  green: { fcKHz: 6, symbolRateKHz: 4.5, bandExplicit: false },
  black: { fcKHz: 18, symbolRateKHz: 12.5, bandExplicit: false },
  pink: { fcKHz: 6, symbolRateKHz: 4, bandExplicit: false },
  brown: { fcKHz: 0.075, symbolRateKHz: 0.0375, bandExplicit: false },
};

function formatKHz(khz: number): string {
  if (khz < 1) return `${Math.round(khz * 1000)} Hz`;
  return `${Number(khz.toFixed(3))} kHz`;
}

/** Human-readable band summary for a channel color, e.g.
 * "10.5-15.5 kHz (fc 13 kHz)", or null if the color isn't recognized. */
export function formatChannelBand(color: string): string | null {
  const band = CHANNEL_BANDS[color];
  if (!band) return null;
  const lo = band.fcKHz - band.symbolRateKHz / 2;
  const hi = band.fcKHz + band.symbolRateKHz / 2;
  const range = `${formatKHz(lo)}–${formatKHz(hi)}`;
  const approx = band.bandExplicit ? "" : "~";
  return `${approx}${range} (fc ${formatKHz(band.fcKHz)})`;
}
