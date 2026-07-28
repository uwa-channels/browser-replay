import type { ChannelEntry } from "../lib/zenodo";
import { formatChannelBand } from "../lib/channelBands";

function formatSize(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(0)} KB`;
  return `${bytes} B`;
}

export function ChannelPicker({
  entries,
  selected,
  onSelect,
  disabled,
}: {
  entries: ChannelEntry[];
  selected: string | null;
  onSelect: (name: string) => void;
  disabled: boolean;
}) {
  return (
    <select
      className="channel-picker"
      value={selected ?? ""}
      disabled={disabled || entries.length === 0}
      onChange={(e) => onSelect(e.target.value)}
    >
      <option value="" disabled>
        {entries.length === 0 ? "Loading catalog…" : "Select a channel…"}
      </option>
      {entries.map((e) => {
        const band = formatChannelBand(e.color);
        return (
          <option key={e.name} value={e.name}>
            {e.name} ({formatSize(e.file.size)}){band ? ` — ${band}` : ""}
            {e.noiseFile ? "" : " — no paired noise file"}
          </option>
        );
      })}
    </select>
  );
}
