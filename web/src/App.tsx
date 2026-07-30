import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import { ChannelPicker } from "./components/ChannelPicker";
import { PlayButton } from "./components/PlayButton";
import { SpectrogramCanvas } from "./components/SpectrogramCanvas";
import { CirCanvas, type CirData } from "./components/CirCanvas";
import { buildChannelList, downloadFile, fetchCatalog, type ChannelEntry } from "./lib/zenodo";
import { parseWav, writeWavMultiChannel } from "./lib/wav";
import { ReplayWorkerClient, type ChannelInfo, type NoiseInfo, type SpectrogramResult } from "./lib/workerClient";

type NoiseMode = "none" | "pink" | "mixing";

/** Everything a replay run needs, kept so the same run can be recomputed at a
 * different carrier (the audible-band option) without re-randomizing the start
 * offset or the noise realization. */
interface RunParams {
  samples: Float64Array;
  sampleRate: number;
  arrayIndex: number[];
  start: string;
  noiseMode: NoiseMode;
  noiseScale: number;
  noiseSeed: string;
}

function randomSeed(): string {
  return BigInt(Math.floor(Math.random() * 2 ** 32)).toString();
}

function formatHz(hz: number): string {
  return hz >= 1000 ? `${Number((hz / 1000).toFixed(3))} kHz` : `${Math.round(hz)} Hz`;
}

// Bigger window for finer frequency bins, finer hop (75% overlap) for more
// time frames -- more resolution on both axes at the cost of more STFT frames
// to compute (still fast: this is a WASM STFT over at most a few hundred
// thousand samples). Shared by every spectrogram of a given run so they're
// directly comparable.
function specParams(inputLen: number): { windowLen: number; hop: number } {
  const windowLen = Math.min(2048, Math.max(128, Math.floor(inputLen / 6)));
  return { windowLen, hop: Math.max(1, Math.floor(windowLen / 4)) };
}

function parseArrayIndex(text: string): number[] {
  return text
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => parseInt(s, 10) - 1) // user enters 1-based indices
    .filter((n) => Number.isInteger(n) && n >= 0);
}

// 1-based receiver indices spread as evenly as possible across [1, m],
// always including the first and last receiver.
function evenlySpacedIndices(m: number, count: number): number[] {
  if (count <= 1) return [1];
  const indices: number[] = [];
  for (let i = 0; i < count; i++) {
    indices.push(Math.round(1 + (i * (m - 1)) / (count - 1)));
  }
  return indices;
}

function App() {
  const workerRef = useRef<ReplayWorkerClient | null>(null);
  if (!workerRef.current) workerRef.current = new ReplayWorkerClient();
  // Bumped by every channel load (Zenodo or local) so a stale in-flight load
  // -- e.g. the auto-restored channel from a previous session, still
  // downloading -- can detect it's been superseded and stop applying its
  // results once a newer one has started.
  const loadSeqRef = useRef(0);
  // Lets the Cancel button abort whichever Zenodo fetch is currently in flight.
  const downloadAbortRef = useRef<AbortController | null>(null);

  const [entries, setEntries] = useState<ChannelEntry[]>([]);
  // What's chosen in the dropdown, independent of whether it's been downloaded yet.
  const [pendingName, setPendingName] = useState<string | null>(null);
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [localChannelName, setLocalChannelName] = useState<string | null>(null);
  const [localNoiseName, setLocalNoiseName] = useState<string | null>(null);
  const [channelProgress, setChannelProgress] = useState<number | null>(null);
  const [channelInfo, setChannelInfo] = useState<ChannelInfo | null>(null);
  const [noiseInfo, setNoiseInfo] = useState<NoiseInfo | null>(null);
  const [cirData, setCirData] = useState<CirData | null>(null);
  const [replaySegment, setReplaySegment] = useState<[number, number] | null>(null);

  // Display/download label: whichever source (Zenodo catalog or a local
  // file) most recently loaded a channel.
  const channelLabel = localChannelName ?? selectedName;

  const [signal, setSignal] = useState<{ sampleRate: number; samples: Float64Array; name: string } | null>(null);

  const [arrayIndexText, setArrayIndexText] = useState("1");
  const [noiseMode, setNoiseMode] = useState<NoiseMode>("none");
  const [noiseScale, setNoiseScale] = useState(0.05);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [output, setOutput] = useState<{ rows: number; cols: number; flat: number[] } | null>(null);
  const [inputSpec, setInputSpec] = useState<SpectrogramResult | null>(null);
  const [outputSpec, setOutputSpec] = useState<SpectrogramResult | null>(null);
  const [lastRun, setLastRun] = useState<RunParams | null>(null);
  // Bumped by every completed run, so playback of a superseded result stops
  // and the cached audible-band replay below is recognized as stale.
  const [runId, setRunId] = useState(0);
  // Channel 1 of the audible-band (fc = B/2) replay. Only computed when it's
  // actually asked for, and kept in a ref because nothing renders from it.
  const audibleRef = useRef<{ runId: number; samples: number[] } | null>(null);
  const [audibleSpec, setAudibleSpec] = useState<SpectrogramResult | null>(null);
  // Which output variant the results section is showing: the replay at the
  // channel file's fc, or the audible-band one. Follows whichever was played.
  const [showAudible, setShowAudible] = useState(false);

  useEffect(() => {
    fetchCatalog()
      .then((files) => setEntries(buildChannelList(files)))
      .catch((err) => setError(`Failed to load Zenodo catalog: ${err.message}`));
  }, []);

  async function loadChannelEntry(entry: ChannelEntry) {
    const seq = ++loadSeqRef.current;
    setSelectedName(entry.name);
    setLocalChannelName(null);
    setLocalNoiseName(null);
    setChannelInfo(null);
    setNoiseInfo(null);
    setCirData(null);
    setReplaySegment(null);
    setError(null);

    setChannelProgress(0);
    const controller = new AbortController();
    downloadAbortRef.current = controller;
    try {
      const worker = workerRef.current!;
      const buf = await downloadFile(
        entry.file,
        (loaded, total) => {
          if (loadSeqRef.current === seq) setChannelProgress(total > 0 ? loaded / total : null);
        },
        controller.signal,
      );
      if (loadSeqRef.current !== seq) return;
      setChannelProgress(1);
      const info = await worker.loadChannel(buf, (fraction) => {
        if (loadSeqRef.current === seq) setChannelProgress(fraction);
      });
      if (loadSeqRef.current !== seq) return;
      setChannelInfo(info);
      const cir = await worker.cirMagnitude(0);
      if (loadSeqRef.current !== seq) return;
      setCirData({ l: info.l, t: info.t, magnitude: cir.magnitude });

      if (entry.noiseFile) {
        const noiseBuf = await downloadFile(entry.noiseFile, undefined, controller.signal);
        if (loadSeqRef.current !== seq) return;
        const nInfo = await worker.loadNoise(noiseBuf);
        if (loadSeqRef.current !== seq) return;
        setNoiseInfo(nInfo);
      }
    } catch (err) {
      const cancelled = err instanceof DOMException && err.name === "AbortError";
      if (loadSeqRef.current === seq && !cancelled) setError(err instanceof Error ? err.message : String(err));
    } finally {
      if (loadSeqRef.current === seq) {
        setChannelProgress(null);
        downloadAbortRef.current = null;
      }
    }
  }

  async function handleDownloadChannel() {
    const entry = entries.find((e) => e.name === pendingName);
    if (!entry) return;
    await loadChannelEntry(entry);
  }

  function handleCancelDownload() {
    // Bump the generation first so the aborted (or already-past-fetch)
    // load's remaining awaits see themselves as stale and skip applying
    // any results, then actually stop the in-flight network request.
    loadSeqRef.current++;
    downloadAbortRef.current?.abort();
    downloadAbortRef.current = null;
    setChannelProgress(null);
  }

  async function handleLoadLocalChannel(file: File) {
    const seq = ++loadSeqRef.current;
    setPendingName(null);
    setSelectedName(null);
    setLocalChannelName(file.name.replace(/\.mat$/i, ""));
    setLocalNoiseName(null);
    setChannelInfo(null);
    setNoiseInfo(null);
    setCirData(null);
    setReplaySegment(null);
    setError(null);
    setChannelProgress(0);
    try {
      const worker = workerRef.current!;
      const buf = await file.arrayBuffer();
      if (loadSeqRef.current !== seq) return;
      const info = await worker.loadChannel(buf, (fraction) => {
        if (loadSeqRef.current === seq) setChannelProgress(fraction);
      });
      if (loadSeqRef.current !== seq) return;
      setChannelInfo(info);
      const cir = await worker.cirMagnitude(0);
      if (loadSeqRef.current !== seq) return;
      setCirData({ l: info.l, t: info.t, magnitude: cir.magnitude });
    } catch (err) {
      if (loadSeqRef.current === seq) setError(err instanceof Error ? err.message : String(err));
    } finally {
      if (loadSeqRef.current === seq) setChannelProgress(null);
    }
  }

  async function handleLoadLocalNoise(file: File) {
    const seq = loadSeqRef.current;
    setError(null);
    try {
      const buf = await file.arrayBuffer();
      const nInfo = await workerRef.current!.loadNoise(buf);
      if (loadSeqRef.current !== seq) return;
      setNoiseInfo(nInfo);
      setLocalNoiseName(file.name.replace(/\.mat$/i, ""));
    } catch (err) {
      if (loadSeqRef.current === seq) setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleFileUpload(file: File) {
    setError(null);
    try {
      const buf = await file.arrayBuffer();
      const wav = parseWav(buf);
      setSignal({ sampleRate: wav.sampleRate, samples: wav.samples, name: file.name });
      setOutput(null);
      setInputSpec(null);
      setOutputSpec(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  /** Runs the replay for `p` and adds its noise, at the channel's own carrier
   * (`fcOverride` null) or at a substituted one. */
  async function computeOutput(p: RunParams, fcOverride: number | null) {
    const worker = workerRef.current!;
    const input = Array.from(p.samples);
    const replayResult = await worker.runReplay(input, p.sampleRate, p.arrayIndex, p.start, fcOverride);

    let flat = replayResult.flat;
    if (p.noiseMode === "pink") {
      const n = await worker.runNoisePink(replayResult.rows, replayResult.cols, p.sampleRate, p.noiseSeed);
      flat = flat.map((v, i) => v + p.noiseScale * n.flat[i]);
    } else if (p.noiseMode === "mixing") {
      const n = await worker.runNoiseMixing(replayResult.rows, p.arrayIndex, p.sampleRate, p.noiseSeed);
      flat = flat.map((v, i) => v + p.noiseScale * n.flat[i]);
    }
    return { rows: replayResult.rows, cols: replayResult.cols, flat };
  }

  async function handleRun() {
    if (!channelInfo || !signal) return;
    setBusy(true);
    setError(null);
    try {
      const worker = workerRef.current!;
      const arrayIndex = parseArrayIndex(arrayIndexText);
      if (arrayIndex.length === 0) throw new Error("Enter at least one hydrophone index (e.g. 1 or 1,2,3)");
      if (noiseMode === "mixing" && !noiseInfo) throw new Error("This channel has no paired noise file for the mixing model");

      const inputLen = signal.samples.length;
      const [lo, hi] = await worker.validStartRange(signal.sampleRate, inputLen);
      const loBig = BigInt(lo);
      const hiBig = BigInt(hi);
      const span = hiBig - loBig;
      const start = span > 0n ? (loBig + BigInt(Math.floor(Math.random() * Number(span)))).toString() : loBig.toString();

      const [segStart, segEnd] = await worker.replayTimeRange(signal.sampleRate, inputLen, start);
      setReplaySegment([segStart, segEnd]);

      const params: RunParams = {
        samples: signal.samples,
        sampleRate: signal.sampleRate,
        arrayIndex,
        start,
        noiseMode,
        noiseScale,
        noiseSeed: randomSeed(),
      };
      const result = await computeOutput(params, null);
      setOutput(result);
      setLastRun(params);
      // The previous run's shifted replay no longer applies.
      audibleRef.current = null;
      setAudibleSpec(null);
      setShowAudible(false);
      setRunId((n) => n + 1);

      const input = Array.from(signal.samples);
      const firstCol = result.flat.slice(0, result.rows);
      const { windowLen, hop } = specParams(input.length);

      const [inSpec, outSpec] = await Promise.all([
        worker.spectrogram(input, signal.sampleRate, windowLen, hop),
        worker.spectrogram(firstCol, signal.sampleRate, windowLen, hop),
      ]);
      setInputSpec(inSpec);
      setOutputSpec(outSpec);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  // Channel 1 of the output -- the channel whose spectrogram is shown, and the
  // one the play button auditions.
  const outputCh1 = useMemo(() => (output ? output.flat.slice(0, output.rows) : null), [output]);

  // The channel's bandwidth B is fs_delay / 2 (the delay domain is sampled at
  // two samples per symbol -- the same relationship SpectrogramCanvas's
  // zoom-to-bandwidth assumes), so a carrier of B/2 puts the replayed band at
  // [0, B]: audible, unlike the measured carrier.
  const bandwidth = channelInfo ? channelInfo.fsDelay / 2 : null;
  const audibleFc = bandwidth != null ? bandwidth / 2 : null;

  /** Re-runs the last replay with fc = B/2 (same start offset, same noise
   * realization, so the carrier is the only difference), returning channel 1
   * and computing its spectrogram. Both are cached for as long as that run is
   * the current one. */
  async function prepareAudibleBand(): Promise<number[] | null> {
    if (!lastRun || audibleFc == null) return null;
    const cached = audibleRef.current;
    if (cached && cached.runId === runId) return cached.samples;
    setBusy(true);
    setError(null);
    try {
      const result = await computeOutput(lastRun, audibleFc);
      const ch1 = result.flat.slice(0, result.rows);
      audibleRef.current = { runId, samples: ch1 };
      const { windowLen, hop } = specParams(lastRun.samples.length);
      setAudibleSpec(await workerRef.current!.spectrogram(ch1, lastRun.sampleRate, windowLen, hop));
      return ch1;
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      return null;
    } finally {
      setBusy(false);
    }
  }

  function downloadOutput() {
    if (!output || !signal) return;
    const channels: number[][] = [];
    for (let c = 0; c < output.cols; c++) {
      channels.push(output.flat.slice(c * output.rows, (c + 1) * output.rows));
    }
    const buf = writeWavMultiChannel(signal.sampleRate, channels);
    const blob = new Blob([buf], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${channelLabel ?? "replay"}_output.wav`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="app">
      <header>
        <h1>Channel Replay</h1>
        <p>
          Replay a signal through a real, measured underwater acoustic channel — entirely in your browser, WASM-accelerated.
          Data from{" "}
          <a href="https://zenodo.org/records/21287414" target="_blank" rel="noreferrer">
            Zenodo
          </a>
          .
        </p>
      </header>

      {error && <div className="error">{error}</div>}

      <section className="card">
        <h2>1. Pick a channel</h2>
        <ChannelPicker
          entries={entries}
          selected={pendingName}
          onSelect={setPendingName}
          disabled={busy || channelProgress != null}
        />
        <button onClick={handleDownloadChannel} disabled={busy || channelProgress != null || !pendingName}>
          Download channel
        </button>

        <div className="divider">or load local files</div>
        <label>
          Channel .mat file:
          <input
            type="file"
            accept=".mat"
            disabled={busy || channelProgress != null}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void handleLoadLocalChannel(f);
            }}
          />
        </label>
        <label>
          Noise .mat file (optional):
          <input
            type="file"
            accept=".mat"
            disabled={busy || channelProgress != null || !channelInfo}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void handleLoadLocalNoise(f);
            }}
          />
        </label>

        {channelProgress != null && (
          <div className="progress-row">
            <div className="progress">
              <div className="progress-bar" style={{ width: `${Math.round(channelProgress * 100)}%` }} />
              <span>{Math.round(channelProgress * 100)}%</span>
            </div>
            <button onClick={handleCancelDownload}>Cancel</button>
          </div>
        )}
        {channelInfo && (
          <table className="channel-info-table">
            <tbody>
              <tr>
                <td>Channel</td>
                <td>{channelLabel}</td>
              </tr>
              <tr>
                <td>Delay taps (L)</td>
                <td>{channelInfo.l}</td>
              </tr>
              <tr>
                <td>Receivers (M)</td>
                <td>{channelInfo.m}</td>
              </tr>
              <tr>
                <td>Time snapshots (T)</td>
                <td>{channelInfo.t}</td>
              </tr>
              <tr>
                <td>fs_delay</td>
                <td>{channelInfo.fsDelay} Hz</td>
              </tr>
              <tr>
                <td>fs_time</td>
                <td>{channelInfo.fsTime} Hz</td>
              </tr>
              <tr>
                <td>fc</td>
                <td>{channelInfo.fc} Hz</td>
              </tr>
              <tr>
                <td>Tracking</td>
                <td>
                  {channelInfo.trackingKind ?? "none"}
                  {channelInfo.hasFResamp ? " + f_resamp" : ""}
                </td>
              </tr>
              <tr>
                <td>Noise</td>
                <td>{noiseInfo ? `α=${noiseInfo.alpha}${localNoiseName ? ` (${localNoiseName})` : ""}` : "not loaded"}</td>
              </tr>
            </tbody>
          </table>
        )}
        {channelInfo && (
          <CirCanvas
            data={cirData}
            fsDelay={channelInfo.fsDelay}
            fsTime={channelInfo.fsTime}
            label="Time-varying CIR, |h(τ, t)| (receiver 1)"
          />
        )}
      </section>

      <section className="card">
        <h2>2. Upload a passband signal (WAV)</h2>
        <input
          type="file"
          accept="audio/wav,.wav"
          disabled={busy}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void handleFileUpload(f);
          }}
        />
        {signal && (
          <div className="info">
            {signal.name}: {signal.samples.length} samples @ {signal.sampleRate} Hz ({(signal.samples.length / signal.sampleRate).toFixed(2)}s)
          </div>
        )}
      </section>

      <section className="card">
        <h2>3. Replay</h2>
        <label>
          Hydrophone index/indices (1-based, comma-separated):
          <input type="text" value={arrayIndexText} onChange={(e) => setArrayIndexText(e.target.value)} disabled={busy} />
        </label>
        {channelInfo && (
          <div className="preset-buttons">
            <button type="button" onClick={() => setArrayIndexText("1")} disabled={busy}>
              First channel
            </button>
            <button type="button" onClick={() => setArrayIndexText(`${channelInfo.m}`)} disabled={busy}>
              Last channel
            </button>
            <button
              type="button"
              onClick={() =>
                setArrayIndexText(Array.from({ length: channelInfo.m }, (_, i) => i + 1).join(","))
              }
              disabled={busy}
            >
              All channels
            </button>
            <button
              type="button"
              onClick={() => setArrayIndexText(evenlySpacedIndices(channelInfo.m, 2).join(","))}
              disabled={busy || channelInfo.m < 2}
            >
              Max spaced (2)
            </button>
            <button
              type="button"
              onClick={() => setArrayIndexText(evenlySpacedIndices(channelInfo.m, 3).join(","))}
              disabled={busy || channelInfo.m < 3}
            >
              Max spaced (3)
            </button>
            <button
              type="button"
              onClick={() => setArrayIndexText(evenlySpacedIndices(channelInfo.m, 4).join(","))}
              disabled={busy || channelInfo.m < 4}
            >
              Max spaced (4)
            </button>
          </div>
        )}
        <label>
          Noise:
          <select value={noiseMode} onChange={(e) => setNoiseMode(e.target.value as NoiseMode)} disabled={busy}>
            <option value="none">None</option>
            <option value="pink">Textbook pink noise</option>
            <option value="mixing" disabled={!noiseInfo}>
              Measured noise (mixing model)
            </option>
          </select>
        </label>
        {noiseMode !== "none" && (
          <label>
            Noise scale:
            <input type="number" step="0.01" value={noiseScale} onChange={(e) => setNoiseScale(parseFloat(e.target.value))} disabled={busy} />
          </label>
        )}
        <button onClick={handleRun} disabled={busy || !channelInfo || !signal}>
          {busy ? "Running…" : "Run replay"}
        </button>
      </section>

      {(inputSpec || outputSpec) && (
        <section className="card">
          <h2>4. Results</h2>
          {channelInfo && cirData && (
            <CirCanvas
              data={cirData}
              fsDelay={channelInfo.fsDelay}
              fsTime={channelInfo.fsTime}
              label="Replayed CIR segment (receiver 1)"
              highlightRangeSec={replaySegment}
            />
          )}
          <div className="spectrograms">
            <SpectrogramCanvas
              spec={inputSpec}
              label="Input signal"
              fc={channelInfo?.fc}
              fsDelay={channelInfo?.fsDelay}
            />
            <SpectrogramCanvas
              spec={showAudible ? audibleSpec : outputSpec}
              label={
                showAudible && bandwidth != null && audibleFc != null
                  ? `Output signal (channel 1), shifted to 0–${formatHz(bandwidth)} (fc = ${formatHz(audibleFc)})`
                  : `Output signal (channel 1)${channelInfo ? `, at the file's fc = ${formatHz(channelInfo.fc)}` : ""}`
              }
              fc={showAudible ? (audibleFc ?? undefined) : channelInfo?.fc}
              fsDelay={channelInfo?.fsDelay}
            />
          </div>
          {output && signal && (
            <>
              <div className="downloads">
                <PlayButton
                  samples={outputCh1}
                  sampleRate={signal.sampleRate}
                  label={output.cols > 1 ? "Play output (channel 1)" : "Play output"}
                  disabled={busy}
                  onPlay={() => setShowAudible(false)}
                  invalidateKey={runId}
                />
                {bandwidth != null && audibleFc != null && (
                  <PlayButton
                    prepare={prepareAudibleBand}
                    sampleRate={signal.sampleRate}
                    label={`Play shifted to 0–${formatHz(bandwidth)}`}
                    disabled={busy}
                    onPlay={() => setShowAudible(true)}
                    invalidateKey={runId}
                  />
                )}
                <button onClick={downloadOutput}>
                  Download output at the file's fc ({output.cols}-channel WAV)
                </button>
              </div>
              <div className="info">
                {showAudible && bandwidth != null && audibleFc != null ? (
                  <>
                    Showing the shifted replay: fc = B/2 = {formatHz(audibleFc)} instead of the file's fc ={" "}
                    {formatHz(channelInfo!.fc)}, so the output lands in 0–{formatHz(bandwidth)} and is audible. Same start offset and noise
                    realization as the run above, so the carrier is the only difference. Play the unshifted output to switch this plot back;
                    the download always writes the replay at the file's fc.
                  </>
                ) : (
                  <>
                    Showing the replay at the channel file's fc.
                    {bandwidth != null && audibleFc != null && (
                      <> Playing the shifted version re-runs it with fc = B/2 = {formatHz(audibleFc)} and updates this plot.</>
                    )}
                  </>
                )}{" "}
                Playback is peak-normalized to ±1; the downloaded WAV keeps the raw amplitudes.
              </div>
            </>
          )}
        </section>
      )}
    </div>
  );
}

export default App;
