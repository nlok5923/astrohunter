"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { CandidateCard } from "./components/CandidateCard";
import {
  buildPSV,
  downloadText,
  DetectResult,
  runDemo,
  runDetect,
} from "./lib/api";

type Status = "undecided" | "accept" | "reject";

export default function Home() {
  const [result, setResult] = useState<DetectResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statuses, setStatuses] = useState<Record<number, Status>>({});
  const [current, setCurrent] = useState(1);

  const [files, setFiles] = useState<File[]>([]);
  const [offline, setOffline] = useState(false);
  const [obscode, setObscode] = useState("XXX");
  const [observer, setObserver] = useState("astrohunt web");
  const [hintRa, setHintRa] = useState("");
  const [hintDec, setHintDec] = useState("");

  const reset = (r: DetectResult | null) => {
    setResult(r);
    setStatuses({});
    setCurrent(1);
    setError(null);
  };

  const doDemo = async () => {
    setLoading(true);
    setError(null);
    try {
      reset(await runDemo());
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const doDetect = async () => {
    if (files.length < 2) {
      setError("Select at least 2 FITS frames of the same field.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      reset(
        await runDetect(files, { offline, obscode, observer, hintRa, hintDec })
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const setStatus = useCallback((idx: number, s: Status) => {
    setStatuses((prev) => ({ ...prev, [idx]: s }));
    setCurrent(idx);
  }, []);

  const counts = useMemo(() => {
    const vals = Object.values(statuses);
    const acc = vals.filter((s) => s === "accept").length;
    const rej = vals.filter((s) => s === "reject").length;
    const n = result?.candidates.length ?? 0;
    return { acc, rej, undecided: n - acc - rej, n };
  }, [statuses, result]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!result) return;
      const n = result.candidates.length;
      if (e.key === "j") setCurrent((c) => Math.min(n, c + 1));
      else if (e.key === "k") setCurrent((c) => Math.max(1, c - 1));
      else if (e.key === "a") setStatus(current, "accept");
      else if (e.key === "r") setStatus(current, "reject");
      else if (e.key === "u") setStatus(current, "undecided");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [result, current, setStatus]);

  const exportAccepted = () => {
    if (!result) return;
    const accepted = new Set(
      Object.entries(statuses)
        .filter(([, s]) => s === "accept")
        .map(([i]) => Number(i))
    );
    downloadText("accepted.psv", buildPSV(result, accepted));
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="mx-auto max-w-5xl px-6 py-10 pb-28">
        <h1 className="bg-gradient-to-r from-amber-300 via-orange-400 to-fuchsia-500 bg-clip-text text-5xl font-bold tracking-tight text-transparent">
          🔭 astrohunt
        </h1>
        <p className="mt-2 max-w-2xl text-zinc-400">
          Find asteroids in telescope images — the same way NASA&apos;s citizen-science
          hunters do, but automated. Hit{" "}
          <span className="font-medium text-emerald-400">Run demo</span> to watch the
          pipeline spot a moving object, then vet it and export an MPC-ready ADES report.
        </p>

        <div className="mt-6 rounded-xl border border-zinc-800 bg-zinc-900/60 p-5">
          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={doDemo}
              disabled={loading}
              className="rounded-md bg-emerald-600 px-4 py-2 font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
            >
              ▶ Run demo (synthetic)
            </button>
            <span className="text-zinc-600">or</span>
            <input
              type="file"
              multiple
              accept=".fits,.fit,.fts"
              onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
              className="text-sm text-zinc-400 file:mr-3 file:rounded-md file:border-0 file:bg-zinc-800 file:px-3 file:py-2 file:text-zinc-100"
            />
            <button
              onClick={doDetect}
              disabled={loading}
              className="rounded-md bg-sky-600 px-4 py-2 font-medium text-white hover:bg-sky-500 disabled:opacity-50"
            >
              Detect {files.length > 0 ? `(${files.length})` : ""}
            </button>
          </div>

          <div className="mt-4 flex flex-wrap items-center gap-4 text-sm">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={offline}
                onChange={(e) => setOffline(e.target.checked)}
              />
              offline (no Gaia/SkyBoT)
            </label>
            <label className="flex items-center gap-1">
              obs code
              <input
                value={obscode}
                onChange={(e) => setObscode(e.target.value)}
                className="w-16 rounded bg-zinc-800 px-2 py-1"
              />
            </label>
            <label className="flex items-center gap-1">
              observer
              <input
                value={observer}
                onChange={(e) => setObserver(e.target.value)}
                className="w-40 rounded bg-zinc-800 px-2 py-1"
              />
            </label>
            <label className="flex items-center gap-1">
              hint RA
              <input
                value={hintRa}
                onChange={(e) => setHintRa(e.target.value)}
                placeholder="deg"
                className="w-20 rounded bg-zinc-800 px-2 py-1"
              />
            </label>
            <label className="flex items-center gap-1">
              hint Dec
              <input
                value={hintDec}
                onChange={(e) => setHintDec(e.target.value)}
                placeholder="deg"
                className="w-20 rounded bg-zinc-800 px-2 py-1"
              />
            </label>
          </div>
          <p className="mt-2 text-xs text-zinc-500">
            Keys: <kbd>j</kbd>/<kbd>k</kbd> move · <kbd>a</kbd> accept ·{" "}
            <kbd>r</kbd> reject · <kbd>u</kbd> reset
          </p>
        </div>

        {loading && (
          <p className="mt-6 animate-pulse text-sky-400">Running pipeline…</p>
        )}
        {error && (
          <pre className="mt-6 whitespace-pre-wrap rounded-lg border border-red-900 bg-red-950/40 p-4 text-sm text-red-300">
            {error}
          </pre>
        )}

        {result && (
          <>
            {result.field_gif && (
              <div className="mt-8 overflow-hidden rounded-2xl border border-zinc-800 bg-black">
                <div className="flex flex-col items-center gap-4 p-5 sm:flex-row sm:items-start">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`data:image/gif;base64,${result.field_gif}`}
                    alt="full-field blink"
                    className="rounded-lg border border-zinc-800"
                    style={{ width: 380, maxWidth: "100%" }}
                  />
                  <div className="flex-1">
                    <h2 className="text-lg font-semibold text-zinc-100">
                      Spot the mover 👀
                    </h2>
                    <p className="mt-1 text-sm text-zinc-400">
                      The stars stay fixed while one object{" "}
                      <span className="text-emerald-400">(circled)</span> creeps
                      across the field, frame by frame. That&apos;s the asteroid —
                      exactly what a human &ldquo;blinking&rdquo; the frames looks for,
                      found here automatically.
                    </p>
                    <p className="mt-3 text-xs text-zinc-500">
                      {result.n_candidates} candidate
                      {result.n_candidates === 1 ? "" : "s"} · astrometric
                      consistency {result.median_rms_arcsec.toFixed(3)}″ · asinh
                      stretch + inferno colormap
                    </p>
                  </div>
                </div>
              </div>
            )}

            <div className="mt-6 flex flex-wrap gap-6 rounded-xl border border-zinc-800 bg-zinc-900/60 p-5 text-sm">
              <Stat label="frames" value={result.frames.length} />
              <Stat
                label="astrometric consistency"
                value={`${result.median_rms_arcsec.toFixed(3)}″`}
              />
              <Stat label="candidates" value={result.n_candidates} />
              <Stat label="known" value={result.n_known} />
            </div>

            {result.warnings.length > 0 && (
              <ul className="mt-3 space-y-1 text-xs text-amber-400/80">
                {result.warnings.map((w, i) => (
                  <li key={i}>⚠ {w}</li>
                ))}
              </ul>
            )}

            <h2 className="mt-8 text-xl font-semibold">
              Candidates ({result.candidates.length})
            </h2>
            <div className="mt-4 space-y-4">
              {result.candidates.map((c) => (
                <CandidateCard
                  key={c.idx}
                  cand={c}
                  current={c.idx === current}
                  status={statuses[c.idx] ?? "undecided"}
                  onSet={(s) => setStatus(c.idx, s)}
                  onFocus={() => setCurrent(c.idx)}
                />
              ))}
              {result.candidates.length === 0 && (
                <p className="text-zinc-500">No moving objects found.</p>
              )}
            </div>
          </>
        )}
      </div>

      {result && result.candidates.length > 0 && (
        <div className="fixed bottom-0 left-0 right-0 flex items-center gap-4 border-t border-zinc-800 bg-zinc-900/95 px-6 py-3 backdrop-blur">
          <span className="flex-1 text-sm text-zinc-400">
            {counts.acc} accepted · {counts.rej} rejected · {counts.undecided}{" "}
            undecided (of {counts.n})
          </span>
          <button
            onClick={exportAccepted}
            disabled={counts.acc === 0}
            className="rounded-md bg-emerald-600 px-4 py-2 font-medium text-white hover:bg-emerald-500 disabled:opacity-40"
          >
            ⬇ Export accepted → ADES PSV
          </button>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <div className="text-2xl font-semibold tabular-nums">{value}</div>
      <div className="text-xs uppercase tracking-wide text-zinc-500">{label}</div>
    </div>
  );
}
