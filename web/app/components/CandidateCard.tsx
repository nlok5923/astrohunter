"use client";

import { Candidate } from "../lib/api";

type Status = "undecided" | "accept" | "reject";

export function CandidateCard({
  cand,
  status,
  current,
  onSet,
  onFocus,
}: {
  cand: Candidate;
  status: Status;
  current: boolean;
  onSet: (s: Status) => void;
  onFocus: () => void;
}) {
  const border =
    status === "accept"
      ? "border-l-emerald-500"
      : status === "reject"
      ? "border-l-red-500"
      : "border-l-zinc-700";
  const ring = current ? "ring-2 ring-sky-500" : "";
  const dim = status === "reject" ? "opacity-60" : "";

  return (
    <div
      onClick={onFocus}
      className={`rounded-xl border border-zinc-800 border-l-4 ${border} ${ring} ${dim} bg-zinc-900/70 overflow-hidden transition`}
    >
      <div className="flex items-center justify-between gap-3 border-b border-zinc-800 px-4 py-3">
        <span
          className="rounded-full px-3 py-1 text-xs font-bold text-zinc-950"
          style={{ background: cand.known ? "#9ca3af" : "#2ec27e" }}
        >
          #{cand.idx} {cand.known ? `KNOWN: ${cand.known}` : "NEW CANDIDATE"}
        </span>
        <span className="flex-1 text-xs text-zinc-400">
          rate {cand.rate.toFixed(2)}″/min · PA {cand.pa.toFixed(0)}° · fit-rms{" "}
          {cand.rms.toFixed(3)}″ · {cand.n} pts
        </span>
        <span className="text-xs uppercase tracking-wide text-zinc-500">
          {status}
        </span>
      </div>

      <div className="flex flex-wrap items-start gap-5 p-4">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`data:image/gif;base64,${cand.gif}`}
          alt={`candidate ${cand.idx} blink`}
          className="rounded-md border border-zinc-800"
          style={{ imageRendering: "pixelated", width: 220 }}
        />
        <div className="flex-1">
          <table className="mb-3 text-sm tabular-nums">
            <thead className="text-zinc-500">
              <tr>
                <th className="border border-zinc-800 px-2 py-1 text-right">frame</th>
                <th className="border border-zinc-800 px-2 py-1 text-right">RA</th>
                <th className="border border-zinc-800 px-2 py-1 text-right">Dec</th>
                <th className="border border-zinc-800 px-2 py-1 text-right">mag</th>
              </tr>
            </thead>
            <tbody>
              {cand.detections.map((d, i) => (
                <tr key={i}>
                  <td className="border border-zinc-800 px-2 py-1 text-right">{d.frame}</td>
                  <td className="border border-zinc-800 px-2 py-1 text-right">{d.ra.toFixed(6)}</td>
                  <td className="border border-zinc-800 px-2 py-1 text-right">{d.dec.toFixed(6)}</td>
                  <td className="border border-zinc-800 px-2 py-1 text-right">
                    {d.mag == null ? "—" : d.mag.toFixed(1)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="flex gap-2">
            <button
              onClick={() => onSet("accept")}
              className="rounded-md border border-emerald-700 bg-emerald-800/40 px-3 py-1.5 text-sm hover:bg-emerald-700/50"
            >
              ✓ Accept <kbd className="text-zinc-400">a</kbd>
            </button>
            <button
              onClick={() => onSet("reject")}
              className="rounded-md border border-red-800 bg-red-900/40 px-3 py-1.5 text-sm hover:bg-red-800/50"
            >
              ✗ Reject <kbd className="text-zinc-400">r</kbd>
            </button>
            <button
              onClick={() => onSet("undecided")}
              className="rounded-md border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-sm hover:bg-zinc-700"
            >
              Reset <kbd className="text-zinc-400">u</kbd>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
