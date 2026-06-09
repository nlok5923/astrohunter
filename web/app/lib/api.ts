// API client + types for the astrohunt backend (FastAPI).

// The browser always calls same-origin /api/*; next.config.ts rewrites those to
// the real backend (set via NEXT_PUBLIC_API_URL at build, default localhost:8000).
export const API_BASE = "";

export type AdesRow = Record<string, string>;

export interface Detection {
  frame: number;
  ra: number;
  dec: number;
  mag: number | null;
}

export interface Candidate {
  idx: number;
  known: string | null;
  rate: number;
  pa: number;
  rms: number;
  n: number;
  gif: string; // base64 GIF
  rows: AdesRow[];
  detections: Detection[];
}

export interface FrameReport {
  index: number;
  n_sources: number;
  n_matched: number;
  n_vetoed: number;
  rms_arcsec: number;
}

export interface DetectResult {
  frames: FrameReport[];
  median_rms_arcsec: number;
  n_candidates: number;
  n_known: number;
  ades: { metaLines: string[]; columns: string[] };
  candidates: Candidate[];
  warnings: string[];
}

export interface DetectOptions {
  offline: boolean;
  obscode: string;
  observer: string;
  hintRa?: string;
  hintDec?: string;
  refCatalog?: File | null;
}

export async function runDemo(): Promise<DetectResult> {
  const r = await fetch(`${API_BASE}/api/demo`);
  if (!r.ok) throw new Error(`demo failed: ${r.status} ${await r.text()}`);
  return r.json();
}

export async function runDetect(
  files: File[],
  opts: DetectOptions
): Promise<DetectResult> {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  fd.append("offline", String(opts.offline));
  fd.append("obscode", opts.obscode);
  fd.append("observer", opts.observer);
  if (opts.hintRa) fd.append("hint_ra", opts.hintRa);
  if (opts.hintDec) fd.append("hint_dec", opts.hintDec);
  if (opts.refCatalog) fd.append("ref_catalog", opts.refCatalog);

  const r = await fetch(`${API_BASE}/api/detect`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`detect failed: ${r.status} ${await r.text()}`);
  return r.json();
}

// Build an ADES PSV document for the accepted candidates (browser-side export).
export function buildPSV(
  result: DetectResult,
  acceptedIdx: Set<number>
): string {
  const lines = [...result.ades.metaLines, result.ades.columns.join("|")];
  result.candidates
    .filter((c) => acceptedIdx.has(c.idx))
    .forEach((c) =>
      c.rows.forEach((row) =>
        lines.push(result.ades.columns.map((col) => row[col] ?? "").join("|"))
      )
    );
  return lines.join("\n") + "\n";
}

export function downloadText(filename: string, text: string) {
  const blob = new Blob([text], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}
