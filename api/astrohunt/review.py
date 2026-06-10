"""Interactive candidate review app: blink GIFs + accept/reject + ADES export.

The 'vet, don't hunt' experience, as a single self-contained HTML file (no server,
no build step — just open it). For each candidate tracklet we cut the same sky
patch from every frame and animate it (the mover walks across a fixed starfield).
The reviewer accepts/rejects with the mouse or keyboard; decisions persist in
localStorage; and 'Export accepted' downloads an ADES PSV containing only the
confirmed candidates — ready for a human to submit. Pillow only, no GPU.
"""

from __future__ import annotations

import base64
import hashlib
import io as _io
import json

import numpy as np
from PIL import Image, ImageDraw


# --- "inferno"-style colormap + asinh stretch for a real-telescope look ----

_INFERNO_ANCHORS = [
    (0.00, (0, 0, 4)),
    (0.15, (40, 11, 84)),
    (0.30, (101, 21, 110)),
    (0.50, (159, 42, 99)),
    (0.70, (212, 72, 66)),
    (0.85, (245, 125, 21)),
    (1.00, (252, 255, 164)),
]


def _build_lut(anchors):
    xs = np.array([a[0] for a in anchors])
    cols = np.array([a[1] for a in anchors], float)
    grid = np.linspace(0, 1, 256)
    lut = np.zeros((256, 3), np.uint8)
    for c in range(3):
        lut[:, c] = np.clip(np.interp(grid, xs, cols[:, c]), 0, 255).astype(np.uint8)
    return lut


_INFERNO_LUT = _build_lut(_INFERNO_ANCHORS)


def _asinh_stretch(stack, lo_pct=10.0, hi_pct=99.6, q=8.0):
    """Percentile-clip + arcsinh stretch -> 8-bit, shared scaling across frames.

    arcsinh compresses the huge star-to-background dynamic range so faint stars
    stay visible while bright cores don't blow out — the classic astro look.
    """
    finite = stack[np.isfinite(stack)]
    if finite.size == 0:
        return np.zeros_like(stack, dtype=np.uint8)
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((stack - lo) / (hi - lo), 0, 1)
    s = np.arcsinh(norm * q) / np.arcsinh(q)
    return (s * 255).astype(np.uint8)


def _colorize(gray8):
    """Map an 8-bit array to RGB through the inferno LUT."""
    return _INFERNO_LUT[gray8]


def _normalize(stack):
    finite = stack[np.isfinite(stack)]
    if finite.size == 0:
        return np.zeros_like(stack, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((stack - lo) / (hi - lo), 0, 1) * 255.0).astype(np.uint8)


def _square_box(detections, image_shape, pad=18):
    xs = [d.x for d in detections]
    ys = [d.y for d in detections]
    side = max(max(xs) - min(xs), max(ys) - min(ys)) + 2 * pad
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    h, w = image_shape
    x0 = int(np.clip(cx - side / 2, 0, w - 1))
    y0 = int(np.clip(cy - side / 2, 0, h - 1))
    side = int(min(side, w - x0, h - y0))
    return x0, y0, max(side, 8)


def _candidate_frame_images(frames, candidate, upscale=5, pad=18):
    """Render each frame's cutout (with a reticle on the mover) as a PIL image."""
    by_frame = {d.frame: d for d in candidate.detections}
    x0, y0, side = _square_box(candidate.detections, frames[0].data.shape, pad)
    stack = np.array([f.data[y0:y0 + side, x0:x0 + side] for f in frames], float)
    g8 = _asinh_stretch(stack)

    images = []
    for f, gg in zip(frames, g8):
        img = Image.fromarray(_colorize(gg), "RGB")
        img = img.resize((side * upscale, side * upscale), Image.NEAREST)
        draw = ImageDraw.Draw(img)
        if f.index in by_frame:
            d = by_frame[f.index]
            rx, ry = (d.x - x0) * upscale, (d.y - y0) * upscale
            r = 6 * upscale
            draw.ellipse([rx - r, ry - r, rx + r, ry + r], outline=(80, 240, 130), width=2)
        draw.text((5, 5), f"frame {f.index}", fill=(255, 235, 120))
        images.append(img)
    return images


def _candidate_gif(frames, candidate, upscale=5, pad=18):
    images = _candidate_frame_images(frames, candidate, upscale, pad)
    buf = _io.BytesIO()
    images[0].save(buf, format="GIF", save_all=True, append_images=images[1:],
                   duration=500, loop=0)
    return base64.b64encode(buf.getvalue()).decode()


def candidate_frame_pngs(frames, candidate, upscale=4, pad=18, max_frames=4):
    """Per-frame cutouts as base64 PNGs (for sending to a vision model)."""
    frames = sorted(frames, key=lambda f: f.index)
    out = []
    for img in _candidate_frame_images(frames, candidate, upscale, pad)[:max_frames]:
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out


def field_blink_gif(frames, tracklets, max_size=460, duration=450):
    """Colorized blink of the WHOLE field with each candidate circled — the
    'money shot' for a demo: the mover sweeps across a fixed starfield."""
    frames = sorted(frames, key=lambda f: f.index)
    h, w = frames[0].data.shape
    stack = np.array([f.data for f in frames], float)
    # darker background (stars-on-black) — prettier and compresses far better
    g8 = _asinh_stretch(stack, lo_pct=32.0, hi_pct=99.7, q=7.0)
    scale = max_size / max(h, w)
    out_w, out_h = int(w * scale), int(h * scale)

    positions: dict[int, list] = {}
    for t in tracklets:
        for d in t.detections:
            positions.setdefault(d.frame, []).append((d.x, d.y, bool(t.known_name)))

    images = []
    for f, gg in zip(frames, g8):
        img = Image.fromarray(_colorize(gg), "RGB").resize((out_w, out_h), Image.BILINEAR)
        draw = ImageDraw.Draw(img)
        for x, y, known in positions.get(f.index, []):
            cx, cy = x * scale, y * scale
            r = 13
            color = (160, 160, 170) if known else (90, 240, 140)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=2)
        draw.text((7, 6), f"frame {f.index}", fill=(255, 235, 120))
        images.append(img)

    buf = _io.BytesIO()
    images[0].save(buf, format="GIF", save_all=True, append_images=images[1:],
                   duration=duration, loop=0, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _rows_html(candidate):
    out = []
    for d in sorted(candidate.detections, key=lambda d: d.time_mjd):
        mag = f"{d.mag:.1f}" if np.isfinite(d.mag) else "—"
        out.append(f"<tr><td>{d.frame}</td><td>{d.ra:.6f}</td>"
                   f"<td>{d.dec:+.6f}</td><td>{mag}</td></tr>")
    return "\n".join(out)


def _card_html(i, cand, gif_b64):
    tag = f"KNOWN: {cand.known_name}" if cand.known_name else "NEW CANDIDATE"
    badge = "#888" if cand.known_name else "#2ec27e"
    return f"""
    <div class="card" id="card-{i}" data-idx="{i}">
      <div class="head">
        <span class="badge" style="background:{badge}">#{i} {tag}</span>
        <span class="meta">rate {cand.rate_total:.2f}&Prime;/min &middot;
          PA {cand.pa_deg:.0f}&deg; &middot; fit-rms {cand.rms_arcsec:.3f}&Prime; &middot;
          {cand.n} pts</span>
        <span class="status" id="status-{i}">undecided</span>
      </div>
      <div class="body">
        <img class="blink" src="data:image/gif;base64,{gif_b64}"/>
        <div class="side">
          <table>
            <tr><th>frame</th><th>RA</th><th>Dec</th><th>mag</th></tr>
            {_rows_html(cand)}
          </table>
          <div class="btns">
            <button onclick="setStatus({i},'accept')">✓ Accept (a)</button>
            <button onclick="setStatus({i},'reject')">✗ Reject (r)</button>
            <button onclick="setStatus({i},'undecided')">Reset (u)</button>
          </div>
        </div>
      </div>
    </div>"""


_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
  body { background:#0e1117; color:#e6e6e6; font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; padding:24px 24px 80px; }
  h1 { font-weight:600; margin:0 0 4px; }
  .sub { color:#9aa0a6; margin-bottom:16px; }
  .card { background:#161b22; border:1px solid #30363d; border-left:5px solid #30363d; border-radius:10px; margin:0 0 20px; overflow:hidden; }
  .card.accept { border-left-color:#2ec27e; }
  .card.reject { border-left-color:#f85149; opacity:.55; }
  .card.current { box-shadow:0 0 0 2px #58a6ff; }
  .head { display:flex; justify-content:space-between; align-items:center; gap:12px; padding:12px 16px; border-bottom:1px solid #30363d; }
  .badge { color:#0e1117; font-weight:700; padding:3px 10px; border-radius:20px; font-size:13px; }
  .meta { color:#9aa0a6; font-size:13px; flex:1; }
  .status { font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:#9aa0a6; }
  .body { display:flex; gap:20px; padding:16px; align-items:flex-start; flex-wrap:wrap; }
  .blink { image-rendering:pixelated; border:1px solid #30363d; border-radius:6px; }
  table { border-collapse:collapse; font-size:13px; margin-bottom:12px; }
  th,td { border:1px solid #30363d; padding:5px 10px; text-align:right; }
  th { color:#9aa0a6; }
  .btns button { background:#21262d; color:#e6e6e6; border:1px solid #30363d; border-radius:6px; padding:6px 12px; margin-right:6px; cursor:pointer; font-size:13px; }
  .btns button:hover { background:#30363d; }
  .bar { position:fixed; bottom:0; left:0; right:0; background:#161b22; border-top:1px solid #30363d; padding:12px 24px; display:flex; gap:16px; align-items:center; }
  .bar .counts { color:#9aa0a6; font-size:14px; flex:1; }
  .bar button { background:#238636; color:#fff; border:none; border-radius:6px; padding:8px 16px; cursor:pointer; font-weight:600; }
  .bar button.ghost { background:#21262d; color:#e6e6e6; border:1px solid #30363d; }
  kbd { background:#30363d; border-radius:4px; padding:1px 6px; font-size:12px; }
</style></head>
<body>
  <h1>🔭 __TITLE__</h1>
  <div class="sub">__NCAND__ candidate(s) &middot; green reticle = measured position &middot;
    keys: <kbd>j</kbd>/<kbd>k</kbd> move &middot; <kbd>a</kbd> accept &middot;
    <kbd>r</kbd> reject &middot; <kbd>u</kbd> reset</div>
  __CARDS__
  <div class="bar">
    <span class="counts" id="counts"></span>
    <button class="ghost" onclick="resetAll()">Reset all</button>
    <button onclick="exportAccepted()">⬇ Export accepted → ADES PSV</button>
  </div>
<script>
const DATA = __DATA__;
let current = 1;
const key = i => `astrohunt:${DATA.reportId}:${i}`;
const getStatus = i => localStorage.getItem(key(i)) || "undecided";
function setStatus(i, s){
  if (s === "undecided") localStorage.removeItem(key(i)); else localStorage.setItem(key(i), s);
  current = i; render();
}
function render(){
  let acc = 0, rej = 0;
  DATA.candidates.forEach(c => {
    const s = getStatus(c.idx);
    if (s === "accept") acc++; else if (s === "reject") rej++;
    const card = document.getElementById("card-" + c.idx);
    card.classList.toggle("accept", s === "accept");
    card.classList.toggle("reject", s === "reject");
    card.classList.toggle("current", c.idx === current);
    document.getElementById("status-" + c.idx).textContent = s;
  });
  const n = DATA.candidates.length;
  document.getElementById("counts").textContent =
    `${acc} accepted · ${rej} rejected · ${n - acc - rej} undecided  (of ${n})`;
}
function buildPSV(){
  const accepted = DATA.candidates.filter(c => getStatus(c.idx) === "accept");
  const lines = DATA.metaLines.slice();
  lines.push(DATA.columns.join("|"));
  accepted.forEach(c => c.rows.forEach(r => lines.push(DATA.columns.map(col => r[col]).join("|"))));
  return lines.join("\\n") + "\\n";
}
function exportAccepted(){
  const blob = new Blob([buildPSV()], {type: "text/plain"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "accepted.psv"; a.click();
}
function resetAll(){ DATA.candidates.forEach(c => localStorage.removeItem(key(c.idx))); render(); }
document.addEventListener("keydown", e => {
  const n = DATA.candidates.length;
  if (e.key === "j") current = Math.min(n, current + 1);
  else if (e.key === "k") current = Math.max(1, current - 1);
  else if (e.key === "a") return setStatus(current, "accept");
  else if (e.key === "r") return setStatus(current, "reject");
  else if (e.key === "u") return setStatus(current, "undecided");
  else return;
  document.getElementById("card-" + current).scrollIntoView({behavior:"smooth", block:"center"});
  render();
});
render();
</script>
</body></html>"""


def write_html_report(frames, tracklets, out_path, meta=None, title="astrohunt review"):
    """Write a single interactive HTML review app for all candidate tracklets."""
    from .ades import ADES_COLUMNS, ObsMetadata, ades_metadata_lines, tracklet_to_ades_rows

    meta = meta or ObsMetadata()
    frames = sorted(frames, key=lambda f: f.index)

    cards, cand_data = [], []
    for i, cand in enumerate(tracklets, 1):
        gif_b64 = _candidate_gif(frames, cand)
        cards.append(_card_html(i, cand, gif_b64))
        cand_data.append({
            "idx": i,
            "known": cand.known_name,
            "rows": tracklet_to_ades_rows(cand, meta, i),
        })

    digest_src = "".join(r["obsTime"] + r["ra"] for c in cand_data for r in c["rows"])
    report_id = hashlib.md5(digest_src.encode()).hexdigest()[:10] if digest_src else "empty"

    payload = {
        "reportId": report_id,
        "metaLines": ades_metadata_lines(meta),
        "columns": list(ADES_COLUMNS),
        "candidates": cand_data,
    }

    html = (_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__NCAND__", str(len(tracklets)))
            .replace("__CARDS__", "".join(cards) or '<p class="sub">No candidates.</p>')
            .replace("__DATA__", json.dumps(payload)))

    with open(out_path, "w") as f:
        f.write(html)
    return out_path
