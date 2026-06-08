"""Visual candidate review: blink GIFs + a self-contained HTML report.

The 'vet, don't hunt' experience. For each candidate tracklet we cut the same
sky patch from every frame, animate it (the mover walks across a fixed
starfield — a classic blink), and lay everything out in a single HTML file with
the measurements, so a human can confirm or reject at a glance. Pillow only — no
GPU, no browser automation, opens anywhere.
"""

from __future__ import annotations

import base64
import io as _io

import numpy as np
from PIL import Image, ImageDraw


def _normalize(stack):
    """Percentile-clip a stack of patches to a shared 0-255 scale."""
    finite = stack[np.isfinite(stack)]
    if finite.size == 0:
        return np.zeros_like(stack, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((stack - lo) / (hi - lo), 0, 1) * 255.0
    return out.astype(np.uint8)


def _square_box(detections, image_shape, pad=18):
    xs = [d.x for d in detections]
    ys = [d.y for d in detections]
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h, w = image_shape
    x0 = int(np.clip(cx - side / 2, 0, w - 1))
    y0 = int(np.clip(cy - side / 2, 0, h - 1))
    side = int(min(side, w - x0, h - y0))
    return x0, y0, max(side, 8)


def _candidate_gif(frames, candidate, upscale=5, pad=18):
    """Return (gif_bytes, n_frames) for one candidate's blink animation."""
    by_frame = {d.frame: d for d in candidate.detections}
    x0, y0, side = _square_box(candidate.detections, frames[0].data.shape, pad)

    patches = [f.data[y0:y0 + side, x0:x0 + side] for f in frames]
    stack = np.array(patches, dtype=float)
    norm = _normalize(stack)

    images = []
    for f, p8 in zip(frames, norm):
        img = Image.fromarray(p8, mode="L").convert("RGB")
        img = img.resize((side * upscale, side * upscale), Image.NEAREST)
        draw = ImageDraw.Draw(img)
        # reticle on the mover where it was measured this frame
        if f.index in by_frame:
            d = by_frame[f.index]
            rx = (d.x - x0) * upscale
            ry = (d.y - y0) * upscale
            r = 6 * upscale
            draw.ellipse([rx - r, ry - r, rx + r, ry + r], outline=(80, 220, 120), width=2)
        draw.text((4, 4), f"frame {f.index}", fill=(255, 230, 90))
        images.append(img)

    buf = _io.BytesIO()
    images[0].save(
        buf, format="GIF", save_all=True, append_images=images[1:],
        duration=500, loop=0,
    )
    return buf.getvalue(), len(images)


def _candidate_rows_html(candidate):
    rows = []
    for d in sorted(candidate.detections, key=lambda d: d.time_mjd):
        mag = f"{d.mag:.1f}" if np.isfinite(d.mag) else "—"
        rows.append(
            f"<tr><td>{d.frame}</td><td>{d.ra:.6f}</td><td>{d.dec:+.6f}</td>"
            f"<td>{mag}</td></tr>"
        )
    return "\n".join(rows)


def write_html_report(frames, candidates, out_path, title="astrohunt candidates"):
    """Write a single self-contained HTML file reviewing all candidates."""
    frames = sorted(frames, key=lambda f: f.index)
    cards = []
    for i, cand in enumerate(candidates, 1):
        gif_bytes, _n = _candidate_gif(frames, cand)
        b64 = base64.b64encode(gif_bytes).decode()
        tag = (f"KNOWN: {cand.known_name}" if cand.known_name
               else "NEW CANDIDATE")
        badge = "#2ec27e" if not cand.known_name else "#888"
        cards.append(f"""
        <div class="card">
          <div class="head">
            <span class="badge" style="background:{badge}">#{i} {tag}</span>
            <span class="meta">rate {cand.rate_total:.2f}&Prime;/min &nbsp;|&nbsp;
              PA {cand.pa_deg:.0f}&deg; &nbsp;|&nbsp;
              fit-rms {cand.rms_arcsec:.3f}&Prime; &nbsp;|&nbsp;
              {cand.n} points</span>
          </div>
          <div class="body">
            <img class="blink" src="data:image/gif;base64,{b64}"/>
            <table>
              <tr><th>frame</th><th>RA (deg)</th><th>Dec (deg)</th><th>mag</th></tr>
              {_candidate_rows_html(cand)}
            </table>
          </div>
        </div>""")

    if not candidates:
        cards.append('<p class="empty">No candidates to review.</p>')

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ background:#0e1117; color:#e6e6e6; font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; padding:24px; }}
  h1 {{ font-weight:600; }}
  .sub {{ color:#9aa0a6; margin-bottom:24px; }}
  .card {{ background:#161b22; border:1px solid #30363d; border-radius:10px; margin:0 0 20px; overflow:hidden; }}
  .head {{ display:flex; justify-content:space-between; align-items:center; padding:12px 16px; border-bottom:1px solid #30363d; }}
  .badge {{ color:#0e1117; font-weight:700; padding:3px 10px; border-radius:20px; font-size:13px; }}
  .meta {{ color:#9aa0a6; font-size:13px; }}
  .body {{ display:flex; gap:20px; padding:16px; align-items:flex-start; flex-wrap:wrap; }}
  .blink {{ image-rendering:pixelated; border:1px solid #30363d; border-radius:6px; }}
  table {{ border-collapse:collapse; font-size:13px; }}
  th,td {{ border:1px solid #30363d; padding:5px 10px; text-align:right; }}
  th {{ color:#9aa0a6; font-weight:600; }}
  .empty {{ color:#9aa0a6; }}
</style></head>
<body>
  <h1>🔭 {title}</h1>
  <div class="sub">{len(candidates)} candidate(s) &middot; green reticle marks the measured position &middot; the mover walks across a fixed starfield</div>
  {''.join(cards)}
</body></html>"""

    with open(out_path, "w") as f:
        f.write(html)
    return out_path
