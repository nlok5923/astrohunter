# Deploying astrohunt (web UI)

Two pieces: a **Next.js frontend** (`web/`) and a **Python FastAPI backend** that
runs the real `astrohunt` pipeline. **Live now:**

- Frontend: **https://web-zeta-bay-43.vercel.app**
- Backend:  **https://asteriod-naming.vercel.app** (`/api/demo`, `/api/detect`, `/api/health`)

## Local development
```bash
# 1) backend — from the repo root
pip install -e ".[web]"
astrohunt serve                 # → http://127.0.0.1:8000

# 2) frontend — in another terminal
cd web && npm install && npm run dev    # → http://localhost:3000
```
The frontend calls same-origin `/api/*`; `next.config.ts` rewrites that to the
backend (`NEXT_PUBLIC_API_URL`, default `http://127.0.0.1:8000`). So you don't set
anything for local dev — just run both.

## How it's deployed on Vercel (two projects, one repo)

**Backend** — Vercel project rooted at the **repo root**:
- `api/index.py` exposes the FastAPI ASGI `app`. The `astrohunt` package is
  **vendored into `api/astrohunt/`** (synced from `src/astrohunt`, git-ignored) so
  Vercel's FastAPI bundler includes it without pip-installing the heavy package.
- `requirements.txt` is a **lean** set (numpy/scipy/astropy/photutils/pillow +
  fastapi/python-multipart). `.vercelignore` excludes `pyproject.toml` so Vercel
  installs from `requirements.txt` (not the package's full core deps). astroquery
  + lxml are omitted, so the hosted backend supports the **synthetic demo and
  offline detection**; online Gaia/SkyBoT and strict-XSD validation are local-only.
- Deploy: `vercel deploy --prod` from the repo root. (Re-sync the vendored copy
  first: `rm -rf api/astrohunt && cp -r src/astrohunt api/astrohunt`.)

**Frontend** — Vercel project rooted at **`web/`**:
- Set env `NEXT_PUBLIC_API_URL` = the backend URL. `next.config.ts` rewrites
  `/api/*` → `${NEXT_PUBLIC_API_URL}/api/*` (server-side proxy — no CORS, no
  client-side URL inlining).
- Deploy: `cd web && vercel deploy --prod`.

```bash
# backend (repo root)
rm -rf api/astrohunt && cp -r src/astrohunt api/astrohunt
vercel deploy --prod --yes
# frontend (web/)
cd web
printf "https://<your-backend>.vercel.app" | vercel env add NEXT_PUBLIC_API_URL production
vercel deploy --prod --yes
```

> **Note on the heavy stack:** the bundled scientific deps exceed Vercel's static
> bundle limit, so Vercel switches to **runtime dependency installation** (first
> cold start is slower; warm requests are ~3–4s). This works today but is the main
> thing to watch if you change deps.

## Alternative: frontend on Vercel, backend elsewhere
Deploy only `web/` to Vercel and run the backend on any Python host
(`astrohunt serve` on a VM/container/Fly/Railway/Render — no 250MB function
limit, so the full `[web]` stack incl. astroquery works). Point the frontend at
it via `NEXT_PUBLIC_API_URL`.
