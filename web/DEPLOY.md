# Deploying astrohunt (web UI)

The app is two pieces: a **Next.js frontend** (`web/`) and a **Python FastAPI
backend** that runs the real `astrohunt` pipeline (`src/astrohunt/webapi.py`,
exposed for Vercel at `api/index.py`).

## Local development (verified)
```bash
# 1) backend — from the repo root, with the venv that has astrohunt + [web]
pip install -e ".[web]"
astrohunt serve                 # → http://127.0.0.1:8000  (try /api/demo)

# 2) frontend — in another terminal
cd web
npm install
npm run dev                     # → http://localhost:3000
```
Open http://localhost:3000, click **Run demo** (or upload FITS frames), vet the
candidates, and **Export accepted → ADES PSV**. The frontend talks to the backend
via `NEXT_PUBLIC_API_URL` (defaults to `http://127.0.0.1:8000`).

## Vercel — two projects from this one repo (recommended)
A single Vercel project can't cleanly host a Next.js app in `web/` *and* a Python
function at the repo root, so deploy them as two linked projects:

**Backend project** (Root Directory = repo root)
- Vercel installs `requirements.txt` (`.[web]` → astrohunt + FastAPI) and serves
  `api/index.py`; `vercel.json` rewrites `/api/*` to that ASGI function.
- Gives you e.g. `https://astrohunt-api.vercel.app/api/demo`.

**Frontend project** (Root Directory = `web/`)
- Auto-detected Next.js. Set env `NEXT_PUBLIC_API_URL` = the backend URL above.
- Gives you the shareable UI URL.

```bash
# from repo root — backend
vercel --prod
# from web/ — frontend (set NEXT_PUBLIC_API_URL in the dashboard or:)
cd web && vercel --prod -e NEXT_PUBLIC_API_URL=https://astrohunt-api.vercel.app
```

> **Heads-up (not yet deploy-tested):** the backend bundles a heavy scientific
> stack (numpy/scipy/astropy/photutils/astroquery/lxml/pillow). It runs great
> locally; on Vercel it relies on Fluid Compute and should fit, but verify the
> function size/cold-start on first deploy. The optional PyTorch CNN vetter is
> excluded from `requirements.txt` to keep the bundle lean.

## Alternative: frontend on Vercel, backend anywhere
Deploy only `web/` to Vercel and run the backend on any Python host (a VM,
container, Fly.io, Railway, etc.) with `astrohunt serve`. Point the frontend at
it with `NEXT_PUBLIC_API_URL`.
