"""Vercel Python serverless function exposing the astrohunt web API.

Deploy this repo as a Vercel project (root = repo root). Vercel installs
`requirements.txt` (which installs the astrohunt package + web deps) and serves
this ASGI app. The vercel.json rewrite funnels all /api/* requests here, so the
FastAPI routes (/api/demo, /api/detect, /api/health) resolve normally.

See web/DEPLOY.md for the full deployment guide.
"""

from astrohunt.webapi import app  # noqa: F401  (Vercel serves this ASGI `app`)
