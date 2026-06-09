"""Vercel Python serverless function exposing the astrohunt web API.

`astrohunt/` is vendored alongside this file (synced from ../src/astrohunt) so the
Vercel FastAPI bundler includes it without pip-installing the heavy package. The
ASGI `app` below is what Vercel serves; FastAPI routes (/api/demo, /api/detect,
/api/health) resolve directly. See web/DEPLOY.md.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from astrohunt.webapi import app  # noqa: E402,F401  (Vercel serves this ASGI `app`)
