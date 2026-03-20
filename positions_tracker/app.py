"""
FastAPI application for the positions tracker web dashboard.

Serves static frontend files and exposes a single ``/api/data`` endpoint
that returns all dashboard data as JSON.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure the project root is on sys.path so `positions_tracker` resolves
# when running directly via `python positions_tracker/app.py`
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from positions_tracker.build_response import build_dashboard_data
from positions_tracker.data_fetcher import fetch_ohlcv, init_clients

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize exchange clients on startup."""
    init_clients()
    yield


app = FastAPI(title="Positions Tracker", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the main dashboard page."""
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/api/data")
async def get_data():
    """Return all dashboard data as JSON.

    Wraps ``build_dashboard_data()`` in a try/except so that any unhandled
    exception is returned as a descriptive JSON error instead of an HTTP 500.
    """
    try:
        data = build_dashboard_data()
        return data
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={"error": str(e)},
        )


@app.get("/api/ohlcv")
async def get_ohlcv(symbol: str, timeframe: str = "15m", limit: int = 60):
    """Fetch OHLCV data for a specific symbol and timeframe on demand."""
    try:
        data = fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        return {"ohlcv": data or []}
    except Exception as e:
        return JSONResponse(status_code=200, content={"error": str(e)})


if __name__ == "__main__":
    import os
    import uvicorn

    dev = os.environ.get("DEV", "").lower() in ("1", "true", "yes")
    uvicorn.run("positions_tracker.app:app", host="0.0.0.0", port=8080,
                reload=dev,
                reload_dirs=[str(Path(__file__).resolve().parent)] if dev else [])
