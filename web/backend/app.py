import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from seizure_v2.data.channels import CHANNELS
from seizure_v2.data.splits import split_manifest
from seizure_v2.inference.predictor import Predictor

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


def create_app(bundle=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = None
        app.state.error = "Verified model artifacts have not been installed."
        root = Path(bundle or os.environ.get("SEIZURE_BUNDLE", "artifacts/demo"))
        if (root / "manifest.json").exists():
            try:
                app.state.predictor = Predictor(root)
                app.state.error = None
            except Exception:
                logging.exception("Artifact verification failed")
                app.state.error = "Model artifacts failed verification."
        yield

    app = FastAPI(title="Pediatric EEG · Research Explorer", lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    def predictor():
        if app.state.predictor is None:
            raise HTTPException(503, app.state.error)
        return app.state.predictor

    @app.get("/healthz")
    def health():
        if app.state.predictor is None:
            raise HTTPException(503, {"status": "unavailable", "reason": app.state.error})
        return {"status": "ready", "study_sha256": app.state.predictor.manifest["study_sha256"]}

    @app.get("/api/research")
    def research():
        active = app.state.predictor
        return {"status": "ready" if active else "awaiting_artifacts", "message": app.state.error,
                "models": active.manifest["models"] if active else {},
                "default_model": active.manifest["default_model"] if active else None,
                "split": split_manifest(), "channels": CHANNELS, "sampling_rate": 256, "window_seconds": 8,
                "disclaimer": "Research/educational prototype. Not a diagnostic system. Not for clinical decision-making.",
                "attribution": active.manifest["attribution"] if active else "CHB-MIT Scalp EEG Database v1.0.0 · PhysioNet"}

    @app.get("/api/examples")
    def examples():
        return {"examples": list(predictor().examples.values())}

    @app.get("/api/examples/{example_id}")
    def example(example_id: str, start_seconds: float | None = None):
        active = predictor()
        if example_id not in active.examples:
            raise HTTPException(404, "Unknown example")
        metadata, signal, _ = active.clip(example_id)
        start = metadata["focus_start_seconds"] - 4 if start_seconds is None else start_seconds
        if not np.isfinite(start):
            raise HTTPException(422, "Time must be finite")
        begin = max(0, min(signal.shape[1] - 1, round((start - metadata["clip_start_seconds"]) * 256)))
        end = min(signal.shape[1], begin + 16 * 256)
        return {"example": metadata, "channels": CHANNELS, "sampling_rate": 256,
                "view_start_seconds": metadata["clip_start_seconds"] + begin / 256,
                "signal_uv": signal[:, begin:end].tolist()}

    class Request(BaseModel):
        example_id: str = Field(min_length=1, max_length=80)
        model_id: str = Field(min_length=1, max_length=40)

    @app.post("/api/predict")
    def prediction(request: Request):
        active = predictor()
        if request.example_id not in active.examples or request.model_id not in active.sessions:
            raise HTTPException(404, "Unknown example or model")
        return active.predict_example(request.example_id, request.model_id)

    @app.get("/")
    def index():
        return FileResponse(FRONTEND / "index.html")

    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
    return app


app = create_app()
