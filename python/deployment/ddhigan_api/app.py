from __future__ import annotations

import hmac
import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from .runtime import DDHIgANRuntime
from .schemas import PredictionRequest


logger = logging.getLogger("uvicorn.error")
app = FastAPI(
    title="DDHIgAN Research Pilot API", version="1.0",
    docs_url=None, redoc_url=None, openapi_url=None,
)
runtime = None
startup_error = None
try:
    bundle = os.environ["DDHIGAN_BUNDLE_DIR"]
    if (Path(bundle) / "latest_manifest.json").exists():
        from .latest_runtime import LatestRuntime
        runtime = LatestRuntime(bundle)
    else:
        runtime = DDHIgANRuntime(bundle)
except Exception as exc:  # readiness reports failure without leaking path/details
    startup_error = type(exc).__name__
def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("DDHIGAN_API_KEY", "")
    if not expected or x_api_key is None or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="invalid API key")


@app.middleware("http")
async def audit_metadata_only(request: Request, call_next):
    started = time.perf_counter()
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    status = 500
    category = "internal"
    try:
        response = await call_next(request)
        status = response.status_code
        category = "ok" if status < 400 else f"http_{status}"
    except Exception:
        response = JSONResponse(status_code=500, content={"detail": "internal error"})
    response.headers["X-Request-ID"] = request_id
    source_ip = request.headers.get("x-real-ip") or (request.client.host if request.client else "unknown")
    version = runtime.provenance["model_version"] if runtime is not None else "unavailable"
    response.headers["X-Error-Category"] = category
    response.headers["X-Model-Version"] = version
    logger.info(
        "request_id=%s source_ip=%s status=%s latency_ms=%.3f category=%s model_version=%s",
        request_id, source_ip, status, (time.perf_counter() - started) * 1000,
        category, version,
    )
    return response


@app.get("/health/ready")
def ready():
    if runtime is None:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "category": startup_error})
    return {
        "status": "ready",
        "model_version": runtime.provenance["model_version"],
        "api_release": runtime.api_release,
    }


@app.get("/v1/model-info", dependencies=[Depends(require_api_key)])
def model_info():
    if runtime is None:
        raise HTTPException(status_code=503, detail="model unavailable")
    return runtime.model_info()


@app.post("/ddhigan/v1/predict", dependencies=[Depends(require_api_key)])
def predict(payload: PredictionRequest):
    if runtime is None:
        raise HTTPException(status_code=503, detail="model unavailable")
    if payload.model_id != "DDHIgAN" and not hasattr(runtime, "models"):
        raise HTTPException(status_code=422, detail="selected model unavailable")
    return runtime.predict(payload)
