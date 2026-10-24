"""Prediction service.  Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Env: MODEL_DIR (default ./artifacts)  API_KEY (optional, enables X-API-Key auth)  MAX_UPLOAD_MB (default 8)"""
import json
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import Depends, FastAPI, File, Header, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from PIL import UnidentifiedImageError

from cropdx.constants import DISCLAIMER, MSG_CONFIDENT, MSG_LOW
from cropdx.preprocess import load_rgb_from_bytes, preprocess_pil, quality_warnings
from .schemas import ClassInfo, ClassScore, ErrorResponse, PredictResponse

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[1] / "artifacts"))
MAX_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "8")) * 1024 * 1024)


class ApiError(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    try:
        so = ort.SessionOptions(); so.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))
        app.state.session = ort.InferenceSession(str(MODEL_DIR / "model.onnx"), so, providers=["CPUExecutionProvider"])
        app.state.input_name = app.state.session.get_inputs()[0].name
        app.state.meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
        app.state.classes = json.loads((MODEL_DIR / "labels.json").read_text())
        app.state.ready = True
    except Exception as e:  # service starts, /health reports DOWN
        print("MODEL LOAD FAILED:", e)
    yield


app = FastAPI(title="Crop Disease Prediction Service", version="1.0.0", lifespan=lifespan,
              description=DISCLAIMER)


@app.middleware("http")
async def request_id_mw(request: Request, call_next):
    rid = (request.headers.get("x-request-id") or str(uuid.uuid4()))[:64]
    request.state.request_id = rid
    resp = await call_next(request)
    resp.headers["X-Request-Id"] = rid
    return resp


def _err(request, status, code, message):
    return JSONResponse(status_code=status, content={"requestId": request.state.request_id,
                                                     "error": {"code": code, "message": message}})


@app.exception_handler(ApiError)
async def api_error_handler(request, exc):
    return _err(request, exc.status, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def validation_handler(request, exc):
    return _err(request, 422, "VALIDATION_ERROR", "Send multipart/form-data with an image in the 'file' field (topK must be 1-5).")


@app.exception_handler(Exception)
async def unhandled(request, exc):
    return _err(request, 500, "INTERNAL_ERROR", "Unexpected server error.")


def check_key(x_api_key: str | None = Header(default=None)):
    expected = os.getenv("API_KEY")
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise ApiError(401, "UNAUTHORIZED", "Missing or invalid X-API-Key.")


def _score(app, i, p):
    c = app.state.classes[i]
    return ClassScore(class_id=c["classId"], crop=c["crop"], condition=c["condition"],
                      display_name=c["displayName"], healthy=c["healthy"], confidence=round(float(p), 4))


def infer(app, img, top_k):
    m = app.state.meta
    x = preprocess_pil(img, m["imageSize"], m["mean"], m["std"])[None]
    t0 = time.perf_counter()
    z = app.state.session.run(None, {app.state.input_name: x})[0][0].astype(np.float64) / m["temperature"]
    ms = (time.perf_counter() - t0) * 1000
    z -= z.max(); p = np.exp(z); p /= p.sum()
    order = np.argsort(-p)[:top_k]
    return [_score(app, int(i), p[i]) for i in order], ms


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return JSONResponse(status_code=204, content=None)


@app.get("/health", tags=["ops"])
async def health(request: Request):
    ok = request.app.state.ready
    return JSONResponse(status_code=200 if ok else 503,
                        content={"status": "UP" if ok else "DOWN", "modelLoaded": ok})


def _need_model(request):
    if not request.app.state.ready:
        raise ApiError(503, "MODEL_UNAVAILABLE", "Model is not loaded.")


@app.get("/v1/model-info", tags=["meta"], dependencies=[Depends(check_key)])
async def model_info(request: Request):
    _need_model(request)
    m = dict(request.app.state.meta); m.pop("mean", None); m.pop("std", None); m.pop("temperature", None)
    m["supportedCrops"] = sorted({c["crop"] for c in request.app.state.classes})
    return m


@app.get("/v1/classes", response_model=list[ClassInfo], tags=["meta"], dependencies=[Depends(check_key)])
async def classes(request: Request):
    _need_model(request)
    return [ClassInfo(class_id=c["classId"], crop=c["crop"], condition=c["condition"],
                      display_name=c["displayName"], healthy=c["healthy"]) for c in request.app.state.classes]


@app.post("/v1/predict", response_model=PredictResponse, tags=["prediction"], dependencies=[Depends(check_key)],
          responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
                     422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def predict(request: Request, file: UploadFile = File(..., description="JPEG/PNG/WebP photo of a single leaf"),
                  top_k: int = Query(3, alias="topK", ge=1, le=5)):
    _need_model(request)
    data = await file.read(MAX_BYTES + 1)
    if not data:
        raise ApiError(400, "EMPTY_FILE", "The uploaded file is empty.")
    if len(data) > MAX_BYTES:
        raise ApiError(413, "FILE_TOO_LARGE", f"Image larger than {MAX_BYTES // (1024*1024)} MB.")
    try:
        img = await run_in_threadpool(load_rgb_from_bytes, data)
    except (UnidentifiedImageError, OSError, ValueError, Exception) as e:
        raise ApiError(400, "INVALID_IMAGE", "File is not a readable JPEG/PNG/WebP image.") from e
    if min(img.size) < 32:
        raise ApiError(400, "INVALID_IMAGE", "Image is too small (minimum 32 px per side).")
    cands, ms = await run_in_threadpool(infer, request.app, img, top_k)
    warns = await run_in_threadpool(quality_warnings, img)
    thr = request.app.state.meta["confidenceThreshold"]
    top = cands[0]
    ok = top.confidence >= thr
    return PredictResponse(
        request_id=request.state.request_id, model_version=request.app.state.meta["modelVersion"],
        status="CONFIDENT" if ok else "LOW_CONFIDENCE", prediction=top if ok else None, candidates=cands,
        confidence_threshold=thr, quality_warnings=warns, disclaimer=DISCLAIMER, inference_ms=round(ms, 1),
        message=MSG_CONFIDENT.format(crop=top.crop, condition=top.condition, pct=top.confidence * 100) if ok else MSG_LOW)
