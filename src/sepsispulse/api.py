from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from sepsispulse.features import PatientDataError
from sepsispulse.inference import load_model_bundle, predict_patient
from sepsispulse.schema import WINDOW_HOURS

LOGGER = logging.getLogger("sepsispulse")
STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 2_000_000
DEFAULT_MODEL_PATH = Path("artifacts/sepsispulse.joblib")


class PatientRequest(BaseModel):
    patient_id: str = Field(min_length=1, max_length=128)
    records: list[dict[str, float | None]] = Field(
        min_length=WINDOW_HOURS, max_length=10_000
    )


@asynccontextmanager
async def lifespan(application: FastAPI):
    model_path = Path(os.environ.get("SEPSISPULSE_MODEL_PATH", DEFAULT_MODEL_PATH))
    try:
        application.state.bundle = load_model_bundle(model_path)
        application.state.model_error = None
        LOGGER.info("Loaded SepsisPulse model artifact from %s", model_path)
    except (FileNotFoundError, ValueError) as error:
        application.state.bundle = None
        application.state.model_error = str(error)
        LOGGER.warning("%s", error)
    yield


app = FastAPI(
    title="SepsisPulse",
    description=(
        "Research/Clinical Decision-Support Prototype — Not for Medical Diagnosis."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _bundle_or_503(request: Request) -> dict[str, Any]:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        model_error = getattr(
            request.app.state, "model_error", "Model has not been loaded."
        )
        raise HTTPException(status_code=503, detail=model_error)
    return bundle


def _predict(
    request: Request, frame: pd.DataFrame, patient_id: str
) -> dict[str, Any]:
    bundle = _bundle_or_503(request)
    try:
        return predict_patient(bundle, frame, patient_id)
    except PatientDataError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=None)
async def health(request: Request) -> dict[str, Any] | Response:
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "model_status": "unavailable",
                "detail": getattr(
                    request.app.state, "model_error", "Model has not been loaded."
                ),
            },
        )
    return {
        "status": "ok",
        "model_status": "ready",
        "feature_version": bundle["metadata"]["feature_version"],
        "window_hours": bundle["metadata"]["window_hours"],
        "prediction_horizon_hours": bundle["metadata"][
            "forecast_horizon_hours"
        ],
    }


@app.post("/api/predict/json")
async def predict_json(
    patient_request: PatientRequest, request: Request
) -> dict[str, Any]:
    patient_id = patient_request.patient_id.strip()
    if not patient_id:
        raise HTTPException(status_code=400, detail="Patient ID cannot be blank.")
    frame = pd.DataFrame(patient_request.records)
    return _predict(request, frame, patient_id)


@app.post("/api/predict")
async def predict_upload(
    request: Request,
    file: UploadFile = File(...),
    patient_id: str = Form(..., min_length=1, max_length=128),
) -> dict[str, Any]:
    patient_id = patient_id.strip()
    if not patient_id:
        raise HTTPException(status_code=400, detail="Patient ID cannot be blank.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload must be no larger than {MAX_UPLOAD_BYTES} bytes.",
        )
    try:
        frame = pd.read_csv(
            io.BytesIO(content), sep="|", na_values=["NaN"], encoding="utf-8-sig"
        )
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise HTTPException(
            status_code=400, detail=f"Could not parse the PSV upload: {error}"
        ) from error
    return _predict(request, frame, patient_id)
