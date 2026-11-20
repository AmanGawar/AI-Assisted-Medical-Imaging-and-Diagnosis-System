import asyncio
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import audit
from .inference import ChestXrayClassifier
from .segmentation import LungVolumeSegmenter

ORTHANC_URL = os.getenv("ORTHANC_URL", "http://localhost:8042").rstrip("/")
ORTHANC_USERNAME = os.getenv("ORTHANC_USERNAME", "demo")
ORTHANC_PASSWORD = os.getenv("ORTHANC_PASSWORD", "demo-only-change-me")
AUDIT_DB = os.getenv("AUDIT_DB", "audit.sqlite3")
MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "")
SEGMENTATION_CHECKPOINT = os.getenv("SEGMENTATION_CHECKPOINT", "")
ID_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")
DISCLAIMER = "For education only, not for clinical use."
AUTO_ANALYZE = os.getenv("AUTO_ANALYZE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}
POLL_INTERVAL_SECONDS = float(os.getenv("ORTHANC_POLL_INTERVAL_SECONDS", "5"))
STUDY_STABILITY_SECONDS = float(os.getenv("STUDY_STABILITY_SECONDS", "15"))
logger = logging.getLogger(__name__)
classifier: ChestXrayClassifier | None = None
segmenter: LungVolumeSegmenter | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global classifier, segmenter
    audit.initialize(AUDIT_DB)
    if MODEL_CHECKPOINT and os.path.isfile(MODEL_CHECKPOINT):
        classifier = ChestXrayClassifier(MODEL_CHECKPOINT)
    if SEGMENTATION_CHECKPOINT and os.path.isfile(SEGMENTATION_CHECKPOINT):
        segmenter = LungVolumeSegmenter(SEGMENTATION_CHECKPOINT)
    automation_task = (
        asyncio.create_task(watch_orthanc_studies())
        if AUTO_ANALYZE
        else None
    )
    try:
        yield
    finally:
        if automation_task is not None:
            automation_task.cancel()
            try:
                await automation_task
            except asyncio.CancelledError:
                pass


app = FastAPI(title="Educational Medical Imaging API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ReviewRequest(BaseModel):
    analysis_id: str
    reviewer: str = Field(min_length=1, max_length=120)
    decision: Literal["approve", "reject"]
    note: str = Field(default="", max_length=2000)


def validate_id(value: str) -> str:
    if not ID_PATTERN.fullmatch(value):
        raise HTTPException(status_code=400, detail="Invalid identifier")
    return value


async def orthanc_get(path: str) -> httpx.Response:
    try:
        async with httpx.AsyncClient(
            auth=(ORTHANC_USERNAME, ORTHANC_PASSWORD), timeout=20
        ) as client:
            response = await client.get(f"{ORTHANC_URL}{path}")
            response.raise_for_status()
            return response
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502, detail=f"Orthanc returned HTTP {exc.response.status_code}"
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail="Orthanc is unavailable") from exc


async def first_instance_id(study_id: str) -> str:
    response = await orthanc_get(f"/studies/{validate_id(study_id)}/instances")
    instances = response.json()
    if not instances:
        raise HTTPException(status_code=404, detail="Study contains no instances")
    return validate_id(instances[0]["ID"])


async def study_modalities(study_id: str) -> set[str]:
    study_response = await orthanc_get(f"/studies/{validate_id(study_id)}")
    modalities: set[str] = set()
    for series_id in study_response.json().get("Series", []):
        series_response = await orthanc_get(f"/series/{validate_id(series_id)}")
        modality = series_response.json().get("MainDicomTags", {}).get("Modality")
        if modality:
            modalities.add(str(modality).upper())
    return modalities


@app.get("/api/health")
async def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_ready": classifier is not None,
        "segmentation_ready": segmenter is not None,
        "auto_analysis": AUTO_ANALYZE,
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/studies")
async def list_studies() -> list[dict[str, str | list[str]]]:
    response = await orthanc_get("/studies")
    study_ids = [validate_id(study_id) for study_id in response.json()]
    modalities = await asyncio.gather(
        *(study_modalities(study_id) for study_id in study_ids)
    )
    return [
        {"id": study_id, "modalities": sorted(study_modalities)}
        for study_id, study_modalities in zip(study_ids, modalities)
    ]


async def watch_orthanc_studies() -> None:
    stable_counts: dict[str, tuple[int, float]] = {}
    in_flight: set[tuple[str, int]] = set()
    failed_attempts: set[tuple[str, int]] = set()
    if classifier is None and segmenter is None:
        logger.error("Automatic study processing is enabled but no model checkpoint is loaded.")
    while classifier is not None or segmenter is not None:
        try:
            response = await orthanc_get("/studies")
            study_ids = {validate_id(study_id) for study_id in response.json()}
            stable_counts = {
                study_id: state
                for study_id, state in stable_counts.items()
                if study_id in study_ids
            }
            for study_id in study_ids:
                instances_response = await orthanc_get(
                    f"/studies/{study_id}/instances"
                )
                instance_count = len(instances_response.json())
                if instance_count == 0:
                    continue
                status = audit.get_automation_status(AUDIT_DB, study_id)
                if (
                    status is not None
                    and status["instance_count"] == instance_count
                    and status["status"] == "completed"
                ):
                    continue
                work_key = (study_id, instance_count)
                if work_key in in_flight or work_key in failed_attempts:
                    continue
                previous = stable_counts.get(study_id)
                now = time.monotonic()
                if previous is None or previous[0] != instance_count:
                    stable_counts[study_id] = (instance_count, now)
                    continue
                if now - previous[1] < STUDY_STABILITY_SECONDS:
                    continue

                audit.save_automation_status(
                    AUDIT_DB, study_id, instance_count, "processing"
                )
                in_flight.add(work_key)
                try:
                    modalities = await study_modalities(study_id)
                    if "CT" in modalities:
                        if audit.has_segmentation(AUDIT_DB, study_id):
                            audit.save_automation_status(
                                AUDIT_DB, study_id, instance_count, "completed"
                            )
                            in_flight.discard(work_key)
                            stable_counts.pop(study_id, None)
                            continue
                        result = await segment_study(study_id)
                        audit.save_automation_status(
                            AUDIT_DB,
                            study_id,
                            instance_count,
                            "completed",
                            segmentation_id=str(result["segmentation_id"]),
                        )
                    elif modalities.intersection({"CR", "DX"}):
                        if audit.has_analysis(AUDIT_DB, study_id):
                            audit.save_automation_status(
                                AUDIT_DB, study_id, instance_count, "completed"
                            )
                            in_flight.discard(work_key)
                            stable_counts.pop(study_id, None)
                            continue
                        result = await analyze_study(study_id)
                        audit.save_automation_status(
                            AUDIT_DB,
                            study_id,
                            instance_count,
                            "completed",
                            analysis_id=str(result["analysis_id"]),
                        )
                    else:
                        raise HTTPException(
                            status_code=422,
                            detail=f"No supported inference route for modalities: {sorted(modalities)}",
                        )
                except HTTPException as exc:
                    in_flight.discard(work_key)
                    failed_attempts.add(work_key)
                    logger.error(
                        "Automatic processing failed for study %s: %s",
                        study_id,
                        exc.detail,
                    )
                    audit.save_automation_status(
                        AUDIT_DB,
                        study_id,
                        instance_count,
                        "failed",
                        error=str(exc.detail),
                    )
                else:
                    in_flight.discard(work_key)
                stable_counts.pop(study_id, None)
        except HTTPException as exc:
            logger.error("Orthanc polling failed: %s", exc.detail)
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


@app.get("/api/automation")
async def automation_status() -> dict[str, object]:
    return {
        "enabled": AUTO_ANALYZE,
        "poll_interval_seconds": POLL_INTERVAL_SECONDS,
        "study_stability_seconds": STUDY_STABILITY_SECONDS,
        "studies": audit.list_automation_status(AUDIT_DB),
    }


@app.get("/api/studies/{study_id}/preview")
async def study_preview(study_id: str) -> Response:
    instance_id = await first_instance_id(study_id)
    response = await orthanc_get(f"/instances/{instance_id}/preview")
    return Response(content=response.content, media_type="image/png")


async def ct_series_files(study_id: str) -> list[bytes]:
    study_response = await orthanc_get(f"/studies/{validate_id(study_id)}")
    series_ids = study_response.json().get("Series", [])
    candidates: list[tuple[int, str]] = []
    for series_id in series_ids:
        series_response = await orthanc_get(f"/series/{validate_id(series_id)}")
        series_info = series_response.json()
        if series_info.get("MainDicomTags", {}).get("Modality") != "CT":
            continue
        instances_response = await orthanc_get(
            f"/series/{validate_id(series_id)}/instances"
        )
        instance_ids = [
            validate_id(instance["ID"]) for instance in instances_response.json()
        ]
        if len(instance_ids) >= 8:
            candidates.append((len(instance_ids), series_id))

    if not candidates:
        raise HTTPException(
            status_code=422,
            detail="This study has no CT series with at least 8 slices for 3D segmentation.",
        )

    _, selected_series = max(candidates)
    instance_response = await orthanc_get(
        f"/series/{validate_id(selected_series)}/instances"
    )
    instance_ids = [
        validate_id(instance["ID"]) for instance in instance_response.json()
    ]
    semaphore = asyncio.Semaphore(8)
    timeout = httpx.Timeout(120.0, connect=10.0)
    try:
        async with httpx.AsyncClient(
            auth=(ORTHANC_USERNAME, ORTHANC_PASSWORD),
            timeout=timeout,
        ) as client:
            async def fetch_instance(instance_id: str) -> bytes:
                async with semaphore:
                    response = await client.get(
                        f"{ORTHANC_URL}/instances/{instance_id}/file"
                    )
                    response.raise_for_status()
                    return response.content

            return await asyncio.gather(
                *(fetch_instance(instance_id) for instance_id in instance_ids)
            )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Orthanc returned HTTP {exc.response.status_code} while reading the CT series.",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail="Orthanc became unavailable while reading the CT series.",
        ) from exc


@app.post("/api/studies/{study_id}/analyze")
async def analyze_study(study_id: str) -> dict[str, str | float]:
    study_id = validate_id(study_id)
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="No classifier checkpoint is configured; analysis was not run.",
        )
    modalities = await study_modalities(study_id)
    if not modalities or not modalities.issubset({"CR", "DX"}):
        raise HTTPException(
            status_code=422,
            detail="2D classification supports CR/DX studies only.",
        )
    instance_id = await first_instance_id(study_id)
    response = await orthanc_get(f"/instances/{instance_id}/file")
    try:
        result = classifier.predict(response.content)
    except (ValueError, KeyError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not analyze DICOM: {exc}") from exc
    analysis_id = audit.save_analysis(AUDIT_DB, study_id, result)
    return {**result, "analysis_id": analysis_id, "disclaimer": DISCLAIMER}


@app.get("/api/studies/{study_id}/analysis")
async def latest_analysis(study_id: str) -> dict[str, object] | None:
    study_id = validate_id(study_id)
    modalities = await study_modalities(study_id)
    if not modalities or not modalities.issubset({"CR", "DX"}):
        return None
    result = audit.get_latest_analysis(AUDIT_DB, study_id)
    if result is None:
        return None
    return {**result, "disclaimer": DISCLAIMER}


@app.post("/api/studies/{study_id}/segment")
async def segment_study(study_id: str) -> dict[str, object]:
    study_id = validate_id(study_id)
    if segmenter is None:
        raise HTTPException(
            status_code=503,
            detail="No 3D segmentation checkpoint is configured; segmentation was not run.",
        )
    if "CT" not in await study_modalities(study_id):
        raise HTTPException(
            status_code=422,
            detail="3D segmentation supports CT studies only.",
        )
    try:
        result = segmenter.predict(await ct_series_files(study_id))
    except (ValueError, KeyError, RuntimeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not segment this CT series: {exc}",
        ) from exc
    summary = {
        key: result[key]
        for key in ("slice_index", "slice_count", "voxel_counts", "model_version")
    }
    segmentation_id = audit.save_segmentation(AUDIT_DB, study_id, summary)
    return {**result, "segmentation_id": segmentation_id}


@app.post("/api/studies/{study_id}/review")
async def record_review(study_id: str, request: ReviewRequest) -> dict[str, str]:
    reviewer = request.reviewer.strip()
    if not reviewer:
        raise HTTPException(status_code=422, detail="Reviewer identifier must not be blank")
    review = audit.save_review(
        AUDIT_DB,
        request.analysis_id,
        validate_id(study_id),
        reviewer,
        request.decision,
        request.note.strip(),
    )
    if review is None:
        raise HTTPException(status_code=404, detail="Analysis not found for this study")
    return review


@app.get("/api/audit")
async def audit_history(study_id: str = Query(min_length=1, max_length=128)):
    return audit.get_reviews(AUDIT_DB, validate_id(study_id))


@app.get("/api/segmentations")
async def segmentation_history(study_id: str = Query(min_length=1, max_length=128)):
    return audit.get_segmentations(AUDIT_DB, validate_id(study_id))
