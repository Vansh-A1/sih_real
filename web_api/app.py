"""Local, single-worker HTTP bridge to the repository's saved EMA model.

Run: .venv/bin/python -m uvicorn web_api.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from uuid import uuid4
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
from pydantic import BaseModel
import rasterio
from rasterio.enums import ColorInterp
from starlette.concurrency import run_in_threadpool
import torch

import inference
from s2_evidencesr.data.fixtures import phantom
from s2_evidencesr.training.state import model_from_checkpoint

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 20 * 1024 * 1024
MAX_SIDE = 1024
MAX_IMAGES = 8
MAX_JOBS = 8
TTL = 60 * 60
LOG = logging.getLogger("rezx.api")


@dataclass
class ImageRecord:
    id: str
    directory: Path
    name: str = ""
    width: int = 0
    height: int = 0
    geo: dict | None = None
    settings: dict = field(default_factory=dict)
    sample: bool = False
    ready: bool = False
    used: float = field(default_factory=time.monotonic)

    def public(self):
        return {"id": self.id, "name": self.name, "width": self.width,
                "height": self.height, "sample": self.sample,
                "url": f"/api/images/{self.id}/preview", "settings": self.settings}


@dataclass
class Job:
    id: str
    image: ImageRecord
    directory: Path
    status: str = "queued"
    progress: int = 0
    message: str = "Waiting for the model"
    error: str | None = None
    result: dict | None = None
    used: float = field(default_factory=time.monotonic)
    cancel: threading.Event = field(default_factory=threading.Event)

    def public(self):
        return {"id": self.id, "status": self.status, "progress": self.progress,
                "message": self.message, "error": self.error, "result": self.result}


class InferenceCancelled(Exception):
    pass


def numbers(value: str, label: str):
    try:
        result = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError(f"{label} must be a number or four comma-separated numbers.") from exc
    if len(result) not in (1, 4) or not np.isfinite(result).all():
        raise ValueError(f"{label} needs one or four finite values.")
    if label == "Scale" and min(result) <= 0:
        raise ValueError("Scale must be positive.")
    return result[0] if len(result) == 1 else result


def prepare_image(record: ImageRecord, path: Path, bands: str, scale: str, offset: str, layout: str):
    try:
        indices = tuple(int(x.strip()) for x in bands.split(","))
    except ValueError as exc:
        raise ValueError("Band order needs four distinct one-based band numbers.") from exc
    if len(indices) != 4 or len(set(indices)) != 4 or min(indices) < 1 or max(indices) > 16:
        raise ValueError("Select four distinct band numbers from 1–16, in B2, B3, B4, B8 order.")
    if layout not in ("CHW", "HWC"):
        raise ValueError("Choose CHW or HWC for the NumPy layout.")
    factor, shift = numbers(scale, "Scale"), numbers(offset, "Offset")
    if path.suffix == ".npy":
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        if raw.ndim != 3 or raw.dtype.kind not in "fiu":
            raise ValueError("Upload a numeric three-dimensional NumPy array with four measured bands.")
        channels, height, width = raw.shape if layout == "CHW" else (raw.shape[2], raw.shape[0], raw.shape[1])
        integer = raw.dtype.kind in "iu"
        del raw
    else:
        with path.open("rb") as handle:
            if handle.read(4) not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
                raise ValueError("This file is not a valid TIFF. Use a four-band TIFF or NumPy array.")
        with rasterio.open(path) as dataset:
            if dataset.driver != "GTiff":
                raise ValueError("Only TIFF rasters and NumPy arrays are supported.")
            channels, height, width = dataset.count, dataset.height, dataset.width
            integer = any(np.dtype(dtype).kind in "iu" for dtype in dataset.dtypes)
            if channels >= max(indices) and dataset.colorinterp[indices[3] - 1] == ColorInterp.alpha:
                raise ValueError("An alpha/transparency channel is not near-infrared. Upload measured B2, B3, B4 and B8 bands.")
            for index, band in zip(indices, ("B2", "B3", "B4", "B8")):
                if index <= channels:
                    label = dataset.descriptions[index - 1]
                    if label in ("B2", "B3", "B4", "B8") and label != band:
                        raise ValueError(f"Band {index} is labeled {label}; select the {band} band in Input settings.")
    if channels < max(indices) or channels > 16:
        raise ValueError("The model needs four measured bands, including near-infrared. An RGB image is insufficient.")
    if not (1 <= height <= MAX_SIDE and 1 <= width <= MAX_SIDE):
        raise ValueError(f"Use an image no larger than {MAX_SIDE} × {MAX_SIDE} pixels.")
    if integer and np.all(np.asarray(factor) == 1) and np.all(np.asarray(shift) == 0):
        raise ValueError("This file stores integer values. Set the scale and offset used during training in Input settings.")
    lr, weights, geo = inference.load_image(path, indices, factor, shift, layout)
    if any(not np.isfinite(band).any() for band in lr):
        raise ValueError("Every selected band must contain valid pixels.")
    np.save(record.directory / "lr.npy", lr, allow_pickle=False)
    np.save(record.directory / "weights.npy", weights, allow_pickle=False)
    limits = inference.display_limits(lr[[2, 1, 0]])
    np.savez(record.directory / "display.npz", low=limits[0], high=limits[1])
    save_preview(lr, limits, record.directory / "original.png")
    record.height, record.width, record.geo = height, width, geo
    record.settings = {"bands": list(indices), "scale": factor, "offset": shift, "layout": layout}
    record.ready = True


def save_preview(cube, limits, destination):
    rgba = inference.rgb_preview(cube[[2, 1, 0]], limits, max_side=MAX_SIDE * 4)
    Image.fromarray(np.rint(rgba * 255).astype(np.uint8)).save(destination)


class Service:
    def __init__(self, checkpoint: Path, device: str, directory: Path):
        self.checkpoint, self.device, self.directory = checkpoint, device, directory
        self.model = None
        self.step = None
        self.digest = None
        self.load_error = None
        self.images: dict[str, ImageRecord] = {}
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rezx-inference")

    def load(self):
        try:
            if not self.checkpoint.is_file():
                raise FileNotFoundError("Checkpoint not found. Set REZX_CHECKPOINT to the repository's last.pt.")
            self.model, checkpoint = model_from_checkpoint(self.checkpoint, device=self.device, ema=True)
            self.step = checkpoint.get("step")
            self.digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
            torch.set_num_threads(self.model.config.train.threads)
            del checkpoint
        except Exception:
            LOG.exception("Could not load the configured checkpoint")
            self.model = None
            self.load_error = "Model unavailable. Check REZX_CHECKPOINT, REZX_DEVICE, and the backend log."

    def health(self):
        return {"status": "ready" if self.model is not None else "unavailable",
                "model": "S2-EvidenceSR-4X", "step": self.step, "ema": True,
                "device": self.device, "checkpoint_sha256": self.digest,
                "error": self.load_error, "bands": ["B2", "B3", "B4", "B8"],
                "scale": 4, "max_bytes": MAX_BYTES, "max_side": MAX_SIDE}

    def reserve_image(self):
        with self.lock:
            self.prune()
            if len(self.images) >= MAX_IMAGES:
                raise HTTPException(429, "The workspace is full. Remove an image or try again later.")
            identifier = uuid4().hex
            record = ImageRecord(identifier, self.directory / identifier)
            record.directory.mkdir()
            self.images[identifier] = record
            return record

    def get_image(self, identifier):
        with self.lock:
            record = self.images.get(identifier)
            if record is None or not record.ready:
                raise HTTPException(404, "This image has expired. Upload it again.")
            record.used = time.monotonic()
            return record

    def get_job(self, identifier):
        with self.lock:
            job = self.jobs.get(identifier)
            if job is None:
                raise HTTPException(404, "This result has expired. Upload the image again.")
            job.used = job.image.used = time.monotonic()
            return job

    def delete_image(self, identifier):
        with self.lock:
            if any(j.image.id == identifier and j.status in ("queued", "processing") for j in self.jobs.values()):
                raise HTTPException(409, "Cancel image processing before removing this image.")
            for job in list(self.jobs.values()):
                if job.image.id == identifier:
                    self.jobs.pop(job.id, None)
            record = self.images.pop(identifier, None)
            if record:
                shutil.rmtree(record.directory, ignore_errors=True)

    def prune(self):
        with self.lock:
            for record in list(self.images.values()):
                if record.ready and time.monotonic() - record.used > TTL:
                    try:
                        self.delete_image(record.id)
                    except HTTPException:
                        pass

    def submit(self, image_id):
        with self.lock:
            if self.model is None:
                raise HTTPException(503, self.load_error)
            image = self.get_image(image_id)
            previous = [j for j in self.jobs.values() if j.image.id == image_id]
            for job in previous:
                if job.status in ("queued", "processing", "complete"):
                    return job
            if sum(j.status in ("queued", "processing") for j in self.jobs.values()) >= MAX_JOBS:
                raise HTTPException(429, "The model is busy. Please try again shortly.")
            for job in previous:
                self.jobs.pop(job.id, None)
                shutil.rmtree(job.directory, ignore_errors=True)
            identifier = uuid4().hex
            job = Job(identifier, image, image.directory / identifier)
            job.directory.mkdir()
            self.jobs[identifier] = job
            self.executor.submit(self.run, job)
            return job

    def update(self, job, **values):
        with self.lock:
            for key, value in values.items():
                setattr(job, key, value)
            job.used = job.image.used = time.monotonic()

    def run(self, job):
        hooks = []
        started = time.monotonic()
        try:
            def check_cancel(*_):
                if job.cancel.is_set():
                    raise InferenceCancelled()
            check_cancel()
            self.update(job, status="processing", progress=2, message="Reading the four-band image")
            lr = np.load(job.image.directory / "lr.npy", allow_pickle=False)
            weights = np.load(job.image.directory / "weights.npy", allow_pickle=False)
            tiling = self.model.config.tiling
            ys = [0] if lr.shape[1] <= tiling.core else list(inference.starts(lr.shape[1], tiling.core, tiling.overlap))
            xs = [0] if lr.shape[2] <= tiling.core else list(inference.starts(lr.shape[2], tiling.core, tiling.overlap))
            total, completed = len(xs) * len(ys), 0

            def after_tile(*_):
                nonlocal completed
                completed += 1
                check_cancel()
                self.update(job, progress=5 + round(85 * completed / total),
                            message=f"Reconstructing image details · Tile {completed} of {total}")

            hooks = [self.model.register_forward_pre_hook(check_cancel), self.model.register_forward_hook(after_tile)]
            self.update(job, progress=5, message=f"Reconstructing image details · Tile 1 of {total}")
            sr = inference.predict_sr(self.model, lr, weights, self.device,
                                      tile_size=tiling.core, halo=tiling.halo, overlap=tiling.overlap)
            check_cancel()
            self.update(job, progress=94, message="Preparing your images and four-band data")
            inference.save_numeric_sr(sr, job.directory, job.image.geo)
            with np.load(job.image.directory / "display.npz") as display:
                save_preview(sr, (display["low"], display["high"]), job.directory / "sr.png")
            report = {"model": "S2-EvidenceSR-4X", "checkpoint_sha256": self.digest,
                      "step": self.step, "ema": True, "mode": "analytical", "device": self.device,
                      "input": job.image.name, "input_settings": job.image.settings,
                      "input_shape": list(lr.shape), "output_shape": list(sr.shape),
                      "bands": ["B2", "B3", "B4", "B8"], "synthetic_sample": job.image.sample,
                      "reference_used_for_inference": False, "physical_accuracy_calibrated": False,
                      "display": "PNG only: shared LR-derived 2–98 percentile RGB stretch. Numeric outputs are unstretched.",
                      "elapsed_seconds": round(time.monotonic() - started, 3)}
            (job.directory / "inference.json").write_text(json.dumps(report, indent=2) + "\n")
            with zipfile.ZipFile(job.directory / "rezx-result.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for name in ("sr.png", "sr.npy", "sr.tif", "inference.json"):
                    if (job.directory / name).exists():
                        archive.write(job.directory / name, name)
            check_cancel()
            self.update(job, status="complete", progress=100, message="Super-resolution complete",
                        result={"url": f"/api/jobs/{job.id}/preview", "download_url": f"/api/jobs/{job.id}/download",
                                "width": sr.shape[2], "height": sr.shape[1], "step": self.step,
                                "elapsed_seconds": report["elapsed_seconds"],
                                "numeric_format": "GeoTIFF + NumPy" if job.image.geo else "NumPy"})
        except InferenceCancelled:
            self.update(job, status="cancelled", message="Processing cancelled")
        except Exception:
            LOG.exception("Inference failed for job %s", job.id)
            self.update(job, status="failed", message="Processing failed",
                        error="The model could not process this image. Check its bands and decoding settings, then retry.")
        finally:
            for hook in hooks:
                hook.remove()
            if job.status in ("cancelled", "failed"):
                shutil.rmtree(job.directory, ignore_errors=True)


class JobRequest(BaseModel):
    image_id: str


def create_app(checkpoint: Path | None = None, device: str | None = None):
    @asynccontextmanager
    async def lifespan(app):
        with tempfile.TemporaryDirectory(prefix="rezx-api-") as directory:
            service = Service(checkpoint or Path(os.getenv("REZX_CHECKPOINT", str(ROOT / "last.pt"))),
                              device or os.getenv("REZX_DEVICE", "cpu"), Path(directory))
            app.state.service = service
            await run_in_threadpool(service.load)

            async def expire():
                while True:
                    await asyncio.sleep(60)
                    await run_in_threadpool(service.prune)

            cleanup = asyncio.create_task(expire())
            try:
                yield
            finally:
                cleanup.cancel()
                for job in service.jobs.values():
                    job.cancel.set()
                await run_in_threadpool(lambda: service.executor.shutdown(wait=True, cancel_futures=True))

    app = FastAPI(title="RezX inference API", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_limits(request, call_next):
        if request.url.path.startswith("/api/"):
            try:
                if int(request.headers.get("content-length", "0")) > MAX_BYTES + 1024 * 1024:
                    return JSONResponse({"detail": "Choose a file smaller than 20 MB."}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "Invalid content length."}, status_code=400)
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/health")
    def health():
        return app.state.service.health()

    @app.post("/api/images", status_code=201)
    async def upload(file: UploadFile = File(...), bands: str = Form("1,2,3,4"),
                     scale: str = Form("1"), offset: str = Form("0"), layout: str = Form("CHW")):
        extension = Path(file.filename or "").suffix.lower()
        if extension not in (".npy", ".tif", ".tiff"):
            await file.close()
            raise HTTPException(422, "Upload a four-band TIFF or NumPy array. PNG/JPG images do not contain measured near-infrared data.")
        service = app.state.service
        record = service.reserve_image()
        record.name = Path((file.filename or "image").replace("\\", "/")).name[:160]
        path = record.directory / ("input" + extension)
        try:
            count = 0
            with path.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    count += len(chunk)
                    if count > MAX_BYTES:
                        raise HTTPException(413, "Choose a file smaller than 20 MB.")
                    handle.write(chunk)
            await run_in_threadpool(prepare_image, record, path, bands, scale, offset, layout)
            return record.public()
        except HTTPException:
            service.delete_image(record.id)
            raise
        except Exception as exc:
            service.delete_image(record.id)
            message = str(exc).replace(str(record.directory), "uploaded image") if isinstance(exc, ValueError) else "This image could not be read. Use a valid four-band TIFF or NumPy array."
            raise HTTPException(422, message) from exc
        finally:
            await file.close()

    @app.post("/api/images/sample", status_code=201)
    def sample():
        service = app.state.service
        record = service.reserve_image()
        record.name, record.sample = "synthetic-four-band-sample.npy", True
        path = record.directory / "input.npy"
        try:
            # Explicit engineering fixture, not a photograph with invented NIR.
            np.save(path, phantom("repeated", 64, 42).numpy(), allow_pickle=False)
            prepare_image(record, path, "1,2,3,4", "1", "0", "CHW")
            return record.public()
        except Exception:
            service.delete_image(record.id)
            raise

    @app.get("/api/images/{image_id}/preview")
    def original(image_id: str):
        record = app.state.service.get_image(image_id)
        return FileResponse(record.directory / "original.png", media_type="image/png")

    @app.delete("/api/images/{image_id}", status_code=204)
    def remove_image(image_id: str):
        app.state.service.delete_image(image_id)

    @app.post("/api/jobs", status_code=202)
    def submit(body: JobRequest):
        service = app.state.service
        with service.lock:
            return service.submit(body.image_id).public()

    @app.get("/api/jobs/{job_id}")
    def status(job_id: str):
        service = app.state.service
        with service.lock:
            return service.get_job(job_id).public()

    @app.delete("/api/jobs/{job_id}", status_code=202)
    def cancel(job_id: str):
        service = app.state.service
        with service.lock:
            job = service.get_job(job_id)
            if job.status in ("queued", "processing"):
                job.cancel.set()
            return job.public()

    def complete_job(identifier):
        job = app.state.service.get_job(identifier)
        if job.status != "complete":
            raise HTTPException(409, "The result is not ready yet.")
        return job

    @app.get("/api/jobs/{job_id}/preview")
    def result_preview(job_id: str):
        return FileResponse(complete_job(job_id).directory / "sr.png", media_type="image/png")

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str):
        job = complete_job(job_id)
        name = Path(job.image.name).stem + "-rezx-4x.zip"
        return FileResponse(job.directory / "rezx-result.zip", media_type="application/zip", filename=name)

    build = ROOT / "frontend" / "dist"
    if build.is_dir():
        app.mount("/", StaticFiles(directory=build, html=True), name="frontend")
    return app


app = create_app()
