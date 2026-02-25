#!/usr/bin/env python3
"""
OpenBanana — FastAPI backend server.

Provides upload, conversion, and result-serving API.
Run with: python server_pa.py
Server runs at http://localhost:8000
"""

import os
import sys
import time
import uuid
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Must be set before any PyTorch import
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Bypass system proxy (e.g. Clash/VPN) for localhost — needed for LM Studio VLM calls
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI(
    title="OpenBanana API",
    description="Image/PDF to editable DrawIO XML",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
app.mount("/static", StaticFiles(directory=os.path.join(PROJECT_ROOT, "static")), name="static")

# In-memory job store
# {job_id: {status, stage, job_name, error, element_count, created_at}}
jobs: dict[str, dict] = {}

# Single-worker executor so GPU is not double-loaded
_executor = ThreadPoolExecutor(max_workers=1)

# Jobs older than this (seconds) that are still "processing" are declared timed out
_JOB_TIMEOUT = 180


# ---------------------------------------------------------------------------
# Background pipeline runner (runs in thread pool)
# ---------------------------------------------------------------------------

def _set_stage(job_id: str, stage: str) -> None:
    """Update the stage label for an in-progress job (thread-safe via GIL)."""
    if job_id in jobs:
        jobs[job_id]["stage"] = stage


def _run_pipeline(job_id: str, tmp_path: str, ext: str, output_dir: str) -> None:
    """Synchronous pipeline run executed in a thread."""
    try:
        from main import load_config, Pipeline

        _set_stage(job_id, "Loading models…")
        config = load_config()
        pipeline = Pipeline(config)

        def stage_cb(msg: str):
            _set_stage(job_id, msg)

        result_path = pipeline.process_image(
            tmp_path,
            output_dir=output_dir,
            with_refinement=False,
            with_text=True,
            stage_callback=stage_cb,
        )
        if not result_path or not os.path.exists(result_path):
            jobs[job_id] = {
                "status": "error", "stage": "Failed",
                "job_name": None, "error": "Pipeline produced no output", "element_count": 0,
                "created_at": jobs[job_id].get("created_at", 0),
            }
            return

        job_name = Path(result_path).parent.name

        # Try to read element count from metadata
        element_count = 0
        meta_path = Path(PROJECT_ROOT) / "output" / job_name / "sam3_metadata.json"
        if meta_path.exists():
            import json
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                element_count = len(meta.get("elements", []))
            except Exception:
                pass

        jobs[job_id] = {
            "status": "done",
            "stage": "Done",
            "job_name": job_name,
            "error": None,
            "element_count": element_count,
            "created_at": jobs[job_id].get("created_at", 0),
        }
    except Exception as e:
        jobs[job_id] = {
            "status": "error", "stage": "Failed",
            "job_name": None, "error": str(e), "element_count": 0,
            "created_at": jobs[job_id].get("created_at", 0),
        }
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Serve the frontend."""
    index_path = os.path.join(PROJECT_ROOT, "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"service": "OpenBanana", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    """Upload an image or PDF; returns a job_id to poll for status."""
    name = file.filename or "image.png"
    ext = Path(name).suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".pdf", ".bmp", ".tiff", ".webp"}:
        raise HTTPException(400, "Unsupported format. Use image or PDF.")

    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    if not os.path.exists(config_path):
        raise HTTPException(503, "Server not configured — copy config/config.yaml.example to config/config.yaml")

    # Read content before the background task (UploadFile closes after response)
    content = await file.read()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(content)
    tmp.close()

    # Resolve output dir from config
    output_dir = "./output"
    try:
        from main import load_config
        cfg = load_config()
        output_dir = cfg.get("paths", {}).get("output_dir", "./output")
    except Exception:
        pass
    os.makedirs(output_dir, exist_ok=True)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "processing",
        "stage": "Queued…",
        "job_name": None,
        "error": None,
        "element_count": 0,
        "created_at": time.time(),
    }

    loop = asyncio.get_running_loop()
    loop.run_in_executor(_executor, _run_pipeline, job_id, tmp.name, ext, output_dir)

    return {"job_id": job_id}


@app.get("/job/{job_id}")
def get_job(job_id: str):
    """Poll the status of a conversion job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]

    # Lazy timeout: mark stuck jobs as failed when polled
    if job["status"] == "processing":
        elapsed = time.time() - job.get("created_at", time.time())
        if elapsed > _JOB_TIMEOUT:
            jobs[job_id] = {
                **job,
                "status": "error",
                "stage": "Timed out",
                "error": f"Processing exceeded {_JOB_TIMEOUT}s timeout. Image may be too complex.",
            }

    return jobs[job_id]


@app.get("/result/{job_name}")
def get_result(job_name: str):
    """Download the DrawIO XML for a completed job."""
    xml_path = Path(PROJECT_ROOT) / "output" / job_name / f"{job_name}_merged.drawio.xml"
    if not xml_path.exists():
        raise HTTPException(404, "Result not found")
    return FileResponse(
        str(xml_path),
        media_type="application/xml",
        filename=f"{job_name}.drawio",
    )


@app.get("/preview/{job_name}")
def get_preview(job_name: str):
    """Serve the SAM3 segmentation preview image."""
    png_path = Path(PROJECT_ROOT) / "output" / job_name / "sam3_extraction.png"
    if not png_path.exists():
        raise HTTPException(404, "Preview not found")
    return FileResponse(str(png_path), media_type="image/png")


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
