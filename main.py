import os
import json
import uuid
import threading
from io import BytesIO
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from engine.pipeline import run_negative_keyword_pipeline

load_dotenv()

app = FastAPI(title="TermTidy API", version="0.2.0")

# If you want CORS (optional when using Next.js proxy). Safe to keep.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Existing JSON /run endpoint
# ----------------------------
class RunRequest(BaseModel):
    search_terms: List[Dict[str, Any]] = Field(..., description="Rows from Google Ads search terms export")
    keywords: List[Dict[str, Any]] = Field(..., description="Rows from Google Ads keywords export")

    min_clicks: int = 3
    min_cost: float = 0.0
    similarity_threshold: float = 0.75
    use_llm: bool = True
    batch_size: int = 5
    currency: str = "GBP"
    brand_terms: List[str] = []

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def root():
    return {"ok": True, "service": "termtidy-api"}

@app.post("/run")
def run_audit(req: RunRequest):
    try:
        search_df = pd.DataFrame(req.search_terms)
        kw_df = pd.DataFrame(req.keywords)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input format: {e}")

    cfg = {
        "min_clicks": req.min_clicks,
        "min_cost": req.min_cost,
        "similarity_threshold": req.similarity_threshold,
        "use_llm": req.use_llm,
        "batch_size": req.batch_size,
        "embedding_model": "text-embedding-3-small",
        "chat_model": "gpt-4.1-mini",
        "brand_terms": req.brand_terms,
        "currency": req.currency,
    }

    try:
        final_df, stats = run_negative_keyword_pipeline(search_df, kw_df, cfg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    if final_df is None or final_df.empty:
        return {"ok": True, "stats": stats, "results": []}

    final_df = final_df.fillna("")
    return {"ok": True, "stats": stats, "results": final_df.to_dict(orient="records")}

# ----------------------------
# Job mode (NEW)
# ----------------------------

# In-memory job store (good for now; later move to Redis/Supabase)
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()

def _set_job(job_id: str, patch: Dict[str, Any]):
    with JOBS_LOCK:
        job = JOBS.get(job_id, {})
        job.update(patch)
        JOBS[job_id] = job

def _read_job(job_id: str) -> Optional[Dict[str, Any]]:
    with JOBS_LOCK:
        return JOBS.get(job_id)

def _run_job(job_id: str, search_df: pd.DataFrame, kw_df: pd.DataFrame, cfg: Dict[str, Any]):
    try:
        _set_job(job_id, {"status": "running", "progress": 10})

        final_df, stats = run_negative_keyword_pipeline(search_df, kw_df, cfg)

        _set_job(job_id, {"progress": 90})

        if final_df is None or final_df.empty:
            _set_job(job_id, {"status": "done", "progress": 100, "stats": stats, "results": []})
            return

        final_df = final_df.fillna("")
        _set_job(
            job_id,
            {
                "status": "done",
                "progress": 100,
                "stats": stats,
                "results": final_df.to_dict(orient="records"),
            },
        )
    except Exception as e:
        _set_job(
            job_id,
            {
                "status": "error",
                "progress": 100,
                "error": {"message": str(e)},
            },
        )

@app.post("/jobs")
async def start_job(
    search_terms_file: UploadFile = File(...),
    keywords_file: UploadFile = File(...),

    min_clicks: int = Form(3),
    min_cost: float = Form(0.0),
    similarity_threshold: float = Form(0.75),
    use_llm: bool = Form(True),
    batch_size: int = Form(5),
    currency: str = Form("GBP"),
    brand_terms: str = Form(""),
):
    # Read CSVs
    try:
        search_bytes = await search_terms_file.read()
        kw_bytes = await keywords_file.read()
        search_df = pd.read_csv(BytesIO(search_bytes))
        kw_df = pd.read_csv(BytesIO(kw_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSVs: {e}")

    cfg = {
        "min_clicks": int(min_clicks),
        "min_cost": float(min_cost),
        "similarity_threshold": float(similarity_threshold),
        "use_llm": bool(use_llm),
        "batch_size": int(batch_size),
        "embedding_model": "text-embedding-3-small",
        "chat_model": "gpt-4.1-mini",
        "brand_terms": [t.strip() for t in brand_terms.split(",") if t.strip()],
        "currency": currency,
    }

    job_id = uuid.uuid4().hex
    _set_job(job_id, {"status": "queued", "progress": 0})

    # Start background thread
    t = threading.Thread(target=_run_job, args=(job_id, search_df, kw_df, cfg), daemon=True)
    t.start()

    return {"ok": True, "job_id": job_id, "status": "queued"}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = _read_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Standard shape your frontend expects
    return {
        "ok": True,
        "job_id": job_id,
        "status": job.get("status", "queued"),
        "progress": job.get("progress", 0),
        "stats": job.get("stats"),
        "results": job.get("results"),
        "error": job.get("error"),
    }
