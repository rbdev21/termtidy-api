import os
import uuid
import threading
import time
from io import BytesIO
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from supabase import create_client, Client

from engine.pipeline import run_negative_keyword_pipeline

load_dotenv()

app = FastAPI(title="TermTidy API", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Supabase client (service role)
# ----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def _require_supabase() -> Client:
    if not supabase:
        raise RuntimeError(
            "Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."
        )
    return supabase


def _job_update(job_id: str, patch: Dict[str, Any]):
    sb = _require_supabase()
    sb.table("audit_jobs").update(patch).eq("id", job_id).execute()


def _job_create(user_id: Optional[str]) -> str:
    sb = _require_supabase()
    job_id = str(uuid.uuid4())
    payload = {
        "id": job_id,
        "user_id": user_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued…",
        "stats": None,
        "results": None,
        "error": None,
    }
    sb.table("audit_jobs").insert(payload).execute()
    return job_id


def _job_read(job_id: str) -> Dict[str, Any]:
    sb = _require_supabase()
    res = sb.table("audit_jobs").select("*").eq("id", job_id).single().execute()
    data = res.data
    if not data:
        raise KeyError("Job not found")
    return data


def _job_is_canceled(job_id: str) -> bool:
    # Small helper so the runner can bail out early.
    try:
        job = _job_read(job_id)
        return (job.get("status") == "canceled")
    except Exception:
        # If we can't read the job, don't hard-crash the worker loop.
        return False


# ----------------------------
# Existing JSON /run endpoint
# ----------------------------
class RunRequest(BaseModel):
    search_terms: List[Dict[str, Any]] = Field(
        ..., description="Rows from Google Ads search terms export"
    )
    keywords: List[Dict[str, Any]] = Field(
        ..., description="Rows from Google Ads keywords export"
    )

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
# Job mode (durable in Supabase)
# ----------------------------
def _run_job(job_id: str, search_df: pd.DataFrame, kw_df: pd.DataFrame, cfg: Dict[str, Any]):
    """
    Background runner.
    We can't get granular progress out of the pipeline without callbacks,
    but we CAN:
      - update message/progress at meaningful milestones
      - honour cancellation requests
    """
    try:
        _job_update(job_id, {"status": "running", "progress": 5, "message": "Starting audit…", "error": None})

        if _job_is_canceled(job_id):
            _job_update(job_id, {"status": "canceled", "progress": 100, "message": "Canceled."})
            return

        _job_update(job_id, {"progress": 10, "message": "Preparing data…"})
        time.sleep(0.05)  # tiny yield so UI can pick it up quickly

        if _job_is_canceled(job_id):
            _job_update(job_id, {"status": "canceled", "progress": 100, "message": "Canceled."})
            return

        _job_update(job_id, {"progress": 20, "message": "Running pipeline (this may take a couple of minutes)…"})

        # Main work
        final_df, stats = run_negative_keyword_pipeline(search_df, kw_df, cfg)

        if _job_is_canceled(job_id):
            _job_update(job_id, {"status": "canceled", "progress": 100, "message": "Canceled."})
            return

        _job_update(job_id, {"progress": 90, "message": "Finalising results…"})
        time.sleep(0.05)

        if final_df is None or final_df.empty:
            _job_update(
                job_id,
                {
                    "status": "done",
                    "progress": 100,
                    "message": "Complete. No negatives found.",
                    "stats": stats,
                    "results": [],
                    "error": None,
                },
            )
            return

        final_df = final_df.fillna("")
        _job_update(
            job_id,
            {
                "status": "done",
                "progress": 100,
                "message": f"Complete. Found {len(final_df)} suggested negatives.",
                "stats": stats,
                "results": final_df.to_dict(orient="records"),
                "error": None,
            },
        )
    except Exception as e:
        # Always store a predictable shape
        _job_update(
            job_id,
            {
                "status": "error",
                "progress": 100,
                "message": "Failed.",
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
    x_user_id: Optional[str] = Header(default=None),  # sent by Next proxy (recommended)
):
    # Ensure Supabase is configured
    try:
        _require_supabase()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Jobs require Supabase config: {e}")

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

    job_id = _job_create(user_id=x_user_id)

    # Start background thread
    t = threading.Thread(target=_run_job, args=(job_id, search_df, kw_df, cfg), daemon=True)
    t.start()

    return {"ok": True, "job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        job = _job_read(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read job: {e}")

    return {
        "ok": True,
        "job_id": job["id"],
        "status": job.get("status", "queued"),
        "progress": job.get("progress", 0),
        "message": job.get("message"),
        "stats": job.get("stats"),
        "results": job.get("results"),
        "error": job.get("error"),
    }


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """
    Best-effort cancel:
    - sets job status to canceled
    - runner checks status at safe points and stops
    """
    try:
        _job_read(job_id)  # ensure exists
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read job: {e}")

    _job_update(job_id, {"status": "canceled", "progress": 100, "message": "Cancel requested."})
    return {"ok": True, "job_id": job_id, "status": "canceled"}
