import os
import uuid
import asyncio
import traceback
from io import BytesIO
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from supabase import create_client, Client

from engine.pipeline import run_negative_keyword_pipeline
from engine.email import send_job_complete_email

load_dotenv()

app = FastAPI(title="TermTidy API", version="0.5.0")

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

# Default bucket name (must match what you created in Supabase)
DEFAULT_UPLOADS_BUCKET = os.getenv("TERMTIDY_UPLOADS_BUCKET", "termtidy-uploads")

# Table names (override via env if you want)
JOBS_TABLE = os.getenv("TERMTIDY_JOBS_TABLE", "audit_jobs")
SUBS_TABLE = os.getenv("TERMTIDY_SUBSCRIPTIONS_TABLE", "subscriptions")
DEFAULT_LLM_BATCH_SIZE = 200

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def _require_supabase() -> Client:
    if not supabase:
        raise RuntimeError(
            "Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."
        )
    return supabase


# ----------------------------
# Job helpers
# ----------------------------
def _job_update(job_id: str, patch: Dict[str, Any]):
    sb = _require_supabase()
    sb.table(JOBS_TABLE).update(patch).eq("id", job_id).execute()


def _maybe_update_job(update_job, job_id, progress):
    if update_job and job_id:
        try:
            update_job(job_id, {"progress": progress})
        except Exception as e:
            print(f"[maybe_update_job] failed to update progress {progress}: {e}")


def _log_task_result(task: "asyncio.Task", job_id: str) -> None:
    try:
        task.result()
    except Exception as e:
        print(f"[job_task] job_id={job_id} background task failed: {e}")


def _send_completion_email(job_id: str, results_list: List[Dict[str, Any]]) -> None:
    try:
        print(f"[email] preparing completion email job_id={job_id}")
        job = _job_read(job_id)
        user_id = job.get("user_id")
        if not user_id:
            print(f"[email] missing user_id for job_id={job_id}")
            return

        sb = _require_supabase()
        user_resp = sb.auth.admin.get_user_by_id(user_id)
        user_email = getattr(getattr(user_resp, "user", None), "email", None)
        if not user_email:
            print(f"[email] missing user email for job_id={job_id}")
            return

        send_job_complete_email(
            to_email=user_email,
            job_id=job_id,
            results=results_list,
        )
    except Exception as e:
        print(f"[email] failed to send completion email: {e}")


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
    sb.table(JOBS_TABLE).insert(payload).execute()
    return job_id


def _job_read(job_id: str) -> Dict[str, Any]:
    sb = _require_supabase()
    res = sb.table(JOBS_TABLE).select("*").eq("id", job_id).limit(1).execute()
    if getattr(res, "error", None):
        print(f"[_job_read] Supabase error for job_id={job_id}: {res.error}")
    data = res.data or []
    if not data:
        raise KeyError("Job not found")
    return data[0]


def _job_is_canceled(job_id: str) -> bool:
    try:
        job = _job_read(job_id)
        return job.get("status") == "canceled"
    except Exception:
        return False


# ----------------------------
# Date helpers (billing window)
# ----------------------------
def _month_start_utc_iso_date(d: Optional[pd.Timestamp] = None) -> str:
    # YYYY-MM-DD of first day of month UTC
    if d is None:
        d = pd.Timestamp.utcnow()
    y = int(d.year)
    m = int(d.month)
    return f"{y:04d}-{m:02d}-01"


def _date_only_utc_from_iso(iso_str: str) -> Optional[str]:
    # Convert an ISO datetime string to YYYY-MM-DD in UTC (date only)
    try:
        dt = pd.to_datetime(iso_str, utc=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def _get_period_start_for_user(user_id: str) -> str:
    """
    Uses subscriptions.current_period_start if present; else falls back to UTC month start.
    Returns YYYY-MM-DD.
    """
    sb = _require_supabase()
    try:
        res = (
            sb.table(SUBS_TABLE)
            .select("current_period_start,status")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        row = res.data
        if row and row.get("current_period_start"):
            d = _date_only_utc_from_iso(row["current_period_start"])
            if d:
                return d
    except Exception:
        pass

    return _month_start_utc_iso_date()


# ----------------------------
# CSV cleaning + filtering helpers
# ----------------------------
def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _find_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def _money_to_float(series: pd.Series) -> pd.Series:
    # Handles £, commas, and "1,234.56" etc.
    s = series.astype(str).fillna("")
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace(r"[^\d.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def _clean_and_filter_search_terms(
    search_df: pd.DataFrame,
    min_clicks: int,
    min_cost: float,
) -> pd.DataFrame:
    """
    1) Normalise column names
    2) Drop rows where the search term is blank
    3) Apply min_clicks and min_cost filters (if those columns exist)
    """
    if search_df is None or search_df.empty:
        return pd.DataFrame()

    df = _norm_cols(search_df)

    term_col = _find_col(
        list(df.columns),
        ["search term", "search term (search query)", "search query", "query"],
    )

    if term_col:
        df[term_col] = df[term_col].astype(str).fillna("").str.strip()
        df = df[df[term_col] != ""]
    else:
        df = df.dropna(how="all")

    clicks_col = _find_col(list(df.columns), ["clicks"])
    cost_col = _find_col(list(df.columns), ["cost", "cost (gbp)"])

    if clicks_col:
        clicks = pd.to_numeric(df[clicks_col], errors="coerce").fillna(0).astype(int)
        df = df[clicks >= int(min_clicks)]

    if cost_col:
        cost = _money_to_float(df[cost_col])
        df = df[cost >= float(min_cost)]

    return df


# ----------------------------
# Supabase Storage helpers
# ----------------------------
def _download_csv_from_storage(bucket: str, path: str) -> pd.DataFrame:
    sb = _require_supabase()

    if not bucket or not path:
        raise ValueError("Missing bucket/path")

    # supabase-py storage download returns bytes
    try:
        file_bytes = sb.storage.from_(bucket).download(path)
    except Exception as e:
        raise RuntimeError(f"Failed to download from storage: {bucket}/{path} ({e})")

    try:
        return pd.read_csv(BytesIO(file_bytes))
    except Exception as e:
        raise RuntimeError(f"Failed to parse CSV: {bucket}/{path} ({e})")


# ----------------------------
# Metering helper (reserve_terms RPC)
# ----------------------------
def _reserve_terms(user_id: str, month_start: str, amount: int) -> Dict[str, Any]:
    sb = _require_supabase()
    try:
        res = sb.rpc(
            "reserve_terms",
            {
                "p_user_id": user_id,
                "p_month_start": month_start,
                "p_amount": int(amount),
            },
        ).execute()
    except Exception as e:
        raise RuntimeError(f"reserve_terms RPC failed: {e}")

    data = res.data
    # Supabase can return list or dict depending on RPC definition
    row = data[0] if isinstance(data, list) and data else data
    if not isinstance(row, dict):
        raise RuntimeError(f"Unexpected reserve_terms response: {data}")

    return row


# ----------------------------
# Existing JSON /run endpoint (kept)
# ----------------------------
class RunRequest(BaseModel):
    search_terms: List[Dict[str, Any]] = Field(..., description="Rows from search terms export")
    keywords: List[Dict[str, Any]] = Field(..., description="Rows from keywords export")

    min_clicks: int = 3
    min_cost: float = 0.0
    similarity_threshold: float = 0.75
    use_llm: bool = True
    currency: str = "GBP"
    brand_terms: List[str] = []


# ----------------------------
# New Job request (Storage paths)
# ----------------------------
class JobStartRequest(BaseModel):
    uploads_bucket: str = Field(default=DEFAULT_UPLOADS_BUCKET)
    search_path: str
    keywords_path: str

    min_clicks: int = 3
    min_cost: float = 0.0
    similarity_threshold: float = 0.75
    use_llm: bool = True
    currency: str = "GBP"
    brand_terms: str = ""  # comma separated (keeps parity with Next)


# ----------------------------
# Estimate request (Storage path)
# ----------------------------
class EstimateRequest(BaseModel):
    uploads_bucket: str = Field(default=DEFAULT_UPLOADS_BUCKET)
    search_path: str
    min_clicks: int = 3
    min_cost: float = 0.0


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
        "batch_size": DEFAULT_LLM_BATCH_SIZE,
        "embedding_model": "text-embedding-3-small",
        "chat_model": "gpt-4.1-mini",
        "brand_terms": req.brand_terms,
        "currency": req.currency,
    }

    search_df = _clean_and_filter_search_terms(search_df, cfg["min_clicks"], cfg["min_cost"])
    if search_df.empty:
        return {
            "ok": True,
            "stats": {
                "initial_rows": 0,
                "filtered_rows": 0,
                "candidates": 0,
                "negatives_before_brand": 0,
                "negatives_after_brand": 0,
            },
            "results": [],
        }

    try:
        final_df, stats = run_negative_keyword_pipeline(
            search_df,
            kw_df,
            cfg,
            job_id=job_id,
            update_job=_job_update,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    if final_df is None or final_df.empty:
        return {"ok": True, "stats": stats, "results": []}

    final_df = final_df.fillna("")
    return {"ok": True, "stats": stats, "results": final_df.to_dict(orient="records")}


@app.post("/estimate")
def estimate_terms(req: EstimateRequest):
    """
    Estimate billable search terms without running the audit.
    """
    try:
        _require_supabase()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase config error: {e}")

    try:
        search_df = _download_csv_from_storage(req.uploads_bucket, req.search_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download search CSV: {e}")

    initial_rows = int(len(search_df)) if search_df is not None else 0

    try:
        filtered_search_df = _clean_and_filter_search_terms(
            search_df, int(req.min_clicks), float(req.min_cost)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to filter search terms: {e}")

    filtered_rows = int(len(filtered_search_df)) if filtered_search_df is not None else 0

    return {
        "ok": True,
        "initial_rows": initial_rows,
        "filtered_rows": filtered_rows,
    }


# ----------------------------
# Job runner
# ----------------------------
async def _run_job(job_id: str, search_df: pd.DataFrame, kw_df: pd.DataFrame, cfg: Dict[str, Any]):
    try:
        print(f"[job {job_id}] starting run")
        _job_update(
            job_id,
            {
                "status": "running",
                "progress": 5,
                "message": f"Starting audit… ({len(search_df):,} rows to process)",
                "error": None,
            },
        )

        if _job_is_canceled(job_id):
            _job_update(job_id, {"status": "canceled", "progress": 100, "message": "Canceled."})
            return

        _job_update(job_id, {"progress": 15, "message": "Preparing data…"})
        await asyncio.sleep(0.05)

        if _job_is_canceled(job_id):
            _job_update(job_id, {"status": "canceled", "progress": 100, "message": "Canceled."})
            return

        _job_update(job_id, {"progress": 25, "message": "Running pipeline (this may take a couple of minutes)…"})

        final_df, stats = await asyncio.to_thread(
            run_negative_keyword_pipeline,
            search_df,
            kw_df,
            cfg,
            job_id,
            _job_update,
        )
        print(
            f"[job {job_id}] pipeline returned rows={len(final_df) if final_df is not None else None}"
        )

        if _job_is_canceled(job_id):
            _job_update(job_id, {"status": "canceled", "progress": 100, "message": "Canceled."})
            return

        _job_update(job_id, {"progress": 90, "message": "Finalising results…"})
        await asyncio.sleep(0.05)

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
        results_list = final_df.to_dict(orient="records")
        print(f"[job {job_id}] updating job row with done + results")
        _job_update(
            job_id,
            {
                "status": "done",
                "progress": 100,
                "message": f"Complete. Found {len(final_df)} suggested negatives.",
                "stats": stats,
                "results": results_list,
                "error": None,
            },
        )
        print(f"[job {job_id}] job row updated with done + results")
        if results_list:
            print(f"[job {job_id}] scheduling completion email")
            task = asyncio.create_task(
                asyncio.to_thread(_send_completion_email, job_id, results_list)
            )
            task.add_done_callback(lambda t: _log_task_result(t, job_id))
        else:
            print(f"[job {job_id}] no results, skipping completion email")
    except Exception as e:
        print(f"[job {job_id}] error: {e}")
        print(traceback.format_exc())
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
    req: JobStartRequest,
    x_user_id: Optional[str] = Header(default=None),
):
    """
    Storage-first job start:
      - downloads CSVs from Supabase Storage
      - removes blank rows + applies min clicks/cost filters
      - meters based on filtered_rows via reserve_terms RPC
      - starts background pipeline task with filtered dataframe
    """
    # Ensure Supabase configured
    try:
        _require_supabase()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Jobs require Supabase config: {e}")

    if not x_user_id:
        raise HTTPException(status_code=401, detail="Missing x-user-id header")

    # Create job row first (durable)
    job_id = _job_create(user_id=x_user_id)

    try:
        _job_update(job_id, {"message": "Downloading files…", "progress": 1})

        # Download CSVs from storage
        search_df = _download_csv_from_storage(req.uploads_bucket, req.search_path)
        kw_df = _download_csv_from_storage(req.uploads_bucket, req.keywords_path)

        initial_rows = int(len(search_df))

        cfg = {
            "min_clicks": int(req.min_clicks),
            "min_cost": float(req.min_cost),
            "similarity_threshold": float(req.similarity_threshold),
            "use_llm": bool(req.use_llm),
            "batch_size": DEFAULT_LLM_BATCH_SIZE,
            "embedding_model": "text-embedding-3-small",
            "chat_model": "gpt-4.1-mini",
            "brand_terms": [t.strip() for t in (req.brand_terms or "").split(",") if t.strip()],
            "currency": req.currency,
        }

        _job_update(job_id, {"message": "Applying filters…", "progress": 3})

        filtered_search_df = _clean_and_filter_search_terms(
            search_df, cfg["min_clicks"], cfg["min_cost"]
        )
        filtered_rows = int(len(filtered_search_df))

        if filtered_rows == 0:
            _job_update(
                job_id,
                {
                    "status": "done",
                    "progress": 100,
                    "message": "No rows left after filters (blank rows removed + min clicks/cost applied).",
                    "stats": {
                        "initial_rows": initial_rows,
                        "filtered_rows": 0,
                        "candidates": 0,
                        "negatives_before_brand": 0,
                        "negatives_after_brand": 0,
                    },
                    "results": [],
                    "error": None,
                },
            )
            return {"ok": True, "job_id": job_id, "status": "done"}

        # Metering based on FILTERED rows
        month_start = _get_period_start_for_user(x_user_id)
        _job_update(
            job_id,
            {
                "message": f"Reserving quota… ({filtered_rows:,} billable rows)",
                "progress": 5,
            },
        )

        reserve_row = _reserve_terms(x_user_id, month_start, filtered_rows)

        if not reserve_row.get("ok", False):
            # Quota exceeded -> mark job error and return 402
            detail = {
                "reason": reserve_row.get("reason") or "quota_exceeded",
                "requested": filtered_rows,
                "month_start": month_start,
                "quota": reserve_row.get("quota"),
                "used": reserve_row.get("used"),
                "remaining": reserve_row.get("remaining"),
            }
            _job_update(
                job_id,
                {
                    "status": "error",
                    "progress": 100,
                    "message": "Quota exceeded.",
                    "error": {"message": "Quota exceeded", "detail": detail},
                    "stats": {
                        "initial_rows": initial_rows,
                        "filtered_rows": filtered_rows,
                        "candidates": 0,
                        "negatives_before_brand": 0,
                        "negatives_after_brand": 0,
                    },
                    "results": [],
                },
            )
            raise HTTPException(status_code=402, detail={"error": "Quota exceeded", "detail": detail, "job_id": job_id})

        # Store pre-pipeline stats so UI can see counts even if pipeline fails later
        _job_update(
            job_id,
            {
                "message": f"Queued… ({filtered_rows:,} rows after filters)",
                "progress": 0,
                "stats": {
                    "initial_rows": initial_rows,
                    "filtered_rows": filtered_rows,
                    "candidates": 0,
                    "negatives_before_brand": 0,
                    "negatives_after_brand": 0,
                },
                "error": None,
            },
        )

        # Start background task with FILTERED dataframe
        task = asyncio.create_task(_run_job(job_id, filtered_search_df, kw_df, cfg))
        task.add_done_callback(lambda t: _log_task_result(t, job_id))

        return {"ok": True, "job_id": job_id, "status": "queued"}

    except HTTPException:
        # already handled / updated job where appropriate
        raise
    except Exception as e:
        _job_update(
            job_id,
            {
                "status": "error",
                "progress": 100,
                "message": "Failed.",
                "error": {"message": str(e)},
            },
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        job = _job_read(job_id)
    except KeyError:
        # Job is legitimately missing
        return {"ok": False, "error": "Job not found", "job_id": job_id}
    except Exception as e:
        # Log the real error in Render logs
        print(f"[get_job] Failed to read job {job_id}: {repr(e)}")
        return {"ok": False, "error": "Failed to read job", "detail": str(e), "job_id": job_id}

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
        _job_read(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read job: {e}")

    _job_update(job_id, {"status": "canceled", "progress": 100, "message": "Cancel requested."})
    return {"ok": True, "job_id": job_id, "status": "canceled"}
