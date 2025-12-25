import os
from typing import List, Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from engine.pipeline import run_negative_keyword_pipeline

load_dotenv()

app = FastAPI(title="TermTidy API", version="0.1.0")

# ✅ CORS: allow your Next.js app to call this API in dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    # Data is sent as arrays of records (JSON), not CSVs
    search_terms: List[Dict[str, Any]] = Field(..., description="Rows from Google Ads search terms export")
    keywords: List[Dict[str, Any]] = Field(..., description="Rows from Google Ads keywords export")

    # Config overrides
    min_clicks: int = 3
    min_cost: float = 0.0
    similarity_threshold: float = 0.75
    use_llm: bool = True
    batch_size: int = 5
    currency: str = "GBP"

    # Brand protection
    brand_terms: List[str] = []


@app.get("/")
def root():
    return {"ok": True, "service": "termtidy-api"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/run")
def run_audit(req: RunRequest):
    # Convert JSON rows to DataFrames
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

    # Make JSON-safe (no NaN)
    if final_df is None or final_df.empty:
        return {"ok": True, "stats": stats, "results": []}

    final_df = final_df.replace([pd.NA, float("nan")], "").fillna("")
    results = final_df.to_dict(orient="records")

    return {"ok": True, "stats": stats, "results": results}
