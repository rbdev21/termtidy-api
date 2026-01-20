import os
import json
import math
from typing import List, Dict, Any, Tuple, Optional, Callable

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, BadRequestError

# -------------------------------------------------------------------
# Environment (do not hard-exit inside libraries used by an API server)
# -------------------------------------------------------------------

load_dotenv()

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """
    Lazy-load the OpenAI client so importing this module doesn't kill uvicorn.
    """
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (set it in .env or env vars).")

    _client = OpenAI(api_key=api_key)
    return _client


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )
    return df


def drop_total_rows(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    if text_col not in df.columns:
        return df
    s = df[text_col].astype(str).str.strip().str.lower()
    return df[~s.str.startswith("total:")].copy()


def clean_numeric(col: pd.Series) -> pd.Series:
    col = col.astype(str)
    col = col.str.replace(r"[£$,\s]", "", regex=True)
    col = col.str.replace("-", "0")
    col = col.replace("", "0")
    return pd.to_numeric(col, errors="coerce").fillna(0.0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.matmul(a_norm, b_norm.T)


def batch_list(items: List[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# -------------------------------------------------------------------
# OpenAI helpers
# -------------------------------------------------------------------

def _sanitize_text_for_embedding(x: Any, max_chars: int = 4000) -> str:
    """
    Convert any value to a safe string for embeddings.
    Key rule: NEVER send empty string to the embeddings API (use " ").
    """
    # Handle pandas / numpy missing values
    try:
        if x is None:
            return " "
        # numpy.nan / float nan
        if isinstance(x, float) and np.isnan(x):
            return " "
        # pandas NA
        if x is pd.NA:
            return " "
    except Exception:
        pass

    s = str(x).strip()
    # Common bad tokens
    if s == "" or s.lower() in {"nan", "none", "<na>"}:
        s = " "

    # Trim excessively long strings
    if len(s) > max_chars:
        s = s[:max_chars]

    return s


def get_embeddings(texts: List[Any], model: str, batch_size: int = 256) -> np.ndarray:
    """
    Robust embeddings helper:
    - sanitizes inputs (no empty strings)
    - batches requests
    - returns NxD numpy array
    """
    if not texts:
        return np.zeros((0, 1536))

    client = get_client()

    cleaned = [_sanitize_text_for_embedding(t) for t in texts]

    vectors: List[List[float]] = []
    try:
        for chunk in batch_list(cleaned, batch_size):
            resp = client.embeddings.create(model=model, input=chunk)
            vectors.extend([item.embedding for item in resp.data])

        return np.array(vectors)
    except BadRequestError as e:
        # This is where '$.input is invalid' usually shows up
        raise RuntimeError(f"OpenAI embeddings request invalid. {e}") from e
    except RateLimitError as e:
        raise RuntimeError(f"OpenAI rate limit while getting embeddings. {e}") from e
    except APIError as e:
        raise RuntimeError(f"OpenAI API error while getting embeddings. {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected embeddings error: {e}") from e


def build_llm_prompt(row: Dict[str, Any]) -> str:
    return f"""
You are an expert Google Ads PPC specialist.

We are auditing search terms to decide which should be added as negative keywords.

Targeted keyword: {row['best_keyword']}
Search term: {row['search_term']}
Clicks: {row['clicks']}
Cost: {row['cost']}
Conversions: {row['conversions']}
Semantic similarity (0 to 1) between search term and keyword: {row['similarity']:.2f}

Rules:
- Mark the search term as "exclude": true only if it is clearly irrelevant or low-intent
  for the given keyword and likely to waste budget.
- Do NOT exclude search terms that are obviously relevant, even if they have no conversions yet.
- The suggested_negative should be a single word or short phrase that blocks similar irrelevant queries.
- Keep the reason short (1–2 sentences).

Respond ONLY as strict JSON in this format:
{{
  "exclude": true or false,
  "suggested_negative": "single word or short phrase",
  "reason": "short explanation"
}}
""".strip()


def build_multirow_llm_prompt(rows: List[Dict[str, Any]]) -> str:
    items = []
    for i, row in enumerate(rows):
        items.append(
            {
                "id": i,
                "search_term": row.get("search_term", ""),
                "best_keyword": row.get("best_keyword", ""),
                "clicks": row.get("clicks", 0),
                "cost": row.get("cost", 0),
                "conversions": row.get("conversions", 0),
                "similarity": float(row.get("similarity", 0.0) or 0.0),
            }
        )

    payload = json.dumps(items, ensure_ascii=True)

    return f"""
You are an expert Google Ads PPC specialist.

We are auditing search terms to decide which should be added as negative keywords.
Each item includes a unique "id" plus context. Decide for EACH item.

Rules:
- Mark "exclude": true only if it is clearly irrelevant or low-intent for the given keyword.
- Do NOT exclude search terms that are obviously relevant, even if they have no conversions yet.
- The suggested_negative should be a single word or short phrase that blocks similar irrelevant queries.
- Keep the reason short (1–2 sentences).

Input items (JSON array):
{payload}

Respond ONLY as strict JSON array. Each item must include:
{{
  "id": number,
  "exclude": true or false,
  "suggested_negative": "single word or short phrase",
  "reason": "short explanation"
}}
""".strip()


def parse_llm_batch_response(text: str, expected_len: int) -> List[Dict[str, Any]]:
    data = json.loads(text)
    if isinstance(data, dict) and "decisions" in data:
        data = data["decisions"]
    if not isinstance(data, list):
        raise ValueError("LLM response was not a JSON array.")
    if expected_len and len(data) != expected_len:
        raise ValueError("LLM response length did not match request length.")

    decisions: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        decisions.append(
            {
                "id": int(item.get("id", -1)),
                "exclude": bool(item.get("exclude", False)),
                "suggested_negative": str(item.get("suggested_negative", "")),
                "reason": str(item.get("reason", "")),
            }
        )

    if expected_len and len(decisions) != expected_len:
        raise ValueError("LLM response missing decisions.")

    return decisions


def llm_decide_for_row(row: Dict[str, Any], model: str) -> Dict[str, Any]:
    client = get_client()
    prompt = build_llm_prompt(row)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = completion.choices[0].message.content
        data = json.loads(content)
        return {
            "exclude": bool(data.get("exclude", False)),
            "suggested_negative": str(data.get("suggested_negative", "")),
            "reason": str(data.get("reason", "")),
        }
    except Exception:
        # Fail-safe: keep if LLM fails
        return {
            "exclude": False,
            "suggested_negative": "",
            "reason": "LLM error; default keep.",
        }


def llm_decide_for_batch(rows: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    client = get_client()
    prompt = build_multirow_llm_prompt(rows)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    content = completion.choices[0].message.content or ""
    return parse_llm_batch_response(content, expected_len=len(rows))


def decide_with_llm(
    candidates_df: pd.DataFrame,
    model: str,
    use_llm: bool,
    batch_size: int,
    progress_cb: Optional[Callable[[int, int, int, int, bool], None]] = None,
) -> pd.DataFrame:
    records = candidates_df.to_dict("records")
    decisions: List[Dict[str, Any]] = []

    if not records:
        return candidates_df.assign(exclude=False, reason="No candidates.")

    total = len(records)
    done = 0
    total_batches = max(1, math.ceil(total / max(1, int(batch_size))))

    for batch_index, batch in enumerate(batch_list(records, batch_size=batch_size), start=1):
        if progress_cb:
            progress_cb(batch_index, total_batches, done, total, True)
        if use_llm:
            try:
                batch_decisions = llm_decide_for_batch(batch, model=model)
                batch_decisions = sorted(batch_decisions, key=lambda d: d.get("id", 0))
                for d in batch_decisions:
                    decisions.append(
                        {
                            "exclude": d.get("exclude", False),
                            "suggested_negative": d.get("suggested_negative", ""),
                            "reason": d.get("reason", ""),
                        }
                    )
            except Exception:
                for row in batch:
                    decisions.append(llm_decide_for_row(row, model=model))
        else:
            for row in batch:
                decisions.append(
                    {
                        "exclude": True,
                        "suggested_negative": str(row.get("search_term", "")),
                        "reason": "Heuristic: below similarity threshold; LLM disabled.",
                    }
                )

        done += len(batch)
        if progress_cb:
            progress_cb(batch_index, total_batches, done, total, False)

    return pd.concat([candidates_df.reset_index(drop=True), pd.DataFrame(decisions)], axis=1)


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

def _maybe_update_job(
    update_job: Optional[Callable[[str, Dict[str, Any]], None]],
    job_id: Optional[str],
    progress: int,
    message: str,
) -> None:
    if not update_job or not job_id:
        return
    try:
        update_job(job_id, {"progress": int(progress), "message": message})
    except Exception:
        pass


def run_negative_keyword_pipeline(
    search_df: pd.DataFrame,
    kw_df: pd.DataFrame,
    config: Dict[str, Any],
    job_id: Optional[str] = None,
    update_job: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    min_clicks = config["min_clicks"]
    min_cost = config["min_cost"]
    similarity_threshold = config["similarity_threshold"]
    use_llm = config["use_llm"]
    llm_batch_size = config["batch_size"]
    embedding_model = config["embedding_model"]
    chat_model = config["chat_model"]
    brand_terms_raw = config.get("brand_terms", [])
    brand_terms = [t.strip().lower() for t in brand_terms_raw if str(t).strip()]

    _maybe_update_job(update_job, job_id, 10, "Normalizing data…")

    # Normalize + drop totals
    search_df = normalize_columns(search_df)
    kw_df = normalize_columns(kw_df)
    search_df = drop_total_rows(search_df, "search_term")
    kw_df = drop_total_rows(kw_df, "keyword")

    # Validate
    for col in ["search_term", "clicks", "cost"]:
        if col not in search_df.columns:
            raise ValueError(f"Search terms input must contain '{col}' column")
    if "keyword" not in kw_df.columns:
        raise ValueError("Keywords input must contain 'keyword' column")

    if "conversions" not in search_df.columns:
        search_df["conversions"] = 0

    # Clean numeric
    search_df["clicks"] = clean_numeric(search_df["clicks"])
    search_df["cost"] = clean_numeric(search_df["cost"])
    search_df["conversions"] = clean_numeric(search_df["conversions"])

    initial_count = len(search_df)
    filtered = search_df[(search_df["clicks"] >= min_clicks) & (search_df["cost"] >= min_cost)].copy()

    stats: Dict[str, Any] = {
        "initial_rows": int(initial_count),
        "filtered_rows": int(len(filtered)),
        "candidates": 0,
        "negatives_before_brand": 0,
        "negatives_after_brand": 0,
        "protected_brand_rows": 0,
        "saving_cost": 0.0,
        "saving_cost_annual": 0.0,
    }

    if filtered.empty:
        return pd.DataFrame(), stats

    _maybe_update_job(update_job, job_id, 25, "Generating embeddings…")

    # Prepare texts (sanitize happens inside get_embeddings anyway)
    search_texts = filtered["search_term"].tolist()
    kw_texts = kw_df["keyword"].tolist()

    # Embeddings (batched + sanitized)
    search_emb = get_embeddings(search_texts, model=embedding_model, batch_size=256)
    kw_emb = get_embeddings(kw_texts, model=embedding_model, batch_size=256)

    _maybe_update_job(update_job, job_id, 40, "Computing similarities…")

    # Similarity
    sim_matrix = cosine_similarity(search_emb, kw_emb)
    best_kw_idx = np.argmax(sim_matrix, axis=1)
    best_sim = sim_matrix[np.arange(len(filtered)), best_kw_idx]

    filtered["best_keyword"] = [str(kw_texts[i]) for i in best_kw_idx]
    filtered["similarity"] = best_sim

    # Candidate selection
    candidates = filtered[filtered["similarity"] < similarity_threshold].copy()
    stats["candidates"] = int(len(candidates))
    if candidates.empty:
        return pd.DataFrame(), stats

    _maybe_update_job(update_job, job_id, 40, "Running LLM decisions…")

    # LLM decision
    total_candidates = int(len(candidates))

    def progress_cb(
        batch_index: int,
        total_batches: int,
        done: int,
        total: int,
        is_start: bool,
    ) -> None:
        if total_batches <= 0:
            return
        start = 40
        end = 80
        step = batch_index - (1 if is_start else 0)
        pct = start + int((step / total_batches) * (end - start))
        _maybe_update_job(
            update_job,
            job_id,
            pct,
            f"LLM batch {batch_index}/{total_batches}",
        )

    decided = decide_with_llm(
        candidates,
        model=chat_model,
        use_llm=use_llm,
        batch_size=llm_batch_size,
        progress_cb=progress_cb,
    )
    negatives_all = decided[decided["exclude"] == True].copy()
    stats["negatives_before_brand"] = int(len(negatives_all))
    if negatives_all.empty:
        return pd.DataFrame(), stats

    _maybe_update_job(update_job, job_id, 85, "Applying brand protection…")

    # Brand protection
    if brand_terms:
        lower_search = negatives_all["search_term"].astype(str).str.lower()
        lower_best = negatives_all["best_keyword"].astype(str).str.lower()
        protected_mask = pd.Series(False, index=negatives_all.index)
        for term in brand_terms:
            protected_mask |= lower_search.str.contains(term) | lower_best.str.contains(term)

        stats["protected_brand_rows"] = int(protected_mask.sum())
        final_negatives = negatives_all[~protected_mask].copy()
    else:
        final_negatives = negatives_all.copy()

    stats["negatives_after_brand"] = int(len(final_negatives))
    if final_negatives.empty:
        return pd.DataFrame(), stats

    _maybe_update_job(update_job, job_id, 90, "Finalizing results…")

    # Exact match only: full search term in []
    final_negatives["match_type"] = "Exact"
    final_negatives["suggested_negative"] = "[" + final_negatives["search_term"].astype(str) + "]"

    # Risk score
    def risk_score_row(row):
        conv = float(row.get("conversions", 0) or 0)
        if conv > 0:
            return "Low"
        cost = float(row.get("cost", 0) or 0)
        clicks = float(row.get("clicks", 0) or 0)
        if cost >= 50 or clicks >= 20:
            return "High"
        return "Medium"

    final_negatives["risk_score"] = final_negatives.apply(risk_score_row, axis=1)

    saving_cost = float(final_negatives["cost"].sum())
    stats["saving_cost"] = saving_cost
    stats["saving_cost_annual"] = saving_cost * 12.0

    export_cols = [
        "campaign",
        "ad_group",
        "search_term",
        "suggested_negative",
        "match_type",
        "risk_score",
        "best_keyword",
        "similarity",
        "clicks",
        "cost",
        "conversions",
        "reason",
    ]
    export_cols = [c for c in export_cols if c in final_negatives.columns]
    final_df = final_negatives[export_cols].copy()

    return final_df, stats
