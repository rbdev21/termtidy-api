import os
import base64
import concurrent.futures
from io import BytesIO
from typing import List, Dict, Any

import pandas as pd
import resend


def send_job_complete_email(
    *,
    to_email: str,
    job_id: str,
    results: List[Dict[str, Any]],
) -> None:
    try:
        enabled_raw = os.getenv("EMAIL_ENABLED", "")
        enabled = enabled_raw.strip().lower() in {"true", "1", "yes", "y"}
        if not enabled:
            print(f"[email] skipped (EMAIL_ENABLED={enabled_raw!r})")
            return

        api_key = os.getenv("RESEND_API_KEY")
        email_from = os.getenv("EMAIL_FROM")
        if not api_key:
            print("[email] missing RESEND_API_KEY")
            return
        if not email_from:
            print("[email] missing EMAIL_FROM")
            return
        if not to_email:
            print("[email] missing recipient email")
            return

        resend.api_key = api_key

        df = pd.DataFrame(results)
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue()
        encoded = base64.b64encode(csv_bytes).decode("ascii")

        payload = {
            "from": email_from,
            "to": [to_email],
            "subject": "Your TermTidy audit is ready",
            "text": (
                "Your TermTidy audit has completed successfully. "
                "Your results are attached as a CSV."
            ),
            "attachments": [
                {
                    "filename": f"termtidy-{job_id}.csv",
                    "content": encoded,
                }
            ],
        }

        print(f"[email] sending completion email job_id={job_id} to={to_email}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(resend.Emails.send, payload)
            resp = future.result(timeout=20)

        print(f"[email] sent completion email for job_id={job_id} resp={resp}")
    except Exception as e:
        print(f"[email] failed to send completion email: {e}")
