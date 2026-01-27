import os
import base64
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
        if os.getenv("EMAIL_ENABLED", "").lower() != "true":
            return

        api_key = os.getenv("RESEND_API_KEY")
        email_from = os.getenv("EMAIL_FROM")
        if not api_key or not email_from or not to_email:
            return

        resend.api_key = api_key

        df = pd.DataFrame(results)
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue()
        encoded = base64.b64encode(csv_bytes).decode("ascii")

        resend.Emails.send(
            {
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
        )
    except Exception as e:
        print(f"[email] failed to send completion email: {e}")
