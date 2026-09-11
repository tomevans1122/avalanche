"""Fetch a configured official regional bulletin without inventing a forecast."""

import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def pilot_region() -> str:
    with (PROJECT_ROOT / "config" / "pilot-study.json").open() as file:
        return json.load(file)["name"]


def fetch_official_report() -> dict[str, object]:
    """Return a JSON bulletin from the explicitly configured provider endpoint.

    The provider must return JSON. `OFFICIAL_REPORT_REGION` is shown beside the
    payload so an operator can verify it is the same geographic region as the
    model configuration; it is not inferred from a resort name.
    """
    url = os.environ.get("OFFICIAL_REPORT_URL", "").strip()
    configured_region = os.environ.get("OFFICIAL_REPORT_REGION", "").strip()
    if not url:
        return {
            "configured": False,
            "message": "No official report API is configured.",
            "model_region": pilot_region(),
        }
    if not url.startswith("https://"):
        return {"configured": True, "error": "Official report URL must use HTTPS."}
    headers = {"Accept": "application/json"}
    token = os.environ.get("OFFICIAL_REPORT_API_KEY", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urlopen(Request(url, headers=headers), timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        return {"configured": True, "error": f"Could not retrieve official report: {exc}"}
    return {
        "configured": True,
        "model_region": pilot_region(),
        "report_region": configured_region or "Not declared",
        "same_region": configured_region == pilot_region(),
        "payload": payload,
    }
