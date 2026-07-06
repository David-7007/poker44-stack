"""Download the full Poker44 public benchmark corpus with local caching.

Saves one JSON file per chunk record under data/raw/<sourceDate>/<chunkId>.json.
Re-runnable: skips files that already exist (chunk records are immutable by chunkHash).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

BASE = "https://api.poker44.net/api/v1/benchmark"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def get(url: str, params: dict | None = None, retries: int = 5) -> dict:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=90)
            r.raise_for_status()
            body = r.json()
            if not body.get("success"):
                raise RuntimeError(f"API error: {body}")
            return body["data"]
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed after {retries} retries: {last}")


def main() -> None:
    status = get(BASE)
    print(f"status: version={status['releaseVersion']} latest={status['latestSourceDate']} "
          f"totalChunks={status['totalChunks']} totalHands={status['totalHands']}")

    releases = []
    before = None
    while True:
        params = {"limit": 30}
        if before:
            params["before"] = before
        page = get(f"{BASE}/releases", params).get("releases", [])
        if not page:
            break
        releases.extend(page)
        before = page[-1]["sourceDate"]
        if len(page) < 30:
            break
    print(f"releases: {len(releases)}")

    total_records = 0
    total_new = 0
    for rel in releases:
        source_date = rel["sourceDate"]
        out_dir = RAW_DIR / source_date
        out_dir.mkdir(parents=True, exist_ok=True)
        expected = int(rel.get("chunkCount") or 0)
        existing = len(list(out_dir.glob("*.json")))
        if expected > 0 and existing >= expected:
            total_records += existing
            continue

        cursor = None
        fetched = 0
        while True:
            params = {"sourceDate": source_date, "limit": 24}
            if cursor:
                params["cursor"] = cursor
            data = get(f"{BASE}/chunks", params)
            for rec in data.get("chunks", []):
                fetched += 1
                path = out_dir / f"{rec['chunkId']}.json"
                if not path.exists():
                    path.write_text(json.dumps(rec))
                    total_new += 1
            cursor = data.get("nextCursor")
            if not cursor:
                break
        total_records += max(fetched, existing)
        print(f"  {source_date}: {fetched} records (expected {expected})")

    print(f"done: {total_records} chunk records on disk, {total_new} newly downloaded")


if __name__ == "__main__":
    sys.exit(main())
