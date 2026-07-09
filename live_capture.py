"""Capture validator-query chunks locally for domain-gap diagnosis.

Validator queries are the only view of the live evaluation distribution.
This module appends the UNLABELED chunks (plus our own scores) to a local
JSONL file so the benchmark->live gap can be measured offline.

Safety contract:
  * enabled only when POKER44_CAPTURE=1;
  * inputs only — live queries carry no labels, nothing here can act as a
    supervised training label;
  * deduplicated by chunk content hash (validators resend the same daily
    snapshot many times);
  * size-capped (POKER44_CAPTURE_MAX_BYTES, default 250MB) and fail-safe:
    any error is swallowed so serving is never affected;
  * capture files are local-only and gitignored.

ATTESTATION: using captures for diagnosis does not change the training-data
statement. If they ever feed training (even unlabeled), update the manifest
attestations truthfully first.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()
_DIR = Path(os.getenv("POKER44_CAPTURE_DIR", "/root/poker44/live_capture"))
_MAX_BYTES = int(os.getenv("POKER44_CAPTURE_MAX_BYTES", str(250 * 1024 * 1024)))
_ENABLED = os.getenv("POKER44_CAPTURE", "0").strip() == "1"
_state = {"seen": None, "full": False}


def _chunk_key(chunk) -> str:
    blob = json.dumps(chunk, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _load_seen(path: Path) -> set:
    seen = set()
    try:
        with open(path, errors="ignore") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["key"])
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return seen


def capture(chunks, scores, caller: str) -> None:
    """Append new unique chunks from this request; never raises."""
    if not _ENABLED or _state["full"]:
        return
    try:
        _DIR.mkdir(parents=True, exist_ok=True)
        path = _DIR / "live_chunks.jsonl"
        with _LOCK:
            if _state["seen"] is None:
                _state["seen"] = _load_seen(path)
            if path.exists() and path.stat().st_size >= _MAX_BYTES:
                _state["full"] = True
                return
            new = 0
            with open(path, "a") as f:
                for chunk, score in zip(chunks or [], scores or []):
                    key = _chunk_key(chunk)
                    if key in _state["seen"]:
                        continue
                    _state["seen"].add(key)
                    f.write(json.dumps({
                        "key": key,
                        "ts": int(time.time()),
                        "caller": str(caller or "")[:64],
                        "our_score": float(score),
                        "chunk": chunk,
                    }, default=str) + "\n")
                    new += 1
            if new:
                import bittensor as bt
                bt.logging.info(f"live_capture: {new} new unique chunks stored")
    except Exception:
        pass
