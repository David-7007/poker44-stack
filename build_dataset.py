"""Flatten raw benchmark chunk records into a feature matrix.

Output: data/dataset.npz with X (float32), y (int), plus per-row metadata
(sourceDate, apiSplit, chunkId, groupIndex) and feature names.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from features import to_vector

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "data" / "dataset.npz"


def main() -> None:
    rows_X: list[np.ndarray] = []
    rows_y: list[int] = []
    meta_date: list[str] = []
    meta_split: list[str] = []
    meta_chunk: list[str] = []
    meta_idx: list[int] = []
    names: list[str] | None = None

    files = sorted(RAW_DIR.glob("*/*.json"))
    print(f"parsing {len(files)} chunk records...")
    for path in files:
        rec = json.loads(path.read_text())
        groups = rec.get("chunks") or []
        labels = rec.get("groundTruth") or []
        if len(groups) != len(labels):
            print(f"  WARN {path.name}: {len(groups)} groups vs {len(labels)} labels; skipping")
            continue
        for i, (hands, label) in enumerate(zip(groups, labels)):
            vec, vec_names = to_vector(hands)
            if names is None:
                names = vec_names
            rows_X.append(vec)
            rows_y.append(int(label))
            meta_date.append(str(rec.get("sourceDate") or path.parent.name))
            meta_split.append(str(rec.get("split") or ""))
            meta_chunk.append(str(rec.get("chunkId") or ""))
            meta_idx.append(i)

    X = np.vstack(rows_X)
    y = np.asarray(rows_y, dtype=np.int64)
    print(f"dataset: X={X.shape} bots={int(y.sum())} humans={int((y == 0).sum())}")
    per_date = {}
    for d in meta_date:
        per_date[d] = per_date.get(d, 0) + 1
    for d in sorted(per_date):
        print(f"  {d}: {per_date[d]} groups")

    np.savez_compressed(
        OUT,
        X=X,
        y=y,
        feature_names=np.asarray(names),
        source_date=np.asarray(meta_date),
        api_split=np.asarray(meta_split),
        chunk_id=np.asarray(meta_chunk),
        group_index=np.asarray(meta_idx),
    )
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
