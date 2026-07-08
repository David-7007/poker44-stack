"""Validate the crossing-confidence cutoff under the 2026-07-08 reward formula.

Simulates live-like scoring windows (100 chunks) from new-regime benchmark
groups and compares the legacy uniform mapping against the confidence-gated
mapping at several CROSS_CONFIDENCE values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, "/root/poker44/src")
from poker44_infer import StackPredictor  # noqa: E402

DATA_RAW = Path("/root/poker44/data/raw")
NEW_REGIME = ("2026-07-06", "2026-07-07", "2026-07-08")
WINDOW = 100


def recall_at_fpr(y, s, max_fpr=0.05):
    y, s = np.asarray(y, int), np.asarray(s, float)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0:
        return 0.0
    o = np.argsort(-s, kind="mergesort")
    rec = np.cumsum(y[o] == 1) / pos
    fpr = np.cumsum(y[o] == 0) / neg
    ok = fpr <= max_fpr
    return float(rec[ok].max()) if ok.any() else 0.0


def new_reward(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float)
    ap = average_precision_score(y, s) if y.sum() else 0.0
    rec = recall_at_fpr(y, s)
    hard = s >= 0.5
    tp = int((hard & (y == 1)).sum())
    fp = int((hard & (y == 0)).sum())
    neg = max(int((y == 0).sum()), 1)
    hfpr = fp / neg
    if tp <= 0:
        sanity = 0.0
    elif hfpr <= 0.10:
        sanity = 1.0
    else:
        sanity = max(0.0, 1.0 - (hfpr - 0.10) / 0.90)
    if sanity <= 0:
        return 0.0, ap, rec, hfpr, sanity
    rew = 0.35 * ap + 0.30 * rec + 0.20 * sanity + 0.10 * sanity + 0.05 * 1.0
    return float(np.clip(rew, 0, 1)), ap, rec, hfpr, sanity


def main():
    groups, labels = [], []
    for date in NEW_REGIME:
        for path in sorted((DATA_RAW / date).glob("*.json")):
            rec = json.loads(path.read_text())
            gs, ls = rec.get("chunks") or [], rec.get("groundTruth") or []
            if len(gs) == len(ls):
                groups.extend(gs)
                labels.extend(int(x) for x in ls)
    y = np.asarray(labels)
    print(f"new-regime groups: {len(groups)} (bots {y.sum()})", flush=True)

    model = StackPredictor("/root/poker44/models/poker44_stack_v2.joblib")

    rng = np.random.default_rng(44)
    idx = rng.permutation(len(groups))
    windows = [idx[i:i + WINDOW] for i in range(0, len(idx) - WINDOW + 1, WINDOW)]

    for conf in (None, 0.70, 0.75, 0.80, 0.85, 0.90):
        rews, hfprs, sanities = [], [], []
        for w in windows:
            chunk_set = [groups[i] for i in w]
            if conf is None:
                # legacy mapping: bypass gate by scoring with rank-only path
                model.CROSS_CONFIDENCE = 10.0  # nothing crosses on confidence
                s = np.asarray(model.predict_chunk_scores(chunk_set))
                # rebuild legacy uniform mapping from the ordering
                order = np.argsort(np.argsort(-s))
                s = 0.98 - 0.96 * order / max(len(s) - 1, 1)
            else:
                model.CROSS_CONFIDENCE = conf
                s = np.asarray(model.predict_chunk_scores(chunk_set))
            r, ap, rec, hfpr, sanity = new_reward(y[w], s)
            rews.append(r)
            hfprs.append(hfpr)
            sanities.append(sanity)
        name = "legacy-uniform" if conf is None else f"gate@{conf}"
        print(f"{name:15s} reward={np.mean(rews):.4f} "
              f"hardFPR={np.mean(hfprs):.3f} sanity={np.mean(sanities):.3f}",
              flush=True)


if __name__ == "__main__":
    main()
