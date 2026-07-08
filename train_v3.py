"""Train poker44_stack_v3: recency-weighted recipe on the full corpus.

Same architecture as v2 (logreg + xgb + catboost + seq x3, rank-mean at
inference), new-regime rows (>= 2026-07-06) weighted x3, trained on all data
including the 2026-07-08 release. Writes models/poker44_stack_v3.joblib.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from catboost import CatBoostClassifier

sys.path.insert(0, "/root/poker44/src")
from train_seq import ChunkModel, subsample_hands  # noqa: E402

DATA_DIR = Path("/root/poker44/data")
OUT = Path("/root/poker44/models/poker44_stack_v3.joblib")
NEW_REGIME = "2026-07-06"
SEED = 44
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    d = np.load(DATA_DIR / "dataset.npz", allow_pickle=False)
    X, y, dates = d["X"], d["y"], d["source_date"]
    feature_names = [str(n) for n in d["feature_names"]]
    s = np.load(DATA_DIR / "seq_dataset.npz", allow_pickle=False)
    tokens = torch.from_numpy(s["tokens"])
    scalars = torch.from_numpy(s["hand_scalars"])
    hmask = torch.from_numpy(s["hand_mask"])
    amask = torch.from_numpy(s["action_mask"])
    ys = torch.from_numpy(s["y"]).float()
    assert np.array_equal(s["y"], y)
    n = len(y)
    w = np.where(dates >= NEW_REGIME, 3.0, 1.0)
    print(f"n={n} new_regime={int((dates >= NEW_REGIME).sum())} device={DEV}", flush=True)

    t0 = time.time()
    logreg = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=3000, C=0.1, random_state=SEED))
    logreg.fit(X, y, logisticregression__sample_weight=w)
    print(f"logreg ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8,
        colsample_bytree=0.7, reg_lambda=1.0, random_state=SEED,
        eval_metric="logloss", verbosity=0,
        device="cuda" if DEV == "cuda" else "cpu", tree_method="hist", n_jobs=8)
    xgb_clf.fit(X, y, sample_weight=w)
    print(f"xgb ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    cat_clf = CatBoostClassifier(
        iterations=600, learning_rate=0.03, depth=5, random_seed=SEED, verbose=0,
        task_type="GPU" if DEV == "cuda" else "CPU", devices="0", thread_count=8)
    cat_clf.fit(X, y, sample_weight=w)
    print(f"catboost ({time.time()-t0:.0f}s)", flush=True)

    reps = np.maximum(1, np.round(w).astype(int))
    pool = np.repeat(np.arange(n), reps)
    seq_blobs = []
    for seed in (44, 45, 46):
        torch.manual_seed(seed)
        model = ChunkModel(d=96).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        t0 = time.time()
        for _ in range(60):
            model.train()
            perm = pool[torch.randperm(len(pool)).numpy()]
            for i in range(0, len(perm), 32):
                b = torch.from_numpy(perm[i:i + 32].copy())
                hm = subsample_hands(hmask[b]).to(DEV)
                logits = model(tokens[b].to(DEV), scalars[b].to(DEV), hm, amask[b].to(DEV))
                loss = loss_fn(logits, ys[b].to(DEV))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
        buf = io.BytesIO()
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, buf)
        seq_blobs.append(buf.getvalue())
        print(f"seq seed {seed} ({time.time()-t0:.0f}s)", flush=True)

    artifact = {
        "schema": "poker44-stack-v1",
        "feature_names": feature_names,
        "models": {"logreg": logreg, "xgb": xgb_clf, "catboost": cat_clf},
        "seq_states": seq_blobs,
        "seq_arch": {"d": 96},
        "metadata": {
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "n_groups": int(n),
            "recipe": "recency_weighted_x3",
            "training_dates": sorted(set(str(x) for x in dates)),
        },
    }
    joblib.dump(artifact, OUT, compress=3)
    print(f"saved {OUT} ({OUT.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
