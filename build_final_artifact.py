"""Train the deployment artifact on ALL benchmark data.

Components (best OOF stack): catboost + logreg + xgb on 242 chunk features,
plus the sequence transformer (3 seeds, averaged). Saved as one joblib dict;
inference combines components by within-request rank-mean.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from catboost import CatBoostClassifier

from train_seq import ChunkModel, subsample_hands

DATA_DIR = Path("/root/poker44/data")
OUT = Path("/root/poker44/models/poker44_stack_v1.joblib")
SEED = 44
SEQ_SEEDS = [44, 137, 2026]
SEQ_EPOCHS = 60
D_MODEL = 96


def train_seq_full(tokens, hand_scalars, hand_mask, action_mask, y, seed: int) -> dict:
    torch.manual_seed(seed)
    dev = "cuda"
    model = ChunkModel(d=D_MODEL).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SEQ_EPOCHS)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    n = len(y)
    idx = torch.arange(n)
    for epoch in range(SEQ_EPOCHS):
        model.train()
        perm = idx[torch.randperm(n)]
        for i in range(0, n, 32):
            b = perm[i:i + 32]
            hm = subsample_hands(hand_mask[b]).to(dev)
            logits = model(tokens[b].to(dev), hand_scalars[b].to(dev), hm,
                           action_mask[b].to(dev))
            loss = loss_fn(logits, y[b].to(dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    return {k: v.cpu() for k, v in model.state_dict().items()}


def main():
    run = wandb.init(project="poker44", name="deploy-artifact-v1",
                     config={"components": ["catboost", "logreg", "xgb", f"seq x{len(SEQ_SEEDS)}"],
                             "seq_epochs": SEQ_EPOCHS, "d_model": D_MODEL, "seed": SEED})
    d = np.load(DATA_DIR / "dataset.npz", allow_pickle=False)
    X, y = d["X"], d["y"]
    feature_names = [str(n) for n in d["feature_names"]]
    print(f"features X={X.shape}", flush=True)

    t0 = time.time()
    models = {}
    models["logreg"] = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000, C=0.1, random_state=SEED)
    ).fit(X, y)
    print(f"logreg done {time.time()-t0:.0f}s", flush=True)
    models["xgb"] = xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8,
        colsample_bytree=0.7, reg_lambda=1.0, random_state=SEED,
        eval_metric="logloss", verbosity=0, device="cuda", tree_method="hist",
        n_jobs=2).fit(X, y)
    print(f"xgb done {time.time()-t0:.0f}s", flush=True)
    models["catboost"] = CatBoostClassifier(
        iterations=600, learning_rate=0.03, depth=5, random_seed=SEED,
        verbose=0, task_type="GPU", devices="0", thread_count=2).fit(X, y)
    print(f"catboost done {time.time()-t0:.0f}s", flush=True)

    s = np.load(DATA_DIR / "seq_dataset.npz", allow_pickle=False)
    tokens = torch.from_numpy(s["tokens"])
    hand_scalars = torch.from_numpy(s["hand_scalars"])
    hand_mask = torch.from_numpy(s["hand_mask"])
    action_mask = torch.from_numpy(s["action_mask"])
    ys = torch.from_numpy(s["y"]).float()
    assert np.array_equal(s["y"], y), "dataset row mismatch"

    seq_states = []
    for seed in SEQ_SEEDS:
        seq_states.append(train_seq_full(tokens, hand_scalars, hand_mask,
                                         action_mask, ys, seed))
        print(f"seq seed {seed} done {time.time()-t0:.0f}s", flush=True)

    # serialize torch states as bytes so joblib stays torch-version tolerant
    seq_blobs = []
    for st in seq_states:
        buf = io.BytesIO()
        torch.save(st, buf)
        seq_blobs.append(buf.getvalue())

    artifact = {
        "schema": "poker44-stack-v1",
        "feature_names": feature_names,
        "models": models,
        "seq_states": seq_blobs,
        "seq_arch": {"d": D_MODEL},
        "combine": "rank_mean(catboost,logreg,xgb,seq_mean)",
        "metadata": {
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "n_groups": int(len(y)),
            "training_dates": sorted(set(d["source_date"].tolist())),
            "oof_reference": "stack-v1 reward=0.8723 (catboost+logreg+seq+xgb)",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, OUT, compress=3)
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"saved {OUT} sha256={sha}", flush=True)
    wandb.summary.update({"artifact_sha256": sha, "artifact_path": str(OUT)})
    run.finish()
    print("done", flush=True)


if __name__ == "__main__":
    main()
