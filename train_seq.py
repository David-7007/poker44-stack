"""Hierarchical sequence transformer for Poker44 chunk classification.

Architecture:
  action tokens -> summed embeddings -> hand transformer (2 layers) -> hand vec
  hand vec + hand scalars -> chunk transformer over the set of hands (2 layers)
  -> attention-pooled chunk vec -> MLP head -> bot logit

Training: 5-fold GroupKFold by sourceDate, BCE loss, hand-subsampling
augmentation, cosine LR. Logs per-epoch metrics to W&B; saves OOF predictions.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

DATA = Path(__file__).resolve().parent.parent / "data" / "seq_dataset.npz"
OOF_OUT = Path(__file__).resolve().parent.parent / "data" / "oof_seq.npz"
SEED = 44


def recall_at_fpr(y_true, y_score, max_fpr=0.05):
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    pos, neg = int((labels == 1).sum()), int((labels == 0).sum())
    if pos == 0 or neg == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sl = labels[order]
    recall = np.cumsum(sl == 1) / pos
    fpr = np.cumsum(sl == 0) / neg
    ok = fpr <= max_fpr
    return float(recall[ok].max()) if ok.any() else 0.0


def reward_metric(y_true, y_score):
    ap = float(average_precision_score(y_true, y_score)) if y_true.sum() else 0.0
    rec = recall_at_fpr(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score)) if 0 < y_true.sum() < len(y_true) else 0.0
    return {"ap": ap, "recall_at_fpr5": rec, "reward": 0.75 * ap + 0.25 * rec, "auc": auc}


class ChunkModel(nn.Module):
    def __init__(self, d=96, heads=4, drop=0.15):
        super().__init__()
        self.emb_type = nn.Embedding(8, d)
        self.emb_street = nn.Embedding(6, d)
        self.emb_amount = nn.Embedding(18, d)
        self.emb_pot = nn.Embedding(18, d)
        self.emb_hero = nn.Embedding(2, d)
        self.emb_pos = nn.Embedding(12, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d * 4, dropout=drop,
            batch_first=True, norm_first=True)
        self.hand_enc = nn.TransformerEncoder(layer, num_layers=2)
        self.hand_proj = nn.Sequential(
            nn.Linear(d + 5, d), nn.GELU(), nn.Dropout(drop))
        chunk_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d * 4, dropout=drop,
            batch_first=True, norm_first=True)
        self.chunk_enc = nn.TransformerEncoder(chunk_layer, num_layers=2)
        self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d, 1))

    def forward(self, tokens, hand_scalars, hand_mask, action_mask):
        B, H, A, _ = tokens.shape
        t = tokens.reshape(B * H, A, 6).long()
        x = (self.emb_type(t[..., 0]) + self.emb_street(t[..., 1])
             + self.emb_amount(t[..., 2]) + self.emb_pot(t[..., 3])
             + self.emb_hero(t[..., 4]) + self.emb_pos(t[..., 5]))
        am = action_mask.reshape(B * H, A)
        # hands with zero actions: unmask first slot to avoid NaN attention
        empty = ~am.any(dim=1)
        am = am.clone()
        am[empty, 0] = True
        x = self.hand_enc(x, src_key_padding_mask=~am)
        x = (x * am.unsqueeze(-1)).sum(1) / am.sum(1, keepdim=True).clamp(min=1)
        x = x.reshape(B, H, -1)
        x = self.hand_proj(torch.cat([x, hand_scalars], dim=-1))
        hm = hand_mask.clone()
        empty_h = ~hm.any(dim=1)
        hm[empty_h, 0] = True
        x = self.chunk_enc(x, src_key_padding_mask=~hm)
        q = self.pool_q.expand(B, 1, -1)
        pooled, _ = self.pool_attn(q, x, x, key_padding_mask=~hm)
        return self.head(pooled.squeeze(1)).squeeze(-1)


def subsample_hands(hm: torch.Tensor, min_keep=18) -> torch.Tensor:
    """Randomly drop hands as augmentation; returns modified mask."""
    out = hm.clone()
    for b in range(hm.shape[0]):
        idx = torch.nonzero(hm[b]).squeeze(-1)
        n = len(idx)
        if n > min_keep:
            keep = torch.randperm(n)[:torch.randint(min_keep, n + 1, (1,)).item()]
            mask = torch.zeros_like(hm[b])
            mask[idx[keep]] = True
            out[b] = mask
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="seq-transformer-v1")
    parser.add_argument("--project", default="poker44")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d", type=int, default=96)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(DATA, allow_pickle=False)
    tokens = torch.from_numpy(d["tokens"])
    hand_scalars = torch.from_numpy(d["hand_scalars"])
    hand_mask = torch.from_numpy(d["hand_mask"])
    action_mask = torch.from_numpy(d["action_mask"])
    y = torch.from_numpy(d["y"]).float()
    dates = d["source_date"]
    api_split = d["api_split"]
    n = len(y)

    run = wandb.init(
        project=args.project, name=args.run_name,
        config={"n_groups": n, "epochs": args.epochs, "batch": args.batch,
                "lr": args.lr, "d_model": args.d, "arch": "hand-tf(2)+chunk-tf(2)+attnpool",
                "augment": "hand subsampling >=18", "cv": "GroupKFold(5) by sourceDate",
                "device": dev, "seed": SEED},
    )
    wandb.define_metric("epoch/*", step_metric="global_epoch")
    print(f"wandb run: {run.url}", flush=True)

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(n)
    global_epoch = 0
    for fold, (tr, te) in enumerate(gkf.split(np.zeros(n), y.numpy(), groups=dates)):
        model = ChunkModel(d=args.d).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        loss_fn = nn.BCEWithLogitsLoss()
        tr_idx = torch.from_numpy(tr)
        te_idx = torch.from_numpy(te)
        best_reward, best_scores = -1.0, None

        for epoch in range(args.epochs):
            model.train()
            perm = tr_idx[torch.randperm(len(tr_idx))]
            total_loss = 0.0
            for i in range(0, len(perm), args.batch):
                b = perm[i:i + args.batch]
                hm = subsample_hands(hand_mask[b]).to(dev)
                logits = model(tokens[b].to(dev), hand_scalars[b].to(dev),
                               hm, action_mask[b].to(dev))
                loss = loss_fn(logits, y[b].to(dev))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += float(loss) * len(b)
            sched.step()

            model.eval()
            with torch.no_grad():
                scores = []
                for i in range(0, len(te_idx), 64):
                    b = te_idx[i:i + 64]
                    logits = model(tokens[b].to(dev), hand_scalars[b].to(dev),
                                   hand_mask[b].to(dev), action_mask[b].to(dev))
                    scores.append(torch.sigmoid(logits).cpu())
                scores = torch.cat(scores).numpy()
            m = reward_metric(y[te_idx].numpy(), scores)
            if m["reward"] > best_reward:
                best_reward, best_scores = m["reward"], scores
            wandb.log({"global_epoch": global_epoch, "epoch/fold": fold,
                       "epoch/train_loss": total_loss / len(perm),
                       **{f"epoch/val_{k}": v for k, v in m.items()}})
            global_epoch += 1
            if epoch % 10 == 0 or epoch == args.epochs - 1:
                print(f"fold {fold} epoch {epoch}: loss={total_loss/len(perm):.4f} "
                      f"val_reward={m['reward']:.4f} (best {best_reward:.4f})", flush=True)

        # OOF uses final-epoch scores (no test-fold model selection);
        # best-epoch reward is logged separately as an optimistic upper bound.
        oof[te] = scores
        m = reward_metric(y[te_idx].numpy(), scores)
        wandb.summary.update({f"fold{fold}/{k}": v for k, v in m.items()})
        wandb.summary.update({f"fold{fold}/best_epoch_reward": best_reward})
        print(f"fold {fold} final: {m} | best-epoch upper bound {best_reward:.4f}", flush=True)

    m = reward_metric(y.numpy(), oof)
    wandb.summary.update({f"cv/seq/{k}": v for k, v in m.items()})
    print(f"CV seq-transformer OOF: {m}", flush=True)

    np.savez_compressed(OOF_OUT, y=y.numpy(), dates=dates, api_split=api_split, oof_seq=oof)
    print(f"OOF saved to {OOF_OUT}", flush=True)
    run.finish()
    print(f"done: {run.url}", flush=True)


if __name__ == "__main__":
    main()
