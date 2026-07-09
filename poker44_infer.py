"""Inference wrapper: raw DetectionSynapse chunks -> one risk score per chunk.

Loads the poker44_stack_v1 artifact. Per request: extract features + tokens,
run all components, rank-normalize each within the request, average.
Fails safe: any per-component error drops that component; total failure
returns 0.5 for every chunk.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from features import to_vector
from build_seq_dataset import encode_hand, MAX_HANDS, MAX_ACTIONS
from train_seq import ChunkModel


def _rank(x: np.ndarray) -> np.ndarray:
    if len(x) <= 1:
        return np.full(len(x), 0.5)
    return np.argsort(np.argsort(x)) / (len(x) - 1)


class StackPredictor:
    def __init__(self, artifact_path: str | Path, device: str | None = None):
        art = joblib.load(artifact_path)
        self.schema = art.get("schema")
        assert self.schema in ("poker44-stack-v1", "poker44-blend-v5"), "unknown artifact schema"
        self.feature_names = art["feature_names"]
        self.metadata = art.get("metadata", {})
        if self.schema == "poker44-blend-v5":
            self.stack_models = art["stack_models"]
            self.mono_model = art["mono_model"]
            self.mono_cols = np.asarray(art["mono_cols"], dtype=int)
            self.mlp_model = art["mlp_model"]
            self.blend_weights = art["blend_weights"]
            self.models = {}
        else:
            self.models = art["models"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_models = []
        d = art.get("seq_arch", {}).get("d", 96)
        for blob in art.get("seq_states", []):
            m = ChunkModel(d=d)
            m.load_state_dict(torch.load(io.BytesIO(blob), map_location="cpu"))
            m.eval()
            self.seq_models.append(m.to(self.device))

    def _feature_scores(self, chunks: list[list[dict]]) -> dict[str, np.ndarray]:
        rows = []
        for chunk in chunks:
            vec, names = to_vector(chunk)
            if names != self.feature_names:
                # align by name; missing -> 0
                lookup = dict(zip(names, vec))
                vec = np.asarray([lookup.get(n, 0.0) for n in self.feature_names],
                                 dtype=np.float32)
            rows.append(vec)
        X = np.vstack(rows)
        if self.schema == "poker44-blend-v5":
            out = {}
            try:
                out["stack"] = np.mean(
                    [m.predict_proba(X)[:, 1] for m in self.stack_models], axis=0)
            except Exception:
                pass
            try:
                out["mono"] = self.mono_model.predict_proba(X[:, self.mono_cols])[:, 1]
            except Exception:
                pass
            try:
                out["mlp"] = self.mlp_model.predict_proba(X)[:, 1]
            except Exception:
                pass
            return out
        out = {}
        for name, model in self.models.items():
            try:
                out[name] = model.predict_proba(X)[:, 1]
            except Exception:
                pass
        return out

    def _seq_scores(self, chunks: list[list[dict]]) -> np.ndarray | None:
        if not self.seq_models:
            return None
        try:
            B = len(chunks)
            tokens = np.zeros((B, MAX_HANDS, MAX_ACTIONS, 6), dtype=np.int16)
            scal = np.zeros((B, MAX_HANDS, 5), dtype=np.float32)
            amask = np.zeros((B, MAX_HANDS, MAX_ACTIONS), dtype=bool)
            hmask = np.zeros((B, MAX_HANDS), dtype=bool)
            for i, chunk in enumerate(chunks):
                hands = [h for h in chunk if isinstance(h, dict)]
                if len(hands) > MAX_HANDS:
                    # live chunks carry ~80 hands vs 30-40 in training: take an
                    # even spread so the set size matches the trained regime.
                    step = len(hands) / MAX_HANDS
                    hands = [hands[int(j * step)] for j in range(MAX_HANDS)]
                for j, hand in enumerate(hands):
                    t, am, sc = encode_hand(hand)
                    tokens[i, j], amask[i, j], scal[i, j] = t, am, sc
                    hmask[i, j] = True
            tt = torch.from_numpy(tokens)
            ts = torch.from_numpy(scal)
            ta = torch.from_numpy(amask)
            th = torch.from_numpy(hmask)
            scores = np.zeros(B)
            with torch.no_grad():
                for m in self.seq_models:
                    logits = []
                    for i in range(0, B, 64):
                        sl = slice(i, i + 64)
                        lg = m(tt[sl].to(self.device), ts[sl].to(self.device),
                               th[sl].to(self.device), ta[sl].to(self.device))
                        logits.append(torch.sigmoid(lg).cpu().numpy())
                    scores += np.concatenate(logits)
            return scores / len(self.seq_models)
        except Exception:
            return None

    # Absolute-probability confidence needed for a chunk to cross the 0.5
    # operational threshold. The 2026-07-08 validator formula zeroes the
    # reward if no true bot scores >= 0.5 and decays it when hard FPR@0.5
    # exceeds 10%, so only high-confidence chunks may cross while the
    # ranking (AP / recall@FPR) stays intact.
    CROSS_CONFIDENCE = 0.80
    # Floor on the crossing set: live-domain confidence collapses (measured
    # 2026-07-09: 1/100 live chunks >= 0.80), and a window with zero true-bot
    # hard positives scores 0 reward. The top 12% by rank are overwhelmingly
    # bots for any usable ranking, keeping hard-FPR well under the 10% cliff.
    GATE_MIN_FRACTION = 0.12

    def predict_chunk_scores(self, chunks: list[list[dict[str, Any]]]) -> list[float]:
        if not chunks:
            return []
        try:
            feats = self._feature_scores(chunks)
            seq = self._seq_scores(chunks)
            if seq is not None:
                feats["seq"] = seq
            if not feats:
                return [0.5] * len(chunks)
            if self.schema == "poker44-blend-v5":
                weights = {k: float(self.blend_weights.get(k, 0.0)) for k in feats}
                if sum(weights.values()) <= 0:
                    weights = {k: 1.0 for k in feats}
                total = sum(weights.values())
                ranked = sum(w * _rank(feats[k]) for k, w in weights.items()) / total
                mean_prob = sum(w * feats[k] for k, w in weights.items()) / total
            else:
                components = list(feats.values())
                ranked = np.mean([_rank(c) for c in components], axis=0)
                mean_prob = np.mean(components, axis=0)

            # K = chunks confident enough to cross 0.5; crossing set is the
            # top-K by rank so ordering is preserved exactly. Always cross at
            # least the strongest chunk: zero hard positives => zero reward.
            import math
            k = max(
                1,
                int(np.sum(mean_prob >= self.CROSS_CONFIDENCE)),
                math.ceil(self.GATE_MIN_FRACTION * len(chunks)),
            )
            k = min(k, len(chunks))
            order = np.argsort(-ranked, kind="stable")
            final = np.empty(len(chunks))
            n_low = max(len(chunks) - k, 1)
            for pos, idx in enumerate(order):
                if pos < k:  # confident bots: 0.98 down to 0.55, monotone
                    final[idx] = 0.98 - (0.43 * pos / max(k, 1))
                else:  # everything else: 0.45 down to 0.02, monotone
                    final[idx] = 0.45 - (0.43 * (pos - k) / n_low)
            return [float(round(s, 6)) for s in final]
        except Exception:
            return [0.5] * len(chunks)
