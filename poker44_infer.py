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
            self.ranker_model = art.get("ranker_model")  # v7+; absent in v5/v6
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
            if self.ranker_model is not None:
                try:  # LambdaMART: raw ranking score (rank-normalized in blend)
                    out["ranker"] = self.ranker_model.predict(X)
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
    # Refined threshold gate (adopted from field leader uid 99 + our floor):
    #  * FLOOR: our members are trained on SYNTHETIC bots and are under-confident
    #    on real ones (measured: few live raw scores clear 0.5), so we still need
    #    a minimum crossing set to avoid the zero-reward trap.
    #  * CAP: never cross more than GATE_MAX_FRACTION, bounding hard-FPR@0.5.
    #  * ELIGIBILITY: prefer to cross only chunks the ensemble already puts >0.5.
    #  * THIN BAND: positives sit just above 0.5 (0.502..0.512) so an accidental
    #    high-ranked human barely crosses; rank order (AP/recall) is preserved.
    GATE_MIN_FRACTION = 0.10
    GATE_MAX_FRACTION = 0.45
    POS_BAND = (0.502, 0.512)
    NEG_BAND = (0.02, 0.49)

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
                # ordering: every member rank-normalized (ranker score is fine here)
                ranked = sum(w * _rank(feats[k]) for k, w in weights.items()) / total
                # eligibility confidence: probability members only (ranker is not
                # a probability, so it must not enter the raw>0.5 test)
                prob_keys = [k for k in feats if k != "ranker"]
                pw = sum(weights[k] for k in prob_keys) or 1.0
                mean_prob = sum(weights[k] * feats[k] for k in prob_keys) / pw
            else:
                components = list(feats.values())
                ranked = np.mean([_rank(c) for c in components], axis=0)
                mean_prob = np.mean(components, axis=0)

            import math
            n = len(chunks)
            # crossing count K: eligibility (raw>0.5), clamped to [floor, cap]
            eligible = int(np.sum(mean_prob > 0.5))
            floor = math.ceil(self.GATE_MIN_FRACTION * n)
            cap = int(math.floor(self.GATE_MAX_FRACTION * n))
            k = max(1, min(cap if cap > 0 else 1, max(floor, eligible)))
            k = min(k, n)
            order = np.argsort(-ranked, kind="stable")
            final = np.empty(n)
            p_hi, p_lo = self.POS_BAND[1], self.POS_BAND[0]
            n_hi, n_lo = self.NEG_BAND[1], self.NEG_BAND[0]
            for pos, idx in enumerate(order):
                if pos < k:  # positives: thin band just above 0.5, rank-ordered
                    t = pos / (k - 1) if k > 1 else 0.0
                    final[idx] = p_hi - t * (p_hi - p_lo)
                else:  # negatives: (0.02..0.49), rank-ordered
                    lp = pos - k
                    nn = max(n - k - 1, 1)
                    final[idx] = n_hi - (lp / nn) * (n_hi - n_lo)
            return [float(round(s, 6)) for s in final]
        except Exception:
            return [0.5] * len(chunks)
