"""Poker44 miner serving the poker44-stack model (catboost+logreg+xgb+seq).

NOTE: no `from __future__ import annotations` here — bt.Axon.attach inspects
the forward() signature at runtime and needs real class annotations.
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import bittensor as bt

SRC_DIR = os.getenv("POKER44_SRC_DIR", "/root/poker44/src")
sys.path.insert(0, SRC_DIR)

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

from poker44_infer import StackPredictor  # noqa: E402  (from SRC_DIR)

MODEL_PATH = os.getenv("POKER44_MODEL_PATH", "/root/poker44/models/poker44_stack_v1.joblib")
MODEL_NAME = os.getenv("POKER44_MODEL_NAME", "poker44-stack")
MODEL_VERSION = os.getenv("POKER44_MODEL_VERSION", "1.0.0")


class Miner(BaseMinerNeuron):
    """Serves rank-mean stacked scores; fails safe to 0.5 on any error."""

    def __init__(self, config=None):
        super().__init__(config=config)
        bt.logging.info(f"Loading stack artifact: {MODEL_PATH}")
        self.predictor = StackPredictor(MODEL_PATH)
        bt.logging.info(f"Artifact metadata: {self.predictor.metadata}")

        repo_root = Path(__file__).resolve().parents[1]
        impl_files = [
            Path(__file__).resolve(),
            Path(SRC_DIR) / "poker44_infer.py",
            Path(SRC_DIR) / "features.py",
            Path(SRC_DIR) / "build_seq_dataset.py",
            Path(SRC_DIR) / "train_seq.py",
        ]
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[p for p in impl_files if p.exists()],
            defaults={
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "framework": "sklearn+xgboost+catboost+torch (rank-mean stack)",
                "license": "MIT",
                "open_source": False,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained exclusively on the public Poker44 training benchmark "
                    "(api.poker44.net/api/v1/benchmark, releases 2026-05-26..2026-07-06)."
                ),
                "training_data_sources": ["poker44-public-benchmark"],
                "private_data_attestation": (
                    "This model does not train on validator-only evaluation data."
                ),
                "data_attestation": (
                    "All training data is the public benchmark; no scraped or private data."
                ),
                "notes": "Public repo publication pending; will be updated in a future manifest.",
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        bt.logging.info(
            f"Manifest status={self.manifest_compliance['status']} digest={self.manifest_digest}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        t0 = time.time()
        try:
            scores = await asyncio.to_thread(self.predictor.predict_chunk_scores, chunks)
        except Exception as exc:  # noqa: BLE001
            bt.logging.error(f"Inference failed, returning neutral scores: {exc}")
            scores = [0.5] * len(chunks)
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(
            f"Scored {len(chunks)} chunks in {time.time()-t0:.1f}s "
            f"(caller={getattr(synapse.dendrite, 'hotkey', '?')})"
        )
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("poker44-stack miner running...")
        while True:
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}"
            )
            time.sleep(5 * 60)
