# poker44-stack

Open-source Poker44 (Bittensor netuid 126) bot-detection miner: a rank-mean
stacked ensemble over chunk-level behavioral features plus a hierarchical
sequence transformer.

Served by hotkey `5DqDQjvcPurkMVToocEmjAvBufqHZjtfef8WfepHQRwaaFNx` (uid 67).

## Architecture

One score per chunk (a group of ~30–40 sanitized hands from one player
context), combined by within-request rank-mean of four components:

| Component | Input | Details |
|---|---|---|
| logistic regression | 242 chunk-level features | standardized, C=0.1 |
| XGBoost | same features | 600 trees, depth 5, lr 0.03 |
| CatBoost | same features | 600 iterations, depth 5, lr 0.03 |
| sequence transformer ×3 seeds | tokenized action sequences | 2-layer hand encoder → 2-layer chunk (set) encoder → attention pooling, d=96 |

Feature layers (`features.py`): per-hand scalars aggregated across the chunk
(mean/std/min/max/quartiles), group-level action/street/seat distributions and
entropies, and hashed 2/3-gram distributions over per-hand action-type
sequences. Only miner-visible fields are used.

All validator reward metrics (average precision, recall@FPR≤5%) are
rank-based, so rank-mean combination inside a request is monotone-safe.

## Training data

Trained **exclusively** on the public Poker44 training benchmark
(`https://api.poker44.net/api/v1/benchmark`), all releases 2026-05-26 through
2026-07-06 (724 labeled chunk groups). No validator-only evaluation data, no
scraped or private data.

Validation: 5-fold GroupKFold by release date. Out-of-fold reward
0.75·AP + 0.25·recall@FPR≤5% = **0.872** (AP 0.932) for the served
combination.

## Files

- `our_miner.py` — the miner neuron served on-chain (forward/blacklist/priority)
- `poker44_infer.py` — `StackPredictor`: artifact loading + per-request scoring
- `features.py` — chunk-level feature extraction (242 features)
- `build_seq_dataset.py` — action-sequence tokenizer (shared by training and inference)
- `train_seq.py` — sequence-transformer architecture + CV training
- `build_final_artifact.py` — trains all components on the full benchmark and writes the served artifact
- `download_benchmark.py`, `build_dataset.py` — benchmark corpus download + feature dataset build
- `models/poker44_stack_v1.joblib` — the exact served artifact

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_benchmark.py     # full public benchmark corpus
python build_dataset.py          # 242-feature tabular dataset
python build_seq_dataset.py      # tokenized sequence dataset
python build_final_artifact.py   # trains + writes models/poker44_stack_v1.joblib
```

Run the miner:

```bash
python our_miner.py \
  --netuid 126 --subtensor.network finney \
  --wallet.name <cold> --wallet.hotkey <hot> \
  --axon.port <port> --blacklist.force_validator_permit
```

## License

MIT
