"""Tokenize raw benchmark chunks into tensors for the sequence model.

Each group -> (hands, actions) tensor block:
  action token fields: action_type, street, amount_bucket, pot_bucket, is_hero, position
  hand scalar fields:  n_players, n_streets, stack_mean, stack_std, hero_seat_known

Output: data/seq_dataset.npz
  tokens  int16 [n_groups, MAX_HANDS, MAX_ACTIONS, 6]
  hand_scalars float32 [n_groups, MAX_HANDS, 5]
  hand_mask    bool [n_groups, MAX_HANDS]
  action_mask  bool [n_groups, MAX_HANDS, MAX_ACTIONS]
  y, source_date, api_split
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "data" / "seq_dataset.npz"

MAX_HANDS = 40
MAX_ACTIONS = 12

ACTION_TYPES = {"": 0, "check": 1, "call": 2, "bet": 3, "raise": 4, "fold": 5}
STREETS = {"": 0, "preflop": 1, "flop": 2, "turn": 3, "river": 4}
# visible bb bucket edges from the subnet's payload sanitizer
BB_BUCKETS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0, 56.0, 84.0, 126.0]
VISIBLE_BB = 0.02


def bucketize(value_bb: float, edges=BB_BUCKETS) -> int:
    for i, edge in enumerate(edges):
        if value_bb <= edge + 1e-9:
            return i
    return len(edges)


def _f(v, default=0.0):
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def encode_hand(hand: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = [a for a in (hand.get("actions") or []) if isinstance(a, dict)][:MAX_ACTIONS]
    players = [p for p in (hand.get("players") or []) if isinstance(p, dict)]
    streets = hand.get("streets") or []
    meta = hand.get("metadata") or {}
    hero = int(_f(meta.get("hero_seat")))

    tok = np.zeros((MAX_ACTIONS, 6), dtype=np.int16)
    amask = np.zeros(MAX_ACTIONS, dtype=bool)
    for i, a in enumerate(actions):
        amount_bb = _f(a.get("normalized_amount_bb"))
        pot_bb = _f(a.get("pot_after")) / VISIBLE_BB
        tok[i, 0] = ACTION_TYPES.get(str(a.get("action_type") or ""), 0)
        tok[i, 1] = STREETS.get(str(a.get("street") or ""), 0)
        tok[i, 2] = bucketize(amount_bb)
        tok[i, 3] = min(bucketize(pot_bb), 16)
        tok[i, 4] = 1 if (hero > 0 and int(_f(a.get("actor_seat"))) == hero) else 0
        tok[i, 5] = min(i, MAX_ACTIONS - 1)
        amask[i] = True

    stacks = [_f(p.get("starting_stack")) / VISIBLE_BB for p in players]
    scal = np.asarray([
        len(players) / 6.0,
        len(streets) / 4.0,
        (float(np.mean(stacks)) if stacks else 0.0) / 200.0,
        (float(np.std(stacks)) if len(stacks) > 1 else 0.0) / 100.0,
        1.0 if hero > 0 else 0.0,
    ], dtype=np.float32)
    return tok, amask, scal


def main() -> None:
    tokens, hand_scalars, hand_mask, action_mask = [], [], [], []
    ys, dates, splits = [], [], []

    files = sorted(RAW_DIR.glob("*/*.json"))
    print(f"tokenizing {len(files)} chunk records...")
    for path in files:
        rec = json.loads(path.read_text())
        groups = rec.get("chunks") or []
        labels = rec.get("groundTruth") or []
        if len(groups) != len(labels):
            continue
        for hands, label in zip(groups, labels):
            hands = [h for h in hands if isinstance(h, dict)][:MAX_HANDS]
            g_tok = np.zeros((MAX_HANDS, MAX_ACTIONS, 6), dtype=np.int16)
            g_scal = np.zeros((MAX_HANDS, 5), dtype=np.float32)
            g_amask = np.zeros((MAX_HANDS, MAX_ACTIONS), dtype=bool)
            g_hmask = np.zeros(MAX_HANDS, dtype=bool)
            for j, hand in enumerate(hands):
                t, am, sc = encode_hand(hand)
                g_tok[j], g_amask[j], g_scal[j] = t, am, sc
                g_hmask[j] = True
            tokens.append(g_tok)
            hand_scalars.append(g_scal)
            action_mask.append(g_amask)
            hand_mask.append(g_hmask)
            ys.append(int(label))
            dates.append(str(rec.get("sourceDate") or path.parent.name))
            splits.append(str(rec.get("split") or ""))

    np.savez_compressed(
        OUT,
        tokens=np.stack(tokens),
        hand_scalars=np.stack(hand_scalars),
        hand_mask=np.stack(hand_mask),
        action_mask=np.stack(action_mask),
        y=np.asarray(ys, dtype=np.int64),
        source_date=np.asarray(dates),
        api_split=np.asarray(splits),
    )
    print(f"saved {OUT}: groups={len(ys)} bots={sum(ys)} "
          f"tokens shape={np.stack(tokens).shape}")


if __name__ == "__main__":
    main()
