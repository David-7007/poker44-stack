"""Chunk-group feature extraction for Poker44 bot detection.

A "group" is one scoring unit: a list of ~30-40 sanitized hands from one player
context. Features are built in three layers:

1. per-hand scalar features -> aggregated across the group (mean/std/quantiles)
2. group-level action-type / street distributions and entropies
3. action-sequence n-gram distribution features (hashed)

Only miner-visible fields are used (matches live payload sanitization):
action_type, street, actor_seat, normalized_amount_bb, pot_before, pot_after,
players' starting_stack, streets list, metadata.hero_seat / max_seats.
"""

from __future__ import annotations

import math
import zlib
from collections import Counter
from typing import Any

import numpy as np

ACTION_TYPES = ["check", "call", "bet", "raise", "fold"]
STREETS = ["preflop", "flop", "turn", "river"]
NGRAM_BUCKETS = 64


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0 or len(counter) <= 1:
        return 0.0
    ent = 0.0
    for c in counter.values():
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent / math.log(max(len(counter), 2))


def _stats(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {f"{prefix}_{s}": 0.0 for s in ("mean", "std", "min", "max", "q25", "q75")}
    arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_q25": float(np.quantile(arr, 0.25)),
        f"{prefix}_q75": float(np.quantile(arr, 0.75)),
    }


def hand_scalars(hand: dict) -> dict[str, float]:
    actions = [a for a in (hand.get("actions") or []) if isinstance(a, dict)]
    players = [p for p in (hand.get("players") or []) if isinstance(p, dict)]
    streets = [s for s in (hand.get("streets") or []) if isinstance(s, dict)]
    meta = hand.get("metadata") or {}

    types = [str(a.get("action_type") or "") for a in actions]
    tc = Counter(types)
    n = max(len(actions), 1)

    amounts = [_f(a.get("normalized_amount_bb")) for a in actions]
    nonzero_amounts = [x for x in amounts if x > 0]
    pots_after = [_f(a.get("pot_after")) for a in actions]
    pots_before = [_f(a.get("pot_before")) for a in actions]
    hero = int(_f(meta.get("hero_seat")))
    hero_actions = [a for a in actions if int(_f(a.get("actor_seat"))) == hero]
    hero_types = Counter(str(a.get("action_type") or "") for a in hero_actions)
    hn = max(len(hero_actions), 1)

    street_of_action = [str(a.get("street") or "") for a in actions]
    sc = Counter(street_of_action)

    stacks = [_f(p.get("starting_stack")) for p in players]

    out: dict[str, float] = {
        "n_actions": float(len(actions)),
        "n_players": float(len(players)),
        "n_streets": float(len(streets)),
        "reached_flop": 1.0 if len(streets) >= 2 else 0.0,
        "reached_turn": 1.0 if len(streets) >= 3 else 0.0,
        "reached_river": 1.0 if len(streets) >= 4 else 0.0,
        "action_type_entropy": _entropy(tc),
        "zero_amount_share": sum(1 for x in amounts if x <= 0) / n,
        "amount_mean": float(np.mean(nonzero_amounts)) if nonzero_amounts else 0.0,
        "amount_std": float(np.std(nonzero_amounts)) if len(nonzero_amounts) > 1 else 0.0,
        "amount_max": max(nonzero_amounts) if nonzero_amounts else 0.0,
        "pot_final": pots_after[-1] if pots_after else 0.0,
        "pot_start": pots_before[0] if pots_before else 0.0,
        "pot_growth": (pots_after[-1] - pots_before[0]) if pots_after else 0.0,
        "hero_action_share": len(hero_actions) / n,
        "hero_aggr": (hero_types.get("bet", 0) + hero_types.get("raise", 0)) / hn,
        "hero_fold": hero_types.get("fold", 0) / hn,
        "stack_mean": float(np.mean(stacks)) if stacks else 0.0,
        "stack_std": float(np.std(stacks)) if len(stacks) > 1 else 0.0,
    }
    for t in ACTION_TYPES:
        out[f"rate_{t}"] = tc.get(t, 0) / n
    for s in STREETS:
        out[f"street_share_{s}"] = sc.get(s, 0) / n
    aggr = tc.get("bet", 0) + tc.get("raise", 0)
    passive = tc.get("call", 0) + tc.get("check", 0)
    out["aggression_factor"] = aggr / max(passive, 1)
    return out


def _ngram_features(hands: list[dict]) -> dict[str, float]:
    """Hashed 2/3-gram distribution over per-hand action-type sequences."""
    counts = np.zeros(NGRAM_BUCKETS, dtype=float)
    total = 0
    for hand in hands:
        seq = [str(a.get("action_type") or "?") for a in (hand.get("actions") or [])
               if isinstance(a, dict)]
        seq = ["<s>"] + seq + ["</s>"]
        for k in (2, 3):
            for i in range(len(seq) - k + 1):
                gram = "|".join(seq[i:i + k])
                counts[zlib.crc32(gram.encode()) % NGRAM_BUCKETS] += 1.0
                total += 1
    if total > 0:
        counts /= total
    return {f"ngram_{i}": float(counts[i]) for i in range(NGRAM_BUCKETS)}


def group_features(hands: list[dict]) -> dict[str, float]:
    hands = [h for h in hands if isinstance(h, dict)]
    per_hand = [hand_scalars(h) for h in hands]
    out: dict[str, float] = {"group_n_hands": float(len(hands))}
    if per_hand:
        keys = per_hand[0].keys()
        for key in keys:
            vals = [ph[key] for ph in per_hand]
            out.update(_stats(vals, key))

    all_types = Counter()
    all_streets = Counter()
    seat_counter = Counter()
    for h in hands:
        for a in (h.get("actions") or []):
            if isinstance(a, dict):
                all_types[str(a.get("action_type") or "")] += 1
                all_streets[str(a.get("street") or "")] += 1
                seat_counter[int(_f(a.get("actor_seat")))] += 1
    out["grp_action_entropy"] = _entropy(all_types)
    out["grp_street_entropy"] = _entropy(all_streets)
    out["grp_seat_entropy"] = _entropy(seat_counter)
    out.update(_ngram_features(hands))
    return out


FEATURE_NAMES: list[str] | None = None


def to_vector(hands: list[dict]) -> tuple[np.ndarray, list[str]]:
    feats = group_features(hands)
    names = sorted(feats.keys())
    return np.asarray([feats[k] for k in names], dtype=np.float32), names
