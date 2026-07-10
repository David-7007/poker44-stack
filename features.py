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
    stat_names = ("mean", "std", "min", "max", "q10", "q25", "q50", "q75", "q90")
    if not values:
        return {f"{prefix}_{s}": 0.0 for s in stat_names}
    arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_q10": float(np.quantile(arr, 0.10)),
        f"{prefix}_q25": float(np.quantile(arr, 0.25)),
        f"{prefix}_q50": float(np.quantile(arr, 0.50)),
        f"{prefix}_q75": float(np.quantile(arr, 0.75)),
        f"{prefix}_q90": float(np.quantile(arr, 0.90)),
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
    out.update(_signature_features(hands))
    return out


# ---- bot-regularity / cross-hand-signature families -----------------------
# Bots replay near-identical action/sizing sequences across hands; humans do
# not. These families (adapted from the leading open miners) capture that tell
# directly and tend to survive the synthetic->real-bot distribution gap.
_BB_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0,
             36.0, 56.0, 84.0, 126.0]


def _bucket_idx(bb: float) -> int:
    for i, e in enumerate(_BB_EDGES):
        if bb <= e + 1e-9:
            return i
    return len(_BB_EDGES)


def _max_run_share(seq: list) -> float:
    if not seq:
        return 0.0
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return best / len(seq)


def _switch_rate(seq: list) -> float:
    if len(seq) < 2:
        return 0.0
    return sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1]) / (len(seq) - 1)


def _signature_features(hands: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    n = len(hands)
    if n == 0:
        for name in ("action", "role", "street", "bucket"):
            out[f"sig_{name}_top_share"] = 0.0
            out[f"sig_{name}_unique_share"] = 0.0
        for k in ("run_share", "switch_rate", "actor_switch"):
            out[f"reg_{k}_mean"] = 0.0
            out[f"reg_{k}_std"] = 0.0
        out["high_aggr_hand_rate"] = 0.0
        out["low_entropy_hand_rate"] = 0.0
        out["zero_hero_action_rate"] = 0.0
        return out

    sigs = {"action": [], "role": [], "street": [], "bucket": []}
    run_shares, switch_rates, actor_switches = [], [], []
    n_high_aggr = n_low_ent = n_zero_hero = 0
    for h in hands:
        actions = [a for a in (h.get("actions") or []) if isinstance(a, dict)]
        meta = h.get("metadata") or {}
        hero = int(_f(meta.get("hero_seat")))
        a_types = [str(a.get("action_type") or "") for a in actions]
        roles = ["H" if int(_f(a.get("actor_seat"))) == hero and hero > 0 else "o"
                 for a in actions]
        streets = [str(a.get("street") or "") for a in actions]
        buckets = [_bucket_idx(_f(a.get("normalized_amount_bb"))) for a in actions]
        actors = [int(_f(a.get("actor_seat"))) for a in actions]
        sigs["action"].append(tuple(a_types))
        sigs["role"].append(tuple(roles))
        sigs["street"].append(tuple(streets))
        sigs["bucket"].append(tuple(buckets))
        run_shares.append(_max_run_share(a_types))
        switch_rates.append(_switch_rate(a_types))
        actor_switches.append(_switch_rate(actors))
        m = max(len(a_types), 1)
        aggr = sum(1 for t in a_types if t in ("bet", "raise")) / m
        ent = _entropy(Counter(a_types))
        if aggr > 0.5:
            n_high_aggr += 1
        if ent < 0.3:
            n_low_ent += 1
        if not any(r == "H" for r in roles):
            n_zero_hero += 1

    for name in ("action", "role", "street", "bucket"):
        c = Counter(sigs[name])
        out[f"sig_{name}_top_share"] = max(c.values()) / n
        out[f"sig_{name}_unique_share"] = len(c) / n
    for k, series in (("run_share", run_shares), ("switch_rate", switch_rates),
                      ("actor_switch", actor_switches)):
        arr = np.asarray(series, dtype=float)
        out[f"reg_{k}_mean"] = float(arr.mean())
        out[f"reg_{k}_std"] = float(arr.std())
    out["high_aggr_hand_rate"] = n_high_aggr / n
    out["low_entropy_hand_rate"] = n_low_ent / n
    out["zero_hero_action_rate"] = n_zero_hero / n
    return out


FEATURE_NAMES: list[str] | None = None


def to_vector(hands: list[dict]) -> tuple[np.ndarray, list[str]]:
    feats = group_features(hands)
    names = sorted(feats.keys())
    return np.asarray([feats[k] for k in names], dtype=np.float32), names
