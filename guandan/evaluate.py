"""Duplicate-deal, team-swapped evaluation with pair-cluster uncertainty."""
from __future__ import annotations

import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fabledan.engine import GuandanRound, random_tribute_mode

from .agents import make_agent
from .runtime import RULESET, atomic_json, file_sha


def cluster_interval(pair_means, seed=0):
    """Bootstrap deals, not correlated individual seat-swap games."""
    x = np.asarray(pair_means, dtype=np.float64)
    if len(x) < 2:
        return [0.0, 1.0]
    # Report a conservative interval in degenerate samples (bootstrap alone
    # would misleadingly return [1, 1] after all wins).
    p, n, z = float(x.mean()), len(x), 1.95996398454
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    rng = np.random.default_rng(seed)
    boot = np.mean(x[rng.integers(n, size=(3000, n))], axis=1)
    lo, hi = np.quantile(boot, [.025, .975])
    return [max(0.0, min(float(lo), center-radius)), min(1.0, max(float(hi), center+radius))]


def run_evaluation(agent_a, agent_b, pairs=100, seed=1_000_003, progress=None):
    if pairs < 1:
        raise ValueError("pairs must be >= 1")
    rng = random.Random(seed)
    records = []
    start = time.monotonic()
    for pair in range(pairs):
        deal_seed = rng.getrandbits(48)
        setup = random.Random(deal_seed)
        level = setup.randrange(13)
        tribute = random_tribute_mode(setup)
        deck = list(range(108))
        setup.shuffle(deck)
        deal = [deck[i*27:(i+1)*27] for i in range(4)]
        for swap in (0, 1):
            a_seats = (swap, swap + 2)
            agents = [agent_a if p in a_seats else agent_b for p in range(4)]
            rnd = GuandanRound(level, random.Random(deal_seed), tribute, deal)
            rewards, ranking = rnd.play(agents)
            records.append(dict(pair=pair, swap=swap, deal_seed=deal_seed, level=level,
                                tribute=tribute, reward=rewards[a_seats[0]], ranking=ranking,
                                win=int(rewards[a_seats[0]] > 0)))
        if progress and (pair + 1) % max(1, pairs // 10) == 0:
            progress(pair + 1, pairs, sum(r["win"] for r in records)/len(records))
    means = [(records[2*i]["win"]+records[2*i+1]["win"])/2 for i in range(pairs)]
    return dict(ruleset=RULESET, seed=seed, pairs=pairs, games=2*pairs,
                win_rate=float(np.mean(means)), ci95=cluster_interval(means, seed),
                ci_method="paired-deal cluster bootstrap with conservative Wilson envelope",
                average_reward=float(np.mean([r["reward"] for r in records])),
                elapsed_seconds=round(time.monotonic()-start, 2), records=records,
                human_strength="unverified", created_at=datetime.now(timezone.utc).isoformat())


def evaluate_specs(a, b, pairs, seed, out, device="cpu"):
    aa, bb = make_agent(a, seed+1, device), make_agent(b, seed+2, device)
    result = run_evaluation(aa, bb, pairs, seed,
                            lambda n, total, wr: print(f"评测 {n}/{total} 组 · 胜率 {wr:.1%}", flush=True))
    result.update(agent_a=a, agent_b=b)
    for label, spec in [("a", a), ("b", b)]:
        if Path(spec).is_file():
            result[f"sha256_{label}"] = file_sha(spec)
        elif spec.startswith("program:"):
            from .programs import source_fingerprint
            result[f"source_sha256_{label}"] = source_fingerprint(spec.split(":", 1)[1])
    atomic_json(out, result)
    print(f"胜率 {result['win_rate']:.1%} · 95% 区间 {result['ci95'][0]:.1%}–{result['ci95'][1]:.1%}")
    return result
