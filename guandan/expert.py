"""Optional, isolated adapter to the upstream Mac-only DanLM distribution."""
from __future__ import annotations

import importlib
import os
import platform
import sys
import time
from pathlib import Path

from .runtime import ROOT, atomic_json, file_sha

DANLM_ROOT = ROOT / "vendor/DanLM"
WEIGHTS = DANLM_ROOT / "ckpts/DanLM_v1/dansformer_v1_best_eval.pt"


def available():
    return (platform.system() == "Darwin" and platform.machine() == "arm64"
            and sys.version_info[:2] == (3, 12) and WEIGHTS.is_file())


def prepare():
    if not available():
        raise RuntimeError("DanLM 公开版本需要 Apple Silicon + Python 3.12；本地自训练模型支持 CPU/MPS/CUDA。")
    for path in (DANLM_ROOT, DANLM_ROOT / "ui"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ["GUANDAN_IDLE_TIMEOUT"] = "0"
    # The published binary still points at the author's old src/ layout.
    # Resolve only the baseline action.py files actually shipped in this pin.
    from danzero.eval import baseline_adapter
    for name, (_, method) in list(baseline_adapter.BASELINE_BOTS.items()):
        directory = DANLM_ROOT / "baselines" / name
        candidates = sorted(directory.rglob("action.py"), key=lambda p: len(p.parts))
        if candidates:
            baseline_adapter.BASELINE_BOTS[name] = (str(candidates[0].parent), method)


def web_app():
    prepare()
    return importlib.import_module("server").app


def evaluate_expert(games, seed, opponent, out):
    import torch
    prepare()
    from danzero.eval.agents import create_agent
    from danzero.eval.evaluator import evaluate
    if games < 1:
        raise ValueError("games must be positive")
    torch.set_num_threads(2)
    a = create_agent(str(WEIGHTS), "cpu")
    # Baseline adapters discover files relative to the upstream working dir.
    previous = Path.cwd()
    start = time.monotonic()
    try:
        os.chdir(DANLM_ROOT)
        b = create_agent(opponent, "cpu")
        result = evaluate(a, b, num_games=games, seed=seed, log_interval=max(1, games//10))
    finally:
        os.chdir(previous)
    result.update(agent_a="DanLM V1 public pretrained", agent_b=opponent, seed=seed,
                  elapsed_seconds=time.monotonic()-start, sha256=file_sha(WEIGHTS),
                  engine="upstream DanLM; separate from local training ruleset",
                  protocol="upstream single-round evaluator, not local duplicate-deal protocol",
                  source="https://github.com/dashidhy/DanLM", human_strength="unverified_locally")
    result["requested_games"] = games
    result["valid"] = result.get("num_games") == games and (games != 1 or abs(result.get("avg_reward_a", 0)) > 0)
    atomic_json(out, result)
    if not result["valid"]:
        raise RuntimeError("上游评测未完成请求的有效对局；记录已标记 invalid，不能用来判断棋力。")
    print(f"DanLM 对 {opponent}: {result['win_rate_a']:.1%} ({games} 局)")
    return result
