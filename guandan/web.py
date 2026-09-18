from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from fabledan.cards import RANK_NAMES, is_wildcard, order_of, rank_of, suit_of
from fabledan.engine import GuandanRound

from .agents import make_agent
from .runtime import ROOT, atomic_json, system_info

TYPE_NAMES = ["过", "单张", "对子", "三张", "三带二", "顺子", "三连对", "钢板", "炸弹", "同花顺", "四王炸"]
PLAYERS = ["你", "右侧对手", "你的搭档", "左侧对手"]


def card_json(c, level):
    r = rank_of(c)
    return dict(id=c, rank=RANK_NAMES[r] if r < 13 else ("小王" if r == 13 else "大王"),
                suit=["♥", "♦", "♠", "♣"][suit_of(c)] if r < 13 else "★",
                red=r == 14 or (r < 13 and suit_of(c) < 2), wild=is_wildcard(c, level))


def move_json(m, lv, index=None):
    return dict(index=index, type=TYPE_NAMES[m.type], size=m.size,
                cards=[card_json(c, lv) for c in m.cards],
                claims=[RANK_NAMES[r] for r in m.claim_ranks],
                label=TYPE_NAMES[m.type] + (" · " + " ".join(RANK_NAMES[r] for r in m.claim_ranks) if m.size else ""))


class Session:
    def __init__(self, seed, level, agent, agent_label):
        import random
        self.id = secrets.token_hex(12)
        self.round = GuandanRound(level, random.Random(seed))
        self.gen = self.round.play_steps()
        self.obs = next(self.gen)
        self.agent = agent
        self.agent_label = agent_label
        self.version = 0
        self.result = None
        self.touched = time.monotonic()
        self.lock = threading.Lock()
        self._advance_ai()

    def _send(self, idx):
        try:
            self.obs = self.gen.send(int(idx))
        except StopIteration as e:
            rewards, ranking = e.value
            self.result = dict(rewards=rewards, ranking=ranking, won=rewards[0] > 0)
            self.obs = None

    def _advance_ai(self):
        while self.obs is not None and self.obs["player"] != 0:
            self._send(self.agent.act(self.obs))

    def play(self, index, version):
        with self.lock:
            if version != self.version:
                raise HTTPException(409, "牌局已更新，请刷新后重新出牌。")
            if self.obs is None or not 0 <= index < len(self.obs["legal"]):
                raise HTTPException(400, "无效的出牌选择。")
            self._send(index)
            self._advance_ai()
            self.version += 1
            self.touched = time.monotonic()
            return self.state()

    def state(self):
        lv = self.round.lv
        played = []
        for event in self.round.events[-32:]:
            if event[0] == "play":
                played.append(dict(player=event[1], move=move_json(event[2], lv)))
            elif event[0] == "pass":
                played.append(dict(player=event[1], move=dict(type="过", label="过", cards=[], claims=[])))
        hand = sorted(self.round.hands[0], key=lambda c: (-order_of(rank_of(c), lv), suit_of(c), c))
        # Never send other players' hidden cards to the browser.
        return dict(id=self.id, version=self.version, level=RANK_NAMES[lv],
                    hand=[card_json(c, lv) for c in hand],
                    remaining=[len(h) for h in self.round.hands],
                    done=self.round.done_order, current_player=self.obs["player"] if self.obs else None,
                    legal=[move_json(m, lv, i) for i, m in enumerate(self.obs["legal"])] if self.obs else [],
                    history=played, result=self.result, agent=self.agent_label)


class NewGame(BaseModel):
    seed: int = Field(default_factory=lambda: secrets.randbelow(2**31))
    level: int = Field(default=1, ge=0, le=12)
    agent: str = "team-rule"


class Play(BaseModel):
    session: str
    index: int = Field(ge=0)
    version: int = Field(ge=0)


# Resuming rejects a checkpoint whose optimisation settings moved, so the form
# keeps these on the values the run was created with.
FROZEN_ON_RESUME = ("seed", "eval_seed", "epsilon", "epsilon_final", "model_size",
                    "ntp_weight", "belief_weight", "replay_size", "learning_rate")
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,39}")


class TrainOverrides(BaseModel):
    """Every knob the console may set, bounded before TrainConfig sees it."""
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    iterations: int | None = Field(default=None, ge=1, le=1_000_000)
    games_per_iteration: int | None = Field(default=None, ge=1, le=4096)
    workers: int | None = Field(default=None, ge=1, le=64)
    batch_size: int | None = Field(default=None, ge=1, le=4096)
    updates: int | None = Field(default=None, ge=1, le=1024)
    replay_size: int | None = Field(default=None, ge=64, le=1_048_576)
    learning_rate: float | None = Field(default=None, gt=0, le=1)
    epsilon: float | None = Field(default=None, ge=0, le=1)
    epsilon_final: float | None = Field(default=None, ge=0, le=1)
    ntp_weight: float | None = Field(default=None, ge=0, le=10)
    belief_weight: float | None = Field(default=None, ge=0, le=10)
    pool_fraction: float | None = Field(default=None, ge=0, le=1)
    pool_size: int | None = Field(default=None, ge=1, le=256)
    snapshot_every: int | None = Field(default=None, ge=1, le=100_000)
    eval_every: int | None = Field(default=None, ge=1, le=100_000)
    eval_pairs: int | None = Field(default=None, ge=1, le=5000)
    seed: int | None = Field(default=None, ge=0, le=2**31-1)
    threads: int | None = Field(default=None, ge=1, le=64)
    bucket_batches: bool | None = None
    pipeline: bool | None = None
    save_every: int | None = Field(default=None, ge=1, le=10_000)
    max_minutes: float | None = Field(default=None, ge=0, le=10_080)
    device: str | None = Field(default=None, max_length=16)
    model_size: Literal["small", "full"] | None = None
    rule_opponents: list[str] | None = Field(default=None, min_length=1, max_length=8)


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile: str = Field(default="league", max_length=40)
    resume: str | None = Field(default=None, max_length=64)
    name: str | None = Field(default=None, max_length=40)
    overrides: TrainOverrides = TrainOverrides()


class RunLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(default="", max_length=40)


class RunArchive(BaseModel):
    model_config = ConfigDict(extra="forbid")
    archived: bool = True


class NodeExport(BaseModel):
    """A checkpoint of this run, exported as portable weights."""
    model_config = ConfigDict(extra="forbid")
    path: str = Field(max_length=200)


class EvalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: str = Field(max_length=200)
    b: str = Field(default="team-rule", max_length=200)
    pairs: int = Field(default=50, ge=1, le=2000)
    seed: int = Field(default=9_000_019, ge=0, le=2**31-1)
    device: Literal["cpu", "mps", "cuda"] = "cpu"


def report_label(spec):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", spec.replace("runs/", "").removesuffix(".pt")).strip("-")[:40] or "agent"


def profile_training(profile):
    path = (ROOT / "configs" / (profile + ".toml")).resolve()
    if not RUN_NAME.fullmatch(profile) or not path.is_relative_to((ROOT / "configs").resolve()) or not path.is_file():
        raise HTTPException(400, "未知的基础配置。")
    return tomllib.loads(path.read_text()).get("training", {})


def training_config(req):
    from .runtime import device_for
    from .train import TrainConfig
    if req.resume:
        base = (read_json(run_directory(req.resume) / "config.json", {}) or {}).get("training")
        if not base:
            raise HTTPException(400, "该训练记录没有保存配置，无法继续训练。")
    else:
        base = profile_training(req.profile)
    data = {**base, **{k: v for k, v in req.overrides.model_dump().items() if v is not None}}
    if req.resume:
        data.update({k: base[k] for k in FROZEN_ON_RESUME if k in base})
    try:
        cfg = TrainConfig(**data)
        cfg.validate()
        device_for(cfg.device)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return cfg


def training_toml(cfg):
    lines = ["# 由网页控制台生成；可直接用于 uv run python -m guandan train --config", "[training]"]
    for key, value in asdict(cfg).items():
        if isinstance(value, list):
            lines.append(f"{key} = [" + ", ".join(json.dumps(item) for item in value) + "]")
        elif isinstance(value, bool):
            lines.append(f"{key} = {str(value).lower()}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value!r}")
        else:
            lines.append(f"{key} = {json.dumps(value)}")
    return "\n".join(lines) + "\n"


def read_json(path, fallback=None):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return fallback


def read_metrics(directory, limit=240):
    rows = []
    try:
        for line in (directory / "metrics.jsonl").read_text().splitlines()[-limit:]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return rows


_EVALUATIONS = {}
_TEAMWORK = {}


def read_evaluations(directory):
    """Development-split curves; per-deal records stay on disk, parses are cached."""
    paths = sorted(directory.glob("eval-*.json")) + sorted(directory.glob("gate-*.json"))
    signature = tuple((path.name, path.stat().st_mtime_ns) for path in paths)
    cached = _EVALUATIONS.get(directory.name)
    if cached and cached[0] == signature:
        return cached[1]
    reports = []
    for path in paths:
        report = read_json(path, None)
        if isinstance(report, dict) and "win_rate" in report:
            reports.append({k: v for k, v in report.items() if k != "records"})
    reports.sort(key=lambda r: (r.get("iteration", 0), r.get("opponent", "")))
    _EVALUATIONS[directory.name] = (signature, reports)
    return reports


def read_teamwork(directory):
    """对家意识诊断曲线：同一批固定局面上，牌权翻转后改判让牌的比例。"""
    paths = sorted(directory.glob("teamwork-*.json"))
    signature = tuple((path.name, path.stat().st_mtime_ns) for path in paths)
    cached = _TEAMWORK.get(directory.name)
    if cached and cached[0] == signature:
        return cached[1]
    reports = []
    for path in paths:
        report = read_json(path, None)
        if isinstance(report, dict) and "flip_rate" in report:
            reports.append(report)
    reports.sort(key=lambda r: r.get("iteration", 0))
    _TEAMWORK[directory.name] = (signature, reports)
    return reports


def development_scores(evaluations):
    """Per-iteration strength, plus the iteration the champion was promoted at."""
    scores, promoted = {}, None
    for report in evaluations:
        iteration, opponent = report.get("iteration"), report.get("opponent")
        if opponent == "champion":
            if (report.get("ci95") or [0])[0] > .5:
                promoted = iteration
            continue
        if iteration is None:
            continue
        # team-rule is the yardstick every run shares.
        if iteration not in scores or opponent == "team-rule":
            scores[iteration] = dict(iteration=iteration, opponent=opponent,
                                     win_rate=report["win_rate"], ci95=report.get("ci95"),
                                     games=report.get("games"))
    return scores, promoted


def active_pool(directory):
    """Snapshot files the resumable checkpoint still uses as frozen opponents."""
    path = directory / "latest.pt"
    if path.is_file():
        try:
            import torch
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            return [name for name in checkpoint.get("pool", []) if isinstance(name, str)]
        except Exception:
            pass
    size = ((read_json(directory / "config.json", {}) or {}).get("training") or {}).get("pool_size", 8)
    return [f"pool/{item.name}" for item in sorted((directory / "pool").glob("*.pt"))[-size:]]


def prune_plan(directory, keep_pool=True):
    """Keep the best-scoring snapshot (and, by default, the live opponent pool)."""
    scores, _ = development_scores(read_evaluations(directory))
    best = max(scores.values(), key=lambda item: item["win_rate"], default=None)
    reasons = {}
    if best is not None:
        reasons[directory / f"pool/iteration-{best['iteration']:06}.pt"] = \
            f"开发集最高胜率 {best['win_rate']:.1%}（迭代 {best['iteration']}）"
    if keep_pool:
        for name in active_pool(directory):
            reasons.setdefault(directory / name, "当前对手池，续训需要")
    kept, removed = [], []
    for path in sorted((directory / "pool").glob("*.pt"), reverse=True):
        entry = dict(path=str(path.relative_to(ROOT)), name=path.name, size=path.stat().st_size)
        if path in reasons:
            kept.append({**entry, "reason": reasons[path]})
        else:
            removed.append(entry)
    return dict(run=directory.name, keep_pool=keep_pool, best=best, kept=kept, removed=removed,
                freed=sum(item["size"] for item in removed),
                total=sum(item["size"] for item in kept + removed))


def run_activity(directory):
    """Last training activity, so renaming a record does not reorder the list."""
    status = directory / "status.json"
    return (status if status.is_file() else directory).stat().st_mtime


def run_flags(directory):
    """Console-owned bookkeeping: the display name and whether the run is archived."""
    data = read_json(directory / "label.json", {}) or {}
    label = data.get("label")
    return dict(label=label.strip() if isinstance(label, str) and label.strip() else None,
                archived=bool(data.get("archived")))


def write_flags(directory, **changes):
    """Merge, so renaming a record never silently clears its archived state."""
    data = {k: v for k, v in {**(read_json(directory / "label.json", {}) or {}), **changes}.items()
            if v not in (None, False, "")}
    if data:
        atomic_json(directory / "label.json", data)
    else:
        (directory / "label.json").unlink(missing_ok=True)
    return data


def directory_size(directory):
    total = 0
    for item in directory.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:  # a live run rotates files underneath the walk
            continue
    return total


def lock_state(directory):
    """The training lock with its owner resolved, so a crash is recoverable here."""
    path = directory / ".train.lock"
    if not path.is_file():
        return None
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        # Not written by this trainer, so no process can be proven to own it.
        return dict(pid=None, alive=False)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return dict(pid=pid, alive=False)
    except OSError:
        pass  # running under another user, or unknowable: assume it is alive
    return dict(pid=pid, alive=True)


def checkpoint_list(directory, status):
    """Every playable node of a run, each carrying what it is and how it scored."""
    scores, promoted = development_scores(read_evaluations(directory))
    newest = max(scores) if scores else None

    def node(path, label, kind, iteration):
        stat = path.stat()
        # A node's own evaluation when it has one; otherwise the closest earlier
        # measurement, kept labelled with the iteration it was actually measured at.
        score = scores.get(iteration)
        if score is None and kind == "latest" and newest is not None:
            score = scores[newest]
        return dict(path=str(path.relative_to(ROOT)), label=label, kind=kind, iteration=iteration,
                    score=score, size=stat.st_size, created_at=stat.st_mtime)

    items = []
    if (directory / "latest.pt").is_file():
        items.append(node(directory / "latest.pt", "最新模型", "latest", (status or {}).get("iteration")))
    if (directory / "champion.pt").is_file():
        items.append(node(directory / "champion.pt", "当前冠军", "champion", promoted))
    for path in sorted((directory / "pool").glob("*.pt"), reverse=True):
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        items.append(node(path, f"迭代 {int(digits)} 快照" if digits else "初始模型",
                          "pool", int(digits) if digits else 0))
    return items


def agent_spec(spec):
    from .programs import REGISTRY
    if spec in {"team-rule", "rule", "random"} | {"program:" + name for name in REGISTRY}:
        return spec
    path = (ROOT / spec).resolve()
    if not path.is_relative_to((ROOT / "runs").resolve()) or not path.is_file() or path.suffix != ".pt":
        raise HTTPException(400, "请选择本项目 runs 目录下的模型。")
    return str(path)


def agent_ref(spec):
    """Validated spec in project-relative form; reports then name the node, not a path."""
    resolved = agent_spec(spec)
    return str(Path(resolved).relative_to(ROOT)) if resolved.endswith(".pt") else resolved


def run_directory(name):
    """A single directory directly inside runs/ — never runs/ itself, never a nested path."""
    directory = (ROOT / "runs" / name).resolve()
    if directory.parent != (ROOT / "runs").resolve() or not directory.is_dir():
        raise HTTPException(404, "训练记录不存在。")
    return directory


def create_app():
    app = FastAPI(title="掼蛋研究室")
    sessions = {}
    session_lock = threading.Lock()
    process_lock = threading.Lock()
    eval_lock = threading.Lock()
    training = {"process": None, "out": None}
    evaluation = {"process": None, "report": None, "log": None, "request": None}
    web_root = ROOT / "web"

    @app.get("/")
    def index():
        return FileResponse(web_root / "index.html")

    @app.get("/api/overview")
    def overview(run: str | None = None):
        from .expert import available
        runs = []
        # Archived records sink below the working ones, each group newest first.
        directories = [(directory, run_flags(directory))
                       for directory in (ROOT / "runs").glob("*") if directory.is_dir()]
        directories.sort(key=lambda item: (item[1]["archived"], -run_activity(item[0])))
        # Curves are sent for the inspected run only; the rest stay list entries.
        detailed = run if any(d.name == run for d, _ in directories) else next(
            (d.name for d, _ in directories), None)
        for directory, flags in directories:
            status = read_json(directory / "status.json", {})
            config = read_json(directory / "config.json", {}) or {}
            full = directory.name == detailed
            runs.append(dict(name=directory.name, status=status,
                             metrics=read_metrics(directory) if full else [],
                             evaluations=read_evaluations(directory) if full else [],
                             teamwork=read_teamwork(directory) if full else [],
                             training=config.get("training", {}),
                             checkpoints=checkpoint_list(directory, status),
                             label=flags["label"], archived=flags["archived"],
                             size=directory_size(directory), lock=lock_state(directory),
                             snapshots=len(list((directory / "pool").glob("*.pt"))),
                             champion=(directory / "champion.pt").is_file(),
                             console=(directory / "console.log").is_file(),
                             updated_at=run_activity(directory),
                             checkpoint=(directory / "latest.pt").is_file()))
        reports = []
        for path in sorted((ROOT / "reports").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            report = read_json(path, {})
            if isinstance(report, dict):
                reports.append({"name": path.name, **{k: v for k, v in report.items() if k != "records"}})
        proc = training["process"]
        live = bool(proc and proc.poll() is None)
        return dict(system=system_info(), expert_available=available(), runs=runs, reports=reports,
                    managed_training=live, managed_run=training["out"].name if live and training["out"] else None,
                    server_time=time.time())

    @app.put("/api/run/{name}/label")
    def set_run_label(name: str, req: RunLabel):
        directory = run_directory(name)
        label = re.sub(r"[\x00-\x1f]", "", req.label).strip()
        write_flags(directory, label=label or None)
        return dict(name=name, label=label or None)

    @app.put("/api/run/{name}/archive")
    def set_run_archive(name: str, req: RunArchive):
        """Archiving only moves a record out of the way; nothing on disk is touched."""
        write_flags(run_directory(name), archived=req.archived)
        return dict(name=name, archived=req.archived)

    def live_console_run(name):
        with process_lock:
            proc, directory = training["process"], training["out"]
            return bool(proc and proc.poll() is None and directory and directory.name == name)

    def idle_or_409(name, action):
        """Every destructive edit needs the record to be nobody's working directory."""
        directory = run_directory(name)
        if live_console_run(name):
            raise HTTPException(409, f"该训练正在运行，请先「保存并停止」再{action}。")
        state = lock_state(directory)
        if state and state["alive"]:
            raise HTTPException(409, f"进程 {state['pid']} 仍在写入该目录，请先结束它。")
        return directory

    @app.delete("/api/run/{name}/lock")
    def clear_run_lock(name: str):
        """Remove a lock left behind by a killed trainer, never one still in use."""
        directory = run_directory(name)
        if live_console_run(name):
            raise HTTPException(409, "该训练正在本控制台运行，请先「保存并停止」。")
        state = lock_state(directory)
        if state is None:
            raise HTTPException(404, "该训练记录没有训练锁。")
        if state["alive"]:
            raise HTTPException(409, f"进程 {state['pid']} 仍持有该训练锁，请先结束它。")
        (directory / ".train.lock").unlink(missing_ok=True)
        return dict(name=name, removed=True, pid=state["pid"])

    @app.delete("/api/run/{name}")
    def delete_run(name: str):
        """Irreversible: the caller confirms, the server only refuses live records."""
        directory = idle_or_409(name, "删除")
        freed = directory_size(directory)
        shutil.rmtree(directory)
        _EVALUATIONS.pop(directory.name, None)
        return dict(name=name, deleted=True, freed=freed)

    @app.delete("/api/run/{name}/node")
    def delete_node(name: str, path: str):
        """Remove one checkpoint file. Only files the node list itself offers."""
        directory = idle_or_409(name, "删除模型文件")
        status = read_json(directory / "status.json", {})
        nodes = {ROOT / node["path"]: node for node in checkpoint_list(directory, status)}
        target = (ROOT / path).resolve()
        if target not in nodes:
            raise HTTPException(400, "请选择该训练记录中的模型节点。")
        freed = target.stat().st_size
        target.unlink()
        return dict(name=name, path=path, freed=freed, label=nodes[target]["label"],
                    kind=nodes[target]["kind"], remaining=len(nodes) - 1)

    @app.post("/api/run/{name}/export")
    def export_node(name: str, req: NodeExport):
        """Weights-only .npz: portable, and free of optimiser state and replay."""
        directory = run_directory(name)
        source = (ROOT / req.path).resolve()
        if not source.is_relative_to(directory) or source.suffix != ".pt" or not source.is_file():
            raise HTTPException(400, "请选择该训练记录中的检查点文件。")
        from fabledan.model_torch import export_npz

        from .agents import load_model
        try:
            model, _ = load_model(source)
        except Exception as exc:
            raise HTTPException(400, f"无法读取该检查点：{exc}") from exc
        target = ROOT / "exports" / f"{directory.name}-{source.stem}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        export_npz(model, target)
        return dict(path=str(target.relative_to(ROOT)), size=target.stat().st_size,
                    source=str(source.relative_to(ROOT)))

    @app.get("/api/run/{name}/prune")
    def prune_preview(name: str, keep_pool: bool = True):
        return prune_plan(run_directory(name), keep_pool)

    @app.post("/api/run/{name}/prune")
    def prune_snapshots(name: str, keep_pool: bool = True):
        directory = run_directory(name)
        if (directory / ".train.lock").exists():
            raise HTTPException(409, "该训练记录正在训练，停止后再清理快照。")
        plan = prune_plan(directory, keep_pool)
        for item in plan["removed"]:
            path = (ROOT / item["path"]).resolve()
            if path.is_relative_to((directory / "pool").resolve()):
                path.unlink(missing_ok=True)
        return {**plan, "done": True}

    @app.get("/api/run/{name}/log")
    def run_log(name: str, lines: int = 160):
        path = run_directory(name) / "console.log"
        if not path.is_file():
            return dict(name=name, lines=[])
        with path.open(errors="replace") as handle:
            tail = handle.readlines()[-max(1, min(lines, 500)):]
        return dict(name=name, lines=[line.rstrip("\n") for line in tail])

    @app.post("/api/game")
    def new_game(req: NewGame):
        spec = agent_spec(req.agent)
        try:
            agent = make_agent(spec, req.seed)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc
        session = Session(req.seed, req.level, agent, req.agent)
        with session_lock:
            for key in list(sessions):
                if time.monotonic() - sessions[key].touched > 3600:
                    del sessions[key]
            if len(sessions) >= 16:
                del sessions[min(sessions, key=lambda k: sessions[k].touched)]
            sessions[session.id] = session
        return session.state()

    @app.post("/api/play")
    def play(req: Play):
        session = sessions.get(req.session)
        if not session:
            raise HTTPException(404, "牌局不存在或已过期，请重新开局。")
        return session.play(req.index, req.version)

    @app.post("/api/hint")
    def hint(req: Play):
        session = sessions.get(req.session)
        if not session:
            raise HTTPException(404, "牌局不存在。")
        with session.lock:
            if req.version != session.version:
                raise HTTPException(409, "牌局已更新。")
            if session.obs is None:
                raise HTTPException(400, "本局已结束。")
            return dict(index=session.agent.act(session.obs), version=session.version)

    @app.get("/api/train/options")
    def training_options():
        import torch

        from .programs import REGISTRY
        from .train import TrainConfig
        devices = ["auto", "cpu"] + (["mps"] if torch.backends.mps.is_available() else []) \
            + (["cuda"] if torch.cuda.is_available() else [])
        profiles = {path.stem: profile_training(path.stem)
                    for path in sorted((ROOT / "configs").glob("*.toml"))}
        return dict(defaults=asdict(TrainConfig()), profiles=profiles, devices=devices,
                    opponents=["team-rule", "rule", "random"] + ["program:" + name for name in REGISTRY],
                    frozen_on_resume=list(FROZEN_ON_RESUME))

    @app.post("/api/train/preview")
    def preview_training(req: TrainRequest):
        cfg = training_config(req)
        return dict(training=asdict(cfg), toml=training_toml(cfg))

    @app.post("/api/train")
    def start_training(req: TrainRequest):
        cfg = training_config(req)
        with process_lock:
            proc = training["process"]
            if proc and proc.poll() is None:
                raise HTTPException(409, "已有训练正在运行。")
            if req.resume:
                directory = run_directory(req.resume)
                if not (directory / "latest.pt").is_file():
                    raise HTTPException(400, "该训练记录没有检查点，无法继续训练。")
                # A console stop leaves this marker; continuing means clearing it.
                (directory / "STOP").unlink(missing_ok=True)
            else:
                name = req.name or "web-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
                if not RUN_NAME.fullmatch(name):
                    raise HTTPException(400, "训练名称只能使用字母、数字、点、下划线和连字符，且不超过 40 个字符。")
                directory = ROOT / "runs" / name
                if directory.exists():
                    raise HTTPException(409, "同名训练记录已存在，请换一个名称。")
                directory.mkdir(parents=True)
            (directory / "request.toml").write_text(training_toml(cfg))
            command = [sys.executable, "-u", "-m", "guandan", "train",
                       "--config", str(directory / "request.toml"), "--out", str(directory)]
            if req.resume:
                command += ["--resume", str(directory / "latest.pt")]
            with (directory / "console.log").open("a" if req.resume else "w") as log:
                proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            training.update(process=proc, out=directory)
            return dict(name=directory.name, pid=proc.pid, resumed=bool(req.resume), training=asdict(cfg))

    @app.post("/api/evaluate")
    def start_evaluation(req: EvalRequest):
        a, b = agent_ref(req.a), agent_ref(req.b)
        if a == b:
            raise HTTPException(400, "两侧选择了同一个模型，评测结果没有意义。")
        with eval_lock:
            proc = evaluation["process"]
            if proc and proc.poll() is None:
                raise HTTPException(409, "已有评测正在运行。")
            name = (f"web-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    f"-{report_label(req.a)}-vs-{report_label(req.b)}")
            report = ROOT / "reports" / (name + ".json")
            report.parent.mkdir(parents=True, exist_ok=True)
            log = report.with_suffix(".log")
            command = [sys.executable, "-u", "-m", "guandan", "eval", "--a", a, "--b", b,
                       "--pairs", str(req.pairs), "--seed", str(req.seed),
                       "--device", req.device, "--out", str(report)]
            with log.open("w") as handle:
                proc = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            evaluation.update(process=proc, report=report, log=log,
                              request=dict(a=req.a, b=req.b, pairs=req.pairs, seed=req.seed,
                                           device=req.device, started_at=time.time()))
            return dict(report=report.name, pid=proc.pid)

    @app.get("/api/evaluate")
    def evaluation_status():
        with eval_lock:
            proc, report, log = evaluation["process"], evaluation["report"], evaluation["log"]
            request = evaluation["request"]
        if proc is None:
            return dict(state="idle")
        running = proc.poll() is None
        lines = []
        if log and log.is_file():
            with log.open(errors="replace") as handle:
                lines = [line.rstrip("\n") for line in handle.readlines()[-40:]]
        progress = None
        for line in reversed(lines):
            found = re.search(r"评测 (\d+)/(\d+) 组 · 胜率 ([\d.]+)%", line)
            if found:
                progress = dict(done=int(found[1]), total=int(found[2]), win_rate=float(found[3])/100)
                break
        result = None
        if not running and report and report.is_file():
            result = {k: v for k, v in (read_json(report, {}) or {}).items() if k != "records"}
        state = "running" if running else ("completed" if result else "failed")
        return dict(state=state, request=request, report=report.name if report else None,
                    progress=progress, lines=lines, result=result)

    @app.post("/api/evaluate/stop")
    def stop_evaluation():
        with eval_lock:
            proc = evaluation["process"]
            if proc is None or proc.poll() is not None:
                raise HTTPException(409, "没有正在运行的评测。")
            proc.terminate()
            return {"status": "terminated"}

    @app.post("/api/train/stop")
    def stop_training():
        with process_lock:
            proc, directory = training["process"], training["out"]
            if proc is None or proc.poll() is not None:
                raise HTTPException(409, "没有由控制台启动的活动训练。")
            (directory / "STOP").touch()
            return {"status": "stopping_after_current_iteration"}

    app.mount("/assets", StaticFiles(directory=web_root), name="assets")

    from .expert import DANLM_ROOT, available, web_app
    if available():
        # Serve the upstream UI under a prefix without altering vendored files.
        @app.get("/expert/", response_class=HTMLResponse)
        def expert_index():
            html = (DANLM_ROOT / "ui/static/index.html").read_text()
            return html.replace('"/static/', '"/expert/static/')

        @app.get("/expert/static/app.js")
        def expert_js():
            js = (DANLM_ROOT / "ui/static/app.js").read_text()
            js = js.replace("/api/", "/expert/api/")
            js = js.replace("localStorage.getItem('lang') || 'en'", "localStorage.getItem('lang') || 'zh'")
            return Response(js, media_type="application/javascript")

        app.mount("/expert", web_app())
    return app
