"""DMC self-play with public observations, auxiliary learning and a frozen pool.

CPU actor processes collect complete rounds. The learner runs on CPU/MPS/CUDA.
Only learner actions enter replay in mixed-opponent games. Oracle hidden cards
are labels for the auxiliary belief loss, never policy inputs.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fabledan.encode import ENCODING_VERSION, encode_decision
from fabledan.engine import GuandanRound, random_tribute_mode
from fabledan.model_torch import FableDanNet, ModelConfig, export_npz
from fabledan.ring import _belief_label
from fabledan.train import Replay

from .agents import ModelAgent, TeamRuleAgent, load_model, make_agent
from .diagnose import PROBE_SEED, probe_states, teamwork_probe
from .evaluate import run_evaluation
from .runtime import RULESET, atomic_json, device_for, seed_all, system_info


@dataclass
class TrainConfig:
    iterations: int = 200
    games_per_iteration: int = 8
    workers: int = 1
    batch_size: int = 32
    updates: int = 8
    replay_size: int = 8192
    learning_rate: float = 0.0001
    epsilon: float = 0.10
    epsilon_final: float = 0.02
    ntp_weight: float = 0.02
    belief_weight: float = 0.05
    pool_fraction: float = 0.5
    pool_size: int = 8
    pool_recent: int = 4            # newest snapshots the pool always keeps
    pool_archive_every: int = 200   # minimum iteration gap between older ones
    gate_threshold: float = 0.5     # champion promotion, on paired-deal win rate
    snapshot_every: int = 10
    eval_every: int = 10
    eval_pairs: int = 12
    seed: int = 17
    eval_seed: int = 1_000_003
    threads: int = 2
    device: str = "auto"
    model_size: str = "small"
    max_minutes: float = 30
    actor_timeout_seconds: int = 180
    # Throughput knobs. None of them change what is learned, only how fast the
    # same computation runs; all three are safe to flip on an existing run.
    bucket_batches: bool = True     # draw each batch from one length bucket
    pipeline: bool = True           # actors collect while the learner updates
    save_every: int = 1             # iterations between full replay checkpoints
    rule_opponents: list[str] = field(default_factory=lambda: ["team-rule"])

    def validate(self):
        for name in ("iterations", "games_per_iteration", "workers", "batch_size", "updates",
                     "replay_size", "pool_size", "pool_recent", "pool_archive_every",
                     "snapshot_every", "eval_every", "eval_pairs", "threads",
                     "actor_timeout_seconds", "save_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.workers > self.games_per_iteration:
            raise ValueError("workers cannot exceed games_per_iteration")
        if self.pool_recent > self.pool_size:
            raise ValueError("pool_recent cannot exceed pool_size")
        if not 0 < self.gate_threshold < 1:
            raise ValueError("gate_threshold must lie strictly between 0 and 1")
        if not 0 <= self.pool_fraction <= 1 or not 0 <= self.epsilon_final <= self.epsilon <= 1:
            raise ValueError("invalid probability or epsilon schedule")
        if self.learning_rate <= 0 or self.max_minutes < 0:
            raise ValueError("learning_rate must be positive; max_minutes must be nonnegative")
        if self.ntp_weight < 0 or self.belief_weight < 0:
            raise ValueError("auxiliary weights must be nonnegative")
        from .programs import REGISTRY
        allowed = {"team-rule", "rule", "random"} | {"program:" + name for name in REGISTRY}
        if not self.rule_opponents or not set(self.rule_opponents) <= allowed:
            raise ValueError(f"rule_opponents must use {sorted(allowed)}")


def network_config(size):
    if size == "small":
        return ModelConfig(d_model=64, n_blocks=2, n_heads=2, qk_dim=32, v_dim=32,
                           ffn_hidden=128, hand_hidden=128, n_hand_layers=2,
                           q_hidden=128, n_q_layers=2)
    if size == "full":
        return ModelConfig()
    raise ValueError("model_size must be small or full")


def _collect(payload):
    """Process-safe bounded actor job; exceptions propagate to the learner."""
    cfg_dict, state, games, seed, epsilon, pool_fraction, opponent_path = payload
    torch.set_num_threads(1)
    seed_all(seed)
    rng = random.Random(seed)
    model = FableDanNet(ModelConfig.from_dict(cfg_dict))
    # NumPy is intentionally the wire format: torch Tensor queue reduction
    # opens shared-memory file descriptors and can hang in macOS sandboxes.
    model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
    policy = ModelAgent(model)
    opponent_path = opponent_path or "team-rule"
    opponent = make_agent(opponent_path, seed+1)
    samples, returns = [], []
    modes = {"self": 0}
    for _ in range(games):
        round_rng = random.Random(rng.getrandbits(48))
        rnd = GuandanRound(round_rng.randrange(13), round_rng, random_tribute_mode(round_rng))
        mixed = rng.random() < pool_fraction
        team = rng.randrange(2)
        mode = "self" if not mixed else (opponent_path if opponent_path in ("team-rule", "rule", "random") or opponent_path.startswith("program:") else "snapshot")
        modes[mode] = modes.get(mode, 0) + 1
        episode = []
        gen = rnd.play_steps()
        obs = next(gen)
        try:
            for _step in range(600):
                learner = not mixed or obs["player"] % 2 == team
                if learner:
                    tokens, feats = encode_decision(obs)
                    # Uniform exploration retains support for ALL legal actions.
                    if rng.random() < epsilon:
                        action = rng.randrange(len(obs["legal"]))
                    else:
                        action = policy.act(obs)
                    episode.append((np.asarray(tokens, dtype=np.int16), feats[action],
                                    obs["player"], _belief_label(rnd, obs["player"])))
                else:
                    action = opponent.act(obs)
                obs = gen.send(action)
            raise RuntimeError("round exceeded 600 decisions")
        except StopIteration as result:
            rewards, _ = result.value
        for tokens, feat, player, belief in episode:
            samples.append((tokens, feat, rewards[player] / 3.0, belief))
        returns.append(rewards[team])
    return samples, games, modes


def _save(path, model, opt, replay, cfg, meta, rng, pool):
    n = replay.n
    ck = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
              optimizer=opt.state_dict(), config=model.cfg.to_dict(), training_config=asdict(cfg),
              meta=meta, encoding=ENCODING_VERSION, ruleset=RULESET, pool=pool,
              numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
              replay=dict(tokens=[torch.from_numpy(t) for t in replay.toks[:n]],
                          feat=torch.from_numpy(replay.feat[:n].copy()),
                          target=torch.from_numpy(replay.targ[:n].copy()),
                          belief=torch.from_numpy(replay.belief[:n].copy()), ptr=replay.ptr))
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ck, tmp)
    tmp.replace(path)


def _snapshot(path, model, iteration):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    config=model.cfg.to_dict(), meta={"iteration": iteration},
                    encoding=ENCODING_VERSION, ruleset=RULESET), tmp)
    tmp.replace(path)


def _best_evaluated(out):
    """Highest development win rate already on disk, on the team-rule yardstick."""
    best, rate = None, -1.0
    for path in out.glob("eval-*.json"):
        if not re.fullmatch(r"eval-\d+\.json", path.name):
            continue
        report = json.loads(path.read_text()) if path.is_file() else {}
        if isinstance(report.get("win_rate"), (int, float)) and report["win_rate"] > rate:
            best, rate = report.get("iteration"), float(report["win_rate"])
    return best, rate


def _pool_iteration(name):
    """Iteration encoded in a pool filename; None for the random-init entry."""
    match = re.search(r"iteration-(\d+)\.pt$", name)
    return int(match.group(1)) if match else None


def _update_pool(pool, name, cfg):
    """Recent snapshots, plus a thinned archive of the older ones.

    A pool holding only the last few checkpoints lets the policy cycle: it
    beats the selves it just came from while losing to what it was a thousand
    iterations ago. Older entries are kept at `pool_archive_every` spacing
    instead, so the pool reaches back about
    (pool_size - pool_recent) * pool_archive_every iterations at bounded size.
    The random-init reference drops out as soon as real snapshots exist --
    games against it stop being informative long before it would age out.
    """
    pool = pool + [name]
    recent, older = pool[-cfg.pool_recent:], pool[:-cfg.pool_recent]
    archive = []
    for path in older:
        iteration = _pool_iteration(path)
        if iteration is None:
            continue
        previous = _pool_iteration(archive[-1]) if archive else None
        if previous is None or iteration - previous >= cfg.pool_archive_every:
            archive.append(path)
    room = max(0, cfg.pool_size - len(recent))
    return (archive[-room:] if room else []) + recent


def _prune_pool(out, pool, keep_iteration=None):
    """Snapshots outside the active opponent pool are dead weight on disk."""
    keep = {out / name for name in pool}
    if keep_iteration is not None:
        keep.add(out / f"pool/iteration-{keep_iteration:06}.pt")
    removed = 0
    for path in (out / "pool").glob("*.pt"):
        if path not in keep:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def train(cfg: TrainConfig, out, resume=None):
    cfg.validate()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "pool").mkdir(exist_ok=True)
    if (out / "latest.pt").exists() and not resume:
        raise ValueError("输出目录已有模型，请使用 --resume 或选择新的目录。")
    lock_path = out / ".train.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError(f"目录存在训练锁 {lock_path}；确认旧进程已退出后再移除。") from exc
    os.write(lock_fd, str(os.getpid()).encode())
    os.close(lock_fd)
    try:
        return _train_locked(cfg, out, resume)
    finally:
        lock_path.unlink(missing_ok=True)


def _train_locked(cfg, out, resume):
    device = device_for(cfg.device)
    torch.set_num_threads(cfg.threads)
    seed_all(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    ck = None
    if resume:
        model, ck = load_model(resume, device)
        # Preserve the stochastic training definition across a restart. The
        # caller may change budget, device and CPU parallelism.
        previous = TrainConfig(**ck["training_config"])
        for field in ("seed", "eval_seed", "epsilon", "epsilon_final", "model_size",
                      "ntp_weight", "belief_weight", "replay_size", "learning_rate"):
            if getattr(cfg, field) != getattr(previous, field):
                raise ValueError(f"续训参数 {field} 必须与检查点一致 ({getattr(previous, field)})")
    else:
        model = FableDanNet(network_config(cfg.model_size)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    replay = Replay(cfg.replay_size, belief_dim=45)
    meta = dict(iteration=0, games=0, samples=0, updates=0)
    pool = []
    if ck:
        optimizer.load_state_dict(ck["optimizer"])
        meta.update(ck["meta"])
        rng.bit_generator.state = ck["numpy_rng"]
        torch.set_rng_state(ck["torch_rng"])
        rp = ck["replay"]
        for t, feat, target, belief in zip(rp["tokens"], rp["feat"], rp["target"], rp["belief"]):
            replay.add(t.numpy(), feat.numpy(), float(target), belief.numpy())
        replay.ptr = rp["ptr"]
        # Pool filenames are relative so a complete run directory is portable.
        pool = [p for p in ck.get("pool", []) if (out / p).is_file()]
    if not pool:
        _snapshot(out / "pool/initial.pt", model, meta["iteration"])
        pool = ["pool/initial.pt"]
    atomic_json(out / "config.json", dict(training=asdict(cfg), network=model.cfg.to_dict(),
                                         system=system_info(), encoding=ENCODING_VERSION))
    requested_stop = False
    old_handlers = {}

    def request_stop(*_):
        nonlocal requested_stop
        requested_stop = True
        print("收到停止请求，将在当前一批采样结束后保存。", flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, request_stop)
    best_iteration, best_win = _best_evaluated(out)
    probe = None          # fixed diagnostic states, built at the first evaluation
    start, session_games = time.monotonic(), 0
    executor = None

    def build_jobs(iteration):
        """Actor payloads for one iteration, and the epsilon they sample at."""
        epsilon = max(cfg.epsilon_final, cfg.epsilon * math.exp(-iteration/1000))
        state = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
        jobs = []
        for worker in range(cfg.workers):
            games = cfg.games_per_iteration // cfg.workers + (worker < cfg.games_per_iteration % cfg.workers)
            # Half mixed games use the rule opponent; the rest use a
            # uniformly sampled frozen checkpoint, including older ones.
            opponent = (str(out / pool[int(rng.integers(len(pool)))]) if rng.random() < .5
                        else str(rng.choice(cfg.rule_opponents)))
            jobs.append((model.cfg.to_dict(), state, games,
                         cfg.seed + iteration * 100003 + worker * 997,
                         epsilon, cfg.pool_fraction, opponent))
        return jobs, epsilon

    def dispatch(jobs, epsilon):
        """In-flight actor work: futures when pooled, raw payloads otherwise."""
        return ([executor.submit(_collect, job) for job in jobs] if executor else jobs), epsilon

    def gather(handles):
        if not executor:
            return [_collect(job) for job in handles]
        # The timeout starts when the learner begins WAITING, so prefetched
        # actors are not penalised for the learner step they ran under.
        deadline = time.monotonic() + cfg.actor_timeout_seconds
        return [future.result(timeout=max(.1, deadline-time.monotonic())) for future in handles]

    status = "completed"
    last_loss = None
    if cfg.workers > 1:
        executor = ProcessPoolExecutor(cfg.workers, mp_context=get_context("spawn"))
    prefetch = cfg.pipeline and executor is not None
    pending = None              # actor work already in flight for this iteration
    print(f"设备 {device} · 参数 {sum(p.numel() for p in model.parameters()):,} · {cfg.workers} 个采样进程"
          + (" · 采样与学习并行" if prefetch else ""), flush=True)
    try:
        # --iterations is the number of ADDITIONAL iterations on resume.
        target = meta["iteration"] + cfg.iterations
        while meta["iteration"] < target:
            if requested_stop or (out / "STOP").exists():
                status = "stopped"
                break
            if cfg.max_minutes and time.monotonic() - start >= cfg.max_minutes * 60:
                status = "budget_reached"
                break
            iteration = meta["iteration"] + 1
            collect_start = time.monotonic()
            if pending is None:
                pending = dispatch(*build_jobs(iteration))
            handles, epsilon = pending
            pending = None
            batches = gather(handles)
            # _collect limits CPU threads; restore the learner setting.
            torch.set_num_threads(cfg.threads)
            modes = {"self": 0}
            for samples, games, batch_modes in batches:
                for tokens, feat, target_z, belief in samples:
                    replay.add(tokens, feat, target_z, belief)
                meta["samples"] += len(samples)
                meta["games"] += games
                session_games += games
                for mode, count in batch_modes.items():
                    modes[mode] = modes.get(mode, 0) + count
            # collect_seconds is time the learner spent WAITING on actors; with
            # a pipeline it drops to zero once the actors keep ahead.
            collect_seconds = time.monotonic() - collect_start
            # Start the next batch before updating, so the actors work through
            # the learner step. They then sample from weights one iteration
            # old, which is the ordinary actor-learner trade.
            if prefetch and iteration < target and not requested_stop and not (out / "STOP").exists():
                pending = dispatch(*build_jobs(iteration + 1))
            model.train()
            losses = []
            for _ in range(cfg.updates):
                tokens, lengths, feats, targets, beliefs = replay.sample(
                    cfg.batch_size, rng, cfg.bucket_batches)
                t = torch.as_tensor(tokens, device=device)
                lengths = torch.as_tensor(lengths, device=device)
                f = torch.as_tensor(feats, device=device).unsqueeze(1)
                z = torch.as_tensor(targets, device=device)
                b = torch.as_tensor(beliefs, device=device)
                ctx, hid = model.encode_seq(t, lengths)
                q = model.q_values(ctx, f)[:, 0]
                q_loss = F.mse_loss(q, z)
                ntp_loss = model.ntp_loss(t, hid)
                belief_loss = model.belief_loss(ctx, b)
                loss = q_loss + cfg.ntp_weight * ntp_loss + cfg.belief_weight * belief_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append([loss.item(), q_loss.item(), ntp_loss.item(), belief_loss.item()])
                meta["updates"] += 1
            model.eval()
            meta["iteration"] = iteration
            last_loss = np.mean(losses, axis=0).tolist()
            elapsed = time.monotonic() - start
            row = dict(**meta, loss=last_loss[0], q_loss=last_loss[1], ntp_loss=last_loss[2],
                       belief_loss=last_loss[3], epsilon=epsilon, replay_samples=replay.n,
                       games_per_minute=session_games/max(elapsed, 1e-6)*60,
                       collect_seconds=collect_seconds, elapsed_seconds=elapsed, modes=modes,
                       timestamp=datetime.now(timezone.utc).isoformat())
            with (out / "metrics.jsonl").open("a") as log:
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            if iteration % cfg.snapshot_every == 0:
                name = f"pool/iteration-{iteration:06}.pt"
                _snapshot(out / name, model, iteration)
                pool = _update_pool(pool, name, cfg)
                # Keep the active opponent pool and the best-scoring snapshot only.
                dropped = _prune_pool(out, pool, best_iteration)
                if dropped:
                    print(f"清理 {dropped} 个不再使用的历史快照", flush=True)
            # The replay dominates checkpoint size, so a run may trade a few
            # iterations of crash recovery for the write. Stops and the final
            # exit always write a complete checkpoint below.
            if iteration % cfg.save_every == 0:
                _save(out / "latest.pt", model, optimizer, replay, cfg, meta, rng, pool)
            atomic_json(out / "status.json", dict(status="running", device=device, **row))
            print(f"迭代 {iteration} · {meta['games']} 局 · {meta['samples']} 样本 · loss {last_loss[0]:.4f} · {row['games_per_minute']:.1f} 局/分钟", flush=True)
            if iteration % cfg.eval_every == 0 and not requested_stop:
                candidate = ModelAgent(copy.deepcopy(model).cpu())
                evaluation = run_evaluation(candidate, TeamRuleAgent(), cfg.eval_pairs, cfg.eval_seed)
                evaluation.update(iteration=iteration, opponent="team-rule", split="development")
                atomic_json(out / f"eval-{iteration:06}.json", evaluation)
                if evaluation["win_rate"] > best_win:
                    best_iteration, best_win = iteration, evaluation["win_rate"]
                for spec in cfg.rule_opponents:
                    if spec == "team-rule":
                        continue
                    opponent_result = run_evaluation(candidate, make_agent(spec, cfg.eval_seed),
                                                     cfg.eval_pairs, cfg.eval_seed)
                    opponent_result.update(iteration=iteration, opponent=spec, split="development")
                    label = spec.replace(":", "-")
                    atomic_json(out / f"eval-{label}-{iteration:06}.json", opponent_result)
                    print(f"开发集对 {spec} 胜率 {opponent_result['win_rate']:.1%}", flush=True)
                # Partner awareness on one fixed set of states: a move here is
                # the policy changing, not the state distribution drifting.
                if probe is None:
                    probe = probe_states()
                teamwork = teamwork_probe(candidate.scores, probe)
                teamwork.update(iteration=iteration, split="development", seed=PROBE_SEED,
                                reference=teamwork_probe(TeamRuleAgent().scores, probe))
                atomic_json(out / f"teamwork-{iteration:06}.json", teamwork)
                print(f"对家意识：牌权翻转后改判让牌 {teamwork['flip_rate']:.1%}"
                      f"（团队规则参照 {teamwork['reference']['flip_rate']:.1%}）", flush=True)
                # A candidate is promoted only after beating the current
                # champion on paired deals with the interval above 50%.
                best = out / "champion.pt"
                if not best.exists():
                    _snapshot(best, model, iteration)
                    promoted = "initial_reference_only"
                else:
                    champion = ModelAgent(load_model(best)[0])
                    gate = run_evaluation(candidate, champion, cfg.eval_pairs, cfg.eval_seed+1)
                    gate.update(iteration=iteration, split="development", opponent="champion")
                    atomic_json(out / f"gate-{iteration:06}.json", gate)
                    # An interval gate freezes the champion: over 40 games
                    # ci95[0] only clears 0.5 at a ~72% win rate, so a candidate
                    # that is genuinely a few points better never promotes.
                    # Promote on the point estimate; eval_pairs sets its noise.
                    promoted = gate["win_rate"] > cfg.gate_threshold
                    if promoted:
                        _snapshot(best, model, iteration)
                print(f"开发集对 team-rule 胜率 {evaluation['win_rate']:.1%} · 晋级 {promoted}", flush=True)
    except BaseException:
        status = "failed"
        raise
    finally:
        if executor:
            for future in (pending[0] if pending else []):
                future.cancel()
            if status == "failed":
                # A failed/hung actor must not make final checkpointing hang.
                processes = list(executor._processes.values())
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                executor.shutdown(wait=False, cancel_futures=True)
                for process in processes:
                    process.join(timeout=2)
                    if process.is_alive():
                        process.kill()
            else:
                executor.shutdown(wait=True, cancel_futures=True)
        _save(out / "latest.pt", model, optimizer, replay, cfg, meta, rng, pool)
        export_npz(model, out / "latest.npz")
        atomic_json(out / "status.json", dict(status=status, device=device, **meta, loss=last_loss,
                                              elapsed_seconds=time.monotonic()-start,
                                              pid=os.getpid(), human_strength="unverified"))
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    print(f"已保存 {out / 'latest.pt'} ({status})", flush=True)
    return meta
