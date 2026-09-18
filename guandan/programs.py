"""Pure-Python competition strategies in the LOCAL rules engine.

No torch, checkpoint, DanLM binary or oracle hand information is needed here.
Upstream imports are isolated per agent; source and author headers stay intact.
"""
from __future__ import annotations

import builtins
import hashlib
import random
import types
from pathlib import Path

from fabledan.cards import SEQV_TO_RANK, rank_of, suit_of
from fabledan.combos import FULL, PASS, PLATE, ROCKET, SFLUSH, STRAIGHT, TUBE

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = {
    "njupt": {"directory": "fin-njupt-guandan-ai", "name": "南邮掼蛋 AI", "method": "parse"},
    "egg-pancake": {"directory": "2nd-egg-pancake", "name": "蛋饼 / NUAA", "method": "parse_AI"},
}
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "B", "R"]
TYPES = ["PASS", "Single", "Pair", "Trips", "ThreeWithTwo", "Straight", "ThreePair", "TwoTrips", "Bomb", "StraightFlush", "Bomb"]


def source_fingerprint(name):
    directory = ROOT / 'vendor/DanLM/baselines' / REGISTRY[name]['directory']
    digest = hashlib.sha256()
    for path in sorted(directory.rglob('*.py')):
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(path.read_bytes())
    # Include the protocol adapter and its documented counter correction.
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def card_string(c):
    rank = rank_of(c)
    if rank >= 13:
        return "SB" if rank == 13 else "HR"
    return ["H", "D", "S", "C"][suit_of(c)] + RANKS[rank]


def action_message(move):
    if move.type == PASS:
        return ["PASS", "PASS", "PASS"]
    if move.type in (STRAIGHT, SFLUSH, PLATE, TUBE):
        length = {STRAIGHT: 5, SFLUSH: 5, PLATE: 3, TUBE: 2}[move.type]
        rank = SEQV_TO_RANK[move.key + length - 1]
    elif move.type == ROCKET:
        # Match the archived competition protocol's joker-bomb convention.
        rank = 1
    elif move.type == FULL:
        from collections import Counter
        rank = next(r for r, n in Counter(move.claim_ranks).items() if n == 3)
    else:
        rank = move.claim_ranks[0]
    return [TYPES[move.type], RANKS[rank], [card_string(c) for c in move.cards]]


class SourceLoader:
    """Small local module loader for archives using absolute sibling imports.

    It doesn't mutate sys.path, sys.modules, process cwd, global stdout or the
    global random generator, so two differently named bot archives can coexist.
    This is import isolation, not a security sandbox for untrusted code.
    """
    def __init__(self, root, seed):
        self.root = root
        self.cache = {}
        rng = random.Random(seed)
        self.random_module = types.ModuleType("random")
        self.random_module.__dict__.update(vars(random))
        for name in vars(random):
            if hasattr(rng, name):
                setattr(self.random_module, name, getattr(rng, name))

    def path_for(self, name):
        path = self.root.joinpath(*name.split('.'))
        return path / '__init__.py' if path.is_dir() else path.with_suffix('.py')

    def import_hook(self, name, globals=None, locals=None, fromlist=(), level=0):
        if name == "random" and level == 0:
            return self.random_module
        if level == 0 and self.path_for(name.split('.')[0]).is_file():
            mod = self.load(name)
            if fromlist:
                for item in fromlist:
                    child = name + '.' + item
                    if item != '*' and not hasattr(mod, item) and self.path_for(child).is_file():
                        setattr(mod, item, self.load(child))
                return mod
            return self.load(name.split('.')[0])
        return builtins.__import__(name, globals, locals, fromlist, level)

    def load(self, name):
        if name in self.cache:
            return self.cache[name]
        path = self.path_for(name)
        if not path.is_file():
            raise ImportError(f"missing archived bot module: {path}")
        mod = types.ModuleType('guandan_archived.' + name)
        mod.__file__ = str(path)
        mod.__package__ = name.rpartition('.')[0]
        if path.name == '__init__.py':
            mod.__path__ = [str(path.parent)]
        local_builtins = dict(vars(builtins))
        local_builtins['__import__'] = self.import_hook
        local_builtins['print'] = lambda *a, **kw: None
        mod.__dict__['__builtins__'] = local_builtins
        self.cache[name] = mod
        exec(compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec'), mod.__dict__)
        if '.' in name:
            parent, child = name.rsplit('.', 1)
            setattr(self.load(parent), child, mod)
        return mod


class ProgramAgent:
    def __init__(self, name, seed=0):
        if name not in REGISTRY:
            raise ValueError(f"unknown program strategy {name}; choices: {list(REGISTRY)}")
        self.name, self.seed = name, seed
        self.source = ROOT / 'vendor/DanLM/baselines' / REGISTRY[name]['directory']
        self.loader = SourceLoader(self.source, seed)
        self.action_class = self.loader.load('action').Action
        self.actions = {}
        self.events_ref = None
        self.decisions = 0

    def act(self, obs):
        # Source objects are seat-specific and reset at the start of every deal.
        if self.events_ref is not obs['events']:
            self.events_ref = obs['events']
            self.actions = {}
        p = obs['player']
        if p not in self.actions:
            self.actions[p] = self.action_class(p) if self.name == 'njupt' else self.action_class()
        actor = self.actions[p]
        actions = [action_message(m) for m in obs['legal']]
        msg = dict(type='act', stage='play', myPos=p, curRank=RANKS[obs['level']],
                   selfRank=RANKS[obs['level']], oppoRank=RANKS[obs['level']],
                   handCards=[card_string(c) for c in obs['hand']], actionList=actions,
                   indexRange=len(actions)-1,
                   greaterPos=-1 if obs['lead_owner'] is None else obs['lead_owner'],
                   greaterAction=None if obs['lead'] is None else action_message(obs['lead']),
                   curPos=-1 if obs['lead_owner'] is None else obs['lead_owner'],
                   publicInfo=[dict(rest=count, pos=i) for i, count in enumerate(obs['left'])])
        if self.name == 'njupt':
            # Upstream accumulates all_match across decisions instead of
            # recounting the current hand. Resetting it fixes that counter.
            actor.all_match = 0
            index = actor.parse(msg, p, list(obs['left']))
        else:
            index = actor.parse_AI(msg, p)
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(actions):
            raise ValueError(f"program:{self.name} returned illegal action {index}")
        self.decisions += 1
        return index
