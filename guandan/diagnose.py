"""对家意识诊断：同一个局面只翻转牌权归属，看策略会不会改为让牌。

局面集合由固定种子下的 TeamRuleAgent 对局生成，每次诊断面对同一批局面，
所以指标的变化只能来自策略本身，不来自状态分布漂移。除牌权归属外，手牌、
出牌历史与合法动作完全不变，因此测到的是「谁坐庄」这一条信息的因果影响。
"""
from __future__ import annotations

import random

import numpy as np

from fabledan.combos import PASS
from fabledan.engine import GuandanRound, random_tribute_mode

from .agents import TeamRuleAgent

PROBE_SEED = 20_260_918
PROBE_STATES = 160
PROBE_ROUNDS = 24


def _usable(obs):
    """敌方坐庄，且出牌与过牌都合法——只有这种局面才谈得上让不让。"""
    owner = obs["lead_owner"]
    return (owner is not None and owner != (obs["player"] + 2) % 4
            and any(m.type == PASS for m in obs["legal"])
            and any(m.type != PASS for m in obs["legal"]))


def probe_states(count=PROBE_STATES, seed=PROBE_SEED, rounds=PROBE_ROUNDS):
    """固定种子下采集的决策局面。

    先打满 `rounds` 局再等距抽取，局面因此覆盖开局到残局；只取前 count 个
    会把样本全压在最初几局。events 是引擎的活动列表，必须复制后留存。
    """
    collected = []
    driver = TeamRuleAgent()

    class _Collect:
        def act(self, obs):
            if _usable(obs):
                collected.append(dict(obs, events=list(obs["events"])))
            return driver.act(obs)

    rng = random.Random(seed)
    for _ in range(rounds):
        deal = random.Random(rng.getrandbits(48))
        GuandanRound(deal.randrange(13), deal, random_tribute_mode(deal)).play([_Collect()] * 4)
    if not collected:
        raise RuntimeError("采集不到可用于对家意识诊断的局面")
    if len(collected) <= count:
        return collected
    stride = len(collected) / count
    return [collected[int(i * stride)] for i in range(count)]


def _pass_gap(scores, pass_index):
    """Q(过) − max Q(出牌)：越大越倾向让牌。"""
    best = max(score for i, score in enumerate(scores) if i != pass_index)
    return float(scores[pass_index] - best)


def teamwork_probe(scores_of, states):
    """牌权由敌方翻转为对家后的让牌倾向变化。

    flip_rate 与 Q 值尺度无关，可以跨迭代、跨模型比较；deference 是原始 Q
    位移，要配合 q_margin（该策略自身的典型抉择差距）才有意义。
    """
    shifts, margins, flips = [], [], 0
    for obs in states:
        legal = obs["legal"]
        pass_index = next(i for i, m in enumerate(legal) if m.type == PASS)
        before = scores_of(obs)
        after = scores_of(dict(obs, lead_owner=(obs["player"] + 2) % 4))
        shifts.append(_pass_gap(after, pass_index) - _pass_gap(before, pass_index))
        margins.append(abs(_pass_gap(before, pass_index)))
        flips += int(int(np.argmax(before)) != pass_index and int(np.argmax(after)) == pass_index)
    return dict(states=len(states), flip_rate=flips / len(states),
                deference=float(np.mean(shifts)), q_margin=float(np.mean(margins)))
