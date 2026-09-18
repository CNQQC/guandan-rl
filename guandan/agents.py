from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fabledan.agents import RandomAgent, RuleAgent
from fabledan.cards import is_wildcard, rank_of
from fabledan.combos import PASS
from fabledan.encode import ENCODING_VERSION, encode_decision
from fabledan.model_torch import FableDanNet, ModelConfig


class TeamRuleAgent(RuleAgent):
    """A reproducible, team-aware heuristic. It is NOT a human-strength proxy."""

    def scores(self, obs):
        me = obs["player"]
        teammate = (me + 2) % 4
        enemies = [(me + 1) % 4, (me + 3) % 4]
        urgent = min((obs["left"][p] for p in enemies if obs["left"][p]), default=27) <= 2
        partner_leads = obs["lead_owner"] == teammate
        counts = np.bincount([rank_of(c) for c in obs["hand"]], minlength=15)
        values = []
        for m in obs["legal"]:
            if m.type == PASS:
                values.append(8.0 if partner_leads and not urgent else -4.0)
                continue
            if m.size == len(obs["hand"]):
                values.append(100.0)
                continue
            score = 1.8 * m.size - 0.12 * m.key
            score -= (6.0 if not urgent else 1.5) * m.is_bombish()
            score -= 1.1 * sum(is_wildcard(c, obs["level"]) for c in m.cards)
            played = np.bincount([rank_of(c) for c in m.cards], minlength=15)
            score -= sum(2.0 for r in range(15) if counts[r] >= 4 and 0 < played[r] < counts[r])
            if partner_leads and not urgent:
                score -= 10
            values.append(score)
        return np.asarray(values, dtype=np.float32)

    def act(self, obs):
        return int(np.argmax(self.scores(obs)))


class ModelAgent:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device).eval()
        self.device = device

    def scores(self, obs):
        tokens, features = encode_decision(obs)
        with torch.inference_mode():
            t = torch.tensor([tokens], dtype=torch.long, device=self.device)
            lengths = torch.tensor([len(tokens)], device=self.device)
            ctx, _ = self.model.encode_seq(t, lengths)
            # Do not allocate an enormous candidate tensor for rare big hands.
            scores = []
            for start in range(0, len(features), 256):
                f = torch.as_tensor(features[start:start + 256], device=self.device).unsqueeze(0)
                scores.append(self.model.q_values(ctx, f)[0].cpu().numpy())
        return np.concatenate(scores)

    def act(self, obs):
        return int(np.argmax(self.scores(obs)))


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("encoding") != ENCODING_VERSION:
        raise ValueError("模型编码版本不匹配；DanLM/FableDan 原始权重不能当作本地训练权重加载。")
    model = FableDanNet(ModelConfig.from_dict(ck["config"]))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def make_agent(spec, seed=0, device="cpu"):
    if spec.startswith("program:"):
        from .programs import ProgramAgent
        return ProgramAgent(spec.split(":", 1)[1], seed)
    if spec == "random":
        return RandomAgent(seed)
    if spec == "rule":
        return RuleAgent()
    if spec == "team-rule":
        return TeamRuleAgent()
    model, _ = load_model(Path(spec), device)
    return ModelAgent(model, device)
