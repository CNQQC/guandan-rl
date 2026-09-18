import random
import subprocess
import sys

import pytest

from fabledan.engine import play_round
from guandan.agents import TeamRuleAgent
from guandan.programs import ProgramAgent, action_message
from guandan.train import _collect, network_config


@pytest.mark.parametrize('name', ['njupt', 'egg-pancake'])
def test_competition_program_full_rounds_and_reproducibility(name):
    def run():
        bot = ProgramAgent(name, 17)
        results = []
        for seed in range(13):
            rewards, ranking, rnd = play_round([bot, TeamRuleAgent(), bot, TeamRuleAgent()],
                                              random.Random(seed), level=seed)
            assert sum(rewards) == 0 and sorted(ranking) == [0, 1, 2, 3]
            results.append((rewards, ranking, [(ev[1], ev[2].cards) for ev in rnd.events if ev[0]=='play']))
        assert bot.decisions > 0
        return results
    assert run() == run()


def test_program_adapter_imports_without_torch_or_danlm_binary():
    code = "from guandan.programs import ProgramAgent; a=ProgramAgent('njupt'); import sys; assert 'torch' not in sys.modules; assert not any(n.startswith('danzero') for n in sys.modules)"
    subprocess.run([sys.executable, '-c', code], check=True, timeout=15)


def test_program_message_uses_high_end_of_sequence():
    from fabledan.combos import STRAIGHT, Move
    move = Move(STRAIGHT, 2, [4, 9, 14, 19, 20], [1, 2, 3, 4, 5])
    assert action_message(move)[:2] == ['Straight', '6']


def test_rl_collects_only_learning_team_actions_against_programs():
    import torch

    from fabledan.model_torch import FableDanNet
    torch.manual_seed(7)
    model = FableDanNet(network_config('small'))
    state = {k:v.detach().numpy().copy() for k,v in model.state_dict().items()}
    samples, games, modes = _collect((model.cfg.to_dict(), state, 2, 42, .2, 1.0, 'program:njupt'))
    assert games == 2 and modes['program:njupt'] == 2 and modes['self'] == 0
    assert samples and all(-1 <= s[2] <= 1 for s in samples)
