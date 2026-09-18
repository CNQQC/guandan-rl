import copy
import random

import numpy as np
import pytest
import torch

from fabledan.encode import ENCODING_VERSION, FEAT_DIM, encode_decision
from fabledan.engine import GuandanRound
from fabledan.model_torch import FableDanNet, export_npz
from guandan.agents import ModelAgent, TeamRuleAgent, load_model
from guandan.evaluate import cluster_interval, run_evaluation
from guandan.train import TrainConfig, network_config, train


def test_observation_contains_no_other_hands():
    rnd = GuandanRound(1, random.Random(7))
    obs = next(rnd.play_steps())
    before = encode_decision(obs)
    rnd.hands[1], rnd.hands[3] = rnd.hands[3], rnd.hands[1]
    after = encode_decision(obs)
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert 'hands' not in obs and 'belief' not in obs
    assert before[1].shape[1] == FEAT_DIM


def test_actual_suit_choices_produce_different_features():
    rnd = GuandanRound(1, deal=[[20, 21, 22, 4], [53], [52], [107]])
    obs = next(rnd.play_steps())
    _, feats = encode_decision(obs)
    singles = [i for i, m in enumerate(obs['legal']) if m.size == 1 and m.cards[0] in [20, 21]]
    assert len(singles) == 2 and not np.array_equal(feats[singles[0]], feats[singles[1]])


def test_torch_numpy_export_parity(tmp_path):
    from fabledan.model_np import NumpyModel
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = FableDanNet(network_config('small')).eval()
    path = tmp_path / 'model.npz'
    export_npz(model, path)
    agent = ModelAgent(model)
    obs = next(GuandanRound(1, random.Random(3)).play_steps())
    toks, feats = encode_decision(obs)
    expected = agent.scores(obs)
    actual = NumpyModel(path).q_values(toks, feats)
    np.testing.assert_allclose(expected, actual, rtol=2e-4, atol=2e-5)


def test_evaluation_same_policy_is_exactly_balanced():
    result = run_evaluation(TeamRuleAgent(), TeamRuleAgent(), pairs=6, seed=55)
    assert result['win_rate'] == .5 and result['average_reward'] == 0
    assert result['games'] == 12 and len(result['records']) == 12
    for i in range(0, 12, 2):
        a, b = result['records'][i:i+2]
        assert a['deal_seed'] == b['deal_seed'] and a['reward'] == -b['reward']


def test_intervals_do_not_fake_certainty_after_few_wins():
    lo, hi = cluster_interval([1, 1])
    assert 0 < lo < .5 and hi == 1
    assert cluster_interval([.5]) == [0, 1]


def test_train_resume_restores_optimizer_replay_and_progress(tmp_path):
    cfg = TrainConfig(iterations=1, games_per_iteration=2, batch_size=4, updates=1,
                      replay_size=256, eval_every=999, max_minutes=0, threads=1, device='cpu')
    first = train(cfg, tmp_path)
    _, ck = load_model(tmp_path/'latest.pt')
    weights = copy.deepcopy(ck['model'])
    second = train(cfg, tmp_path, tmp_path/'latest.pt')
    model, ck2 = load_model(tmp_path/'latest.pt')
    assert second['iteration'] == 2 and second['games'] == first['games'] + 2
    assert ck2['encoding'] == ENCODING_VERSION and ck2['replay']['tokens']
    assert ck2['optimizer']['state'] and (tmp_path/'latest.npz').is_file()
    assert any(not torch.equal(weights[k], ck2['model'][k]) for k in weights)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert not (tmp_path/'.train.lock').exists()
    with pytest.raises(ValueError, match='已有模型'):
        train(cfg, tmp_path)


def test_config_rejects_bad_probabilities():
    with pytest.raises(ValueError):
        TrainConfig(epsilon=1.5).validate()
    with pytest.raises(ValueError):
        TrainConfig(workers=5, games_per_iteration=2).validate()
