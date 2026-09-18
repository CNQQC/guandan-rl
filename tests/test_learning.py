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


def test_probe_states_are_fixed_and_are_real_snapshots():
    from fabledan.combos import PASS
    from guandan.diagnose import probe_states

    def signature(states):
        return [(s['player'], s['level'], len(s['hand']), len(s['events'])) for s in states]

    first, second = probe_states(count=24), probe_states(count=24)
    assert len(first) == 24 and signature(first) == signature(second)
    for obs in first:
        # Only states where deferring is a real choice qualify.
        assert obs['lead_owner'] not in (None, (obs['player'] + 2) % 4)
        assert any(m.type == PASS for m in obs['legal']) and any(m.type != PASS for m in obs['legal'])
        # events is the engine's live list: an uncopied reference would leave every
        # state holding the finished round, so the last play would not be the leader.
        last_play = next(event for event in reversed(obs['events']) if event[0] == 'play')
        assert last_play[1] == obs['lead_owner']


def test_teamwork_probe_separates_team_aware_from_indifferent():
    from guandan.diagnose import probe_states, teamwork_probe
    states = probe_states(count=24)
    aware = teamwork_probe(TeamRuleAgent().scores, states)
    # A policy that never reads lead ownership scores both variants identically.
    blind = teamwork_probe(lambda obs: np.arange(len(obs['legal']), dtype=np.float32), states)
    assert aware['states'] == 24 and 0 <= aware['flip_rate'] <= 1
    assert aware['flip_rate'] > .3 and aware['deference'] > 0
    assert blind['flip_rate'] == 0 and blind['deference'] == 0


def test_bucketed_batches_cut_padding_without_biasing_the_draw():
    """Bucketing must only change batch GROUPING, never a sample's draw odds."""
    from fabledan.train import Replay
    replay = Replay(512, belief_dim=45)
    lengths = [2 + (i * 37) % 480 for i in range(512)]
    for i, length in enumerate(lengths):
        replay.add(np.full(length, 3, dtype=np.int16), np.zeros(FEAT_DIM, dtype=np.float32),
                   0.0, np.zeros(45, dtype=np.float32))
    rng = np.random.default_rng(0)
    waste = {}
    for bucketed in (False, True):
        rng = np.random.default_rng(0)
        padded = used = 0
        for _ in range(400):
            tokens, L, _, _, _ = replay.sample(32, rng, bucketed)
            padded += tokens.size
            used += int(L.sum())
        waste[bucketed] = 1 - used / padded
    # Padding is what the bucket is for; the win is the ratio, not a constant.
    assert waste[False] > .4 and waste[True] < waste[False] / 3

    # Marginal uniformity: start is uniform over the buffer and the window
    # wraps past the end, so every entry sits in exactly `bs` of the n windows.
    # A non-wrapping window would starve the shortest and longest sequences.
    order = replay.length_order()
    counts = np.zeros(replay.n)
    for start in range(replay.n):
        counts[order[(start + np.arange(32)) % replay.n]] += 1
    assert counts.min() == counts.max() == 32


def test_pipelined_and_serial_training_agree_on_what_was_collected(tmp_path):
    shared = dict(iterations=3, games_per_iteration=4, workers=2, batch_size=4, updates=2,
                  replay_size=256, eval_every=999, snapshot_every=999, max_minutes=0,
                  threads=1, device='cpu')
    serial = train(TrainConfig(**shared, pipeline=False), tmp_path / 'serial')
    piped = train(TrainConfig(**shared, pipeline=True), tmp_path / 'piped')
    # Prefetched actors run on weights one iteration old, so the deals play out
    # differently and the sample COUNT moves; the bookkeeping must not.
    assert {k: v for k, v in serial.items() if k != 'samples'} \
        == {k: v for k, v in piped.items() if k != 'samples'} \
        == dict(iteration=3, games=12, updates=6)
    assert .8 < piped['samples'] / serial['samples'] < 1.25
    for name in ('serial', 'piped'):
        assert (tmp_path / name / 'latest.pt').is_file() and not (tmp_path / name / '.train.lock').exists()


def test_save_every_still_leaves_a_complete_final_checkpoint(tmp_path):
    cfg = TrainConfig(iterations=3, games_per_iteration=2, workers=1, batch_size=4, updates=1,
                      replay_size=256, eval_every=999, max_minutes=0, threads=1, device='cpu',
                      save_every=10)
    meta = train(cfg, tmp_path)
    _, ck = load_model(tmp_path / 'latest.pt')
    # save_every skipped every in-loop write; the exit path must still persist
    # the full replay and the real iteration count, or --resume would rewind.
    assert ck['meta']['iteration'] == meta['iteration'] == 3
    assert len(ck['replay']['tokens']) == min(meta['samples'], cfg.replay_size)


def test_pool_keeps_history_instead_of_only_the_newest_snapshots():
    from guandan.train import _pool_iteration, _update_pool
    cfg = TrainConfig(pool_size=8, pool_recent=3, pool_archive_every=100, snapshot_every=20)
    pool = ['pool/initial.pt']
    for iteration in range(20, 1001, 20):
        pool = _update_pool(pool, f'pool/iteration-{iteration:06}.pt', cfg)
    kept = [_pool_iteration(name) for name in pool]
    archive, recent = kept[:-cfg.pool_recent], kept[-cfg.pool_recent:]
    assert len(pool) == len(set(pool)) <= cfg.pool_size
    assert recent == [960, 980, 1000]
    # Sliding the window alone would leave the pool spanning 860..1000; the
    # archive has to reach far enough back that self-play cannot cycle.
    assert archive == [520, 620, 720, 820, 920]
    assert all(b - a >= cfg.pool_archive_every for a, b in zip(archive, archive[1:]))
    # Games against the random-init reference stop being informative.
    assert 'pool/initial.pt' not in pool


def test_pool_never_exceeds_its_cap_on_degenerate_settings():
    from guandan.train import _update_pool
    for recent, size in ((8, 8), (1, 1), (2, 3)):
        cfg = TrainConfig(pool_size=size, pool_recent=recent, pool_archive_every=50)
        pool = ['pool/initial.pt']
        for iteration in range(20, 601, 20):
            pool = _update_pool(pool, f'pool/iteration-{iteration:06}.pt', cfg)
        assert len(pool) == len(set(pool)) <= size


def test_config_rejects_inconsistent_pool_and_gate():
    with pytest.raises(ValueError):
        TrainConfig(pool_size=4, pool_recent=8).validate()
    with pytest.raises(ValueError):
        TrainConfig(gate_threshold=1.0).validate()
    with pytest.raises(ValueError):
        TrainConfig(pool_archive_every=0).validate()
