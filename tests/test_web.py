import os
import secrets
import shutil
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from guandan.runtime import ROOT, atomic_json
from guandan.web import create_app


@pytest.fixture(scope='module')
def client():
    with TestClient(create_app()) as client:
        yield client


@pytest.fixture
def scratch_run():
    """A disposable record, so management tests never touch real training output."""
    name = 'pytest-manage-' + secrets.token_hex(3)
    directory = ROOT / 'runs' / name
    (directory / 'pool').mkdir(parents=True)
    (directory / 'pool' / 'iteration-000010.pt').write_bytes(b'x' * 2048)
    atomic_json(directory / 'status.json', {'status': 'completed', 'iteration': 10})
    atomic_json(directory / 'config.json', {'training': {'eval_every': 10}})
    yield name, directory
    shutil.rmtree(directory, ignore_errors=True)


def test_web_game_roundtrip_and_stale_action(client):
    assert client.get('/').status_code == 200
    state = client.post('/api/game', json={'seed': 42, 'level': 1}).json()
    assert len(state['hand']) == 27 and 'hands' not in state
    assert state['remaining'] == [27]*4
    hint = client.post('/api/hint', json={'session':state['id'], 'index':0, 'version':0}).json()
    req = dict(session=state['id'], index=hint['index'], version=0)
    response = client.post('/api/play', json=req)
    assert response.status_code == 200 and response.json()['version'] == 1
    assert client.post('/api/play', json=req).status_code == 409
    assert client.post('/api/play', json={**req, 'version':1, 'index':999999}).status_code == 400


def test_local_game_can_finish(client):
    state = client.post('/api/game', json={'seed': 91, 'level': 5}).json()
    for _ in range(150):
        if state['result']:
            break
        req = dict(session=state['id'], version=state['version'], index=0)
        hint = client.post('/api/hint', json=req).json()
        state = client.post('/api/play', json={**req, 'index':hint['index']}).json()
    assert state['result'] is not None
    assert sorted(state['result']['ranking']) == [0, 1, 2, 3]


def test_arbitrary_checkpoint_path_rejected(client):
    response = client.post('/api/game', json={'agent':'../some-other-file.pt'})
    assert response.status_code == 400


def test_expert_static_prefix(client):
    from guandan.expert import available
    if not available():
        pytest.skip('public DanLM binaries require macOS ARM64')
    html = client.get('/expert/').text
    js = client.get('/expert/static/app.js').text
    assert '/expert/static/app.js' in html
    assert "'/api/" not in js and '/expert/api/' in js
    assert '`/api/' not in js
    assert client.get('/expert/api/agents').status_code == 200


def test_overview_exposes_run_detail(client):
    data = client.get('/api/overview').json()
    assert 'managed_run' in data and isinstance(data['runs'], list)
    for run in data['runs']:
        assert {'metrics', 'evaluations', 'training', 'snapshots', 'console',
                'size', 'archived', 'lock'} <= set(run)
        assert run['size'] >= 0 and (run['lock'] is None or 'alive' in run['lock'])
        # per-deal records stay on disk; the curve only needs the summary
        assert all('records' not in report for report in run['evaluations'])
    detailed = [run for run in data['runs'] if run['metrics'] or run['evaluations']]
    assert len(detailed) <= 1


def test_run_log_is_confined_to_the_runs_directory(client):
    assert client.get('/api/run/..%2F..%2Fguandan/log').status_code == 404
    assert client.get('/api/run/does-not-exist/log').status_code == 404


def test_training_options_and_preview(client):
    options = client.get('/api/train/options').json()
    assert 'league' in options['profiles'] and 'cpu' in options['devices']
    assert 'learning_rate' in options['frozen_on_resume']
    preview = client.post('/api/train/preview',
                          json={'profile': 'mac', 'overrides': {'workers': 2, 'games_per_iteration': 8}})
    assert preview.status_code == 200 and 'games_per_iteration = 8' in preview.json()['toml']
    bad = {'workers': 9, 'games_per_iteration': 4}
    assert client.post('/api/train/preview', json={'profile': 'mac', 'overrides': bad}).status_code == 400
    assert client.post('/api/train/preview',
                       json={'profile': 'mac', 'overrides': {'rule_opponents': ['nope']}}).status_code == 400
    assert client.post('/api/train/preview', json={'profile': '../configs/mac'}).status_code == 400
    assert client.post('/api/train/preview',
                       json={'profile': 'mac', 'overrides': {'unknown': 1}}).status_code == 422


def test_evaluation_rejects_bad_agents(client):
    assert client.post('/api/evaluate', json={'a': 'team-rule', 'b': 'team-rule'}).status_code == 400
    assert client.post('/api/evaluate', json={'a': '../guandan/web.py'}).status_code == 400
    assert client.post('/api/evaluate', json={'a': 'team-rule', 'b': 'random', 'pairs': 0}).status_code == 422
    assert client.get('/api/evaluate').json()['state'] in {'idle', 'running', 'completed', 'failed'}


def test_checkpoints_stay_inside_their_run(client):
    for run in client.get('/api/overview').json()['runs']:
        for node in run['checkpoints']:
            assert node['path'].startswith('runs/' + run['name'] + '/') and node['path'].endswith('.pt')


def test_run_label_roundtrip(client):
    name = client.get('/api/overview').json()['runs'][0]['name']
    assert client.put(f'/api/run/{name}/label', json={'label': ' 夜间长训 '}).json()['label'] == '夜间长训'
    labels = {r['name']: r['label'] for r in client.get('/api/overview').json()['runs']}
    assert labels[name] == '夜间长训'
    assert client.put(f'/api/run/{name}/label', json={'label': ''}).json()['label'] is None
    assert client.put(f'/api/run/{name}/label', json={'label': 'x'*80}).status_code == 422


def test_prune_keeps_the_best_and_the_live_pool(client):
    for run in client.get('/api/overview').json()['runs']:
        plan = client.get(f"/api/run/{run['name']}/prune").json()
        kept = {item['name'] for item in plan['kept']}
        removed = {item['name'] for item in plan['removed']}
        assert not (kept & removed)
        if plan['best'] is not None:
            best = 'iteration-%06d.pt' % plan['best']['iteration']
            assert best not in removed
        assert plan['freed'] <= plan['total']


def entry(client, name):
    return next(run for run in client.get('/api/overview').json()['runs'] if run['name'] == name)


def test_archiving_sorts_a_record_away_without_touching_disk(client, scratch_run):
    name, directory = scratch_run
    assert entry(client, name)['archived'] is False
    assert entry(client, name)['size'] >= 2048
    assert client.put(f'/api/run/{name}/archive', json={'archived': True}).json()['archived'] is True
    runs = [run['name'] for run in client.get('/api/overview').json()['runs']]
    active = [run['name'] for run in client.get('/api/overview').json()['runs'] if not run['archived']]
    assert runs.index(name) >= len(active) and directory.is_dir()
    # Renaming an archived record must not quietly un-archive it.
    client.put(f'/api/run/{name}/label', json={'label': '归档实验'})
    assert entry(client, name)['label'] == '归档实验'
    assert entry(client, name)['archived'] is True
    assert client.put(f'/api/run/{name}/archive', json={'archived': False}).json()['archived'] is False


def test_a_live_lock_blocks_deletion_and_a_stale_one_can_be_cleared(client, scratch_run):
    name, directory = scratch_run
    lock = directory / '.train.lock'
    lock.write_text(str(os.getpid()))
    assert entry(client, name)['lock'] == {'pid': os.getpid(), 'alive': True}
    assert client.delete(f'/api/run/{name}/lock').status_code == 409
    assert client.delete(f'/api/run/{name}').status_code == 409
    finished = subprocess.Popen([sys.executable, '-c', 'pass'])
    finished.wait()
    lock.write_text(str(finished.pid))
    assert entry(client, name)['lock']['alive'] is False
    assert client.delete(f'/api/run/{name}/lock').json()['removed'] is True
    assert not lock.exists() and entry(client, name)['lock'] is None
    assert client.delete(f'/api/run/{name}/lock').status_code == 404


def test_delete_removes_the_record_and_only_that_record(client, scratch_run):
    name, directory = scratch_run
    freed = client.delete(f'/api/run/{name}').json()['freed']
    assert freed >= 2048 and not directory.exists()
    assert name not in [run['name'] for run in client.get('/api/overview').json()['runs']]
    assert client.delete(f'/api/run/{name}').status_code == 404


def test_delete_never_escapes_a_single_run_directory(client, scratch_run):
    name, _ = scratch_run
    assert client.delete('/api/run/..%2F..%2Fguandan').status_code == 404
    assert client.delete(f'/api/run/{name}%2Fpool').status_code == 404
    assert client.delete('/api/run/.').status_code == 404
    assert (ROOT / 'runs' / name).is_dir()


def test_export_is_confined_to_checkpoints_of_the_named_run(client, scratch_run):
    name, directory = scratch_run
    bad = [f'runs/{name}/status.json', 'guandan/web.py', f'runs/{name}/pool/../../../pyproject.toml']
    for path in bad:
        assert client.post(f'/api/run/{name}/export', json={'path': path}).status_code == 400
    # A real file of the run that is not a loadable checkpoint fails as a request error.
    assert client.post(f'/api/run/{name}/export',
                       json={'path': f'runs/{name}/pool/iteration-000010.pt'}).status_code == 400


def test_export_writes_portable_weights(client):
    found = [(run['name'], node['path']) for run in client.get('/api/overview').json()['runs']
             for node in run['checkpoints'] if node['kind'] == 'latest']
    if not found:
        pytest.skip('no trained checkpoint on this machine')
    name, path = found[0]
    result = client.post(f'/api/run/{name}/export', json={'path': path})
    assert result.status_code == 200, result.text
    target = ROOT / result.json()['path']
    try:
        import numpy as np
        with np.load(target) as weights:
            assert '__config__' in weights and not any(k.startswith('belief_head') for k in weights)
    finally:
        target.unlink(missing_ok=True)


def test_overview_carries_the_teamwork_diagnostic(client):
    runs = client.get('/api/overview').json()['runs']
    assert runs, 'no run directory on this machine'
    detailed = [run for run in runs if run['metrics'] or run['evaluations'] or run['teamwork']]
    assert all('teamwork' in run for run in runs)
    for run in detailed:
        for point in run['teamwork']:
            assert 0 <= point['flip_rate'] <= 1 and point['states'] > 0
            assert 0 <= point['reference']['flip_rate'] <= 1


def test_node_delete_removes_one_checkpoint_and_nothing_else(client, scratch_run):
    name, directory = scratch_run
    (directory / 'latest.pt').write_bytes(b'x' * 4096)
    nodes = {node['kind']: node['path'] for node in entry(client, name)['checkpoints']}
    assert {'latest', 'pool'} <= set(nodes)
    outside = [f'runs/{name}/status.json', 'guandan/web.py', f'runs/{name}/pool/../../../pyproject.toml']
    for path in outside:
        assert client.delete(f'/api/run/{name}/node', params={'path': path}).status_code == 400
    # A live lock protects a single file exactly as it protects the whole record.
    (directory / '.train.lock').write_text(str(os.getpid()))
    assert client.delete(f'/api/run/{name}/node', params={'path': nodes['pool']}).status_code == 409
    (directory / '.train.lock').unlink()
    removed = client.delete(f'/api/run/{name}/node', params={'path': nodes['pool']}).json()
    assert removed['freed'] == 2048 and removed['remaining'] == 1
    assert not (directory / 'pool/iteration-000010.pt').exists()
    assert (directory / 'latest.pt').is_file() and (directory / 'status.json').is_file()
    assert client.delete(f'/api/run/{name}/node', params={'path': nodes['pool']}).status_code == 400


def test_resume_preview_pins_the_settings_the_checkpoint_was_made_with(client, scratch_run):
    name, directory = scratch_run
    atomic_json(directory / 'config.json', {'training': dict(
        learning_rate=0.0003, seed=99, replay_size=4096, model_size='small',
        workers=1, games_per_iteration=8, iterations=50)})
    moved = {'learning_rate': 0.01, 'seed': 1, 'replay_size': 65536,
             'workers': 3, 'games_per_iteration': 6, 'iterations': 20}
    resumed = client.post('/api/train/preview', json={'profile': 'league', 'resume': name,
                                                      'overrides': moved})
    assert resumed.status_code == 200, resumed.text
    config = resumed.json()['training']
    # The nine frozen fields define the experiment; the console may not move them.
    assert (config['learning_rate'], config['seed'], config['replay_size']) == (0.0003, 99, 4096)
    # Budget and parallelism are the caller's to change on every restart.
    assert (config['workers'], config['games_per_iteration'], config['iterations']) == (3, 6, 20)


def test_resume_needs_a_record_that_saved_its_configuration(client, scratch_run):
    name, directory = scratch_run
    (directory / 'config.json').unlink()
    refused = client.post('/api/train/preview', json={'profile': 'league', 'resume': name})
    assert refused.status_code == 400 and '没有保存配置' in refused.json()['detail']
    assert client.post('/api/train/preview',
                       json={'profile': 'league', 'resume': 'does-not-exist'}).status_code == 404
