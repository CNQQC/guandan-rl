import pytest
from fastapi.testclient import TestClient

from guandan.web import create_app


@pytest.fixture(scope='module')
def client():
    with TestClient(create_app()) as client:
        yield client


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
