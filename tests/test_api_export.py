import numpy as np
from fastapi.testclient import TestClient
from web.backend.app import create_app
from seizure_v2.inference.bundle import export_onnx
from seizure_v2.models import make_model
from seizure_v2.inference.predictor import Predictor
from seizure_v2.common import write_json
import pytest


def test_missing_artifacts_are_honest_and_site_still_loads(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get('/').status_code == 200
        assert 'No simulated research results' in client.get('/').text
        assert client.get('/healthz').status_code == 503
        research = client.get('/api/research').json()
        assert research['status'] == 'awaiting_artifacts' and research['models'] == {}
        assert client.get('/api/examples').status_code == 503
        assert client.post('/api/predict', json={'example_id':'x','model_id':'baseline'}).status_code == 503


@pytest.mark.parametrize('name', ['baseline','residual'])
def test_dynamic_batch_onnx_score_parity(tmp_path, name):
    import torch
    torch.set_num_threads(2)
    x = np.random.default_rng(42).normal(size=(3,18,2048)).astype(np.float32)
    parity = export_onnx(make_model(name), tmp_path / 'model.onnx', x)
    assert parity['maximum_score_error'] <= 1e-5


def test_development_artifacts_rejected(tmp_path):
    write_json(tmp_path / 'manifest.json', {'development': True})
    with pytest.raises(ValueError, match='Development'):
        Predictor(tmp_path)


def test_window_order_threshold_and_regions_are_consistent():
    predictor = object.__new__(Predictor)
    signal = np.tile(np.arange(2048 * 3, dtype=np.float32), (18,1))
    predictor.clip = lambda _: ({'clip_start_seconds': 80}, signal, np.array([0,1,1]))
    def infer(model, windows):
        assert windows.shape == (3,18,2048)
        np.testing.assert_array_equal(windows[1,0], np.arange(2048,4096))
        scores=np.array([.2,.6,.8])
        return scores,(scores>=.6).astype(int),.6
    predictor.predict = infer
    result=predictor.predict_example('fixture','baseline')
    assert result['high_score_regions']==[[88,104]]
    assert [w['true_label'] for w in result['windows']]==[0,1,1]
    assert result['windows'][1]['predicted_label']==1


def test_ready_api_passes_model_and_example_and_bounds_waveform(tmp_path, monkeypatch):
    import web.backend.app as server
    class Fake:
        def __init__(self, root):
            self.manifest={'study_sha256':'fixture','models':{'baseline':{'threshold':.6}},'default_model':'baseline','attribution':'Explicit test fixture'}
            self.examples={'fixture':{'id':'fixture'}}
            self.sessions={'baseline':None}
        def clip(self, example_id):
            return {'focus_start_seconds':8,'clip_start_seconds':0},np.zeros((18,8192)),np.zeros(4)
        def predict_example(self, example_id, model_id):
            return {'example_id':example_id,'model_id':model_id,'threshold':.6}
    write_json(tmp_path/'manifest.json',{})
    monkeypatch.setattr(server,'Predictor',Fake)
    with TestClient(server.create_app(tmp_path)) as client:
        assert client.get('/healthz').status_code==200
        assert client.get('/api/examples/absent').status_code==404
        assert len(client.get('/api/examples/fixture').json()['signal_uv'][0])==4096
        assert client.get('/api/examples/fixture?start_seconds=nan').status_code==422
        response=client.post('/api/predict',json={'example_id':'fixture','model_id':'baseline'}).json()
        assert response['threshold']==.6 and response['model_id']=='baseline'
