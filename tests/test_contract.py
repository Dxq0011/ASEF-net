from pathlib import Path
from asef.utils import load_config,make_model


def test_manuscript_contract():
    cfg=load_config(str(Path(__file__).parent.parent/'configs/lol_v1.yaml'))
    assert cfg['model']['channels']==512
    assert cfg['model']['stages']==10
    assert cfg['model']['rk4_steps']==10
    assert cfg['training']['batch_size']==4
    assert cfg['training']['epochs']==300
    assert cfg['loss']['delta']==2.0
    assert cfg['loss']['tau']==.3
    assert cfg['model']['epsilon']==.1
    assert cfg['model']['eta']==1e-6
    assert cfg['dataset']['directory_names']['test']=='eval15'
