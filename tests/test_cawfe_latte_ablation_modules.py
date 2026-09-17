import pytest
pytest.importorskip('torch')
import torch
from src.models.cawfe_latte import build_support_gate, build_temporal_pooling

def test_attention_weights_sum_over_time():
    pool=build_temporal_pooling({'type':'attention','hidden_dim':4,'initialize_uniform':True},dim=8,input_sequence_length=3)
    pool(torch.randn(2,3,8,4,4))
    assert torch.allclose(pool.last_attention_weights.sum(1),torch.ones(2,1,4,4))

def test_support_gate_range():
    gate=build_support_gate({'enabled':True,'gate_min':.05},dim=8)
    _, values=gate(torch.randn(2,8,4,4))
    assert values.min() >= .05 and values.max() <= 1


def test_forward_variants_keep_prediction_contract():
    from src.config import load_config
    from src.models.model_factory import build_model_from_config
    import yaml
    definitions=yaml.safe_load(open('configs/ablations/cawfe_latte_ablations.yaml'))['ablations']
    for name in ('baseline','A_resblocks','B_multiscale_context','C_temporal_attention'):
        config=load_config(definitions[name]['config_path'])
        model=build_model_from_config(config,129).eval()
        with torch.no_grad(): output=model(torch.randn(1,5,129,16,16),terrain=torch.randn(1,4,16,16))
        prediction=output['prediction'] if isinstance(output,dict) else output
        assert prediction.shape == (1,4,16,16)
