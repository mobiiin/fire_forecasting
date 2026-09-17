import yaml
from src.config import load_config

def test_catalogue_keeps_one_active_architecture():
    definitions=yaml.safe_load(open('configs/ablations/cawfe_latte_ablations.yaml'))['ablations']
    for definition in definitions.values():
        config=load_config(definition['config_path'])
        assert config['model']['architecture']=='cawfe_latte'

def test_single_change_ablations_are_explicit():
    definitions=yaml.safe_load(open('configs/ablations/cawfe_latte_ablations.yaml'))['ablations']
    assert definitions['A_resblocks']['changed_component']=='post_fusion_backbone'
    assert definitions['B_multiscale_context']['changed_component']=='post_fusion_backbone'
    assert definitions['C_temporal_attention']['changed_component']=='temporal_pooling'
