import yaml
from src.config import load_config

def test_catalogue_keeps_one_active_architecture():
    definitions=yaml.safe_load(open('configs/ablations/cawfe_latte_ablations.yaml'))['ablations']
    baseline=load_config('configs/experiments/cawfe_latte_baseline.yaml')
    assert baseline['model']['architecture']=='cawfe_latte'
    for definition in definitions.values():
        assert definition['overrides'].get('model.architecture','cawfe_latte')=='cawfe_latte'

def test_single_change_ablations_are_explicit():
    definitions=yaml.safe_load(open('configs/ablations/cawfe_latte_ablations.yaml'))['ablations']
    assert definitions['A_resblocks_only']['changed_from_baseline']==['post_fusion_backbone']
    assert definitions['B1_softplus_only']['changed_from_baseline']==['regression_activation']
    assert definitions['C_temporal_attention_only']['changed_from_baseline']==['temporal_pooling']
