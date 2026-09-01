import numpy as np
import pytest
import torch
from src.training.fusion_vector_logger import FusionVectorLogger, reduce_feature_to_vector


def config(enabled=True):
    return {"model":{"architecture":"cawfe_latte"}, "experiment":{"name":"test"}, "training":{"save_fusion_vectors":{"enabled":enabled,"output_name":"vectors.npy","vector_reduce":"mean_bt_hw","save_metadata":True}}}

class FakeModel(torch.nn.Module):
    def __init__(self, feature): super().__init__(); self.feature=feature
    def forward(self, x, return_features=False, terrain=None): return {"fused_dynamic":self.feature.to(x.device)}


def batch(): return torch.zeros(2,1), torch.zeros(2,1)

def test_disabled_does_not_create_file(tmp_path):
    logger=FusionVectorLogger(config(False),tmp_path)
    assert logger.collect_epoch_vector(FakeModel(torch.ones(1,1,3,2,2)),batch(),torch.device("cpu"),1) is None
    assert logger.save() is None
    assert not (tmp_path/"features"/"vectors.npy").exists()

@pytest.mark.parametrize("feature, expected", [
    (torch.ones(2,3,5,2,2), torch.ones(5)),
    (torch.ones(2,3,4,5), torch.ones(5)),
    (torch.ones(2,5,3,4), torch.ones(5)),
])
def test_reduces_supported_layouts(feature, expected):
    torch.testing.assert_close(reduce_feature_to_vector(feature), expected)

def test_saves_one_vector_per_collection_and_restores_mode(tmp_path):
    model=FakeModel(torch.ones(2,3,4,2,2)); model.train()
    logger=FusionVectorLogger(config(),tmp_path)
    for epoch in range(3): logger.collect_epoch_vector(model,batch(),torch.device("cpu"),epoch)
    path=logger.save()
    assert model.training
    assert np.load(path).shape == (3,4)
    assert logger.metadata_path.exists()

def test_missing_feature_has_clear_error(tmp_path):
    class NoFeature(torch.nn.Module):
        def forward(self, x, return_features=False): return {"prediction":x}
    with pytest.raises(ValueError,match="no fusion feature was found"):
        FusionVectorLogger(config(),tmp_path).collect_epoch_vector(NoFeature(),batch(),torch.device("cpu"),1)
