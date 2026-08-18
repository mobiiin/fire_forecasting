import json
import numpy as np
import pytest
from src.data.processed_dataset import crop_patch, frame_npz_roundtrip, make_patch_id, manifest_split_fires, patch_starts, validate_split_assignments

def test_patch_starts_with_border(): assert patch_starts(144,64,60,True)==[0,60,80]
def test_patch_id_is_deterministic(): assert make_patch_id('FIRE_A',60,80,64,64)=='FIRE_A_y060_x080_h064_w064'
def test_split_validation():
 assert validate_split_assignments({'a':{},'b':{},'c':{}},['a'],['b'],['c'])['test']==['c']
 with pytest.raises(ValueError,match='overlaps'): validate_split_assignments({'a':{}},['a'],['a'],[])
 with pytest.raises(ValueError,match='missing'): validate_split_assignments({},['a'],[],[])
def test_frame_roundtrip_and_crop(tmp_path):
 x=np.arange(3*5*6,dtype=np.float32).reshape(3,5,6); path=tmp_path/'frame.npz'; frame_npz_roundtrip(path,x,x+1)
 with np.load(path,allow_pickle=False) as z: np.testing.assert_array_equal(z['x_engineered'],x)
 assert crop_patch(x,{'y0':1,'x0':2,'height':3,'width':4}).shape==(3,3,4)
def test_manifest_contains_splits(tmp_path):
 path=tmp_path/'dataset_manifest.json'; path.write_text(json.dumps({'splits':{'train_fires':['a'],'val_fires':['b'],'test_fires':['c']}})); assert set(json.loads(path.read_text())['splits'])=={'train_fires','val_fires','test_fires'}


def test_manifest_split_key_compatibility():
 assert manifest_split_fires({"splits": {"train": ["A"]}}, "train") == ["A"]
 assert manifest_split_fires({"splits": {"train_fires": ["B"]}}, "train") == ["B"]
