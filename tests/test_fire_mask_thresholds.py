import json
import numpy as np
import pytest
from src.data.fire_mask_thresholds import resolve_frozen_thresholds, threshold_union_mask

def test_separate_threshold_union():
    m=threshold_union_mask(np.array([0.,2.,0.]),np.array([2.,0.,0.]),np.array([0.,0.,2.]),{'energy_threshold_mw':1.,'surface_fuel_threshold':1.,'canopy_fuel_threshold':1.})
    assert m.tolist()==[True,True,True]

def test_missing_frozen_thresholds_fails():
    with pytest.raises(FileNotFoundError,match='estimate_fire_mask_thresholds'):
        resolve_frozen_thresholds({'target_construction':{'fire_mask':{'require_frozen_thresholds':True}}},require=True)

def test_threshold_file_loads(tmp_path):
    p=tmp_path/'thresholds.json'; p.write_text(json.dumps({'threshold_version':'v1','thresholds':{'energy_threshold_mw':1,'surface_fuel_threshold':2,'canopy_fuel_threshold':3}}))
    values,meta=resolve_frozen_thresholds({'target_construction':{'fire_mask':{'threshold_file':str(p),'require_frozen_thresholds':True}}},require=True)
    assert values['canopy_fuel_threshold']==3 and meta['source']=='file'
