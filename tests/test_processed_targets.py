import numpy as np
from src.data.processed_targets import build_processed_target

def test_target_clipping_energy_and_union_mask():
 c=np.zeros((2,2,86),np.float32); f=c.copy(); c[:,:,84]=1; c[:,:,85]=.5; f[:,:,84]=.2; f[:,:,85]=.7; f[:,:,80]=1e6
 r=build_processed_target(c,f,np.ones((2,2),np.float32),{'target_construction':{'fire_mask':{'energy_threshold_mw':.1,'fuel_threshold':.001}}})
 assert np.allclose(r['surface_consumed'],.8); assert np.all(r['canopy_consumed']==0); assert np.allclose(r['energy_release_mw'],1); assert np.all(r['energy_log']==np.log1p(1)); assert r['fire_mask'].dtype==bool

def test_channel_first_input_is_supported():
 c=np.zeros((86,2,2),np.float32); f=c.copy(); c[84]=1; f[84]=0
 r=build_processed_target(c,f,np.ones((2,2),np.float32),{})
 assert r['surface_consumed'].shape==(2,2)


def test_processed_raw_shape_is_not_misread_as_hwc():
 c=np.zeros((86, 240, 144), np.float32); f=c.copy(); c[84]=1
 r=build_processed_target(c, f, np.ones((240, 144), np.float32), {})
 assert r["surface_consumed"].shape == (240, 144)
