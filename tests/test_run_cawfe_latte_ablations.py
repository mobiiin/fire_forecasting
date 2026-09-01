from scripts.run_cawfe_latte_ablations import FAST_OVERRIDES, set_dotted

def test_fast_mode_limits_epochs_and_batches():
    config={}
    for key,value in FAST_OVERRIDES.items(): set_dotted(config,key,value)
    assert config['training']['max_epochs']==5
    assert config['training']['max_train_batches_per_epoch']==100
    assert config['training']['validation']['max_val_batches_per_epoch']==20
