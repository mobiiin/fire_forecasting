#!/usr/bin/env python3
"""Run named CAWFE-Latte config ablations without mutating baseline configs."""
from __future__ import annotations
import argparse, copy, csv, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import yaml
from src.config import load_config
from src.training.train import train_model_from_config

FAST_OVERRIDES = {
    "training.max_epochs": 5,
    "training.epochs": 5,
    "training.max_train_batches_per_epoch": 100,
    "training.validation.mode": "fixed_subset_every_epoch",
    "training.validation.max_val_batches_per_epoch": 20,
    "training.early_stopping.enabled": False,
    "ablation.fast_mode.max_train_batches_per_epoch": 100,
    "ablation.fast_mode.max_val_batches_per_epoch": 20,
    "ablation.fast_mode.evaluate_best_checkpoint": False,
    "ablation.fast_mode.save_checkpoints": True,
}
SUMMARY_FIELDS = ["rank", "ablation_name", "description", "changed_from_baseline", "best_epoch", "val_loss", "val_mask_dice", "val_mask_iou", "val_energy_log_mae", "val_surface_consumed_mae", "val_canopy_consumed_mae", "val_active_canopy_consumed_mae", "val_no_fire_mask_false_positive_rate", "val_no_fire_energy_log_pred_mean", "run_dir", "config_path"]

def set_dotted(payload: dict[str, Any], key: str, value: Any) -> None:
    current = payload
    bits = key.split('.')
    for bit in bits[:-1]:
        current = current.setdefault(bit, {})
        if not isinstance(current, dict): raise ValueError(f"Cannot apply {key}: {bit} is not a mapping")
    current[bits[-1]] = value

def get_dotted(payload: dict[str, Any], key: str) -> Any:
    current: Any = payload
    for bit in key.split('.'):
        if not isinstance(current, dict): return None
        current = current.get(bit)
    return current

def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ablations', nargs='+', default=['all'])
    p.add_argument('--ablation-file', default='configs/ablations/cawfe_latte_ablations.yaml')
    p.add_argument('--base-config', default='configs/experiments/cawfe_latte_baseline.yaml')
    p.add_argument('--mode', choices=('fast','full'), default='fast')
    p.add_argument('--max-epochs', type=int); p.add_argument('--max-train-batches', type=int); p.add_argument('--max-val-batches', type=int)
    p.add_argument('--seed', type=int); p.add_argument('--dry-run', action='store_true'); p.add_argument('--skip-existing', action='store_true')
    p.add_argument('--output-dir', default='artifacts/ablations/cawfe_latte'); p.add_argument('--device'); p.add_argument('--slurm', action='store_true')
    return p.parse_args()

def main() -> None:
    args=parse_args(); catalogue_path=Path(args.ablation_file); catalogue=yaml.safe_load(catalogue_path.read_text()) or {}; definitions=catalogue.get('ablations', {})
    names=list(definitions) if args.ablations == ['all'] or 'all' in args.ablations else args.ablations
    unknown=[n for n in names if n not in definitions]
    if unknown: raise SystemExit(f"Unknown ablation(s): {', '.join(unknown)}")
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'); root=Path(args.output_dir)/stamp; root.mkdir(parents=True, exist_ok=True)
    rows=[]
    for name in names:
        definition=definitions[name]; baseline_path=Path(definition.get('base_config',args.base_config))
        config=copy.deepcopy(load_config(baseline_path)); config.pop('base_config',None)
        overrides=dict(definition.get('overrides',{})); applied=dict(overrides)
        for key,value in overrides.items(): set_dotted(config,key,value)
        if args.mode=='fast':
            for key,value in FAST_OVERRIDES.items(): set_dotted(config,key,value); applied[key]=value
        for key,value in [('training.max_epochs',args.max_epochs),('training.max_train_batches_per_epoch',args.max_train_batches),('training.validation.max_val_batches_per_epoch',args.max_val_batches),('seed',args.seed),('training.device',args.device)]:
            if value is not None: set_dotted(config,key,value); applied[key]=value
        if get_dotted(config,'model.architecture') != 'cawfe_latte': raise ValueError(f"{name} changed the active architecture")
        run_root=root/name; resolved_path=run_root/'resolved_config.yaml'; write_yaml(resolved_path,config)
        (run_root/'baseline_config.yaml').write_text(Path(baseline_path).read_text(encoding='utf-8'), encoding='utf-8')
        (run_root/'ablation_definition.yaml').write_text(yaml.safe_dump(definition,sort_keys=False)); (run_root/'applied_overrides.yaml').write_text(yaml.safe_dump(applied,sort_keys=False))
        print(f"\n{name}: {definition.get('description','')}\n  changed: {', '.join(definition.get('changed_from_baseline',[])) or 'none'}\n  config: {resolved_path}\n  architecture: cawfe_latte")
        if args.dry_run: continue
        if args.slurm: raise SystemExit('--slurm is intentionally handled by scripts/slurm_run_cawfe_latte_ablations_a10080.sh')
        config['config_path']=str(resolved_path); config['_config_path']=str(resolved_path)
        result=train_model_from_config(config); final=result.get('final_epoch_summary',{}) or {}; row={'ablation_name':name,'description':definition.get('description',''),'changed_from_baseline':', '.join(definition.get('changed_from_baseline',[])),'best_epoch':result.get('best_epoch'),'run_dir':result.get('run_dir'),'config_path':str(resolved_path)}
        for field in SUMMARY_FIELDS:
            if field.startswith('val_'): row[field]=final.get(field)
        rows.append(row)
    if args.dry_run: return
    rows.sort(key=lambda r: (r.get('val_mask_dice') is None, -(r.get('val_mask_dice') or float('-inf'))))
    for rank,row in enumerate(rows,1): row['rank']=rank
    for suffix in ('csv','json','md'):
        path=root/f'ablation_summary.{suffix}'
        if suffix=='csv':
            with path.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=SUMMARY_FIELDS); w.writeheader(); w.writerows(rows)
        elif suffix=='json': path.write_text(json.dumps(rows,indent=2,default=str))
        else:
            cols=['rank','ablation_name','changed_from_baseline','val_mask_dice','val_mask_iou','val_energy_log_mae','val_surface_consumed_mae','val_canopy_consumed_mae','val_active_canopy_consumed_mae','run_dir']; path.write_text('| '+' | '.join(cols)+' |\n|'+ '---|'*len(cols)+'\n'+'\n'.join('| '+' | '.join(str(r.get(c,'')) for c in cols)+' |' for r in rows)+'\n')
    if rows:
        print(f"\nBest by val_mask_dice: {rows[0]['ablation_name']}")
        for key,label in [('val_energy_log_mae','val_energy_log_mae'),('val_no_fire_mask_false_positive_rate','no_fire_false_positive_rate')]:
            candidates=[r for r in rows if r.get(key) is not None]
            if candidates: print(f"Best by {label}: {min(candidates,key=lambda r:r[key])['ablation_name']}")
if __name__=='__main__': main()
