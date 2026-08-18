"""Produce a compact sanity report for the rebuilt processed dataset."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
from src.config import load_config
from src.data.processed_dataset import load_dataset_manifest
from src.data.fire_mask_thresholds import resolve_frozen_thresholds

def main():
	p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--dataset_root'); a=p.parse_args(); c=load_config(a.config); pc=c.get('processed_dataset',{}); root=Path(a.dataset_root or pc.get('root','/scratch/mhabibp/cawfe_datasets/cawfe_engineered_v1')); manifest=load_dataset_manifest(root); thresholds, threshold_meta=resolve_frozen_thresholds(c,a.config,require=False); report={'dataset_root':str(root),'dataset_version':manifest.get('dataset_version'),'splits':manifest.get('splits',{}),'fires':manifest.get('fires',{}),'patches':{},'targets':{},'samples':{},'missing_files':[], 'thresholds': thresholds, 'threshold_source': threshold_meta, 'threshold_warning': None if thresholds else 'Frozen fire-mask thresholds are missing'}; patch_files=sorted((root/'indices'/'patches').glob('*.jsonl')); 
	if patch_files:
		rows=[json.loads(x) for x in patch_files[0].read_text().splitlines() if x.strip()]; report['patches']=dict(Counter(r['split'] for r in rows))
	for path in sorted((root/'indices'/'temporal').glob('samples_*.jsonl')):
		rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]; report['samples'][path.stem]=dict(Counter(r['split'] for r in rows))
	for path in sorted((root/'targets').glob('h*/target_manifest.json')):
		payload=json.loads(path.read_text()); report['targets'][path.parent.name]=dict(Counter(r['split'] for r in payload.get('targets',[])))
	(root/'processed_dataset_inspection.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
