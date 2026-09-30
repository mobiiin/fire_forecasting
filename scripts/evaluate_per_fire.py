#!/usr/bin/env python3
"""Publish locked Table 2 per-fire metrics, optionally rerunning the frozen test evaluator."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.flare_runs import PAPER_NAMES, TEST_FIRES, resolve_run, prepare_cpu_mamba

METRICS = ('dice', 'iou', 'surface_mae', 'canopy_mae', 'energy_log_mae', 'active_canopy_mae')


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rerun(run, destination):
    import torch
    from torch.utils.data import DataLoader
    from src.baselines.table2_deterministic import ProcessedHistoryBaselinePredictor, ProcessedTargetOnlyDataset
    from src.data.dataset import create_dataloaders
    from src.evaluation.held_out_test import evaluate_locked_test
    from src.models.model_factory import build_model_from_config
    from src.training.checkpoints import load_checkpoint, load_model_state_dict_compatible, validate_checkpoint_model_compatibility
    config = dict(run['config'])
    config['return_metadata'] = True
    config['dataloader'] = {**config.get('dataloader', {}), 'return_metadata': True}
    config['data_loader'] = {**config.get('data_loader', {}), 'test': {
        **config.get('data_loader', {}).get('test', {}), 'num_workers': 0, 'persistent_workers': False,
        'pin_memory': False}}
    if run['alias'] == 'persistence':
        root = Path(config['dataloader']['dataset_root'])
        pattern = config['dataloader']['sample_pattern']
        dataset = ProcessedTargetOnlyDataset(root, root / 'indices/temporal' / f'samples_{pattern}.jsonl', 'test')
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, drop_last=False)
        predictor = ProcessedHistoryBaselinePredictor('persistence', root, config)
        return evaluate_locked_test(test_loader=loader, config=config, run_dir=destination,
            model_name='persistence', seed=None, deterministic_predictor=predictor, device=torch.device('cpu'))
    _, _, loader = create_dataloaders(config)
    model = build_model_from_config(config, input_channels=int(config['model']['input_channels'])).to('cpu')
    checkpoint = load_checkpoint(run['checkpoint'], map_location='cpu')
    validate_checkpoint_model_compatibility(model, checkpoint, run['checkpoint'])
    load_model_state_dict_compatible(model, checkpoint, run['checkpoint'])
    if run['alias'] == 'cawfe_st_mamba':
        prepare_cpu_mamba(model)
    return evaluate_locked_test(test_loader=loader, config=config, run_dir=destination,
        model_name=run['alias'], seed=run['seed'], model=model, device=torch.device('cpu'),
        checkpoint_path=run['checkpoint'], checkpoint_epoch=checkpoint.get('epoch'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default=','.join(PAPER_NAMES))
    parser.add_argument('--seeds', default='42,123,2026')
    parser.add_argument('--output-dir', type=Path, default=Path('results/per_fire_generalization'))
    parser.add_argument('--recompute', action='store_true', help='Run full CPU inference instead of reusing verified locked Table 2 results.')
    args = parser.parse_args(argv)
    aliases = [x.strip() for x in args.models.split(',') if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(',') if x.strip()]
    if not aliases or len(set(aliases)) != len(aliases) or any(x not in PAPER_NAMES for x in aliases):
        parser.error(f'Choose unique aliases from {list(PAPER_NAMES)}')
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error('Provide unique checkpoint seeds.')
    output = args.output_dir.resolve()
    raw, provenance = [], []
    timestamp_counts = defaultdict(set)
    index_root = Path(resolve_run('persistence')['config']['dataloader']['dataset_root'])
    index_file = index_root / 'indices/temporal/samples_sparse5_h10.jsonl'
    with index_file.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get('split') == 'test':
                timestamp_counts[record['fire_name']].add((int(record['current_index']), int(record['target_index'])))
    if set(timestamp_counts) != set(TEST_FIRES):
        raise ValueError('Canonical temporal index test fires differ from Table 2.')
    for alias in aliases:
        for seed in ([None] if alias == 'persistence' else seeds):
            run = resolve_run(alias, seed, require_metrics=not args.recompute)
            source = run['run_dir']
            if args.recompute:
                source = output / 'recomputed' / alias / ('deterministic' if seed is None else f'seed_{seed}')
                _rerun(run, source)
            payload = json.loads((source / 'evaluation/test_metrics.json').read_text())
            if payload.get('metric_scope') != 'complete_locked_held_out_test' or payload.get('split') != 'test' or payload.get('test_used_for_model_selection') is not False:
                raise ValueError(f'Not a locked complete test evaluation: {source}')
            if int(payload['dataset_sample_count']) != 59509 or int(payload['unique_sample_id_count']) != 59509:
                raise ValueError(f'Table 2 case count mismatch: {source}')
            with (source / 'evaluation/test_per_fire_metrics.csv').open(newline='') as handle:
                per_fire = list(csv.DictReader(handle))
            if {row['fire_name'] for row in per_fire} != set(TEST_FIRES) or sum(int(row['sample_count']) for row in per_fire) != 59509:
                raise ValueError(f'Test fire coverage mismatch: {source}')
            # Sample-level records provide exact active-pixel and unique forecast-time counts.
            sample_path = source / 'evaluation/test_sample_metrics.csv'
            activity = defaultdict(lambda: {'pixels': 0})
            if sample_path.is_file():
                with sample_path.open(newline='') as handle:
                    for item in csv.DictReader(handle):
                        state = activity[item['fire_name']]
                        state['pixels'] += int(item['active_pixel_count'])
            for row in per_fire:
                fire = row['fire_name']
                raw.append({'paper_model_name': run['paper_model_name'], 'repo_model_name': run['repo_model_name'],
                    'model_alias': alias, 'seed': '' if seed is None else seed, 'fire_name': fire,
                    'checkpoint_path': '' if run['checkpoint'] is None else str(run['checkpoint']),
                    'config_path': str(run['config_path']), 'normalization_path': '' if run['normalization'] is None else str(run['normalization']),
                    'num_samples': int(row['sample_count']), 'num_active_pixels': activity[fire]['pixels'] if sample_path.is_file() else '',
                    'num_forecast_timestamps': len(timestamp_counts[fire]),
                    **{key: row.get(key, '') for key in row if key not in {'model', 'model_name', 'seed', 'fire_name', 'sample_count'}}})
            provenance.append(f"{run['paper_model_name']} seed={seed if seed is not None else 'deterministic'}\n"
                f"  checkpoint: {run['checkpoint']}\n  config: {run['config_path']}\n"
                f"  normalization: {run['normalization']}\n  locked metrics: {source / 'evaluation/test_metrics.json'}")
            print(f"Resolved {alias} seed={seed}: {source}")
    summary = []
    for (name, fire), rows in sorted(defaultdict(list, {key: [r for r in raw if (r['paper_model_name'], r['fire_name']) == key]
            for key in {(r['paper_model_name'], r['fire_name']) for r in raw}}).items()):
        result = {'paper_model_name': name, 'fire_name': fire, 'seed_count': len(rows)}
        for key in METRICS:
            values = [float(r[key]) for r in rows if r[key] not in ('', None)]
            result[f'{key}_mean'] = statistics.mean(values) if values else ''
            result[f'{key}_std'] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else '')
        summary.append(result)
    _csv(output / 'per_fire_raw.csv', raw)
    _csv(output / 'per_fire_summary.csv', summary)
    lines = ['Locked Table 2 per-fire evaluation; global pixel/count aggregation over all fixed test patches.',
             'Overlapping patches remain separate, as in the Table 2 evaluator; no full-domain reconstruction.',
             'Target mask > 0.5; predicted sigmoid(logit) > 0.5; active canopy is target-mask pixels only.',
             'Test fires: ' + ', '.join(TEST_FIRES), '', 'Model | Fire | Dice | IoU | Surface MAE | Canopy MAE | Energy Log MAE | Active Canopy MAE']
    for r in summary:
        lines.append(' | '.join([r['paper_model_name'], r['fire_name']] +
            [f"{r[k+'_mean']:.4f} ± {r[k+'_std']:.4f}" if r[k+'_mean'] != '' else 'N/A' for k in METRICS]))
    (output / 'per_fire_summary.txt').write_text('\n'.join(lines) + '\n')
    tex = ['\\begin{tabular}{llrrrr}', '\\toprule',
           'Fire & Model & Dice & Energy Log MAE & Surface MAE & Active Canopy MAE \\\\', '\\midrule']
    for r in summary:
        values = [r[k+'_mean'] for k in ('dice', 'energy_log_mae', 'surface_mae', 'active_canopy_mae')]
        tex.append(' & '.join([r['fire_name'].replace('_', '\\_'), r['paper_model_name']] +
                   [f'{v:.4f}' if v != '' else '--' for v in values]) + ' \\\\')
    tex += ['\\bottomrule', '\\end{tabular}']
    (output / 'per_fire_table.tex').write_text('\n'.join(tex) + '\n')
    (output / 'provenance.txt').write_text('\n\n'.join(provenance) + '\n\n' + '\n'.join(lines[:4]) + '\n')
    print(f'Wrote {len(raw)} model-seed-fire rows to {output}')


if __name__ == '__main__':
    main()
