#!/usr/bin/env python3
"""GT-selected, paired-seed qualitative figures for frozen Table 2 models."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.export_qualitative_forecasts import checked_case, raw_case
from src.baselines.table2_deterministic import ProcessedHistoryBaselinePredictor
from src.config import load_config
from src.data.dataset import create_dataloaders
from src.evaluation.fire_activity import FIRE_MASK_THRESHOLD, active_fraction_bin_name, classify_fire_masks
from src.evaluation.full_validation import FullValidationAccumulator
from src.evaluation.qualitative_test_figure import (
    draw_comparison, draw_selection_sheet, gt_descriptors, observed_context,
    read_test_records, select_gt_cases, sha256_file,
)
from src.training.batch_utils import unpack_batch
from src.models.model_factory import build_model_from_config
from src.training.checkpoints import load_checkpoint, load_model_state_dict_compatible, validate_checkpoint_model_compatibility
from src.training.input_normalization import apply_input_normalization, build_input_normalizer_for_loader, resolve_input_normalization_stats_path
from src.training.model_outputs import extract_prediction

MODEL_KEYS = ('convlstm', 'baseline', 'final')
TABLE2_NAMES = {'convlstm': 'convlstm_unet', 'baseline': 'cawfe_latte_baseline', 'final': 'cawfe_latte_final'}
EXPECTED_ARCH = {'convlstm': 'convlstm_unet', 'baseline': 'cawfe_latte', 'final': 'cawfe_latte'}
PAPER_NAMES = {'convlstm': 'ConvLSTM U-Net', 'baseline': 'FLARE baseline', 'final': 'FLARE final'}
PRIMARY = ('dice','iou','surface_mae','canopy_mae','energy_log_mae','active_canopy_mae')


def json_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    temporary.replace(path)


def resolve_figure_run(model_key, seed, reference=None):
    summary = json.loads((ROOT / 'artifacts/table2_baselines/summary/table2_results.json').read_text())
    row = next((r for r in summary['rows'] if r['model_key'] == TABLE2_NAMES[model_key]), None)
    if row is None:
        raise FileNotFoundError(f'MISSING CHECKPOINT: {model_key}/{seed}')
    matches = [ROOT / directory for found_seed, directory in zip(row['seed_values'],row['run_dirs']) if found_seed == seed]
    if len(matches) != 1:
        raise FileNotFoundError(f'MISSING CHECKPOINT: {model_key}/{seed}')
    run_dir = matches[0]
    checkpoint = run_dir / 'checkpoints/best_model.pt'
    config_path = run_dir / 'resolved_config.yaml'
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError(f'MISSING CHECKPOINT: {model_key}/{seed} ({checkpoint})')
    config = load_config(config_path)
    if config.get('model',{}).get('architecture') != EXPECTED_ARCH[model_key]:
        raise RuntimeError(f'Wrong architecture in {config_path}; expected {EXPECTED_ARCH[model_key]}')
    if model_key == 'final' and config.get('final_training',{}).get('finalist') != 'GA_Q2':
        raise RuntimeError(f'Final model is not GA_Q2: {config_path}')
    if model_key == 'baseline' and config.get('final_training',{}).get('finalist') != 'baseline':
        raise RuntimeError(f'Baseline is not the original FLARE baseline: {config_path}')
    metrics_path = run_dir / 'evaluation/test_metrics.json'
    if not metrics_path.is_file():
        raise RuntimeError(f'Table 2 test evaluation missing: {metrics_path}')
    metrics = json.loads(metrics_path.read_text())
    if (metrics.get('metric_scope') != 'complete_locked_held_out_test' or metrics.get('split') != 'test'
            or metrics.get('test_used_for_model_selection') is not False
            or Path(metrics.get('checkpoint','')).resolve() != checkpoint.resolve()):
        raise RuntimeError(f'Table 2 checkpoint/evaluation identity mismatch: {run_dir}')
    normalization = resolve_input_normalization_stats_path(config, must_exist=True)
    signature = (str(Path(config['dataloader']['dataset_root']).resolve()), config['dataloader']['sample_pattern'],
                 int(config['dataloader']['patch_size']), int(config['dataloader']['target_horizon']),
                 config['dataloader']['input_key'], config['dataloader']['terrain_key'],
                 str(normalization), int(config['cache']['input_sequence_length']),
                 int(config['cache']['prediction_horizon']), config['cache']['target_definition_version'],
                 int(metrics['dataset_sample_count']))
    if reference is not None and signature != reference:
        raise RuntimeError(f'Table 2 dataset/preprocessing protocol mismatch: {model_key}/{seed}')
    return {'key': model_key, 'seed': seed, 'run_dir': run_dir, 'checkpoint': checkpoint,
            'config_path': config_path, 'config': config, 'normalization': normalization,
            'signature': signature, 'metrics_path': metrics_path}


def _settings(args):
    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict):
        raise ValueError(f'Invalid figure config: {args.config}')
    config['layout']['mode'] = args.layout or config['layout']['mode']
    config['layout']['num_cases'] = 3 if args.include_weak_case else args.num_cases or config['layout']['num_cases']
    if config['layout']['num_cases'] not in (2,3):
        raise ValueError('--num-cases must be 2 or 3')
    if args.terrain_contours is not None:
        config['terrain']['contours'] = args.terrain_contours == 'on'
    if args.wind is not None:
        config['wind']['enabled'] = args.wind == 'on'
    if args.wind_stride is not None:
        config['wind']['stride'] = args.wind_stride
    if args.wind_scale is not None:
        config['wind']['scale'] = args.wind_scale
    if args.wind_width is not None:
        config['wind']['width'] = args.wind_width
    if args.color_percentile is not None:
        config['continuous']['color_percentile'] = args.color_percentile
    if args.show_negative:
        config['continuous']['clip_negative_for_display'] = False
        config['continuous']['show_negative'] = True
    if args.show_threshold_contour:
        config['mask']['show_threshold_contour'] = True
    if args.font_family is not None:
        config['fonts']['family'] = args.font_family
    if args.dpi is not None:
        config['figure']['dpi'] = args.dpi
    if not 0 < float(config['continuous']['color_percentile']) <= 100:
        raise ValueError('Color percentile must be in (0,100].')
    return config


def _validate_cases(root, index, records, manifest, expected_num):
    if manifest.get('split') != 'test' or manifest.get('selection_uses_predictions') is not False:
        raise ValueError('Selection manifest must be GT-only and from the test split.')
    if Path(manifest['sample_index_path']).resolve() != index.resolve() or manifest['sample_index_sha256'] != sha256_file(index):
        raise ValueError('Frozen selection uses a different temporal index.')
    cases = manifest['cases']
    if len(cases) != expected_num or len({c['fire_name'] for c in cases}) != expected_num:
        raise ValueError('Selected cases must have the requested count and distinct test fires.')
    by_id = {r['sample_id']: r for r in records}
    if len(by_id) != len(records):
        raise RuntimeError('Duplicate test sample IDs.')
    for case, category in zip(cases, ('medium','high_energy','weak')[:expected_num]):
        record = by_id.get(case['sample_id'])
        if record is None or record.get('split') != 'test' or case['case_type'] != category:
            raise ValueError(f'Frozen case is absent or in the wrong category: {case["sample_id"]}')
        checked_case(root, record, case)
        desc = gt_descriptors(root, record)
        allowed_bins = {'medium': ('medium_fire',), 'high_energy': ('large_fire',),
                        'weak': ('small_fire', 'tiny_fire')}
        if (desc['target_activity_bin'] != case['target_activity_bin']
                or desc['target_activity_bin'] not in allowed_bins[category]):
            raise ValueError(f'GT activity class changed for {case["sample_id"]}')
        if abs(desc['target_active_fraction'] - case['target_active_fraction']) > 1e-9:
            raise ValueError(f'GT active fraction changed for {case["sample_id"]}')
        print(f"CASE {category}: {case['sample_id']} | {case['fire_name']} | crop={case['patch']} | t={case['latest_input_timestamp']} -> {case['target_timestamp']}")
    return cases, by_id


def _data_for_cases(root, cases, by_id, dataset_config):
    predictor = ProcessedHistoryBaselinePredictor('persistence', root, dataset_config)
    result = []
    for case in cases:
        record = by_id[case['sample_id']]
        _, target, _ = checked_case(root, record, case)
        context = observed_context(root, record, dataset_config, predictor)
        result.append({'target': target, 'context': context, 'predictions': {}})
    return result


def _load_one_model(run, cases, by_id, case_data, reference_inputs, *, expected_counts=None):
    """Run one frozen checkpoint on selected cases from any dataset split."""
    config = copy.deepcopy(run['config'])
    config['return_metadata'] = True
    config['dataloader']['return_metadata'] = True
    requested = {case['split'] for case in cases}
    if not requested <= {'train', 'val', 'test'}:
        raise ValueError(f'Unsupported qualitative split: {requested}')
    counts = {'test': run['signature'][-1]} if expected_counts is None else expected_counts
    for split in requested:
        options = config['data_loader'][split]
        options.update({'num_workers': 0, 'persistent_workers': False, 'pin_memory': False})
    loaders = dict(zip(('train', 'val', 'test'), create_dataloaders(config)))
    datasets, normalizers, lookups = {}, {}, {}
    for split in requested:
        loader = loaders[split]
        if loader is None or split not in counts or loader.dataset.split != split or len(loader.dataset) != counts[split]:
            raise RuntimeError(f'Model loader did not return the complete {split} split for {run["key"]}.')
        if split == 'test' and counts[split] != run['signature'][-1]:
            raise RuntimeError(f'Test split count differs from locked Table 2 evaluation for {run["key"]}.')
        dataset = loader.dataset
        wanted = {case['sample_id'] for case in cases if case['split'] == split}
        lookup = {record['sample_id']: i for i, record in enumerate(dataset.records) if record['sample_id'] in wanted}
        if set(lookup) != wanted:
            raise RuntimeError(f'Selected sample ID is missing from {run["key"]} {split} loader.')
        datasets[split], lookups[split] = dataset, lookup
        normalizers[split] = build_input_normalizer_for_loader(
            loader, torch.device('cpu'), int(config['model']['input_channels']))
    model = build_model_from_config(config, input_channels=int(config['model']['input_channels'])).to('cpu')
    checkpoint = load_checkpoint(run['checkpoint'], map_location='cpu')
    validate_checkpoint_model_compatibility(model, checkpoint, run['checkpoint'])
    load_model_state_dict_compatible(model, checkpoint, run['checkpoint'])
    model.eval()
    for i, case in enumerate(cases):
        split = case['split']
        dataset = datasets[split]
        record = dataset.records[lookups[split][case['sample_id']]]
        if (record['sample_id'] != case['sample_id'] or record['split'] != split
                or record['patch'] != case['patch'] or record['input_indices'] != case['input_timestamps']):
            raise RuntimeError(f'Model-specific split, crop, or time mismatch for {run["key"]}: {case["sample_id"]}')
        x, y, extra = unpack_batch(dataset[lookups[split][case['sample_id']]])
        if not np.array_equal(y.numpy(), case_data[i]['target']):
            raise RuntimeError(f'Model target differs for {run["key"]}: {case["sample_id"]}')
        identity = (split, case['sample_id'])
        input_hash = hashlib.sha256(np.ascontiguousarray(x.numpy()).tobytes()).digest()
        if identity in reference_inputs and input_hash != reference_inputs[identity]:
            raise RuntimeError(f'Model input differs for {run["key"]}: {case["sample_id"]}')
        reference_inputs.setdefault(identity, input_hash)
        terrain = extra.get('terrain')
        if terrain is not None:
            _, _, canonical_terrain = checked_case(dataset.root, record, case)
            if not np.array_equal(terrain.numpy(), canonical_terrain):
                raise RuntimeError(f'Model terrain differs for {run["key"]}: {case["sample_id"]}')
        with torch.no_grad():
            inputs = apply_input_normalization(x.unsqueeze(0), normalizers[split])
            output = model(inputs) if terrain is None else model(inputs, terrain=terrain.unsqueeze(0))
            prediction = extract_prediction(output).float().cpu().numpy()[0]
        if prediction.shape != case_data[i]['target'].shape or not np.isfinite(prediction).all():
            raise RuntimeError(f'Invalid prediction shape or values for {run["key"]}: {case["sample_id"]}')
        case_data[i]['predictions'][run['key']] = prediction
        if len(cases) > 10 and (i + 1) % 10 == 0:
            print(f'MODEL PROGRESS: {run["key"]} {i + 1}/{len(cases)} cases', flush=True)
    del model, loaders, checkpoint


def _panel_metrics(cases, case_data):
    rows = []
    for case,data in zip(cases,case_data):
        target = torch.from_numpy(data['target'][None])
        for key in MODEL_KEYS:
            prediction = torch.from_numpy(data['predictions'][key][None])
            accumulator = FullValidationAccumulator()
            accumulator.update(prediction,target)
            result = accumulator.finalize()
            row = {'model': PAPER_NAMES[key], 'model_key': key, 'sample_id': case['sample_id'], 'fire_name': case['fire_name']}
            for metric in PRIMARY:
                row[metric] = result['full_val_'+metric]
            rows.append(row)
    return rows


def _save_prediction_arrays(path, cases, case_data):
    arrays = {}
    for i,(case,data) in enumerate(zip(cases,case_data)):
        prefix = f'case_{i}'
        arrays[prefix+'_sample_id'] = np.asarray(case['sample_id'])
        context = data['context']; target = data['target']
        for name in ('latest_observed_fire_state','wind_u','wind_v','terrain_elevation'):
            if context.get(name) is not None:
                arrays[prefix+'_'+name] = np.asarray(context[name], dtype=np.float32)
        for name,channel in (('mask',2),('surface',0),('canopy',1),('energy_log',3)):
            arrays[prefix+'_target_'+name] = np.asarray(target[channel], dtype=np.float32)
        for model_key,pred in data['predictions'].items():
            arrays[prefix+'_'+model_key+'_raw'] = np.asarray(pred, dtype=np.float32)
            arrays[prefix+'_'+model_key+'_mask_probability'] = (1/(1+np.exp(-np.clip(pred[2],-60,60)))).astype(np.float32)
            for name,channel in (('surface',0),('canopy',1),('energy_log',3)):
                arrays[prefix+'_'+model_key+'_'+name] = np.asarray(pred[channel], dtype=np.float32)
    np.savez_compressed(path, **arrays)


def _load_prediction_arrays(path, cases):
    result = []
    with np.load(path, allow_pickle=False) as archive:
        for i,case in enumerate(cases):
            prefix = f'case_{i}'
            if str(archive[prefix+'_sample_id']) != case['sample_id']:
                raise ValueError('Saved prediction sample ID differs from frozen selection.')
            target = np.stack([archive[prefix+'_target_'+name] for name in ('surface','canopy','mask','energy_log')])
            context = {name: archive[prefix+'_'+name] if prefix+'_'+name in archive.files else None
                for name in ('latest_observed_fire_state','wind_u','wind_v','terrain_elevation')}
            predictions = {key: archive[prefix+'_'+key+'_raw'] for key in MODEL_KEYS}
            result.append({'target':target,'context':context,'predictions':predictions})
    return result


def _write_metrics(path, rows):
    with path.open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=['model','model_key','sample_id','fire_name',*PRIMARY])
        writer.writeheader(); writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/plots/qualitative_test_comparison.yaml'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/qualitative_test_comparison'))
    parser.add_argument('--selection-seed', type=int)
    parser.add_argument('--model-seed', type=int, choices=(42,123,2026), default=42)
    parser.add_argument('--num-cases', type=int, choices=(2,3))
    parser.add_argument('--include-weak-case', action='store_true')
    parser.add_argument('--layout', choices=('compact','stacked'))
    parser.add_argument('--terrain-contours', choices=('on','off'))
    parser.add_argument('--wind', choices=('on','off'))
    parser.add_argument('--wind-stride', type=int)
    parser.add_argument('--wind-scale', type=float)
    parser.add_argument('--wind-width', type=float)
    parser.add_argument('--color-percentile', type=float)
    parser.add_argument('--show-negative', action='store_true')
    parser.add_argument('--show-threshold-contour', action='store_true')
    parser.add_argument('--annotate-metrics', action='store_true')
    parser.add_argument('--font-family')
    parser.add_argument('--dpi', type=int)
    parser.add_argument('--cpu-threads', type=int)
    parser.add_argument('--selection-only', action='store_true')
    parser.add_argument('--freeze-selection', type=Path)
    parser.add_argument('--render-only', action='store_true', help='Redraw saved predictions.npz without checkpoint loading.')
    parser.add_argument('--reselect', action='store_true', help='Replace a GT-only draft selection before any model run exists.')
    args = parser.parse_args(argv)
    if args.cpu_threads is not None:
        if args.cpu_threads < 1: parser.error('--cpu-threads must be positive')
        torch.set_num_threads(args.cpu_threads)
    config = _settings(args)
    dataset_config_path = ROOT / 'artifacts/table2_baselines/persistence/resolved_config.yaml'
    dataset_config = load_config(dataset_config_path)
    config['_dataset_config'] = dataset_config
    config['_wind_levels'] = dataset_config.get('atmospheric_features',{}).get('low_level_indices',[0,1,2])
    root,index,records,fires = read_test_records(dataset_config)
    print('TEST DATASET:', root)
    print('DETECTED TEST FIRES:', ', '.join(fires))
    if args.freeze_selection:
        manifest = json.loads(args.freeze_selection.read_text())
        seed = int(manifest['selection_seed'])
    else:
        seed = int(args.selection_seed if args.selection_seed is not None else config['selection']['seed'])
        selection_dir = args.output_dir / f'selection_seed_{seed}'
        stored = selection_dir / 'selected_cases.json'
        if args.reselect and any(selection_dir.glob('model_seed_*')):
            raise RuntimeError('Cannot reselect cases after model inference exists for this selection seed.')
        if stored.is_file() and not args.reselect:
            manifest = json.loads(stored.read_text())
        elif args.render_only:
            raise FileNotFoundError(stored)
        else:
            manifest, pool_summary = select_gt_cases(root,index,records,config,seed,int(config['layout']['num_cases']))
            json_write(selection_dir / 'candidate_pool_summary.json',pool_summary)
            json_write(stored,manifest)  # durable before any learned-model checkpoint is opened
    selection_dir = args.output_dir / f'selection_seed_{seed}'
    cases,by_id = _validate_cases(root,index,records,manifest,int(config['layout']['num_cases']))
    stored = selection_dir / 'selected_cases.json'
    if not stored.exists():
        json_write(stored,manifest)
    elif json.loads(stored.read_text())['cases'] != cases:
        raise RuntimeError(f'Existing selection differs from frozen selection: {stored}')
    case_data = _data_for_cases(root,cases,by_id,dataset_config)
    sheet = selection_dir / 'qualitative_selection_sheet.pdf'
    if not sheet.exists() or args.reselect:
        draw_selection_sheet(cases,case_data,sheet,config)
    print('GT-ONLY SELECTION SHEET:',sheet)
    if args.selection_only:
        return selection_dir
    model_dir = selection_dir / f'model_seed_{args.model_seed}'
    model_dir.mkdir(parents=True,exist_ok=True)
    json_write(model_dir / 'selected_cases.json',manifest)
    if args.render_only:
        case_data = _load_prediction_arrays(model_dir / 'predictions.npz',cases)
        with (model_dir / 'panel_metrics.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
    else:
        runs = {}
        reference_signature = None
        for key in MODEL_KEYS:
            run = resolve_figure_run(key,args.model_seed,reference_signature)
            reference_signature = run['signature']
            if reference_signature[-1] != len(records):
                raise RuntimeError(f'Table 2 test count differs from index for {key}')
            runs[key] = run
            print('CHECKPOINT:',key,run['checkpoint'])
        checkpoint_manifest = {'schema_version':1,'model_seed':args.model_seed,'checkpoint_selection':'frozen Table 2 run_dirs, not test metrics',
            'models':{key:{'paper_name':PAPER_NAMES[key], 'architecture':EXPECTED_ARCH[key],
                'finalist':'GA_Q2' if key=='final' else 'baseline' if key=='baseline' else None,
                'seed':args.model_seed,'checkpoint':str(run['checkpoint']),
                'checkpoint_sha256':sha256_file(run['checkpoint']),
                'config':str(run['config_path']),'normalization':str(run['normalization']),
                'locked_test_metrics':str(run['metrics_path'])} for key,run in runs.items()}}
        json_write(model_dir / 'checkpoint_manifest.json',checkpoint_manifest)
        reference_inputs = {}
        for key in MODEL_KEYS:
            _load_one_model(runs[key],cases,by_id,case_data,reference_inputs)
        rows = _panel_metrics(cases,case_data)
        _write_metrics(model_dir / 'panel_metrics.csv',rows)
        _save_prediction_arrays(model_dir / 'predictions.npz',cases,case_data)
    limits = draw_comparison(cases,case_data,model_dir,config,annotate_metrics=args.annotate_metrics,panel_metrics=rows)
    metadata = {'schema_version':1,'selection_seed':seed,'model_seed':args.model_seed,'layout':config['layout']['mode'],
        'test_dataset':str(root),'sample_index':str(index),'selected_cases_path':str(stored),
        'selected_cases_sha256':sha256_file(stored),'figure_config':str(args.config),
        'figure_settings':{k:v for k,v in config.items() if not k.startswith('_')},
        'color_limits_gt_only':limits,'energy_display_units':'log(1 + E_MW)',
        'mask_display':'GT binary; model sigmoid(logits) probabilities',
        'continuous_negative_clipping_for_display':config['continuous']['clip_negative_for_display'],
        'continuous_predictions_saved_raw':True,'selection_uses_predictions':False,
        'checkpoint_manifest':str(model_dir / 'checkpoint_manifest.json')}
    json_write(model_dir / 'figure_metadata.json',metadata)
    print('FINAL FIGURE:',model_dir / 'qualitative_test_comparison.pdf')
    return model_dir


if __name__ == '__main__':
    main()
