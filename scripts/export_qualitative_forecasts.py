#!/usr/bin/env python3
"""Select locked test cases from ground truth and export independent FLARE panels."""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines.table2_deterministic import ProcessedHistoryBaselinePredictor, ProcessedTargetOnlyDataset
from src.config import load_config
from src.data.dataset import create_dataloaders
from src.evaluation.fire_activity import FIRE_MASK_THRESHOLD, PREDICTED_FIRE_THRESHOLD
from src.evaluation.flare_runs import PAPER_NAMES, TEST_FIRES, resolve_run, prepare_cpu_mamba
from src.evaluation.full_validation import FullValidationAccumulator, _sample_metric_rows
from src.models.model_factory import build_model_from_config
from src.training.batch_utils import unpack_batch
from src.training.checkpoints import load_checkpoint, load_model_state_dict_compatible, validate_checkpoint_model_compatibility
from src.training.input_normalization import apply_input_normalization, build_input_normalizer_for_loader
from src.training.model_outputs import extract_prediction

CHANNELS = {'surface': 0, 'canopy': 1, 'mask': 2, 'energy': 3}


def digest(array):
    value = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return hashlib.sha256(value.tobytes()).hexdigest()


def safe_name(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value))


def read_records(config):
    root = Path(config['dataloader']['dataset_root'])
    path = root / 'indices/temporal' / f"samples_{config['dataloader']['sample_pattern']}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    records = [r for r in records if r['split'] == 'test']
    if len(records) != 59509 or {r['fire_name'] for r in records} != set(TEST_FIRES):
        raise RuntimeError('Temporal index differs from the locked Table 2 test split.')
    return root, path, records


@lru_cache(maxsize=2)
def _full_target(path):
    with np.load(path, allow_pickle=False) as archive:
        return tuple(np.asarray(archive[key], dtype=np.float32)
                     for key in ('surface_consumed', 'canopy_consumed', 'fire_mask', 'energy_log'))


def raw_case(root, record, *, include_input=True, include_terrain=True):
    patch = record['patch']
    y0, x0, h, w = (int(patch[k]) for k in ('y0', 'x0', 'height', 'width'))
    fire = record['fire_name']
    x = None
    if include_input:
        frames = []
        for i in record['input_indices']:
            with np.load(root / 'fires' / fire / 'frames' / f'frame_{i:06d}.npz', allow_pickle=False) as archive:
                frames.append(np.asarray(archive['x_engineered'][:, y0:y0+h, x0:x0+w], dtype=np.float32))
        x = np.stack(frames)
    y = np.stack([array[y0:y0+h, x0:x0+w] for array in _full_target(root / record['target_path'])]).astype(np.float32)
    y[2] = (y[2] > FIRE_MASK_THRESHOLD).astype(np.float32)
    terrain_path = root / 'fires' / fire / 'terrain' / 'terrain_features.npy'
    terrain = (np.asarray(np.load(terrain_path, allow_pickle=False)[:, y0:y0+h, x0:x0+w], dtype=np.float32)
               if include_terrain and terrain_path.is_file() else None)
    return x, y, terrain


def activity_rows(root, fire):
    path = ROOT / 'artifacts/table2_baselines/persistence/evaluation/test_sample_metrics.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Canonical ground-truth sample metrics missing: {path}')
    with path.open(newline='') as handle:
        return {r['sample_id']: r for r in csv.DictReader(handle) if r['fire_name'] == fire}


def case_metadata(root, record, stats=None):
    x, y, terrain = raw_case(root, record)
    active = y[2] > FIRE_MASK_THRESHOLD
    return {'case_id': record['sample_id'], 'sample_id': record['sample_id'], 'fire_name': record['fire_name'],
        'fire_id': record['fire_name'], 'split': record['split'], 'latest_input_timestamp': record['current_index'],
        'target_timestamp': record['target_index'], 'input_timestamps': record['input_indices'],
        'forecast_horizon': record['horizon'], 'patch': record['patch'], 'patch_id': record.get('patch_id'),
        'sample_index': record.get('_index'), 'pattern': record['pattern'], 'target_path': record['target_path'],
        'input_sha256': digest(x), 'target_sha256': digest(y),
        'terrain_sha256': digest(terrain) if terrain is not None else None,
        'target_activity': {'active_pixels': int(active.sum()), 'activity_fraction': float(active.mean()),
            'surface_consumption_sum': float(y[0].sum(dtype=np.float64)),
            'canopy_consumption_sum': float(y[1].sum(dtype=np.float64)),
            'energy_log_sum': float(y[3].sum(dtype=np.float64))}}


def choose(records, root, args):
    candidates = [(i, r) for i, r in enumerate(records) if args.fire is None or r['fire_name'] == args.fire]
    if not candidates:
        raise ValueError(f'No held-out test samples for fire {args.fire!r}; test fires: {TEST_FIRES}')
    if args.sample_id:
        candidates = [(i, r) for i, r in candidates if r['sample_id'] == args.sample_id]
    if args.timestamp is not None:
        candidates = [(i, r) for i, r in candidates if str(r['current_index']) == args.timestamp or str(r['target_index']) == args.timestamp]
    if not candidates:
        raise ValueError('No test case matched --sample-id/--timestamp.')
    mode = args.selection_mode
    if mode == 'exact' and not (args.sample_id or args.timestamp):
        raise ValueError('--selection-mode exact requires --sample-id or --timestamp.')
    if args.sample_id and len(candidates) != 1:
        raise ValueError('Sample ID must identify exactly one case.')
    if not args.sample_id:
        if mode in ('active', 'high_activity', 'low_activity', 'no_fire'):
            if args.fire is None:
                raise ValueError(f'--selection-mode {mode} requires --fire.')
            activity = activity_rows(root, args.fire)
            if set(r['sample_id'] for _, r in candidates) - set(activity):
                raise ValueError('Ground-truth activity records do not cover candidate cases.')
            candidates = [(i, r) for i, r in candidates if
                (mode != 'active' or float(activity[r['sample_id']]['active_fraction']) > 0) and
                (mode != 'no_fire' or float(activity[r['sample_id']]['active_fraction']) == 0) and
                (mode not in ('high_activity', 'low_activity') or float(activity[r['sample_id']]['active_fraction']) > 0)]
            if mode == 'high_activity':
                candidates.sort(key=lambda v: (-float(activity[v[1]['sample_id']]['active_fraction']), v[1]['sample_id']))
            elif mode == 'low_activity':
                candidates.sort(key=lambda v: (float(activity[v[1]['sample_id']]['active_fraction']), v[1]['sample_id']))
            else:
                random.Random(args.selection_seed).shuffle(candidates)
        elif mode == 'random':
            random.Random(args.selection_seed).shuffle(candidates)
        elif mode == 'evenly_spaced':
            candidates.sort(key=lambda v: (v[1]['fire_name'], v[1]['current_index'], v[1]['patch']['y0'], v[1]['patch']['x0']))
            count = min(args.num_samples, len(candidates))
            positions = np.linspace(0, len(candidates)-1, count).round().astype(int)
            candidates = [candidates[int(i)] for i in positions]
        else:
            candidates.sort(key=lambda v: v[1]['sample_id'])
    chosen = candidates[:args.num_samples]
    if len(chosen) < args.num_samples and not args.sample_id:
        raise ValueError(f'Requested {args.num_samples} cases; only {len(chosen)} match.')
    return [case_metadata(root, {**r, '_index': i}) for i, r in chosen]


def checked_case(root, record, case):
    for field, key in (('fire_name', 'fire_name'), ('current_index', 'latest_input_timestamp'),
                       ('target_index', 'target_timestamp'), ('horizon', 'forecast_horizon'),
                       ('patch', 'patch'), ('input_indices', 'input_timestamps')):
        if record[field] != case[key]:
            raise ValueError(f'Manifest mismatch in {field} for {case["case_id"]}')
    if record['sample_id'] != case['sample_id'] or record['split'] != case['split']:
        raise ValueError('Manifest sample ID or split mismatch.')
    x, y, terrain = raw_case(root, record)
    for key, array in (('input_sha256', x), ('target_sha256', y), ('terrain_sha256', terrain)):
        actual = digest(array) if array is not None else None
        if actual != case[key]:
            raise ValueError(f'Manifest {key} mismatch for {case["case_id"]}')
    return x, y, terrain


_PREDICT_CACHE = {}

def predict(run, record, index, device):
    config = dict(run['config'])
    config['return_metadata'] = True
    config['dataloader'] = {**config.get('dataloader', {}), 'return_metadata': True}
    if run['alias'] == 'persistence':
        predictor = ProcessedHistoryBaselinePredictor('persistence', config['dataloader']['dataset_root'], config)
        return predictor.predict_one(record), None
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')
    config['data_loader'] = {**config.get('data_loader', {}), 'test': {
        **config.get('data_loader', {}).get('test', {}), 'num_workers': 0,
        'persistent_workers': False, 'pin_memory': False}}
    key = (run['alias'], run['seed'], str(device))
    if key not in _PREDICT_CACHE:
        _, _, loader = create_dataloaders(config)
        model = build_model_from_config(config, input_channels=int(config['model']['input_channels'])).to(device)
        checkpoint = load_checkpoint(run['checkpoint'], map_location=device)
        validate_checkpoint_model_compatibility(model, checkpoint, run['checkpoint'])
        load_model_state_dict_compatible(model, checkpoint, run['checkpoint'])
        model.eval()
        if device.type == 'cpu' and run['alias'] == 'cawfe_st_mamba':
            prepare_cpu_mamba(model)
        normalizer = build_input_normalizer_for_loader(loader, device, int(config['model']['input_channels']))
        _PREDICT_CACHE[key] = (loader, model, normalizer)
    loader, model, normalizer = _PREDICT_CACHE[key]
    dataset = loader.dataset
    if dataset.records[index]['sample_id'] != record['sample_id']:
        raise RuntimeError('Canonical learned-model dataset order differs from the manifest.')
    batch = dataset[index]
    x, target, extra = unpack_batch(batch)
    if not np.array_equal(target.detach().numpy(), raw_case(Path(config['dataloader']['dataset_root']), record, include_input=False, include_terrain=False)[1]):
        raise RuntimeError('Loaded target differs from manifest target.')
    with torch.inference_mode():
        inputs = apply_input_normalization(x.unsqueeze(0).to(device), normalizer)
        terrain = extra.get('terrain')
        terrain = terrain.unsqueeze(0).to(device) if terrain is not None else None
        output = model(inputs) if terrain is None else model(inputs, terrain=terrain)
        prediction = extract_prediction(output).float().cpu().numpy()[0]
    if prediction.shape != target.shape:
        raise RuntimeError(f'Prediction shape {prediction.shape} differs from target {target.shape}.')
    return prediction, x.numpy()


def save_panel(path, array, *, vmin=None, vmax=None, cmap='viridis', label=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(array, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_axis_off()
    colorbar = fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    if label:
        colorbar.set_label(label)
    fig.savefig(path, dpi=180, bbox_inches='tight', pad_inches=.02)
    plt.close(fig)


def export_case(args, run, root, record, case, output_names, device):
    x_raw, target, terrain = checked_case(root, record, case)
    fire_dir = args.output_dir / safe_name(record['fire_name']) / safe_name(case['case_id'])
    model_dir = fire_dir / (run['alias'] if run['seed'] is None else f"{run['alias']}_seed{run['seed']}")
    if model_dir.exists() and not args.overwrite:
        raise FileExistsError(f'Output exists; pass --overwrite to replace this model case: {model_dir}')
    prediction, model_input = predict(run, record, case['sample_index'], device)
    if prediction.shape != target.shape or tuple(prediction.shape[1:]) != (record['patch']['height'], record['patch']['width']):
        raise RuntimeError('Model outputs do not match the canonical target patch.')
    pred_probability = torch.sigmoid(torch.from_numpy(prediction[2])).numpy()
    pred_mask = (pred_probability > PREDICTED_FIRE_THRESHOLD).astype(np.float32)
    accumulator = FullValidationAccumulator()
    accumulator.update(torch.from_numpy(prediction[None]), torch.from_numpy(target[None]))
    metrics = accumulator.finalize()
    per_case = _sample_metric_rows(torch.from_numpy(prediction[None]), torch.from_numpy(target[None]), [record], model_name=run['alias'], seed=-1 if run['seed'] is None else run['seed'])[0]
    if not any((args.save_inputs, args.save_targets, args.save_predictions, args.save_metrics,
                args.save_arrays, args.save_png, args.save_pdf, args.save_metadata, args.context)):
        print(f"{case['case_id']}: {run['paper_model_name']} Dice={metrics['full_val_dice']} canopy MAE={metrics['full_val_canopy_mae']}")
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    if args.save_metadata or args.save_targets or args.save_arrays:
        fire_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = fire_dir / 'manifest.json'
        if not manifest_path.exists() or args.overwrite:
            manifest_path.write_text(json.dumps(case, indent=2) + '\n')
    if args.save_metadata:
        model_meta = {**case, 'paper_model_name': run['paper_model_name'], 'repository_model_name': run['repo_model_name'],
            'checkpoint': str(run['checkpoint']) if run['checkpoint'] else None, 'checkpoint_seed': run['seed'],
            'normalization_path': str(run['normalization']) if run['normalization'] else None,
            'config_path': str(run['config_path']),
            'mask_threshold': PREDICTED_FIRE_THRESHOLD, 'target_mask_threshold': FIRE_MASK_THRESHOLD,
            'energy_units': 'log1p(MW)', 'per_case_metrics': per_case}
        (model_dir / 'metadata.json').write_text(json.dumps(model_meta, indent=2, default=str) + '\n')
    if args.save_metrics:
        (model_dir / 'metrics.json').write_text(json.dumps({'global_patch_metrics': metrics, 'sample_metrics': per_case}, indent=2, default=str) + '\n')
    if args.save_predictions or args.save_arrays:
        np.savez_compressed(model_dir / 'prediction.npz', prediction=prediction, mask_logits=prediction[2],
                            mask_probability=pred_probability, thresholded_mask=pred_mask)
    if args.save_targets or args.save_arrays:
        target_dir = fire_dir / 'target'; target_dir.mkdir(exist_ok=True)
        target_file = target_dir / 'target.npz'
        if not target_file.exists() or args.overwrite:
            np.savez_compressed(target_file, target=target)
    if args.save_inputs:
        np.savez_compressed(model_dir / 'inputs.npz', raw=x_raw, model_input=model_input if model_input is not None else x_raw)
    if args.context:
        context_dir = fire_dir / 'context'; context_dir.mkdir(exist_ok=True)
        if 'low_level_wind' in args.context:
            atmospheric = run['config'].get('atmospheric_features', {})
            levels = [int(v) for v in atmospheric.get('low_level_indices', [0, 1, 2])]
            width = int(atmospheric.get('variables_per_level', 10))
            if not levels or width < 2:
                raise ValueError('Invalid low-level wind layout in resolved config.')
            frame_path = root / 'fires' / record['fire_name'] / 'frames' / f"frame_{record['current_index']:06d}.npz"
            with np.load(frame_path, allow_pickle=False) as archive:
                frame = np.asarray(archive['x_raw'], dtype=np.float32)
            patch = record['patch']; yy, xx, hh, ww = (patch[k] for k in ('y0','x0','height','width'))
            if max(levels)*width+1 >= frame.shape[0]:
                raise ValueError('Low-level wind channels exceed the observed raw frame.')
            u = np.mean(np.stack([frame[z*width, yy:yy+hh, xx:xx+ww] for z in levels]), axis=0)
            v = np.mean(np.stack([frame[z*width+1, yy:yy+hh, xx:xx+ww] for z in levels]), axis=0)
            speed = np.sqrt(u*u+v*v)
            np.savez_compressed(context_dir / 'low_level_wind.npz', speed=speed, u=u, v=v,
                                levels=np.asarray(levels), timestamp=record['current_index'])
            for ext in [name for name, enabled in (('png', args.save_png), ('pdf', args.save_pdf)) if enabled]:
                save_panel(context_dir / f'low_level_wind_speed.{ext}', speed,
                           vmin=0, vmax=float(speed.max()), label='wind speed (m/s)')
        if 'terrain' in args.context and terrain is not None:
            np.savez_compressed(context_dir / 'terrain.npz', terrain=terrain)
        if 'latest_fire' in args.context:
            persistence = ProcessedHistoryBaselinePredictor('persistence', root, run['config'])
            latest_logits = persistence.predict_one(record)[2]
            latest_mask = (torch.sigmoid(torch.from_numpy(latest_logits)).numpy() > PREDICTED_FIRE_THRESHOLD).astype(np.float32)
            np.savez_compressed(context_dir / 'latest_fire.npz', fire_mask=latest_mask)
            for ext in ([name for name, enabled in (('png', args.save_png), ('pdf', args.save_pdf)) if enabled]):
                save_panel(context_dir / f'latest_fire.{ext}', latest_mask, vmin=0, vmax=1, cmap='gray')
    if args.save_png or args.save_pdf:
        formats = [ext for ext, enabled in (('png', args.save_png), ('pdf', args.save_pdf)) if enabled]
        prior = []
        if args.scale_mode == 'shared_case':
            for archive_path in fire_dir.glob('*/prediction.npz'):
                with np.load(archive_path, allow_pickle=False) as archive:
                    prior.append(np.asarray(archive['prediction'], dtype=np.float32))
        for output in output_names:
            channel = CHANNELS[output]
            if output == 'mask':
                target_image, prediction_image, limits = target[2], pred_mask, (0, 1)
                filename = 'fire_mask'
            else:
                target_image, prediction_image = target[channel], prediction[channel]
                vals = [target_image] + ([p[channel] for p in prior] + [prediction_image] if args.scale_mode == 'shared_case' else [])
                limits = (float(np.min(vals)), float(np.max(vals)))
                filename = output
            vmin = args.vmin if args.vmin is not None else limits[0]
            vmax = args.vmax if args.vmax is not None else limits[1]
            for ext in formats:
                if args.save_targets or args.save_arrays:
                    save_panel(fire_dir / 'target' / f'{filename}.{ext}', target_image, vmin=vmin, vmax=vmax,
                               cmap='gray' if output == 'mask' else 'viridis',
                               label='log1p energy (MW)' if output == 'energy' else None)
                if args.save_predictions or args.save_arrays:
                    save_panel(model_dir / f'{filename}.{ext}', prediction_image, vmin=vmin, vmax=vmax,
                               cmap='gray' if output == 'mask' else 'viridis',
                               label='log1p energy (MW)' if output == 'energy' else None)
    print(f'Exported {case["case_id"]} -> {model_dir}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=PAPER_NAMES, default='persistence')
    parser.add_argument('--checkpoint-seed', type=int, default=42)
    parser.add_argument('--fire')
    parser.add_argument('--timestamp', help='Exact current or target frame index.')
    parser.add_argument('--sample-id')
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--selection-seed', type=int, default=123)
    parser.add_argument('--selection-mode', choices=('random','active','high_activity','low_activity','no_fire','evenly_spaced','exact'), default='random')
    parser.add_argument('--list-fire-samples', action='store_true')
    parser.add_argument('--save-case-manifest', type=Path)
    parser.add_argument('--case-manifest', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('qualitative_results'))
    parser.add_argument('--overwrite', action='store_true')
    for flag in ('inputs','targets','predictions','metrics','arrays','png','pdf','metadata'):
        parser.add_argument('--save-' + flag, action='store_true')
    parser.add_argument('--outputs', default='mask,surface,canopy,energy')
    parser.add_argument('--scale-mode', choices=('shared_case','ground_truth'), default='ground_truth')
    parser.add_argument('--vmin', type=float)
    parser.add_argument('--vmax', type=float)
    parser.add_argument('--context', default='', help='Comma-separated: latest_fire,low_level_wind,terrain')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    args = parser.parse_args(argv)
    if args.save_png or args.save_pdf:
        if not (args.save_predictions or args.save_targets or args.save_arrays):
            parser.error('--save-png/--save-pdf requires --save-predictions, --save-targets, or --save-arrays')
    if args.num_samples < 1:
        parser.error('--num-samples must be positive')
    output_names = [o.strip() for o in args.outputs.split(',') if o.strip()]
    if not output_names or any(o not in CHANNELS for o in output_names):
        parser.error('--outputs must be drawn from mask,surface,canopy,energy')
    args.context = [o.strip() for o in args.context.split(',') if o.strip()]
    if any(o not in ('latest_fire','low_level_wind','terrain') for o in args.context):
        parser.error('--context must be latest_fire, low_level_wind, or terrain')
    run = resolve_run(args.model, None if args.model == 'persistence' else args.checkpoint_seed)
    root, index_path, records = read_records(run['config'])
    by_id = {r['sample_id']: (i, r) for i, r in enumerate(records)}
    if args.list_fire_samples:
        if args.fire not in TEST_FIRES:
            parser.error(f'--list-fire-samples requires an exact test fire: {TEST_FIRES}')
        print('sample_id\tlatest_input_time\ttarget_time\tactivity_fraction\tsurface_consumption_sum\tcanopy_consumption_sum\tenergy_log_sum')
        activity = activity_rows(root, args.fire)
        for i, r in enumerate(records):
            if r['fire_name'] != args.fire:
                continue
            _, y, _ = raw_case(root, r, include_input=False, include_terrain=False)
            print(f"{r['sample_id']}\t{r['current_index']}\t{r['target_index']}\t{activity[r['sample_id']]['active_fraction']}\t{y[0].sum():.6g}\t{y[1].sum():.6g}\t{y[3].sum():.6g}")
        return
    if args.case_manifest:
        manifest = json.loads(args.case_manifest.read_text())
        if manifest.get('split') != 'test' or manifest.get('sample_index_path') != str(index_path):
            raise ValueError('Manifest split/index does not match locked Table 2 test cases.')
        cases = manifest['cases']
        if args.fire and any(c['fire_name'] != args.fire for c in cases):
            raise ValueError('--fire disagrees with case manifest.')
    else:
        cases = choose(records, root, args)
    if args.save_case_manifest:
        if args.save_case_manifest.exists() and not args.overwrite:
            raise FileExistsError(args.save_case_manifest)
        args.save_case_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.save_case_manifest.write_text(json.dumps({'schema_version': 1, 'split': 'test',
            'sample_index_path': str(index_path), 'selection_mode': args.selection_mode,
            'selection_seed': args.selection_seed, 'cases': cases}, indent=2) + '\n')
        print(f'Saved {len(cases)} cases to {args.save_case_manifest}')
        if not any((args.save_inputs, args.save_targets, args.save_predictions, args.save_metrics,
                    args.save_arrays, args.save_png, args.save_pdf, args.save_metadata, args.context)):
            return
    device = torch.device(args.device)
    for case in cases:
        if case['case_id'] not in by_id:
            raise ValueError(f"Manifest case absent from test split: {case['case_id']}")
        index, record = by_id[case['case_id']]
        if index != case['sample_index']:
            raise ValueError(f"Manifest sample index mismatch for {case['case_id']}")
        export_case(args, run, root, record, case, output_names, device)


if __name__ == '__main__':
    main()
