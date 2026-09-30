#!/usr/bin/env python3
"""Generate GT-only or fully rendered candidate figures for fixed selection seeds."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.plot_qualitative_test_comparison import main as generate_one


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-seeds', nargs='+', type=int, required=True)
    parser.add_argument('--model-seed', type=int, choices=(42,123,2026), default=42)
    parser.add_argument('--selection-only', action='store_true')
    parser.add_argument('--num-cases', type=int, choices=(2,3), default=2)
    parser.add_argument('--include-weak-case', action='store_true')
    parser.add_argument('--config', type=Path, default=Path('configs/plots/qualitative_test_comparison.yaml'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/qualitative_test_comparison'))
    parser.add_argument('--cpu-threads', type=int)
    args = parser.parse_args(argv)
    if len(set(args.selection_seeds)) != len(args.selection_seeds):
        parser.error('--selection-seeds must be unique')
    output = args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    rows = []
    columns = ['selection_seed','case_A_sample_id','case_A_fire','case_B_sample_id','case_B_fire',
               'case_C_sample_id','case_C_fire','output_pdf']
    for seed in args.selection_seeds:
        commands = ['--selection-seed',str(seed),'--model-seed',str(args.model_seed),
                    '--num-cases',str(args.num_cases),'--config',str(args.config),
                    '--output-dir',str(output)]
        if args.selection_only: commands.append('--selection-only')
        if args.include_weak_case: commands.append('--include-weak-case')
        if args.cpu_threads: commands += ['--cpu-threads',str(args.cpu_threads)]
        generate_one(commands)
        directory = output / f'selection_seed_{seed}'
        cases = json.loads((directory / 'selected_cases.json').read_text())['cases']
        row = {'selection_seed':seed,
            'case_A_sample_id':cases[0]['sample_id'],'case_A_fire':cases[0]['fire_name'],
            'case_B_sample_id':cases[1]['sample_id'],'case_B_fire':cases[1]['fire_name'],
            'case_C_sample_id':cases[2]['sample_id'] if len(cases)>2 else '',
            'case_C_fire':cases[2]['fire_name'] if len(cases)>2 else '',
            'output_pdf':str(directory / ('qualitative_selection_sheet.pdf' if args.selection_only else
                                         f'model_seed_{args.model_seed}/qualitative_test_comparison.pdf'))}
        rows.append(row)
        with (output / 'candidate_index.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=columns)
            writer.writeheader(); writer.writerows(rows)
    print('CANDIDATE INDEX:',output / 'candidate_index.csv')


if __name__ == '__main__':
    main()
