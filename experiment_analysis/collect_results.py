"""
Collect and aggregate experiment results across feature combinations.

Scans saved_results/ for DyGFormer result JSON files, groups by
(dataset, feature_tag), and prints a summary table with mean +/- std.

Usage:
    python collect_results.py
    python collect_results.py --task lp          # link prediction only
    python collect_results.py --task nc          # node classification only
    python collect_results.py --results_root experiments/feature_social_v2_trainfit/saved_results
    python collect_results.py --csv results.csv  # also save to CSV
"""

import os
import json
import argparse
import re
from collections import defaultdict

MODEL = 'DyGFormer'
DEFAULT_RESULTS_ROOT = './saved_results'

TAG_DISPLAY = {
    'base': 'Base',
    'S': 'Base+Style',
    'P': 'Base+Pers',
    'T': 'Base+Topic',
    'So': 'Base+Social',
    'S_P': 'Base+S+P',
    'S_P_T': 'Base+S+P+T',
    'full': 'Full',
}

TAG_ORDER = ['base', 'S', 'P', 'T', 'So', 'S_P', 'S_P_T', 'full']

LP_PATTERN = re.compile(
    rf'^{MODEL}_(?P<tag>[A-Za-z_]+)_seed(?P<seed>\d+)\.json$')
NC_PATTERN = re.compile(
    rf'^node_classification_{MODEL}_(?P<tag>[A-Za-z_]+)_seed(?P<seed>\d+)\.json$')


def _infer_experiment_dir(results_root: str) -> str:
    normalized = os.path.normpath(results_root)
    if normalized == os.path.normpath(DEFAULT_RESULTS_ROOT):
        return ''
    if os.path.basename(normalized) == 'saved_results':
        return os.path.dirname(normalized)
    return normalized


def _result_metadata(data: dict, tag: str, results_root: str) -> dict:
    metadata = data.get('metadata', {})
    if metadata:
        return metadata
    return {
        'experiment_dir': _infer_experiment_dir(results_root),
        'feature_bank_dir': './processed_data',
        'feature_version': 'v1',
        'preprocessing_protocol': 'base' if tag == 'base' else 'legacy_full',
        'feature_tag': tag,
    }


def _group_key(tag: str, metadata: dict):
    return (
        tag,
        metadata.get('experiment_dir', ''),
        metadata.get('feature_version', 'v1'),
        metadata.get('preprocessing_protocol', ''),
        metadata.get('feature_bank_dir', ''),
    )


def scan_results(dataset: str, task: str = 'lp', results_root: str = DEFAULT_RESULTS_ROOT):
    """Return {(tag, experiment_dir, feature_version, protocol, feature_bank_dir): [result_dict_per_seed]}."""
    pattern = LP_PATTERN if task == 'lp' else NC_PATTERN
    folder = os.path.join(results_root, MODEL, dataset)
    if not os.path.isdir(folder):
        return {}

    grouped = defaultdict(list)
    for fname in sorted(os.listdir(folder)):
        m = pattern.match(fname)
        if not m:
            continue
        tag = m.group('tag')
        seed = int(m.group('seed'))
        fpath = os.path.join(folder, fname)
        with open(fpath) as f:
            data = json.load(f)
        metadata = _result_metadata(data, tag, results_root)
        data['_seed'] = seed
        data['_file'] = fname
        data['_metadata'] = metadata
        grouped[_group_key(tag, metadata)].append(data)

    return dict(grouped)


def summarize_metric(results_list, metric_group, metric_name):
    """Extract a metric across seeds, return (mean, std, n)."""
    vals = []
    for r in results_list:
        group = r.get(metric_group, {})
        if metric_name in group:
            vals.append(float(group[metric_name]))
    if not vals:
        return None, None, 0
    import numpy as np
    mean = np.mean(vals)
    std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
    return mean, std, len(vals)


def _sorted_group_keys(grouped):
    tag_rank = {tag: i for i, tag in enumerate(TAG_ORDER)}
    return sorted(grouped.keys(), key=lambda key: (
        tag_rank.get(key[0], len(TAG_ORDER)),
        key[2],
        key[3],
        key[1],
        key[4],
    ))


def print_lp_table(datasets, results_root):
    """Print link prediction summary table."""
    print('\n' + '=' * 120)
    print('LINK PREDICTION RESULTS')
    print('=' * 120)

    header = f'{"dataset":<12} {"setting":<16} {"version":<14} {"protocol":<12} {"experiment_dir":<34} {"runs":>4}  ' \
             f'{"test_AP":>12} {"test_AUC":>12}  ' \
             f'{"nn_test_AP":>12} {"nn_test_AUC":>12}'
    print(header)
    print('-' * 120)

    for dataset in datasets:
        grouped = scan_results(dataset, 'lp', results_root)
        for key in _sorted_group_keys(grouped):
            tag, experiment_dir, feature_version, protocol, _ = key
            results = grouped[key]
            label = TAG_DISPLAY.get(tag, tag)

            ap_m, ap_s, n = summarize_metric(results, 'test metrics', 'average_precision')
            auc_m, auc_s, _ = summarize_metric(results, 'test metrics', 'roc_auc')
            nn_ap_m, nn_ap_s, _ = summarize_metric(results, 'new node test metrics', 'average_precision')
            nn_auc_m, nn_auc_s, _ = summarize_metric(results, 'new node test metrics', 'roc_auc')

            def fmt(m, s):
                if m is None:
                    return f'{"--":>12}'
                return f'{m:.4f}±{s:.4f}'

            print(f'{dataset:<12} {label:<16} {feature_version:<14} {protocol:<12} {experiment_dir:<34} {n:>4}  '
                  f'{fmt(ap_m, ap_s):>12} {fmt(auc_m, auc_s):>12}  '
                  f'{fmt(nn_ap_m, nn_ap_s):>12} {fmt(nn_auc_m, nn_auc_s):>12}')
        print()


def print_nc_table(datasets, results_root):
    """Print node classification summary table."""
    print('\n' + '=' * 80)
    print('NODE CLASSIFICATION RESULTS')
    print('=' * 80)

    header = f'{"dataset":<12} {"setting":<16} {"version":<14} {"protocol":<12} {"experiment_dir":<34} {"runs":>4}  {"test_roc_auc":>16}'
    print(header)
    print('-' * 80)

    for dataset in datasets:
        grouped = scan_results(dataset, 'nc', results_root)
        for key in _sorted_group_keys(grouped):
            tag, experiment_dir, feature_version, protocol, _ = key
            results = grouped[key]
            label = TAG_DISPLAY.get(tag, tag)

            auc_m, auc_s, n = summarize_metric(results, 'test metrics', 'roc_auc')

            def fmt(m, s):
                if m is None:
                    return f'{"--":>16}'
                return f'{m:.4f}±{s:.4f}'

            print(f'{dataset:<12} {label:<16} {feature_version:<14} {protocol:<12} {experiment_dir:<34} {n:>4}  {fmt(auc_m, auc_s):>16}')
        print()


def save_csv(datasets, csv_path, results_root):
    """Save all results to CSV."""
    import csv
    rows = []
    for task in ['lp', 'nc']:
        for dataset in datasets:
            grouped = scan_results(dataset, task, results_root)
            for key in _sorted_group_keys(grouped):
                tag, experiment_dir, feature_version, protocol, feature_bank_dir = key
                results = grouped[key]
                label = TAG_DISPLAY.get(tag, tag)
                for r in results:
                    row = {
                        'task': task,
                        'dataset': dataset,
                        'setting': label,
                        'tag': tag,
                        'seed': r['_seed'],
                        'experiment_dir': experiment_dir,
                        'feature_version': feature_version,
                        'preprocessing_protocol': protocol,
                        'feature_bank_dir': feature_bank_dir,
                    }
                    for group_name, metrics in r.items():
                        if group_name.startswith('_') or group_name == 'metadata' or not isinstance(metrics, dict):
                            continue
                        for metric_name, value in metrics.items():
                            col = f'{group_name}/{metric_name}'
                            row[col] = value
                    rows.append(row)

    if not rows:
        print('No results found.')
        return

    all_keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} result rows to {csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Collect experiment results')
    parser.add_argument('--task', type=str, default='all',
                        choices=['lp', 'nc', 'all'])
    parser.add_argument('--results_root', type=str, default=DEFAULT_RESULTS_ROOT,
                        help='root directory containing MODEL/dataset result JSONs')
    parser.add_argument('--csv', type=str, default=None,
                        help='Optional: save to CSV file')
    args = parser.parse_args()

    datasets = ['wikipedia', 'reddit']

    if args.task in ('lp', 'all'):
        print_lp_table(datasets, args.results_root)
    if args.task in ('nc', 'all'):
        print_nc_table(datasets, args.results_root)
    if args.csv:
        save_csv(datasets, args.csv, args.results_root)
