"""
Analyze the Social v1_trainfit vs v2_trainfit protocol.

This script reads completed experiment result directories and generated feature
banks. It does not launch training.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.DataLoader import get_link_prediction_data


MODEL = 'DyGFormer'
LP_PATTERN = re.compile(rf'^{MODEL}_(?P<tag>[A-Za-z_]+)_seed(?P<seed>\d+)\.json$')
NC_PATTERN = re.compile(rf'^node_classification_{MODEL}_(?P<tag>[A-Za-z_]+)_seed(?P<seed>\d+)\.json$')


def _load_results(results_root: str, dataset: str, task: str, tag: str):
    pattern = LP_PATTERN if task == 'lp' else NC_PATTERN
    folder = os.path.join(results_root, MODEL, dataset)
    by_seed = {}
    if not os.path.isdir(folder):
        return by_seed
    for fname in sorted(os.listdir(folder)):
        match = pattern.match(fname)
        if not match or match.group('tag') != tag:
            continue
        seed = int(match.group('seed'))
        with open(os.path.join(folder, fname)) as f:
            by_seed[seed] = json.load(f)
    return by_seed


def _metric(result: dict, group: str, metric_name: str):
    value = result.get(group, {}).get(metric_name)
    if value is None:
        return None
    return float(value)


def _paired_stats(deltas):
    values = np.asarray(deltas, dtype=np.float64)
    mean = float(values.mean()) if len(values) else np.nan
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    sem = std / np.sqrt(len(values)) if len(values) > 1 else 0.0
    p_value = np.nan
    ci95 = (np.nan, np.nan)
    if len(values) > 1:
        try:
            from scipy import stats
            p_value = float(stats.ttest_1samp(values, 0.0).pvalue)
            tcrit = float(stats.t.ppf(0.975, len(values) - 1))
        except Exception:
            tcrit = 1.96
        ci95 = (float(mean - tcrit * sem), float(mean + tcrit * sem))
    return mean, std, ci95, p_value


def print_paired_delta(dataset: str, task: str, v1_results: dict,
                       v2_results: dict, metric_group: str, metric_name: str):
    seeds = sorted(set(v1_results) & set(v2_results))
    deltas = []
    print(f'\n{task.upper()} paired delta: dataset={dataset}, metric={metric_group}/{metric_name}')
    print(f'{"seed":>4} {"v1":>10} {"v2":>10} {"delta":>10}')
    print('-' * 40)
    for seed in seeds:
        v1_value = _metric(v1_results[seed], metric_group, metric_name)
        v2_value = _metric(v2_results[seed], metric_group, metric_name)
        if v1_value is None or v2_value is None:
            continue
        delta = v2_value - v1_value
        deltas.append(delta)
        print(f'{seed:>4} {v1_value:>10.4f} {v2_value:>10.4f} {delta:>10.4f}')

    if not deltas:
        print('No paired seeds found.')
        return
    mean, std, ci95, p_value = _paired_stats(deltas)
    positives = int(np.sum(np.asarray(deltas) > 0))
    print(f'mean_delta={mean:.6f} std={std:.6f} positives={positives}/{len(deltas)} '
          f'ci95=[{ci95[0]:.6f}, {ci95[1]:.6f}] p={p_value:.6g}')


def _corrcoef(x, y):
    if x.std() < 1e-12 or y.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _univariate_auc(values, labels):
    if len(np.unique(labels)) < 2:
        return np.nan
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels, values))
        return max(auc, 1.0 - auc)
    except Exception:
        return np.nan


def print_feature_diagnostics(dataset: str, processed_dir: str,
                              v1_feature_bank_root: str,
                              v2_feature_bank_root: str,
                              val_ratio: float, test_ratio: float):
    v1_path = os.path.join(v1_feature_bank_root, dataset, 'social_v1_trainfit.npy')
    v2_path = os.path.join(v2_feature_bank_root, dataset, 'social_v2_trainfit.npy')
    if not os.path.isfile(v1_path) or not os.path.isfile(v2_path):
        print(f'\nFeature diagnostics skipped for {dataset}: missing {v1_path} or {v2_path}')
        return

    v1 = np.load(v1_path)
    v2 = np.load(v2_path)
    graph_df = pd.read_csv(os.path.join(processed_dir, dataset, f'ml_{dataset}.csv'))
    _, _, _, train_data, _, _, _, _ = get_link_prediction_data(
        dataset_name=dataset, val_ratio=val_ratio, test_ratio=test_ratio,
        processed_dir=processed_dir)
    edge_ids = train_data.edge_ids.astype(np.longlong)
    labels = graph_df.set_index('idx').loc[edge_ids, 'label'].values

    base_names = ['log_degree', 'log_unique_neighbors', 'log_recency',
                  'activity_rate', 'repeat_ratio']
    extra_names = ['short_count', 'long_count', 'short_repeat_ratio',
                   'decayed_degree', 'burstiness']

    print(f'\nV2 feature diagnostics: dataset={dataset}')
    print(f'{"side":<4} {"extra_feature":<20} {"max_abs_corr_to_v1":>18} {"label_auc":>10}')
    print('-' * 60)
    sides = [
        ('src', v1[edge_ids, 0:5], v2[edge_ids, 5:10]),
        ('dst', v1[edge_ids, 5:10], v2[edge_ids, 15:20]),
    ]
    for side, base_values, extra_values in sides:
        for extra_idx, extra_name in enumerate(extra_names):
            corrs = [
                abs(_corrcoef(extra_values[:, extra_idx], base_values[:, base_idx]))
                for base_idx, _ in enumerate(base_names)
            ]
            finite_corrs = [c for c in corrs if np.isfinite(c)]
            max_corr = max(finite_corrs) if finite_corrs else np.nan
            auc = _univariate_auc(extra_values[:, extra_idx], labels)
            print(f'{side:<4} {extra_name:<20} {max_corr:>18.4f} {auc:>10.4f}')


def main():
    parser = argparse.ArgumentParser('Analyze Social v2 trainfit experiments')
    parser.add_argument('--base_results_root', type=str, default=None,
                        help='optional Base saved_results root for reference summaries')
    parser.add_argument('--v1_results_root', type=str, required=True)
    parser.add_argument('--v2_results_root', type=str, required=True)
    parser.add_argument('--v1_feature_bank_root', type=str, default=None)
    parser.add_argument('--v2_feature_bank_root', type=str, default=None)
    parser.add_argument('--processed_dir', type=str, default='./processed_data')
    parser.add_argument('--datasets', nargs='+', default=['wikipedia', 'reddit'])
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--skip_features', action='store_true', default=False)
    args = parser.parse_args()

    for dataset in args.datasets:
        v1_lp = _load_results(args.v1_results_root, dataset, 'lp', 'So')
        v2_lp = _load_results(args.v2_results_root, dataset, 'lp', 'So')
        print_paired_delta(dataset, 'lp', v1_lp, v2_lp,
                           'test metrics', 'average_precision')
        print_paired_delta(dataset, 'lp', v1_lp, v2_lp,
                           'new node test metrics', 'average_precision')

        v1_nc = _load_results(args.v1_results_root, dataset, 'nc', 'So')
        v2_nc = _load_results(args.v2_results_root, dataset, 'nc', 'So')
        if v1_nc or v2_nc:
            print_paired_delta(dataset, 'nc', v1_nc, v2_nc,
                               'test metrics', 'roc_auc')

        if args.base_results_root:
            base_lp = _load_results(args.base_results_root, dataset, 'lp', 'base')
            if base_lp:
                print(f'\nBase LP reference: dataset={dataset}, seeds={sorted(base_lp.keys())}')

        if not args.skip_features and args.v1_feature_bank_root and args.v2_feature_bank_root:
            print_feature_diagnostics(
                dataset=dataset,
                processed_dir=args.processed_dir,
                v1_feature_bank_root=args.v1_feature_bank_root,
                v2_feature_bank_root=args.v2_feature_bank_root,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
            )


if __name__ == '__main__':
    main()
