"""
Social feature extractor (proxy version -- structural statistics).

Since Wikipedia / Reddit have no multi-relation edges, no user attributes,
and no explicit social graph, this module computes *time-aware structural
statistics* from the dynamic bipartite interaction graph as a proxy.

For each edge (u, v, t), we compute 5 statistics for src (u) and 5 for
dst (v), using ONLY edges with timestamp < t (strict causal ordering,
zero future leakage).

Per-node statistics (computed at the moment just before this edge):
  0. degree           -- log(1 + interaction_count_before_t)
  1. unique_neighbors -- log(1 + distinct_neighbor_count_before_t)
  2. recency          -- log(1 + (t - last_interaction_time)), 0 if first
  3. activity_rate    -- degree / (t - first_interaction_time + 1)
  4. repeat_ratio     -- (degree - unique_neighbors) / max(degree, 1)

Output shape: (num_edges + 1, 10)  -- [src_0..4, dst_0..4]

Implementation: single chronological scan, O(E) time, O(N) space.

This is NOT a heterogeneous social encoder.  It is a dynamic graph
structural statistics channel.  The "sidecar" concept is realized at
the feature computation level, not the model architecture level.

Usage:
    python -m features.social_extractor --dataset_name wikipedia
    python -m features.social_extractor --dataset_name reddit
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import logging
from collections import Counter, defaultdict, deque

from utils.DataLoader import get_link_prediction_data

logger = logging.getLogger(__name__)

SOCIAL_DIM = 10  # 5 per src + 5 per dst, fixed
SOCIAL_V2_DIM = 20  # 10 per src + 10 per dst
EPS = 1e-8


def _compute_node_stats(degree: int, unique_nbrs: int,
                        last_time: float, first_time: float,
                        current_time: float) -> np.ndarray:
    """Compute 5 social statistics for one node at a given timestamp."""
    stats = np.zeros(5, dtype=np.float64)
    stats[0] = np.log1p(degree)
    stats[1] = np.log1p(unique_nbrs)
    if degree > 0:
        stats[2] = np.log1p(current_time - last_time)
        time_span = current_time - first_time + 1.0
        stats[3] = degree / time_span
        stats[4] = (degree - unique_nbrs) / max(degree, 1)
    return stats


def build_social_features(
    dataset_name: str,
    processed_dir: str = './processed_data',
    version: str = 'v1',
) -> np.ndarray:
    """
    Build time-aware per-edge social structural features.

    :param dataset_name: e.g. 'wikipedia', 'reddit'
    :param processed_dir: root of processed_data
    :param version: version tag for output filename
    :return: social_features ndarray, shape (num_edges + 1, SOCIAL_DIM)
    """
    csv_path = os.path.join(
        processed_dir, dataset_name, f'ml_{dataset_name}.csv')
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'Edge list not found at {csv_path}.')

    graph_df = pd.read_csv(csv_path)
    num_edges = len(graph_df)
    logger.info(f'Loaded {num_edges} edges from {csv_path}')

    src_ids = graph_df.u.values
    dst_ids = graph_df.i.values
    timestamps = graph_df.ts.values
    edge_ids = graph_df.idx.values  # 1-indexed

    # Running state per node
    node_degree = defaultdict(int)
    node_unique_neighbors = defaultdict(set)
    node_last_time = {}
    node_first_time = {}

    # Output array: row 0 is zero padding (edge_id=0 unused)
    social_features = np.zeros((num_edges + 1, SOCIAL_DIM), dtype=np.float32)

    for i in range(num_edges):
        u = src_ids[i]
        v = dst_ids[i]
        t = float(timestamps[i])
        eid = edge_ids[i]

        # Read current state BEFORE updating (strict causal)
        src_stats = _compute_node_stats(
            degree=node_degree[u],
            unique_nbrs=len(node_unique_neighbors[u]),
            last_time=node_last_time.get(u, t),
            first_time=node_first_time.get(u, t),
            current_time=t,
        )
        dst_stats = _compute_node_stats(
            degree=node_degree[v],
            unique_nbrs=len(node_unique_neighbors[v]),
            last_time=node_last_time.get(v, t),
            first_time=node_first_time.get(v, t),
            current_time=t,
        )

        social_features[eid, :5] = src_stats
        social_features[eid, 5:] = dst_stats

        # Update running state
        node_degree[u] += 1
        node_degree[v] += 1
        node_unique_neighbors[u].add(v)
        node_unique_neighbors[v].add(u)
        node_last_time[u] = t
        node_last_time[v] = t
        if u not in node_first_time:
            node_first_time[u] = t
        if v not in node_first_time:
            node_first_time[v] = t

        if (i + 1) % 100000 == 0:
            logger.info(f'  processed {i + 1}/{num_edges} edges')

    # Z-score normalize each column (across all actual edges)
    actual = social_features[1:]  # skip padding row
    col_mean = actual.mean(axis=0)
    col_std = actual.std(axis=0)
    col_std[col_std < 1e-10] = 1.0  # avoid division by zero
    social_features[1:] = ((actual - col_mean) / col_std).astype(np.float32)

    out_path = os.path.join(
        processed_dir, dataset_name, f'social_{version}.npy')
    np.save(out_path, social_features)
    logger.info(f'Saved social features to {out_path}, '
                f'shape={social_features.shape}')

    return social_features


class _WindowState:
    """Per-node sliding interaction window with unique-neighbor counts."""

    def __init__(self):
        self.events = deque()
        self.neighbor_counts = Counter()

    def evict_before(self, cutoff: float):
        while self.events and self.events[0][1] < cutoff:
            neighbor, _ = self.events.popleft()
            self.neighbor_counts[neighbor] -= 1
            if self.neighbor_counts[neighbor] == 0:
                del self.neighbor_counts[neighbor]

    def add(self, neighbor: int, timestamp: float):
        self.events.append((neighbor, timestamp))
        self.neighbor_counts[neighbor] += 1

    @property
    def count(self) -> int:
        return len(self.events)

    @property
    def unique_count(self) -> int:
        return len(self.neighbor_counts)


class _NodeHistory:
    """Running strict-causal history for one endpoint."""

    def __init__(self, short_window: float, long_window: float, tau: float):
        self.degree = 0
        self.unique_neighbors = set()
        self.last_time = None
        self.first_time = None
        self.short_window = short_window
        self.long_window = long_window
        self.tau = tau
        self.short_history = _WindowState()
        self.long_history = _WindowState()
        self.decayed_degree = 0.0
        self.decayed_time = None
        self.last_inter_event_gaps = deque(maxlen=20)

    def base_stats(self, current_time: float) -> np.ndarray:
        return _compute_node_stats(
            degree=self.degree,
            unique_nbrs=len(self.unique_neighbors),
            last_time=current_time if self.last_time is None else self.last_time,
            first_time=current_time if self.first_time is None else self.first_time,
            current_time=current_time,
        )

    def v2_stats(self, current_time: float) -> np.ndarray:
        self.short_history.evict_before(current_time - self.short_window)
        self.long_history.evict_before(current_time - self.long_window)

        short_count_raw = self.short_history.count
        long_count_raw = self.long_history.count
        short_unique = self.short_history.unique_count
        decayed_degree = self._decayed_at(current_time)

        if len(self.last_inter_event_gaps) >= 2:
            gaps = np.asarray(self.last_inter_event_gaps, dtype=np.float64)
            burstiness = float(gaps.std() / (gaps.mean() + EPS))
        else:
            burstiness = 0.0

        extra = np.zeros(5, dtype=np.float64)
        extra[0] = np.log1p(short_count_raw)
        extra[1] = np.log1p(long_count_raw)
        extra[2] = (short_count_raw - short_unique) / max(short_count_raw, 1)
        extra[3] = decayed_degree
        extra[4] = burstiness
        return np.concatenate([self.base_stats(current_time), extra])

    def update(self, neighbor: int, current_time: float):
        if self.degree == 0:
            self.first_time = current_time
        elif current_time - self.last_time > 0:
            self.last_inter_event_gaps.append(current_time - self.last_time)

        self.decayed_degree = self._decayed_at(current_time) + 1.0
        self.decayed_time = current_time
        self.degree += 1
        self.unique_neighbors.add(neighbor)
        self.last_time = current_time
        self.short_history.add(neighbor, current_time)
        self.long_history.add(neighbor, current_time)

    def _decayed_at(self, current_time: float) -> float:
        if self.decayed_time is None:
            return 0.0
        return float(self.decayed_degree * np.exp(-(current_time - self.decayed_time) / self.tau))


def _load_graph(processed_dir: str, dataset_name: str) -> pd.DataFrame:
    csv_path = os.path.join(processed_dir, dataset_name, f'ml_{dataset_name}.csv')
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'Edge list not found at {csv_path}.')
    return pd.read_csv(csv_path)


def _build_raw_trainfit_features(graph_df: pd.DataFrame, variant: str,
                                 train_time_span: float) -> np.ndarray:
    src_ids = graph_df.u.values
    dst_ids = graph_df.i.values
    timestamps = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    order = np.argsort(timestamps, kind='mergesort')

    dim = SOCIAL_DIM if variant == 'v1_trainfit' else SOCIAL_V2_DIM
    features = np.zeros((len(graph_df) + 1, dim), dtype=np.float32)

    short_window = 0.01 * train_time_span
    long_window = 0.10 * train_time_span
    tau = 0.05 * train_time_span
    histories = defaultdict(lambda: _NodeHistory(short_window, long_window, tau))

    pos = 0
    while pos < len(order):
        current_time = float(timestamps[order[pos]])
        group_end = pos + 1
        while group_end < len(order) and timestamps[order[group_end]] == current_time:
            group_end += 1
        group = order[pos:group_end]

        for row_idx in group:
            u = src_ids[row_idx]
            v = dst_ids[row_idx]
            eid = edge_ids[row_idx]
            if variant == 'v1_trainfit':
                features[eid, :5] = histories[u].base_stats(current_time)
                features[eid, 5:] = histories[v].base_stats(current_time)
            else:
                features[eid, :10] = histories[u].v2_stats(current_time)
                features[eid, 10:] = histories[v].v2_stats(current_time)

        for row_idx in group:
            u = src_ids[row_idx]
            v = dst_ids[row_idx]
            histories[u].update(v, current_time)
            histories[v].update(u, current_time)

        if group_end % 100000 == 0:
            logger.info(f'  processed {group_end}/{len(order)} edges')
        pos = group_end

    return features


def _normalize_with_train_edges(features: np.ndarray, train_edge_ids: np.ndarray):
    train_values = features[train_edge_ids]
    col_mean = train_values.mean(axis=0)
    col_std = train_values.std(axis=0)
    col_std[col_std < 1e-10] = 1.0
    normalized = features.copy()
    normalized[1:] = ((features[1:] - col_mean) / col_std).astype(np.float32)
    normalized[0] = 0.0
    return normalized, col_mean, col_std


def check_social_features(features: np.ndarray, train_edge_ids: np.ndarray,
                          expected_dim: int) -> dict:
    train_values = features[train_edge_ids]
    return {
        'shape': list(features.shape),
        'expected_dim': expected_dim,
        'row0_all_zero': bool(np.allclose(features[0], 0.0)),
        'finite': bool(np.isfinite(features).all()),
        'train_mean_abs_max': float(np.max(np.abs(train_values.mean(axis=0)))),
        'train_std_min': float(train_values.std(axis=0).min()),
        'train_std_max': float(train_values.std(axis=0).max()),
    }


def _default_manifest_dir(output_dir: str) -> str:
    output_dir = os.path.abspath(output_dir)
    if os.path.basename(output_dir) == 'feature_bank':
        return os.path.join(os.path.dirname(output_dir), 'manifests')
    return os.path.join(output_dir, 'manifests')


def build_trainfit_social_features(
    dataset_name: str,
    processed_dir: str = './processed_data',
    output_dir: str = './processed_data',
    version: str = 'v1_trainfit',
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    manifest_dir: str = None,
) -> np.ndarray:
    """
    Build Social v1/v2 with train-edge-only z-score normalization.

    The normalization fit uses get_link_prediction_data(...).train_data.edge_ids,
    preserving the inductive observed_edges_mask used by LP training.
    """
    if version not in ('v1_trainfit', 'v2_trainfit'):
        raise ValueError('version must be v1_trainfit or v2_trainfit')

    graph_df = _load_graph(processed_dir=processed_dir, dataset_name=dataset_name)
    val_time = float(np.quantile(graph_df.ts.values, 1 - val_ratio - test_ratio))
    min_timestamp = float(graph_df.ts.values.min())
    train_time_span = val_time - min_timestamp + EPS
    logger.info(f'Loaded {len(graph_df)} edges for {dataset_name}; train_time_span={train_time_span:.6f}')

    _, _, _, train_data, _, _, _, _ = get_link_prediction_data(
        dataset_name=dataset_name, val_ratio=val_ratio, test_ratio=test_ratio,
        processed_dir=processed_dir)
    train_edge_ids = train_data.edge_ids.astype(np.longlong)

    raw_features = _build_raw_trainfit_features(
        graph_df=graph_df, variant=version, train_time_span=train_time_span)
    social_features, mean, std = _normalize_with_train_edges(raw_features, train_edge_ids)

    out_dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(out_dataset_dir, exist_ok=True)
    out_path = os.path.join(out_dataset_dir, f'social_{version}.npy')
    np.save(out_path, social_features)

    expected_dim = SOCIAL_DIM if version == 'v1_trainfit' else SOCIAL_V2_DIM
    sanity = check_social_features(social_features, train_edge_ids, expected_dim)
    logger.info(f'Saved social features to {out_path}, shape={social_features.shape}')
    logger.info(f'Sanity check: {sanity}')

    manifest_dir = manifest_dir or _default_manifest_dir(output_dir)
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, f'{dataset_name}_social_{version}.json')
    manifest = {
        'dataset_name': dataset_name,
        'feature_version': version,
        'preprocessing_protocol': 'trainfit',
        'feature_path': out_path,
        'processed_dir': processed_dir,
        'train_edge_count': int(len(train_edge_ids)),
        'train_time_span': train_time_span,
        'normalization': {
            'scope': 'lp_train_edges',
            'mean': mean.tolist(),
            'std': std.tolist(),
        },
        'sanity': sanity,
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
    logger.info(f'Saved manifest to {manifest_path}')

    return social_features


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    parser = argparse.ArgumentParser(
        'Build time-aware proxy social structural features')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['wikipedia', 'reddit'])
    parser.add_argument('--processed_dir', type=str, default='./processed_data')
    parser.add_argument('--output_dir', '--feature_bank_dir', dest='output_dir',
                        type=str, default=None,
                        help='root directory for generated feature files')
    parser.add_argument('--version', type=str, default='v1')
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--manifest_dir', type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.processed_dir
    if args.version in ('v1_trainfit', 'v2_trainfit'):
        build_trainfit_social_features(
            dataset_name=args.dataset_name,
            processed_dir=args.processed_dir,
            output_dir=output_dir,
            version=args.version,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            manifest_dir=args.manifest_dir,
        )
    else:
        build_social_features(
            dataset_name=args.dataset_name,
            processed_dir=args.processed_dir,
            version=args.version,
        )
