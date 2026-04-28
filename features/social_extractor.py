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
import os
import numpy as np
import pandas as pd
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

SOCIAL_DIM = 10  # 5 per src + 5 per dst, fixed


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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    parser = argparse.ArgumentParser(
        'Build time-aware proxy social structural features')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['wikipedia', 'reddit'])
    parser.add_argument('--processed_dir', type=str, default='./processed_data')
    parser.add_argument('--version', type=str, default='v1')
    args = parser.parse_args()

    build_social_features(
        dataset_name=args.dataset_name,
        processed_dir=args.processed_dir,
        version=args.version,
    )
