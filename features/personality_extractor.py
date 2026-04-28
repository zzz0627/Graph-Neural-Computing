"""
Personality feature extractor (proxy version).

Constructs a per-node personality proxy by:
  1. For each node, collecting the LIWC edge features of all its interactions
  2. Averaging them to form a stable per-node linguistic profile
  3. Applying PCA to reduce to `personality_dim` dimensions (default 5,
     loosely corresponding to the Big Five factor structure)

This is a *psycholinguistic proxy variable*, not a true Big Five
measurement.  Pennebaker & King (1999) and Yarkoni (2010) have shown
significant correlations between LIWC dimensions and Big Five traits,
which gives this proxy a reasonable empirical basis.

The key distinction from Style features:
  - Style is per-edge (each interaction's linguistic surface)
  - Personality is per-node (aggregated stable behavioral pattern)

Usage:
    python -m features.personality_extractor --dataset_name wikipedia
    python -m features.personality_extractor --dataset_name reddit
"""

import argparse
import os
import numpy as np
import pandas as pd
import logging
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

DEFAULT_PERSONALITY_DIM = 5


def build_personality_features(
    dataset_name: str,
    processed_dir: str = './processed_data',
    personality_dim: int = DEFAULT_PERSONALITY_DIM,
    version: str = 'v1',
) -> np.ndarray:
    """
    Build PCA-based proxy personality features (per-node).

    :param dataset_name: e.g. 'wikipedia', 'reddit'
    :param processed_dir: root of processed_data
    :param personality_dim: number of PCA components (default 5 for Big Five analogy)
    :param version: version tag for output filename
    :return: personality_features ndarray, shape (num_nodes + 1, personality_dim)
    """
    edge_feat_path = os.path.join(
        processed_dir, dataset_name, f'ml_{dataset_name}.npy')
    csv_path = os.path.join(
        processed_dir, dataset_name, f'ml_{dataset_name}.csv')

    if not os.path.isfile(edge_feat_path):
        raise FileNotFoundError(
            f'Edge features not found at {edge_feat_path}. '
            f'Run preprocess_data.py first.')
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'Edge list not found at {csv_path}.')

    edge_features = np.load(edge_feat_path)  # (num_edges+1, 172)
    graph_df = pd.read_csv(csv_path)
    logger.info(f'Loaded edge features: {edge_features.shape}, '
                f'edges in CSV: {len(graph_df)}')

    src_ids = graph_df.u.values   # node IDs start from 1
    dst_ids = graph_df.i.values
    edge_ids = graph_df.idx.values  # edge IDs start from 1

    max_node_id = max(src_ids.max(), dst_ids.max())
    num_nodes = max_node_id + 1  # +1 because IDs start from 1, index 0 is padding

    # Accumulate sum and count for each node (both as src and dst)
    node_feat_sum = np.zeros((num_nodes, 172), dtype=np.float64)
    node_feat_count = np.zeros(num_nodes, dtype=np.int64)

    for node_id, eid in zip(src_ids, edge_ids):
        node_feat_sum[node_id] += edge_features[eid]
        node_feat_count[node_id] += 1

    for node_id, eid in zip(dst_ids, edge_ids):
        node_feat_sum[node_id] += edge_features[eid]
        node_feat_count[node_id] += 1

    # Compute mean (avoid division by zero for padding node 0)
    active_mask = node_feat_count > 0
    node_feat_mean = np.zeros_like(node_feat_sum)
    node_feat_mean[active_mask] = (
        node_feat_sum[active_mask] / node_feat_count[active_mask, np.newaxis])

    active_nodes = node_feat_mean[active_mask]  # (num_active_nodes, 172)
    logger.info(f'Active nodes with interactions: {active_nodes.shape[0]}')

    # PCA on the per-node mean features
    nonzero_var = np.sum(active_nodes.var(axis=0) > 1e-10)
    effective_dim = min(personality_dim, nonzero_var, active_nodes.shape[0])
    if effective_dim < personality_dim:
        logger.warning(
            f'Requested personality_dim={personality_dim} but only '
            f'{effective_dim} non-trivial components available.')

    pca = PCA(n_components=effective_dim, random_state=42)
    active_transformed = pca.fit_transform(active_nodes)

    explained = pca.explained_variance_ratio_.sum()
    logger.info(f'PCA: {effective_dim} components explain '
                f'{explained:.4f} of variance')

    # Build the full array (num_nodes, personality_dim)
    personality_all = np.zeros((num_nodes, personality_dim), dtype=np.float32)
    if effective_dim < personality_dim:
        active_padded = np.zeros((active_transformed.shape[0], personality_dim),
                                 dtype=np.float32)
        active_padded[:, :effective_dim] = active_transformed
    else:
        active_padded = active_transformed.astype(np.float32)
    personality_all[active_mask] = active_padded

    assert personality_all.shape[0] == num_nodes

    # Save
    out_path = os.path.join(
        processed_dir, dataset_name, f'personality_{version}.npy')
    np.save(out_path, personality_all)
    logger.info(f'Saved personality features to {out_path}, '
                f'shape={personality_all.shape}')

    return personality_all


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    parser = argparse.ArgumentParser(
        'Build proxy personality features via per-node LIWC aggregation + PCA')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['wikipedia', 'reddit'])
    parser.add_argument('--processed_dir', type=str, default='./processed_data')
    parser.add_argument('--personality_dim', type=int,
                        default=DEFAULT_PERSONALITY_DIM)
    parser.add_argument('--version', type=str, default='v1')
    args = parser.parse_args()

    build_personality_features(
        dataset_name=args.dataset_name,
        processed_dir=args.processed_dir,
        personality_dim=args.personality_dim,
        version=args.version,
    )
