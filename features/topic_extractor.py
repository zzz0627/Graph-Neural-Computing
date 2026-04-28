"""
Topic feature extractor (proxy version).

Constructs per-edge topic distributions via Non-negative Matrix
Factorization (NMF) on LIWC edge features.  NMF is chosen over PCA
because:
  1. Non-negativity yields interpretable "topic proportions"
  2. Parts-based decomposition parallels how LDA discovers topics
     from word counts (NMF on document-term matrix)
  3. Sparsity: each edge activates a few topics strongly

This is distinct from Style (PCA), which captures *how* text is
written via orthogonal variance axes.  Topic (NMF) captures *what
content categories* appear, via additive non-negative components.

Since LIWC features can be negative (standardized), we shift each
column to [0, inf) before NMF.

Information leakage note:
  NMF is fit on ALL edges (including val/test), consistent with how
  Style uses PCA on all edges.  Each edge's topic vector depends only
  on its own features.  No temporal or label information leaks.

Usage:
    python -m features.topic_extractor --dataset_name wikipedia
    python -m features.topic_extractor --dataset_name reddit
"""

import argparse
import os
import numpy as np
import logging
from sklearn.decomposition import NMF

logger = logging.getLogger(__name__)

DEFAULT_TOPIC_DIM = 10


def build_topic_features(
    dataset_name: str,
    processed_dir: str = './processed_data',
    topic_dim: int = DEFAULT_TOPIC_DIM,
    version: str = 'v1',
) -> np.ndarray:
    """
    Build NMF-based proxy topic features from edge features.

    :param dataset_name: e.g. 'wikipedia', 'reddit'
    :param processed_dir: root of processed_data
    :param topic_dim: number of NMF components (latent topics)
    :param version: version tag for output filename
    :return: topic_features ndarray, shape (num_edges + 1, topic_dim)
    """
    edge_feat_path = os.path.join(
        processed_dir, dataset_name, f'ml_{dataset_name}.npy')
    if not os.path.isfile(edge_feat_path):
        raise FileNotFoundError(
            f'Edge features not found at {edge_feat_path}. '
            f'Run preprocess_data.py first.')

    edge_features = np.load(edge_feat_path)  # (num_edges+1, 172)
    logger.info(f'Loaded edge features: {edge_features.shape}')

    actual_features = edge_features[1:]  # skip padding row 0

    # Identify columns with non-zero variance
    col_var = actual_features.var(axis=0)
    active_cols = col_var > 1e-10
    features_active = actual_features[:, active_cols]
    logger.info(f'Active columns: {active_cols.sum()} / {actual_features.shape[1]}')

    # Shift each column so minimum is 0 (NMF requires non-negative input)
    col_min = features_active.min(axis=0)
    features_shifted = features_active - col_min  # now all >= 0

    effective_dim = min(topic_dim, features_shifted.shape[1],
                        features_shifted.shape[0])
    if effective_dim < topic_dim:
        logger.warning(
            f'Requested topic_dim={topic_dim} but only {effective_dim} '
            f'components feasible. Using {effective_dim}.')

    nmf = NMF(n_components=effective_dim, init='nndsvda',
              random_state=42, max_iter=300)
    topic_actual = nmf.fit_transform(features_shifted)  # (num_edges, effective_dim)

    reconstruction_err = nmf.reconstruction_err_
    logger.info(f'NMF: {effective_dim} topics, '
                f'reconstruction error = {reconstruction_err:.4f}')

    # Row-normalize to get topic proportions that sum to ~1
    row_sums = topic_actual.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)  # avoid division by zero
    topic_actual = topic_actual / row_sums

    # Pad to topic_dim if effective_dim < topic_dim
    if effective_dim < topic_dim:
        pad = np.zeros((topic_actual.shape[0], topic_dim - effective_dim))
        topic_actual = np.concatenate([topic_actual, pad], axis=1)

    # Prepend zero row for edge_id=0 (padding convention)
    zero_row = np.zeros((1, topic_dim), dtype=np.float32)
    topic_features = np.vstack([zero_row, topic_actual.astype(np.float32)])

    assert topic_features.shape[0] == edge_features.shape[0]

    out_path = os.path.join(
        processed_dir, dataset_name, f'topic_{version}.npy')
    np.save(out_path, topic_features)
    logger.info(f'Saved topic features to {out_path}, '
                f'shape={topic_features.shape}')

    return topic_features


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    parser = argparse.ArgumentParser(
        'Build proxy topic features via NMF on LIWC edge features')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['wikipedia', 'reddit'])
    parser.add_argument('--processed_dir', type=str, default='./processed_data')
    parser.add_argument('--topic_dim', type=int, default=DEFAULT_TOPIC_DIM)
    parser.add_argument('--version', type=str, default='v1')
    args = parser.parse_args()

    build_topic_features(
        dataset_name=args.dataset_name,
        processed_dir=args.processed_dir,
        topic_dim=args.topic_dim,
        version=args.version,
    )
