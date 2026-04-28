"""
Style feature extractor (proxy version).

Since no raw text is available in the Wikipedia / Reddit datasets,
this module constructs a proxy style representation by applying PCA
to the 172-d LIWC edge features.  The leading principal components
of LIWC features correspond to broad stylistic dimensions such as
formality, emotionality, and cognitive complexity (Pennebaker et al.).

The resulting style features are saved as a separate .npy file so that
DyGFormer can consume them as an additional edge channel.

Usage:
    python -m features.style_extractor --dataset_name wikipedia
    python -m features.style_extractor --dataset_name reddit
"""

import argparse
import os
import numpy as np
import logging
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

DEFAULT_STYLE_DIM = 32


def build_style_features(
    dataset_name: str,
    processed_dir: str = './processed_data',
    style_dim: int = DEFAULT_STYLE_DIM,
    version: str = 'v1',
) -> np.ndarray:
    """
    Build PCA-based proxy style features from edge features.

    :param dataset_name: e.g. 'wikipedia', 'reddit'
    :param processed_dir: root of processed_data
    :param style_dim: number of PCA components
    :param version: version tag for output filename
    :return: style_features ndarray, shape (num_edges + 1, style_dim)
    """
    edge_feat_path = os.path.join(
        processed_dir, dataset_name, f'ml_{dataset_name}.npy')
    if not os.path.isfile(edge_feat_path):
        raise FileNotFoundError(
            f'Edge features not found at {edge_feat_path}. '
            f'Run preprocess_data.py first.')

    edge_features = np.load(edge_feat_path)  # shape (num_edges+1, 172)
    logger.info(f'Loaded edge features: {edge_features.shape}')

    # Row 0 is the padding row (all zeros, edge_id starts from 1).
    actual_features = edge_features[1:]  # (num_edges, 172)

    # Clamp style_dim to the number of non-zero-variance features
    nonzero_var = np.sum(actual_features.var(axis=0) > 1e-10)
    effective_dim = min(style_dim, nonzero_var, actual_features.shape[1])
    if effective_dim < style_dim:
        logger.warning(
            f'Requested style_dim={style_dim} but only {effective_dim} '
            f'non-trivial components available. Using {effective_dim}.')

    pca = PCA(n_components=effective_dim, random_state=42)
    style_actual = pca.fit_transform(actual_features)  # (num_edges, effective_dim)

    explained = pca.explained_variance_ratio_.sum()
    logger.info(f'PCA: {effective_dim} components explain '
                f'{explained:.4f} of variance')

    # Pad to style_dim if effective_dim < style_dim
    if effective_dim < style_dim:
        pad = np.zeros((style_actual.shape[0], style_dim - effective_dim))
        style_actual = np.concatenate([style_actual, pad], axis=1)

    # Prepend the zero row to match edge_id indexing convention
    zero_row = np.zeros((1, style_dim), dtype=np.float32)
    style_features = np.vstack([zero_row, style_actual.astype(np.float32)])

    assert style_features.shape[0] == edge_features.shape[0], \
        'Style features must have the same number of rows as edge features'

    # Save
    out_path = os.path.join(
        processed_dir, dataset_name, f'style_{version}.npy')
    np.save(out_path, style_features)
    logger.info(f'Saved style features to {out_path}, shape={style_features.shape}')

    return style_features


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    parser = argparse.ArgumentParser('Build proxy style features via PCA')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['wikipedia', 'reddit'])
    parser.add_argument('--processed_dir', type=str, default='./processed_data')
    parser.add_argument('--style_dim', type=int, default=DEFAULT_STYLE_DIM)
    parser.add_argument('--version', type=str, default='v1')
    args = parser.parse_args()

    build_style_features(
        dataset_name=args.dataset_name,
        processed_dir=args.processed_dir,
        style_dim=args.style_dim,
        version=args.version,
    )
