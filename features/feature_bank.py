"""
FeatureBank: unified cache for pre-computed features.

Supports two granularities:
  - node-level  (e.g. personality, social)   shape [num_nodes+1, feat_dim]
  - edge-level  (e.g. style, topic)          shape [num_edges+1, feat_dim]

Files are loaded lazily from:
  {root_dir}/{dataset_name}/{feature_type}_{version}.npy

When a feature file does not exist the bank returns None, so callers
can gracefully skip features that have not been pre-computed yet.
"""

import os
import logging
import numpy as np
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class FeatureBank:
    """Pre-computed feature store with lazy loading and optional caching."""

    VALID_TYPES = ('style', 'personality', 'topic', 'social')
    GRANULARITY = {
        'style': 'edge',
        'personality': 'node',
        'topic': 'edge',
        'social': 'edge',  # time-aware per-edge structural statistics
    }

    def __init__(self, root_dir: str, dataset_name: str, version: str = 'v1'):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.version = version
        self._cache: Dict[str, Optional[np.ndarray]] = {}

    def _feature_path(self, feature_type: str) -> str:
        return os.path.join(
            self.root_dir, self.dataset_name,
            f'{feature_type}_{self.version}.npy'
        )

    def load(self, feature_type: str) -> Optional[np.ndarray]:
        """Load a feature array.  Returns None if file does not exist."""
        if feature_type not in self.VALID_TYPES:
            raise ValueError(f'Unknown feature type: {feature_type}. '
                             f'Must be one of {self.VALID_TYPES}')
        if feature_type in self._cache:
            return self._cache[feature_type]

        path = self._feature_path(feature_type)
        if os.path.isfile(path):
            data = np.load(path)
            logger.info(f'FeatureBank: loaded {feature_type} from {path}, '
                        f'shape={data.shape}')
            self._cache[feature_type] = data
            return data

        logger.info(f'FeatureBank: {path} not found, {feature_type} '
                    f'features unavailable')
        self._cache[feature_type] = None
        return None

    def has(self, feature_type: str) -> bool:
        """Check whether a feature file exists (without fully loading it)."""
        if feature_type in self._cache:
            return self._cache[feature_type] is not None
        return os.path.isfile(self._feature_path(feature_type))

    def get_dim(self, feature_type: str) -> int:
        """Return feature dimension, or 0 if unavailable."""
        data = self.load(feature_type)
        if data is None:
            return 0
        return data.shape[1]

    def granularity(self, feature_type: str) -> str:
        """Return 'node' or 'edge'."""
        return self.GRANULARITY[feature_type]

    def summary(self) -> Dict[str, dict]:
        """Return a summary dict of all loaded / available features."""
        info = {}
        for ft in self.VALID_TYPES:
            path = self._feature_path(ft)
            exists = os.path.isfile(path)
            loaded = ft in self._cache and self._cache[ft] is not None
            shape = self._cache[ft].shape if loaded else None
            info[ft] = {'exists': exists, 'loaded': loaded, 'shape': shape}
        return info
