"""
FeatureAssembler: top-level coordinator that bridges CLI args, FeatureBank,
and the model.

Usage in training scripts:

    assembler = FeatureAssembler(args)
    assembler.log_status(logger)

    # In the DyGFormer construction phase (Phase 2+):
    extra_edge_features = assembler.get_edge_channel_features()  # dict or empty
    extra_node_features = assembler.get_node_sidecar_features()  # dict or empty
    social_dim          = assembler.get_social_dim()              # int (0 if off)

In Phase 1 all getters return empty / zero since no feature files exist yet.
"""

import logging
import argparse
import numpy as np
from typing import Dict, Optional

from features.feature_bank import FeatureBank

logger = logging.getLogger(__name__)


class FeatureAssembler:

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.enabled = getattr(args, 'enabled_features', [])
        dataset_name = args.dataset_name
        root_dir = getattr(args, 'feature_bank_dir', './processed_data')
        version = getattr(args, 'feature_version', 'v1')

        self.bank = FeatureBank(
            root_dir=root_dir,
            dataset_name=dataset_name,
            version=version,
        )

        self._edge_channel_features: Dict[str, Optional[np.ndarray]] = {}
        self._node_sidecar_features: Dict[str, Optional[np.ndarray]] = {}

        self._load_requested_features()

    def _load_requested_features(self):
        """Attempt to load features that the user requested via --use_* flags."""
        edge_types = [f for f in self.enabled
                      if FeatureBank.GRANULARITY.get(f) == 'edge']
        node_types = [f for f in self.enabled
                      if FeatureBank.GRANULARITY.get(f) == 'node']

        for ft in edge_types:
            data = self.bank.load(ft)
            self._edge_channel_features[ft] = data
            if data is None:
                logger.warning(
                    f'Feature "{ft}" is enabled but .npy file not found. '
                    f'It will be skipped until pre-computed features are generated.')

        for ft in node_types:
            data = self.bank.load(ft)
            self._node_sidecar_features[ft] = data
            if data is None:
                logger.warning(
                    f'Feature "{ft}" is enabled but .npy file not found. '
                    f'It will be skipped until pre-computed features are generated.')

    # ------------------------------------------------------------------
    # Public getters (used by training scripts in Phase 2+)
    # ------------------------------------------------------------------

    def get_edge_channel_features(self) -> Dict[str, np.ndarray]:
        """Return {name: ndarray} for enabled & available edge-level features."""
        return {k: v for k, v in self._edge_channel_features.items()
                if v is not None}

    def get_node_sidecar_features(self) -> Dict[str, np.ndarray]:
        """Return {name: ndarray} for enabled & available node-level features."""
        return {k: v for k, v in self._node_sidecar_features.items()
                if v is not None}

    def get_social_dim(self) -> int:
        """Return the dimension of social features, or 0 if unavailable."""
        if 'social' not in self.enabled:
            return 0
        return self.bank.get_dim('social')

    def get_total_sidecar_dim(self) -> int:
        """Total dimension of all node-level sidecar features (for FusionLayer sizing)."""
        total = 0
        for ft, data in self._node_sidecar_features.items():
            if data is not None:
                total += data.shape[1]
        return total

    def has_any_active(self) -> bool:
        """True if at least one feature is enabled AND its data is available."""
        for ft in self.enabled:
            if ft in self._edge_channel_features and self._edge_channel_features[ft] is not None:
                return True
            if ft in self._node_sidecar_features and self._node_sidecar_features[ft] is not None:
                return True
        return False

    def log_status(self, log: logging.Logger):
        """Print a human-readable summary of feature status."""
        log.info(f'FeatureAssembler: enabled_features={self.enabled}')
        if not self.enabled:
            log.info('FeatureAssembler: running in baseline mode (no extra features)')
            return
        for ft in self.enabled:
            available = self.bank.has(ft)
            dim = self.bank.get_dim(ft) if available else 0
            granularity = FeatureBank.GRANULARITY.get(ft, '?')
            log.info(f'  {ft}: available={available}, dim={dim}, '
                     f'granularity={granularity}')
        log.info(f'  total_sidecar_dim={self.get_total_sidecar_dim()}')
        log.info(f'  has_any_active={self.has_any_active()}')
