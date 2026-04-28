"""
SocialSidecarEncoder: encode node-level social / structural features.

Phase 1: placeholder skeleton.  The actual feature computation and
encoding will be implemented in Phase 5.

Expected input: per-node social feature vectors from FeatureBank.
Expected output: encoded social embeddings to be consumed by FeatureFusion.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class SocialSidecarEncoder(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1,
                 social_mode: str = 'time_aware', device: str = 'cpu'):
        """
        :param input_dim: dimension of raw social feature vectors
        :param output_dim: dimension of encoded social embeddings
        :param dropout: dropout rate
        :param social_mode: 'static' or 'time_aware'
        :param device: device string
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.social_mode = social_mode
        self.device = device

        if input_dim > 0:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim, output_dim),
            )
        else:
            self.encoder = None

    def forward(self, social_features: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        :param social_features: Tensor, shape (batch_size, input_dim) or None
        :return: Tensor, shape (batch_size, output_dim) or None
        """
        if self.encoder is None or social_features is None:
            return None
        return self.encoder(social_features)
