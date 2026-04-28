"""
FeatureFusion: merge DyGFormer backbone embeddings with sidecar feature embeddings.

When no sidecar features are active, forward() is a no-op pass-through.

Supported fusion modes (to be extended):
  - concat: concatenate and project back to backbone_dim
  - gate:   (placeholder, Phase 5+)
  - attention: (placeholder, Phase 5+)
"""

import torch
import torch.nn as nn
from typing import Optional


class FeatureFusion(nn.Module):

    def __init__(self, backbone_dim: int, sidecar_dim: int = 0,
                 fusion_mode: str = 'concat', dropout: float = 0.1):
        """
        :param backbone_dim: dimension of the DyGFormer output embeddings
        :param sidecar_dim: total dimension of all sidecar feature vectors
                            (0 means no sidecar features, forward is identity)
        :param fusion_mode: 'concat' | 'gate' | 'attention'
        :param dropout: dropout rate
        """
        super().__init__()
        self.backbone_dim = backbone_dim
        self.sidecar_dim = sidecar_dim
        self.fusion_mode = fusion_mode
        self.active = sidecar_dim > 0

        if not self.active:
            return

        if fusion_mode == 'concat':
            self.projection = nn.Sequential(
                nn.Linear(backbone_dim + sidecar_dim, backbone_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(backbone_dim, backbone_dim),
            )
        elif fusion_mode in ('gate', 'attention'):
            raise NotImplementedError(
                f'Fusion mode "{fusion_mode}" will be implemented in Phase 5+')
        else:
            raise ValueError(f'Unknown fusion_mode: {fusion_mode}')

    def forward(self, backbone_emb: torch.Tensor,
                sidecar_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param backbone_emb: Tensor, shape (batch_size, backbone_dim)
        :param sidecar_emb:  Tensor, shape (batch_size, sidecar_dim) or None
        :return: Tensor, shape (batch_size, backbone_dim)
        """
        if not self.active or sidecar_emb is None:
            return backbone_emb

        fused = torch.cat([backbone_emb, sidecar_emb], dim=-1)
        return self.projection(fused)
