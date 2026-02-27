from __future__ import annotations

import torch
import torch.nn as nn


DEFAULT_FEATURE_ORDER = [
    "bm25_body",
    "matched_terms",
    "matched_frac",
    "phrase_match",
    "in_title",
]


class FeatureNormalizer(nn.Module):
    """
    Stores mean/std as buffers -> saved with state_dict -> identical at serving time.
    """

    def __init__(self, num_features: int, eps: float = 1e-6, clip_z: float = 8.0):
        super().__init__()
        self.eps = eps
        self.clip_z = clip_z
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("std", torch.ones(num_features))

    def fit(self, x_all: torch.Tensor):
        mean = x_all.mean(dim=0)
        std = x_all.std(dim=0, unbiased=False).clamp_min(self.eps)
        self.mean.copy_(mean)
        self.std.copy_(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mean) / self.std
        if self.clip_z is not None:
            z = torch.clamp(z, -self.clip_z, self.clip_z)
        return z


class TinyLTRModel(nn.Module):
    """Lightweight per-doc scoring model (pointwise scoring, listwise loss)."""

    def __init__(self, num_features: int, hidden: int = 16, dropout: float = 0.1):
        super().__init__()
        self.norm = FeatureNormalizer(num_features=num_features)
        self.mlp = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, F)   mask: (B, L) bool
        returns scores: (B, L)
        """
        x = self.norm(x)
        s = self.mlp(x).squeeze(-1)  # (B, L)
        s = s.masked_fill(~mask, -1e9)
        return s
