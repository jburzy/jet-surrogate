"""Set transformer used for both the track tagger and the truth surrogate."""

from __future__ import annotations

import torch
import torch.nn as nn


class ParticleTransformer(nn.Module):
    """Per-object embedding -> pre-LN transformer encoder with a class token
    -> MLP head -> one logit per jet.

    Inputs: ``x`` float [B, N, n_cont] (already normalized), ``cats`` long
    [B, N, n_cat] (0 = padding / unknown), ``mask`` bool [B, N] (True = real).
    """

    def __init__(self, n_cont: int, cat_sizes: list[int] | None = None, cat_dim: int = 16,
                 d_model: int = 128, n_heads: int = 8, n_layers: int = 4, d_ff: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        cat_sizes = list(cat_sizes or [])
        self.embeds = nn.ModuleList([nn.Embedding(s, cat_dim, padding_idx=0) for s in cat_sizes])
        d_in = n_cont + cat_dim * len(cat_sizes)
        self.embed = nn.Sequential(
            nn.Linear(d_in, d_model), nn.GELU(), nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_model, 1))
        self.config = dict(n_cont=n_cont, cat_sizes=cat_sizes, cat_dim=cat_dim, d_model=d_model,
                           n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor, cats: torch.Tensor | None, mask: torch.Tensor) -> torch.Tensor:
        parts = [x]
        for i, emb in enumerate(self.embeds):
            parts.append(emb(cats[..., i]))
        h = self.embed(torch.cat(parts, dim=-1))
        h = h * mask.unsqueeze(-1)
        b = h.shape[0]
        h = torch.cat([self.cls.expand(b, -1, -1), h], dim=1)
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=mask.device), ~mask], dim=1)
        h = self.encoder(h, src_key_padding_mask=pad)
        return self.head(self.norm(h[:, 0])).squeeze(-1)
