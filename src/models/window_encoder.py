"""Transformer-based window encoder: sequence of beat embeddings -> window embedding."""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=400):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class WindowEncoder(nn.Module):
    """Encode a sequence of beat embeddings + time encoding -> single window vector."""

    def __init__(self, beat_dim=128, embed_dim=256, n_heads=4, n_layers=4,
                 dropout=0.1, max_beats=300, time_dim=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(beat_dim + time_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=max_beats + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, beat_embeds, beat_mask, time_encoding):
        """
        beat_embeds: (B, N, beat_dim)
        beat_mask: (B, N) bool - True for valid beats
        time_encoding: (B, 2) - sin/cos time
        """
        B, N, _ = beat_embeds.shape
        # Expand time encoding to each beat
        time_exp = time_encoding.unsqueeze(1).expand(B, N, -1)  # (B, N, 2)
        x = torch.cat([beat_embeds, time_exp], dim=-1)  # (B, N, beat_dim+2)
        x = self.input_proj(x)  # (B, N, embed_dim)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, embed_dim)
        x = self.pos_enc(x)

        # Create attention mask: CLS always attends, pad beats masked
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        src_key_padding_mask = ~torch.cat([cls_mask, beat_mask], dim=1)  # (B, N+1)

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        cls_out = self.norm(x[:, 0])  # (B, embed_dim)
        return cls_out
