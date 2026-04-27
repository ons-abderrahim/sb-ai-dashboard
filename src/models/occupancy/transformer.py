"""
Domain-Adapted Transformer for Occupancy Prediction
====================================================
Architecture:
  - Transformer encoder (6 layers, 8 attention heads)
  - Cyclic positional encodings for time-of-day and day-of-week
  - Domain adaptation layer for building-specific distributions
  - Calibrated probability output via temperature scaling

Reference: ASHRAE occupancy modeling; Concordia CIISE lab datasets.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OccupancyTransformerConfig:
    """Hyperparameters for the occupancy transformer."""

    # Input
    sensor_feature_dim: int = 8  # CO2, temp, humidity, motion, door, lux, noise, PIR
    time_feature_dim: int = 4    # sin/cos for time-of-day and day-of-week
    seq_len: int = 24            # Input sequence length (e.g. 24 × 5-min = 2h)

    # Architecture
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 512
    dropout: float = 0.1

    # Domain adaptation
    n_domain_tokens: int = 16   # Learnable domain embedding size

    # Output
    n_horizons: int = 3         # Simultaneous prediction at 5, 15, 30 min
    temperature: float = 1.0    # For calibration (updated post-training)

    # Derived
    input_dim: int = field(init=False)

    def __post_init__(self) -> None:
        self.input_dim = self.sensor_feature_dim + self.time_feature_dim


class CyclicPositionalEncoding(nn.Module):
    """
    Encodes time-of-day (minute of day) and day-of-week as sinusoidal features.
    Produces 4-dim vector: [sin(tod), cos(tod), sin(dow), cos(dow)].
    """

    def forward(self, time_of_day_min: torch.Tensor, day_of_week: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_of_day_min: float tensor (B, T) — minute of day [0, 1440)
            day_of_week:     float tensor (B, T) — day index [0, 7)
        Returns:
            (B, T, 4) cyclic encodings
        """
        tod_angle = (time_of_day_min / 1440.0) * 2 * math.pi
        dow_angle = (day_of_week / 7.0) * 2 * math.pi
        return torch.stack(
            [tod_angle.sin(), tod_angle.cos(), dow_angle.sin(), dow_angle.cos()],
            dim=-1,
        )


class DomainAdaptationLayer(nn.Module):
    """
    Injects building-specific domain embeddings into the sequence.
    Allows a model pre-trained on ASHRAE data to adapt to a specific building's
    sensor distribution with minimal fine-tuning data.
    """

    def __init__(self, n_tokens: int, d_model: int) -> None:
        super().__init__()
        self.domain_embedding = nn.Embedding(n_tokens, d_model)
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:         (B, T, d_model) sequence features
            domain_id: (B,) integer building/zone domain identifier
        Returns:
            (B, T, d_model) domain-adapted sequence
        """
        d_emb = self.domain_embedding(domain_id).unsqueeze(1).expand_as(x)  # (B, T, d)
        gate = torch.sigmoid(self.gate(torch.cat([x, d_emb], dim=-1)))
        return x + gate * d_emb


class OccupancyTransformer(nn.Module):
    """
    Full occupancy prediction model.

    Input:
        sensor_seq:      (B, T, sensor_feature_dim) — raw sensor readings
        time_of_day_min: (B, T) — minute of day
        day_of_week:     (B, T) — day of week index
        domain_id:       (B,)   — building/zone domain ID
        src_key_padding_mask: Optional (B, T) — True where padding

    Output:
        logits:  (B, n_horizons) — raw logits for occupied probability
        probs:   (B, n_horizons) — calibrated probabilities
    """

    def __init__(self, config: OccupancyTransformerConfig) -> None:
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.d_model)

        # Positional encoding (learnable absolute positions)
        self.pos_enc = nn.Embedding(config.seq_len, config.d_model)

        # Cyclic time features
        self.cyclic_enc = CyclicPositionalEncoding()

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Domain adaptation
        self.domain_adapt = DomainAdaptationLayer(config.n_domain_tokens, config.d_model)

        # Aggregation & prediction head
        self.aggregator = nn.Linear(config.d_model, config.d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.n_horizons),
        )

        # Temperature scaling parameter (updated during calibration)
        self.log_temperature = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(
        self,
        sensor_seq: torch.Tensor,
        time_of_day_min: torch.Tensor,
        day_of_week: torch.Tensor,
        domain_id: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        B, T, _ = sensor_seq.shape

        # Cyclic time features
        time_feats = self.cyclic_enc(time_of_day_min, day_of_week)  # (B, T, 4)

        # Concatenate sensor + time features, project to model dim
        x = torch.cat([sensor_seq, time_feats], dim=-1)  # (B, T, input_dim)
        x = self.input_proj(x)  # (B, T, d_model)

        # Add learnable positional embeddings
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_enc(positions)

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Domain adaptation
        x = self.domain_adapt(x, domain_id)

        # Mean pooling over sequence (ignoring padding if mask provided)
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).float().unsqueeze(-1)
            x_agg = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            x_agg = x.mean(dim=1)

        x_agg = F.gelu(self.aggregator(x_agg))

        # Predict
        logits = self.head(x_agg)  # (B, n_horizons)
        calibrated_logits = logits / self.temperature
        probs = torch.sigmoid(calibrated_logits)

        return {"logits": logits, "probs": probs}

    def predict(
        self,
        sensor_seq: torch.Tensor,
        time_of_day_min: torch.Tensor,
        day_of_week: torch.Tensor,
        domain_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Convenience inference method (no gradient)."""
        self.eval()
        with torch.no_grad():
            return self.forward(sensor_seq, time_of_day_min, day_of_week, domain_id)
