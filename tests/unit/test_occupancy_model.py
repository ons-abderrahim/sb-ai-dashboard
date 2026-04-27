"""
Unit tests for the OccupancyTransformer model.
"""

import torch
import pytest
from src.models.occupancy.transformer import OccupancyTransformer, OccupancyTransformerConfig


@pytest.fixture
def config() -> OccupancyTransformerConfig:
    return OccupancyTransformerConfig(
        sensor_feature_dim=8,
        time_feature_dim=4,
        seq_len=12,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        n_horizons=3,
    )


@pytest.fixture
def model(config: OccupancyTransformerConfig) -> OccupancyTransformer:
    return OccupancyTransformer(config)


@pytest.fixture
def sample_batch(config: OccupancyTransformerConfig):
    B, T = 4, config.seq_len
    return {
        "sensor_seq": torch.randn(B, T, config.sensor_feature_dim),
        "time_of_day_min": torch.randint(0, 1440, (B, T)).float(),
        "day_of_week": torch.randint(0, 7, (B, T)).float(),
        "domain_id": torch.randint(0, config.n_domain_tokens, (B,)),
    }


class TestOccupancyTransformer:
    def test_output_shapes(self, model, sample_batch, config):
        out = model(**sample_batch)
        B = sample_batch["sensor_seq"].shape[0]
        assert out["logits"].shape == (B, config.n_horizons)
        assert out["probs"].shape == (B, config.n_horizons)

    def test_probs_in_range(self, model, sample_batch):
        out = model(**sample_batch)
        assert out["probs"].min() >= 0.0
        assert out["probs"].max() <= 1.0

    def test_no_nan_in_output(self, model, sample_batch):
        out = model(**sample_batch)
        assert not torch.isnan(out["logits"]).any()
        assert not torch.isnan(out["probs"]).any()

    def test_padding_mask(self, model, sample_batch, config):
        B, T = sample_batch["sensor_seq"].shape[:2]
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, -4:] = True  # Mask last 4 positions

        out_masked = model(**sample_batch, src_key_padding_mask=mask)
        out_unmasked = model(**sample_batch)

        # Outputs should differ when mask is applied
        assert not torch.allclose(out_masked["probs"], out_unmasked["probs"])

    def test_temperature_scaling(self, model, sample_batch):
        """Higher temperature → probabilities closer to 0.5."""
        out_default = model(**sample_batch)

        model.log_temperature.data.fill_(2.0)  # temperature = e^2 ≈ 7.4 (softer)
        out_high_temp = model(**sample_batch)

        # High-temp probs should be more concentrated around 0.5
        assert out_high_temp["probs"].std() < out_default["probs"].std() + 0.1

    def test_predict_no_grad(self, model, sample_batch):
        """predict() should run without gradient tracking."""
        out = model.predict(**sample_batch)
        assert out["probs"].requires_grad is False

    def test_parameter_count(self, model):
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        print(f"\nModel parameters: {n_params:,}")
