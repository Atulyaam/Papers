"""
tests/test_ae_architecture.py
-------------------------------
Sprint 7 — EXP_AE_V1 architecture contract tests.

Verifies:
- Input/output shape = 75
- Exact layer sizes: encoder [75→12→6], decoder [6→12→75]
- Hidden activations = ReLU
- Output activation = Linear (no sigmoid/tanh)
- No BatchNorm layers
- No Dropout layers
- n_params = 2,049
- forward() output shape matches input shape
- encode() returns [N, 6]
- reconstruction_error() returns [N] scalars
- RE is mean over features (not sum)
- architecture_dict() returns correct fields
- Wrong input_dim raises ValueError
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.autoencoder.ae_model import (
    AE_INPUT_DIM,
    AE_ENCODER_SIZES,
    AE_N_PARAMS,
    Autoencoder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ae():
    return Autoencoder()


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.randn(32, 75)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_forward_output_shape_matches_input(ae, batch):
    out = ae(batch)
    assert out.shape == batch.shape, f"Expected {batch.shape}, got {out.shape}"


def test_input_dim_is_75(ae):
    assert AE_INPUT_DIM == 75


def test_encode_output_shape(ae, batch):
    z = ae.encode(batch)
    assert z.shape == (32, 6), f"Expected (32, 6), got {z.shape}"


def test_reconstruction_error_shape(ae, batch):
    re = ae.reconstruction_error(batch)
    assert re.shape == (32,), f"Expected (32,), got {re.shape}"


def test_reconstruction_error_all_nonnegative(ae, batch):
    re = ae.reconstruction_error(batch)
    assert (re >= 0).all(), "Reconstruction errors must be >= 0"


# ---------------------------------------------------------------------------
# RE is MEAN over features, NOT sum
# ---------------------------------------------------------------------------

def test_reconstruction_error_is_mean_not_sum(ae):
    """RE(x) = mean over 75 features, NOT sum."""
    ae.eval()
    x = torch.zeros(1, 75)
    # Manually set output to a known value
    with torch.no_grad():
        x_hat = ae(x)
        diff_sq = ((x - x_hat) ** 2)
        expected_mean = diff_sq.mean(dim=1).item()
        expected_sum = diff_sq.sum(dim=1).item()
        actual = ae.reconstruction_error(x).item()

    assert abs(actual - expected_mean) < 1e-6, (
        f"RE should be mean={expected_mean:.6f}, got {actual:.6f} "
        f"(sum would be {expected_sum:.6f})"
    )


def test_reconstruction_error_uses_dim1_mean(ae):
    """RE is mean over feature dim=1 (75 features)."""
    ae.eval()
    torch.manual_seed(42)
    x = torch.randn(5, 75)
    with torch.no_grad():
        x_hat = ae(x)
        re = ae.reconstruction_error(x)
        manual = ((x - x_hat) ** 2).mean(dim=1)
    assert torch.allclose(re, manual, atol=1e-6)


# ---------------------------------------------------------------------------
# Layer architecture tests
# ---------------------------------------------------------------------------

def test_encoder_first_layer_sizes(ae):
    enc_layers = list(ae.encoder.children())
    linear_layers = [l for l in enc_layers if isinstance(l, nn.Linear)]
    assert len(linear_layers) == 2, f"Encoder should have 2 Linear layers, got {len(linear_layers)}"
    assert linear_layers[0].in_features == 75
    assert linear_layers[0].out_features == 12
    assert linear_layers[1].in_features == 12
    assert linear_layers[1].out_features == 6


def test_decoder_layer_sizes(ae):
    dec_layers = list(ae.decoder.children())
    linear_layers = [l for l in dec_layers if isinstance(l, nn.Linear)]
    assert len(linear_layers) == 2, f"Decoder should have 2 Linear layers, got {len(linear_layers)}"
    assert linear_layers[0].in_features == 6
    assert linear_layers[0].out_features == 12
    assert linear_layers[1].in_features == 12
    assert linear_layers[1].out_features == 75


def test_hidden_activations_are_relu(ae):
    """All non-Linear layers in encoder and decoder must be ReLU."""
    for name, module in [("encoder", ae.encoder), ("decoder", ae.decoder)]:
        non_linear = [
            type(l).__name__ for l in module.children()
            if not isinstance(l, nn.Linear)
        ]
        assert all(t == "ReLU" for t in non_linear), (
            f"{name} has non-ReLU non-Linear layers: {non_linear}"
        )


def test_output_layer_is_linear_no_activation(ae):
    """Last layer of decoder must be Linear (no activation after it)."""
    dec_children = list(ae.decoder.children())
    assert isinstance(dec_children[-1], nn.Linear), (
        f"Last decoder layer must be Linear, got {type(dec_children[-1]).__name__}"
    )


def test_no_batchnorm_anywhere(ae):
    for name, module in ae.named_modules():
        assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)), (
            f"BatchNorm found at {name} — prohibited by design"
        )


def test_no_dropout_anywhere(ae):
    for name, module in ae.named_modules():
        assert not isinstance(module, nn.Dropout), (
            f"Dropout found at {name} — prohibited by design"
        )


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------

def test_parameter_count(ae):
    n = ae.count_parameters()
    assert n == AE_N_PARAMS, f"Expected {AE_N_PARAMS} params, got {n}"


def test_parameter_count_exact_value(ae):
    """Verify: (75*12+12) + (12*6+6) + (6*12+12) + (12*75+75) = 2049"""
    expected = (75 * 12 + 12) + (12 * 6 + 6) + (6 * 12 + 12) + (12 * 75 + 75)
    assert expected == 2049
    assert ae.count_parameters() == 2049


# ---------------------------------------------------------------------------
# Architecture dict
# ---------------------------------------------------------------------------

def test_architecture_dict_fields(ae):
    d = ae.architecture_dict()
    assert d["input_dim"] == 75
    assert d["encoder"] == [75, 12, 6]
    assert d["bottleneck"] == 6
    assert d["decoder"] == [6, 12, 75]
    assert d["hidden_activation"] == "ReLU"
    assert d["output_activation"] == "Linear (none)"
    assert d["batchnorm"] is False
    assert d["dropout"] is False
    assert d["n_params"] == 2049


# ---------------------------------------------------------------------------
# Wrong input dim
# ---------------------------------------------------------------------------

def test_wrong_input_dim_raises():
    with pytest.raises(ValueError, match="75"):
        Autoencoder(input_dim=50)


def test_wrong_batch_dim_raises_at_forward(ae):
    """Passing wrong feature size should raise a runtime shape error."""
    with pytest.raises(Exception):
        ae(torch.randn(10, 50))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_ae_determinism_same_seed():
    """Same seed → same weights → same reconstruction."""
    torch.manual_seed(42)
    ae1 = Autoencoder()
    torch.manual_seed(42)
    ae2 = Autoencoder()

    x = torch.randn(8, 75)
    ae1.eval(); ae2.eval()
    with torch.no_grad():
        out1 = ae1(x)
        out2 = ae2(x)
    assert torch.allclose(out1, out2, atol=1e-6)


# ---------------------------------------------------------------------------
# Frozen feature count
# ---------------------------------------------------------------------------

def test_frozen_feature_count():
    """AE_INPUT_DIM must remain 75 (frozen by EXP_MI_V1_1)."""
    assert AE_INPUT_DIM == 75


def test_encoder_sizes_frozen():
    assert AE_ENCODER_SIZES == [12, 6]
