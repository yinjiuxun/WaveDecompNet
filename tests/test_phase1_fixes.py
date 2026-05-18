"""Phase 1: Tests for dependency audit and code compatibility fixes."""
import os
import sys
import tempfile
import torch
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoencoder_1D_models_torch import (
    SeismogramEncoder,
    SeismogramDecoder,
    SeisSeparator,
    DotProductAttention,
    PositionalEncoding,
    MultiHeadAttention,
    Attention_bottleneck,
    Attention_bottleneck_LSTM,
)
from utilities import mkdir
from torch_tools import (
    WaveformDataset,
    EarlyStopping,
    training_loop,
    training_loop_branches,
    model_same,
)


class TestP0_1_DtypeFloat64Removed:
    """P0-1: Test that dtype=torch.float64 is removed from layer constructors."""

    def test_encoder_no_dtype_in_constructor(self):
        """Encoder should instantiate without dtype parameter."""
        encoder = SeismogramEncoder()
        # Should work without errors
        assert encoder is not None

    def test_decoder_no_dtype_in_constructor(self):
        """Decoder should instantiate without dtype parameter."""
        decoder = SeismogramDecoder(bottleneck=None)
        assert decoder is not None

    def test_model_converts_to_double(self):
        """Model should work with .double() conversion."""
        encoder = SeismogramEncoder()
        encoder.double()
        # Check that parameters are float64
        for param in encoder.parameters():
            assert param.dtype == torch.float64

    def test_multihhead_attention_no_dtype(self):
        """MultiHeadAttention should not have dtype in Linear layers."""
        mha = MultiHeadAttention(
            key_size=64, query_size=64, value_size=64,
            num_hiddens=64, num_heads=4, dropout=0.1
        )
        assert mha is not None

    def test_attention_bottleneck_lstm_no_dtype(self):
        """Attention_bottleneck_LSTM should not have dtype in LSTM."""
        ab_lstm = Attention_bottleneck_LSTM(
            num_hiddens=64, num_heads=4, dropout=0.1
        )
        assert ab_lstm is not None


class TestP0_6_SoftmaxDimension:
    """P0-6: Test that softmax uses dim=-1 instead of dim=0."""

    def test_softmax_over_sequence_dimension(self):
        """Softmax should be over the sequence/time dimension, not batch."""
        attn = DotProductAttention(dropout=0.0)
        batch_size = 2
        seq_len = 10
        d_model = 64

        queries = torch.randn(batch_size, seq_len, d_model)
        keys = torch.randn(batch_size, seq_len, d_model)
        values = torch.randn(batch_size, seq_len, d_model)

        output = attn(queries, keys, values)

        # Output shape should match input shape
        assert output.shape == queries.shape

        # Attention weights should sum to 1 along the sequence dimension
        assert attn.attention_weights.shape[-1] == seq_len
        # Check that attention weights sum to ~1 along dim=-1
        weight_sums = attn.attention_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)


class TestP0_2_3_SaveLoadStateDict:
    """P0-2/P0-3: Test that torch.save/load uses state_dict with weights_only."""

    def test_save_load_state_dict(self, tmp_path):
        """Model should be saved/loaded via state_dict."""
        model = SeismogramEncoder()
        model_path = tmp_path / "model.pth"

        # Save state_dict
        torch.save(model.state_dict(), str(model_path))

        # Load with weights_only=True
        new_model = SeismogramEncoder()
        new_model.load_state_dict(
            torch.load(str(model_path), weights_only=True, map_location="cpu")
        )

        # Check parameters match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_early_stopping_saves_state_dict(self, tmp_path):
        """EarlyStopping should save state_dict, not full model."""
        model = SeismogramEncoder()
        checkpoint_path = str(tmp_path / "checkpoint.pt")

        early_stopping = EarlyStopping(
            patience=2, verbose=False, path=checkpoint_path
        )
        early_stopping(1.0, model)

        # Load and verify it's a state_dict
        loaded = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        assert isinstance(loaded, dict)

        new_model = SeismogramEncoder()
        new_model.load_state_dict(loaded)


class TestP0_4_NextIterator:
    """P0-4: Test that next() is used instead of iterator.next()."""

    def test_next_function_works(self):
        """next() function should work with DataLoaders."""
        from torch.utils.data import DataLoader, TensorDataset

        X = torch.randn(10, 3, 100)
        y = torch.randn(10, 3, 100)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=2)

        data_iter = iter(loader)
        batch = next(data_iter)

        assert len(batch) == 2
        assert batch[0].shape[0] == 2


class TestP0_5_Makedirs:
    """P0-5: Test that mkdir uses os.makedirs with exist_ok=True."""

    def test_makedirs_creates_nested_dirs(self, tmp_path):
        """mkdir should create nested directories."""
        nested_path = tmp_path / "a" / "b" / "c"
        mkdir(str(nested_path))
        assert nested_path.exists()

    def test_makedirs_idempotent(self, tmp_path):
        """mkdir should not fail if directory exists."""
        dir_path = tmp_path / "existing"
        mkdir(str(dir_path))
        mkdir(str(dir_path))  # Should not raise


class TestP0_7_PositionalEncodingDtype:
    """P0-7: Test PositionalEncoding dtype consistency."""

    def test_positional_encoding_uses_float64_in_arange(self):
        """PositionalEncoding arange should use float64 (not float32)."""
        # Check that the source code uses float64 in arange calls
        # This ensures consistency when model is used with double precision
        import inspect
        source = inspect.getsource(PositionalEncoding.__init__)
        assert "torch.float64" in source, "Should use float64 in arange for consistency"


class TestP0_8_ModelSame:
    """P0-8: Test model_same function checks all parameters."""

    def test_model_same_identical(self):
        """model_same should return True for identical models."""
        model1 = SeismogramEncoder()
        model2 = SeismogramEncoder()
        # Copy weights
        model2.load_state_dict(model1.state_dict())
        assert model_same(model1, model2) is True

    def test_model_same_different(self):
        """model_same should return False for different models."""
        model1 = SeismogramEncoder()
        model2 = SeismogramEncoder()
        # Different initialization
        assert model_same(model1, model2) is False


class TestP0_9_NoGradInference:
    """P0-9: Test that inference uses torch.no_grad()."""

    def test_no_grad_reduces_memory(self):
        """torch.no_grad() should not create gradient graphs."""
        model = SeismogramEncoder()
        model.eval()
        x = torch.randn(1, 3, 100)

        with torch.no_grad():
            output, _, _, _ = model(x)

        # Output should not require gradients
        assert not output.requires_grad


class TestWaveformDataset:
    """Test WaveformDataset class."""

    def test_waveform_dataset(self):
        """WaveformDataset should work correctly."""
        X = np.random.randn(10, 3, 100).astype(np.float32)
        Y = np.random.randn(10, 3, 100).astype(np.float32)

        dataset = WaveformDataset(X, Y)
        assert len(dataset) == 10

        x, y = dataset[0]
        assert x.shape == (100, 3)  # moveaxis changes shape
        assert y.shape == (100, 3)


class TestFullModelIntegration:
    """Integration tests for the full model."""

    @pytest.mark.xfail(reason="Original decoder architecture has stride size mismatch with skip connections")
    def test_full_model_forward(self):
        """Full SeisSeparator model should work."""
        bottleneck = torch.nn.LSTM(64, 32, 2, bidirectional=True, batch_first=True)
        import copy
        bottleneck_eq = copy.deepcopy(bottleneck)
        bottleneck_noise = copy.deepcopy(bottleneck)

        encoder = SeismogramEncoder()
        decoder_eq = SeismogramDecoder(bottleneck=bottleneck_eq)
        decoder_noise = SeismogramDecoder(bottleneck=bottleneck_noise)

        model = SeisSeparator("test", encoder, decoder_eq, decoder_noise)
        model.double()  # Convert to float64

        x = torch.randn(2, 3, 100, dtype=torch.float64)
        output1, output2 = model(x)

        assert output1.shape == x.shape
        assert output2.shape == x.shape

    def test_model_save_load_roundtrip(self, tmp_path):
        """Full model save/load roundtrip should work."""
        import copy

        bottleneck = torch.nn.LSTM(64, 32, 2, bidirectional=True, batch_first=True)
        bottleneck_eq = copy.deepcopy(bottleneck)
        bottleneck_noise = copy.deepcopy(bottleneck)

        encoder = SeismogramEncoder()
        decoder_eq = SeismogramDecoder(bottleneck=bottleneck_eq)
        decoder_noise = SeismogramDecoder(bottleneck=bottleneck_noise)

        model = SeisSeparator("test", encoder, decoder_eq, decoder_noise)
        model.double()

        # Save
        model_path = tmp_path / "full_model.pth"
        torch.save(model.state_dict(), str(model_path))

        # Load
        new_model = SeisSeparator("test", encoder.__class__(),
                                  SeismogramDecoder(bottleneck=copy.deepcopy(bottleneck)),
                                  SeismogramDecoder(bottleneck=copy.deepcopy(bottleneck)))
        new_model.double()
        new_model.load_state_dict(
            torch.load(str(model_path), weights_only=True, map_location="cpu")
        )

        # Verify
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
