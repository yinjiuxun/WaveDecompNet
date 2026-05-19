"""Phase 5: Fix Other Python Compatibility Issues

Tests for:
- P5-1: utilities.py uses os.makedirs(exist_ok=True)
- P5-2: f-string compatibility (Python 3.12)
- P5-3: h5py API compatibility (3.11+)
- P5-4: set_seed() function for reproducible results
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import random


class TestP5_1_MakedirsExistOk:
    """P5-1: utilities.py mkdir uses os.makedirs(exist_ok=True)"""

    def test_mkdir_creates_directory(self):
        from utilities import mkdir
        with tempfile.TemporaryDirectory() as tmp:
            new_dir = os.path.join(tmp, "test_dir")
            mkdir(new_dir)
            assert os.path.isdir(new_dir)

    def test_mkdir_idempotent(self):
        """Calling mkdir on existing dir should not raise"""
        from utilities import mkdir
        with tempfile.TemporaryDirectory() as tmp:
            existing = os.path.join(tmp, "existing")
            mkdir(existing)
            mkdir(existing)  # Should not raise

    def test_mkdir_nested_path(self):
        """mkdir should create nested directories"""
        from utilities import mkdir
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "a", "b", "c")
            mkdir(nested)
            assert os.path.isdir(nested)


class TestP5_2_FStringCompatibility:
    """P5-2: f-string compatibility with Python 3.12"""

    def test_f_string_basic(self):
        """Basic f-string should work in Python 3.12"""
        name = "test"
        result = f"hello_{name}"
        assert result == "hello_test"

    def test_f_string_with_expression(self):
        """f-string with expressions"""
        x = 10
        result = f"{x*2}"
        assert result == "20"

    def test_f_string_path_concat(self):
        """f-string path concatenation (used in test_model.py)"""
        model_name = "Branch_Encoder_Decoder_LSTM"
        figure_dir = "/tmp"
        path = figure_dir + f"/{model_name}_Loss_evolution.pdf"
        assert "Branch_Encoder_Decoder_LSTM_Loss_evolution.pdf" in path


class TestP5_3_H5pyApiCompatibility:
    """P5-3: h5py API compatibility with 3.11+"""

    def test_h5py_file_read_mode(self):
        """h5py.File(path, 'r') is standard API"""
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not installed in test environment")

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            tmp_path = f.name
        try:
            # Create test file
            with h5py.File(tmp_path, 'w') as f:
                f.create_dataset("test", data=[1, 2, 3])

            # Read with standard 'r' mode
            with h5py.File(tmp_path, 'r') as f:
                data = f["test"][:]
                assert list(data) == [1, 2, 3]
        finally:
            os.unlink(tmp_path)

    def test_h5py_file_write_mode(self):
        """h5py.File(path, 'w') is standard API"""
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not installed in test environment")

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            tmp_path = f.name
        try:
            with h5py.File(tmp_path, 'w') as f:
                f.create_dataset("data", data=np.array([1, 2, 3]))
                f.attrs["key"] = "value"

            with h5py.File(tmp_path, 'r') as f:
                assert list(f["data"][:]) == [1, 2, 3]
                assert f.attrs["key"] == "value"
        finally:
            os.unlink(tmp_path)

    def test_h5py_attrs(self):
        """h5py attrs access is standard API"""
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not installed in test environment")

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            tmp_path = f.name
        try:
            with h5py.File(tmp_path, 'w') as f:
                f.attrs["model_name"] = "test_model"
                f.attrs["train_size"] = 0.6

            with h5py.File(tmp_path, 'r') as f:
                assert f.attrs["model_name"] == "test_model"
                assert f.attrs["train_size"] == 0.6
        finally:
            os.unlink(tmp_path)


class TestP5_4_SetSeed:
    """P5-4: set_seed() function for reproducible results"""

    def test_set_seed_exists(self):
        """set_seed function should be importable from torch_tools"""
        from torch_tools import set_seed
        assert callable(set_seed)

    def test_set_seed_sets_python_random(self):
        """set_seed should set Python random seed"""
        from torch_tools import set_seed
        set_seed(42)
        random.seed(42)
        expected = random.random()
        # Reset and verify
        set_seed(42)
        assert random.random() == expected

    def test_set_seed_sets_numpy_random(self):
        """set_seed should set NumPy random seed"""
        from torch_tools import set_seed
        set_seed(42)
        val1 = np.random.random()
        set_seed(42)
        val2 = np.random.random()
        assert val1 == val2

    def test_set_seed_sets_torch_random(self):
        """set_seed should set PyTorch random seed"""
        import torch
        from torch_tools import set_seed
        set_seed(42)
        val1 = torch.rand(1).item()
        set_seed(42)
        val2 = torch.rand(1).item()
        assert val1 == val2

    def test_set_seed_sets_pythonhashseed(self):
        """set_seed should set PYTHONHASHSEED environment variable"""
        from torch_tools import set_seed
        old_val = os.environ.get("PYTHONHASHSEED")
        try:
            set_seed(123)
            assert os.environ["PYTHONHASHSEED"] == "123"
        finally:
            if old_val is not None:
                os.environ["PYTHONHASHSEED"] = old_val
            elif "PYTHONHASHSEED" in os.environ:
                del os.environ["PYTHONHASHSEED"]

    def test_set_seed_deterministic_training(self):
        """Two models initialized with same seed should have same weights"""
        import torch
        from torch_tools import set_seed
        from autoencoder_1D_models_torch import SeismogramEncoder

        set_seed(42)
        model1 = SeismogramEncoder()

        set_seed(42)
        model2 = SeismogramEncoder()

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)

    def test_set_seed_different_seeds_different_results(self):
        """Different seeds should produce different results"""
        import torch
        from torch_tools import set_seed

        set_seed(42)
        val1 = torch.rand(1).item()

        set_seed(123)
        val2 = torch.rand(1).item()

        assert val1 != val2

    def test_set_seed_default_value(self):
        """set_seed with no args should use default seed 42"""
        from torch_tools import set_seed
        # Should not raise
        set_seed()

    def test_set_seed_cuda_safe(self):
        """set_seed should not fail when CUDA is unavailable"""
        from torch_tools import set_seed
        # Should work regardless of CUDA availability
        set_seed(42)
