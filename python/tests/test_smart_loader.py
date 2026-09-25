"""
Tests for smart_loader module - RAM-aware model loading.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestSystemResources:
    """Test system resource detection."""

    def test_get_system_resources(self):
        """Test that system resources are detected."""
        from densecore.smart_loader import get_system_resources

        resources = get_system_resources()

        # Basic validation
        assert resources.total_ram_gb > 0
        assert resources.available_ram_gb > 0
        assert resources.cpu_cores > 0
        assert resources.cpu_threads >= resources.cpu_cores

        # Available should be  <= total
        assert resources.available_ram_gb <= resources.total_ram_gb

    def test_system_resources_str(self):
        """Test SystemResources string representation."""
        from densecore.smart_loader import SystemResources

        resources = SystemResources(
            total_ram_gb=16.0,
            available_ram_gb=12.0,
            cpu_cores=4,
            cpu_threads=8,
        )

        str_repr = str(resources)
        assert "16.0" in str_repr
        assert "12.0" in str_repr
        assert "4" in str_repr


class TestMemoryEstimation:
    """Test memory usage estimation."""

    def test_estimate_model_memory(self):
        """Test model memory estimation."""
        from densecore.smart_loader import estimate_model_memory

        # 4 GB file should estimate ~5 GB runtime (safety margin)
        file_size = 4 * 1024**3
        estimated = estimate_model_memory(file_size)

        assert estimated >= 4.0  # At least file size
        assert estimated <= 8.0  # Reasonable upper bound


class TestRecommendQuantization:
    """Test quantization recommendation logic."""

    @patch("densecore.smart_loader.get_system_resources")
    @patch("densecore.hub.list_gguf_files")
    def test_recommends_fitting_quant(self, mock_list, mock_resources):
        """Test that it recommends the best fitting quantization."""
        from densecore.smart_loader import SystemResources, recommend_quantization

        # 16 GB total, 12 GB available
        mock_resources.return_value = SystemResources(16.0, 12.0, 8, 16)

        # Mock available files
        mock_list.return_value = [
            {
                "filename": "model.Q4_K_M.gguf",
                "size": 4 * 1024**3,  # 4 GB
                "quant": "Q4_K_M",
            },
            {
                "filename": "model.Q8_0.gguf",
                "size": 8 * 1024**3,  # 8 GB
                "quant": "Q8_0",
            },
            {
                "filename": "model.Q2_K.gguf",
                "size": 2 * 1024**3,  # 2 GB
                "quant": "Q2_K",
            },
        ]

        filename, msg = recommend_quantization("test/repo", verbose=False)

        # Should recommend Q8_0 (highest quality that fits)
        assert filename == "model.Q8_0.gguf"
        assert "Q8_0" in msg

    @patch("densecore.smart_loader.get_system_resources")
    @patch("densecore.hub.list_gguf_files")
    def test_warns_when_too_large(self, mock_list, mock_resources):
        """Test warning when no model fits."""
        from densecore.smart_loader import SystemResources, recommend_quantization

        # Only 4 GB available
        mock_resources.return_value = SystemResources(8.0, 4.0, 4, 8)

        # All models are too large
        mock_list.return_value = [
            {
                "filename": "huge-model.Q2_K.gguf",
                "size": 20 * 1024**3,  # 20 GB
                "quant": "Q2_K",
            },
        ]

        filename, msg = recommend_quantization("test/repo", verbose=False)

        # Should still return smallest, but with warning
        assert filename == "huge-model.Q2_K.gguf"
        assert "WARNING" in msg or "⚠️" in msg


class TestSmartLoad:
    """Test the smart_load function."""

    @patch("densecore.hub.from_pretrained")
    @patch("densecore.smart_loader.recommend_quantization")
    def test_smart_load_basic(self, mock_recommend, mock_from_pretrained):
        """Test basic smart load functionality."""
        from densecore.smart_loader import smart_load

        # Mock recommendation
        mock_recommend.return_value = ("model.Q4_K_M.gguf", "Selected Q4_K_M")

        # Mock model loading
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        # Call smart_load
        result = smart_load("test/repo", verbose=False)

        # Verify recommendation was called
        mock_recommend.assert_called_once()

        # Verify from_pretrained was called with the recommended file
        mock_from_pretrained.assert_called_once()
        call_kwargs = mock_from_pretrained.call_args[1]
        assert call_kwargs["filename"] == "model.Q4_K_M.gguf"

        assert result == mock_model


class TestLoRAConfig:
    """Test LoRA configuration."""

    def test_lora_config_defaults(self):
        """Test LoRAConfig with defaults."""
        from densecore.lora import LoRAConfig

        config = LoRAConfig(adapter_path="./adapter.gguf")

        assert config.adapter_path == "./adapter.gguf"
        assert config.scale == 1.0
        assert config.enabled is True

    def test_lora_config_validation(self):
        """Test LoRAConfig validation."""
        from densecore.lora import LoRAConfig

        # Negative scale should raise error
        with pytest.raises(ValueError):
            LoRAConfig(adapter_path="./adapter.gguf", scale=-0.5)


class TestLoRAManager:
    """Test LoRA adapter management."""

    def test_lora_manager_load(self, tmp_path):
        """Test loading a LoRA adapter."""
        from densecore.lora import LoRAManager

        # Create a fake adapter file
        adapter_file = tmp_path / "adapter.gguf"
        adapter_file.write_text("fake adapter data")

        manager = LoRAManager()

        config = manager.load(
            name="test_adapter",
            adapter_path=str(adapter_file),
            scale=0.8,
        )

        assert config.scale == 0.8
        assert manager.is_active
        assert "test_adapter" in manager.list_adapters()

    def test_lora_manager_activate_deactivate(self, tmp_path):
        """Test activating/deactivating adapters."""
        from densecore.lora import LoRAManager

        adapter_file = tmp_path / "adapter.gguf"
        adapter_file.write_text("fake adapter data")

        manager = LoRAManager()
        manager.load("test", str(adapter_file), activate=True)

        assert manager.is_active

        manager.deactivate()
        assert not manager.is_active

        manager.activate("test")
        assert manager.is_active
