"""
Tests for DenseCore Python package.
"""

from unittest.mock import MagicMock, patch


class TestConfig:
    """Test configuration classes."""

    def test_generation_config_defaults(self):
        from densecore.config import GenerationConfig

        config = GenerationConfig()
        assert config.max_tokens == 256
        assert config.temperature == 1.0
        assert config.top_p == 1.0

    def test_generation_config_custom(self):
        from densecore.config import GenerationConfig

        config = GenerationConfig(
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
        )
        assert config.max_tokens == 100
        assert config.temperature == 0.7
        assert config.top_p == 0.9

    def test_generation_config_hf_alias(self):
        """Test max_new_tokens alias for HuggingFace compatibility."""
        from densecore.config import GenerationConfig

        config = GenerationConfig(max_new_tokens=128)
        assert config.max_tokens == 128

    def test_model_config_to_dict(self):
        from densecore.config import ModelConfig

        config = ModelConfig(
            model_path="./test.gguf",
            threads=4,
        )
        d = config.to_dict()
        assert d["model_path"] == "./test.gguf"
        assert d["threads"] == 4

    def test_model_config_from_dict(self):
        from densecore.config import ModelConfig

        config = ModelConfig.from_dict(
            {
                "model_path": "./test.gguf",
                "threads": 8,
            }
        )
        assert config.model_path == "./test.gguf"
        assert config.threads == 8


class TestHub:
    """Test HuggingFace Hub integration."""

    def test_format_size(self):
        from densecore.hub import _format_size

        assert "B" in _format_size(500)
        assert "KB" in _format_size(1024)
        assert "MB" in _format_size(1024 * 1024)
        assert "GB" in _format_size(1024 * 1024 * 1024)

    @patch("densecore.hub.HfApi")
    def test_list_gguf_files(self, mock_api_class):
        from densecore.hub import list_gguf_files

        mock_api = MagicMock()
        mock_api.list_repo_files.return_value = [
            "model.Q4_K_M.gguf",
            "model.Q8_0.gguf",
            "README.md",
        ]
        mock_api.get_paths_info.return_value = []
        mock_api_class.return_value = mock_api

        files = list_gguf_files("test/repo")

        assert len(files) == 2
        assert files[0]["quant"] == "Q8_0"  # Higher quality first
        assert files[1]["quant"] == "Q4_K_M"


class TestSamplingParams:
    """Test vLLM-compatible SamplingParams."""

    def test_to_generation_config(self):
        from densecore.config import SamplingParams

        params = SamplingParams(
            max_tokens=100,
            temperature=0.8,
            stop=["END"],
        )

        config = params.to_generation_config()
        assert config.max_tokens == 100
        assert config.temperature == 0.8
        assert config.stop_sequences == ["END"]


class TestPackage:
    """Test package imports."""

    def test_imports(self):
        import densecore

        assert hasattr(densecore, "DenseCore")
        assert hasattr(densecore, "GenerationConfig")
        assert hasattr(densecore, "ModelConfig")
        assert hasattr(densecore, "from_pretrained")
        assert hasattr(densecore, "__version__")

    def test_version(self):
        import densecore

        assert densecore.__version__ == "1.0.0"

    def test_get_device_info(self):
        import densecore

        info = densecore.get_device_info()
        assert "platform" in info
        assert "cpu_count" in info
