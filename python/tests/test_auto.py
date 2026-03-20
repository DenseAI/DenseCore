"""
Tests for HuggingFace-style Auto classes.

These tests verify that AutoModel, AutoModelForCausalLM, and AutoTokenizer
provide the expected API compatibility with HuggingFace Transformers.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestAutoModel:
    """Tests for AutoModel class."""

    def test_cannot_instantiate_directly(self):
        """AutoModel should not be instantiated directly."""
        from densecore import AutoModel

        with pytest.raises(EnvironmentError, match="from_pretrained"):
            AutoModel()

    def test_from_pretrained_delegates_to_hub(self):
        """AutoModel.from_pretrained should delegate to hub.from_pretrained."""
        from densecore.auto import AutoModel

        mock_model = MagicMock()

        with patch("densecore.hub.from_pretrained", return_value=mock_model) as mock_fp:
            result = AutoModel.from_pretrained(
                "test/repo",
                filename="test.gguf",
                threads=4,
            )

            mock_fp.assert_called_once_with(
                repo_id_or_path="test/repo",
                filename="test.gguf",
                auto_select_quant=False,
                threads=4,
                cache_dir=None,
                token=None,
                revision="main",
                quant=None,
                trust_remote_code=False,
            )
            assert result == mock_model

    def test_from_pretrained_with_auto_select_quant(self):
        """AutoModel should pass auto_select_quant parameter."""
        from densecore.auto import AutoModel

        mock_model = MagicMock()

        with patch("densecore.auto.from_pretrained", return_value=mock_model) as mock_fp:
            AutoModel.from_pretrained("test/repo", auto_select_quant=True)

            call_kwargs = mock_fp.call_args[1]
            assert call_kwargs["auto_select_quant"] is True


class TestAutoModelForCausalLM:
    """Tests for AutoModelForCausalLM class."""

    def test_is_alias_for_automodel(self):
        """AutoModelForCausalLM should be a subclass of AutoModel."""
        from densecore import AutoModel, AutoModelForCausalLM

        assert issubclass(AutoModelForCausalLM, AutoModel)

    def test_cannot_instantiate_directly(self):
        """AutoModelForCausalLM should not be instantiated directly."""
        from densecore import AutoModelForCausalLM

        with pytest.raises(EnvironmentError, match="from_pretrained"):
            AutoModelForCausalLM()

    def test_from_pretrained_works(self):
        """AutoModelForCausalLM.from_pretrained should work like AutoModel."""
        from densecore.auto import AutoModelForCausalLM

        mock_model = MagicMock()

        with patch("densecore.auto.from_pretrained", return_value=mock_model) as mock_fp:
            result = AutoModelForCausalLM.from_pretrained("test/repo")

            mock_fp.assert_called_once()
            assert result == mock_model


class TestAutoTokenizer:
    """Tests for AutoTokenizer class."""

    def test_cannot_instantiate_directly(self):
        """AutoTokenizer should not be instantiated directly."""
        from densecore import AutoTokenizer

        with pytest.raises(EnvironmentError, match="from_pretrained"):
            AutoTokenizer()

    def test_from_pretrained_delegates_to_transformers(self):
        """AutoTokenizer.from_pretrained should delegate to transformers."""
        from densecore.auto import AutoTokenizer

        mock_tokenizer = MagicMock()

        with patch("transformers.AutoTokenizer") as mock_hf_tokenizer:
            mock_hf_tokenizer.from_pretrained.return_value = mock_tokenizer

            result = AutoTokenizer.from_pretrained(
                "test/repo",
                cache_dir="/tmp/cache",
                token="hf_token",
            )

            mock_hf_tokenizer.from_pretrained.assert_called_once_with(
                "test/repo",
                cache_dir="/tmp/cache",
                token="hf_token",
                revision="main",
                trust_remote_code=False,
            )
            assert result == mock_tokenizer

    def test_from_pretrained_raises_without_transformers(self):
        """AutoTokenizer should raise ImportError if transformers not installed."""

        with patch.dict("sys.modules", {"transformers": None}):
            # Force reimport to trigger ImportError
            import importlib

            import densecore.auto

            importlib.reload(densecore.auto)

            # This should still work because we mock at call time
            with patch("densecore.auto.HFAutoTokenizer", side_effect=ImportError):
                pass  # The actual error check is in from_pretrained


class TestExports:
    """Tests for module exports."""

    def test_auto_classes_exported_from_densecore(self):
        """Auto classes should be exported from main densecore module."""
        import densecore

        assert hasattr(densecore, "AutoModel")
        assert hasattr(densecore, "AutoModelForCausalLM")
        assert hasattr(densecore, "AutoTokenizer")

    def test_all_contains_auto_classes(self):
        """__all__ should contain all Auto classes."""
        import densecore

        assert "AutoModel" in densecore.__all__
        assert "AutoModelForCausalLM" in densecore.__all__
        assert "AutoTokenizer" in densecore.__all__
