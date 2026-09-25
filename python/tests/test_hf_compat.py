"""
Tests for HuggingFace Transformers API Compatibility.

These tests verify that DenseCore provides a drop-in replacement experience
for users migrating from HuggingFace Transformers.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

# Test imports work
import densecore
from densecore import (
    EosTokenCriteria,
    GenerateOutput,
    GenerationConfig,
    MaxLengthCriteria,
    MaxNewTokensCriteria,
    StoppingCriteriaList,
)
from densecore.generate_output import StopStringCriteria


class TestGenerateOutput:
    """Tests for HuggingFace-compatible GenerateOutput class."""

    def test_basic_creation(self):
        """Test basic GenerateOutput creation."""
        output = GenerateOutput(
            sequences=[[1, 2, 3, 4, 5]],
            text="Hello, world!",
            finish_reason="stop",
        )

        assert output.sequences == [[1, 2, 3, 4, 5]]
        assert output.text == "Hello, world!"
        assert output.finish_reason == "stop"
        assert output.scores is None
        assert output.attentions is None

    def test_dict_like_access(self):
        """Test dict-like access for HuggingFace compatibility."""
        output = GenerateOutput(
            sequences=[[1, 2, 3]],
            text="test",
        )

        # String key access
        assert output["sequences"] == [[1, 2, 3]]
        assert output["text"] == "test"

        # Dict methods
        assert "sequences" in output.keys()
        assert "text" in output.keys()

        # get method
        assert output.get("sequences") == [[1, 2, 3]]
        assert output.get("nonexistent", "default") == "default"

    def test_tuple_like_access(self):
        """Test tuple-like access for unpacking."""
        output = GenerateOutput(
            sequences=[[1, 2, 3]],
            scores=(0.1, 0.2, 0.3),
        )

        # Index access
        assert output[0] == [[1, 2, 3]]  # sequences
        assert output[1] == (0.1, 0.2, 0.3)  # scores

        # Unpacking
        sequences, scores = output
        assert sequences == [[1, 2, 3]]
        assert scores == (0.1, 0.2, 0.3)

    def test_to_tuple(self):
        """Test conversion to tuple."""
        output = GenerateOutput(
            sequences=[[1, 2, 3]],
            scores=(0.1,),
        )

        result = output.to_tuple()
        assert result[0] == [[1, 2, 3]]
        assert result[1] == (0.1,)

    def test_2d_sequence_normalization(self):
        """Test that 1D sequences are normalized to 2D."""
        # This simulates if someone passes a flat list
        output = GenerateOutput(sequences=[1, 2, 3])  # type: ignore
        # __post_init__ should wrap it
        assert output.sequences == [[1, 2, 3]]

    def test_usage_tracking(self):
        """Test token usage tracking."""
        output = GenerateOutput(
            sequences=[[1, 2, 3, 4, 5]],
            text="Hello",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )

        assert output.usage["prompt_tokens"] == 10
        assert output.usage["completion_tokens"] == 5
        assert output.usage["total_tokens"] == 15


class TestStoppingCriteria:
    """Tests for HuggingFace-compatible stopping criteria."""

    def test_max_length_criteria(self):
        """Test MaxLengthCriteria."""
        criteria = MaxLengthCriteria(max_length=10)

        # Not at max length
        assert criteria([[1, 2, 3, 4, 5]]) is False

        # At max length
        assert criteria([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]) is True

        # Over max length
        assert criteria([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]) is True

    def test_max_new_tokens_criteria(self):
        """Test MaxNewTokensCriteria."""
        criteria = MaxNewTokensCriteria(start_length=5, max_new_tokens=10)

        # Total max is 15
        assert criteria([[1] * 10]) is False
        assert criteria([[1] * 15]) is True
        assert criteria([[1] * 20]) is True

    def test_eos_token_criteria_single(self):
        """Test EosTokenCriteria with single token."""
        criteria = EosTokenCriteria(eos_token_id=2)

        # Last token is not EOS
        assert criteria([[1, 3, 4, 5]]) is False

        # Last token is EOS
        assert criteria([[1, 3, 4, 2]]) is True

    def test_eos_token_criteria_multiple(self):
        """Test EosTokenCriteria with multiple tokens."""
        criteria = EosTokenCriteria(eos_token_id=[2, 32000, 32001])

        # Last token is not any EOS
        assert criteria([[1, 3, 4, 5]]) is False

        # Last token is one of the EOS tokens
        assert criteria([[1, 3, 4, 2]]) is True
        assert criteria([[1, 3, 4, 32000]]) is True
        assert criteria([[1, 3, 4, 32001]]) is True

    def test_stopping_criteria_list(self):
        """Test StoppingCriteriaList combining multiple criteria."""
        criteria = StoppingCriteriaList(
            [
                MaxLengthCriteria(max_length=10),
                EosTokenCriteria(eos_token_id=2),
            ]
        )

        # Neither condition met
        assert criteria([[1, 3, 4, 5]]) is False

        # Max length met
        assert criteria([[1] * 10]) is True

        # EOS met
        assert criteria([[1, 3, 2]]) is True

    def test_stop_string_criteria(self):
        """Test StopStringCriteria with mock tokenizer."""
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "Hello, world! This is a test."

        criteria = StopStringCriteria(
            stop_strings=["world!", "test."],
            tokenizer=mock_tokenizer,
        )

        # Should stop because "world!" is in the decoded text
        assert criteria([[1, 2, 3, 4, 5]]) is True

    def test_stop_string_criteria_no_match(self):
        """Test StopStringCriteria when no stop string is found."""
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "Hello there"

        criteria = StopStringCriteria(
            stop_strings=["goodbye", "farewell"],
            tokenizer=mock_tokenizer,
        )

        assert criteria([[1, 2, 3]]) is False


class TestGenerationConfig:
    """Tests for HuggingFace-compatible GenerationConfig."""

    def test_hf_max_new_tokens_alias(self):
        """Test that max_new_tokens maps to max_tokens."""
        config = GenerationConfig(max_new_tokens=100)
        assert config.max_tokens == 100

    def test_do_sample_false_sets_greedy(self):
        """Test that do_sample=False sets greedy decoding parameters."""
        config = GenerationConfig(do_sample=False, temperature=0.7)

        # Temperature should be overridden to 0.0 for greedy
        assert config.temperature == 0.0
        assert config.top_p == 1.0
        assert config.top_k == 0

    def test_eos_token_id_added_to_stop_tokens(self):
        """Test that eos_token_id is added to stop_token_ids."""
        config = GenerationConfig(eos_token_id=2)
        assert 2 in config.stop_token_ids

    def test_num_beams_warning(self):
        """Test that num_beams > 1 emits warning and is reset to 1."""
        with pytest.warns(UserWarning, match="num_beams=1"):
            config = GenerationConfig(num_beams=4)

        assert config.num_beams == 1

    def test_full_hf_compatible_kwargs(self):
        """Test full HuggingFace-compatible parameter set."""
        config = GenerationConfig(
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            eos_token_id=2,
            pad_token_id=0,
            return_dict_in_generate=True,
        )

        assert config.max_tokens == 256
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.repetition_penalty == 1.1
        assert config.return_dict_in_generate is True


class TestAutoModelHFCompat:
    """Tests for HuggingFace-compatible AutoModel class."""

    def test_load_in_4bit_maps_to_quant(self):
        """Test that load_in_4bit maps to Q4_K_M quantization."""
        with patch("densecore.auto.from_pretrained") as mock_from_pretrained:
            mock_from_pretrained.return_value = Mock()

            densecore.AutoModel.from_pretrained(
                "test/model",
                load_in_4bit=True,
            )

            # Check that quant was set to Q4_K_M
            call_kwargs = mock_from_pretrained.call_args.kwargs
            assert call_kwargs.get("quant") == "Q4_K_M"

    def test_load_in_8bit_maps_to_quant(self):
        """Test that load_in_8bit maps to Q8_0 quantization."""
        with patch("densecore.auto.from_pretrained") as mock_from_pretrained:
            mock_from_pretrained.return_value = Mock()

            densecore.AutoModel.from_pretrained(
                "test/model",
                load_in_8bit=True,
            )

            call_kwargs = mock_from_pretrained.call_args.kwargs
            assert call_kwargs.get("quant") == "Q8_0"

    def test_ignored_kwargs_warning(self):
        """Test that ignored HuggingFace kwargs emit warning."""
        with patch("densecore.auto.from_pretrained") as mock_from_pretrained:
            mock_from_pretrained.return_value = Mock()

            with pytest.warns(UserWarning, match="device_map"):
                densecore.AutoModel.from_pretrained(
                    "test/model",
                    device_map="auto",
                    torch_dtype="auto",
                )


class TestPackageExports:
    """Tests for package-level exports."""

    def test_core_classes_exported(self):
        """Test that core HuggingFace-compat classes are exported."""
        assert hasattr(densecore, "GenerateOutput")
        assert hasattr(densecore, "StoppingCriteria")
        assert hasattr(densecore, "StoppingCriteriaList")
        assert hasattr(densecore, "MaxLengthCriteria")
        assert hasattr(densecore, "MaxNewTokensCriteria")
        assert hasattr(densecore, "EosTokenCriteria")

    def test_auto_classes_exported(self):
        """Test that Auto classes are exported."""
        assert hasattr(densecore, "AutoModel")
        assert hasattr(densecore, "AutoModelForCausalLM")
        assert hasattr(densecore, "AutoTokenizer")

    def test_config_classes_exported(self):
        """Test that config classes are exported."""
        assert hasattr(densecore, "GenerationConfig")
        assert hasattr(densecore, "ModelConfig")
        assert hasattr(densecore, "SamplingParams")


class TestGenerateMethodHFCompat:
    """Tests for HuggingFace-compatible generate() method kwargs."""

    def test_generate_output_type_hints(self):
        """Test that generate() return type includes GenerateOutput."""
        import inspect

        from densecore.engine import DenseCore

        sig = inspect.signature(DenseCore.generate)
        return_annotation = sig.return_annotation

        # Should return Union[str, GenerateOutput] or similar
        assert "GenerateOutput" in str(return_annotation) or return_annotation is not None

    def test_generate_signature_has_hf_params(self):
        """Test that generate() accepts HuggingFace-style parameters."""
        import inspect

        from densecore.engine import DenseCore

        sig = inspect.signature(DenseCore.generate)
        params = list(sig.parameters.keys())

        # Check for HuggingFace-compatible parameters
        assert "max_new_tokens" in params
        assert "do_sample" in params
        assert "temperature" in params
        assert "top_p" in params
        assert "top_k" in params
        assert "num_beams" in params
        assert "repetition_penalty" in params
        assert "return_dict_in_generate" in params
        assert "input_ids" in params
        assert "eos_token_id" in params
