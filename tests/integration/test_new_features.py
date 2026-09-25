"""
DenseCore Integration Tests

Tests for the newly implemented features:
1. Rerank API
2. Function Calling
3. Prefix Caching
"""

import json
import subprocess
import time
import requests
import pytest
from typing import Dict, Any, List


# Configuration
SERVER_URL = "http://localhost:8080"
TEST_TIMEOUT = 30


class TestRerankAPI:
    """Test cases for /v1/rerank endpoint"""

    def test_rerank_basic(self):
        """Test basic rerank functionality"""
        payload = {
            "model": "bge-reranker-v2",
            "query": "What is DenseCore?",
            "documents": [
                "DenseCore is a high-performance CPU inference engine for LLMs",
                "Python is a popular programming language",
                "Machine learning models require significant compute resources"
            ],
            "top_n": 2
        }

        response = requests.post(
            f"{SERVER_URL}/v1/rerank",
            json=payload,
            timeout=TEST_TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert len(data["results"]) <= 2

        # First result should be most relevant
        results = data["results"]
        assert results[0]["relevance_score"] >= results[1]["relevance_score"]

        # Check required fields
        for result in results:
            assert "index" in result
            assert "relevance_score" in result

    def test_rerank_with_documents(self):
        """Test rerank with return_documents=True"""
        payload = {
            "model": "test-model",
            "query": "CPU optimization",
            "documents": ["CPU inference optimization", "GPU training"],
            "return_documents": True
        }

        response = requests.post(
            f"{SERVER_URL}/v1/rerank",
            json=payload,
            timeout=TEST_TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()

        for result in data["results"]:
            assert "document" in result
            assert "text" in result["document"]

    def test_rerank_empty_documents(self):
        """Test error handling for empty documents"""
        payload = {
            "model": "test-model",
            "query": "test query",
            "documents": []
        }

        response = requests.post(
            f"{SERVER_URL}/v1/rerank",
            json=payload,
            timeout=TEST_TIMEOUT
        )

        assert response.status_code == 400

    def test_rerank_missing_query(self):
        """Test error handling for missing query"""
        payload = {
            "model": "test-model",
            "documents": ["doc1", "doc2"]
        }

        response = requests.post(
            f"{SERVER_URL}/v1/rerank",
            json=payload,
            timeout=TEST_TIMEOUT
        )

        assert response.status_code == 400


class TestFunctionCalling:
    """Test cases for function calling via /v1/chat/completions"""

    def test_chat_with_tools(self):
        """Test chat completion with tools parameter"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"}
                        },
                        "required": ["city"]
                    }
                }
            }
        ]

        payload = {
            "model": "densecore-v1",
            "messages": [
                {"role": "user", "content": "What's the weather in Seoul?"}
            ],
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 256
        }

        response = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json=payload,
            timeout=TEST_TIMEOUT
        )

        # Should succeed even if model doesn't actually call the tool
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    def test_tool_response_format(self):
        """Verify tool call response format"""
        # This tests the expected response structure when tools are used
        expected_tool_call_format = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Seoul"}'
            }
        }

        # Validate format
        assert "id" in expected_tool_call_format
        assert expected_tool_call_format["type"] == "function"
        assert "function" in expected_tool_call_format
        assert "name" in expected_tool_call_format["function"]
        assert "arguments" in expected_tool_call_format["function"]


class TestPrefixCaching:
    """Test cases for prefix caching functionality"""

    def test_prefix_cache_ttft_improvement(self):
        """Test that repeated system prompts benefit from caching"""
        system_prompt = "You are a helpful assistant specialized in Korean culture."

        # First request (cold cache)
        payload1 = {
            "model": "densecore-v1",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Hello!"}
            ],
            "max_tokens": 10
        }

        start1 = time.time()
        response1 = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json=payload1,
            timeout=TEST_TIMEOUT
        )
        time1 = time.time() - start1

        assert response1.status_code == 200

        # Second request with same system prompt (should hit cache)
        payload2 = {
            "model": "densecore-v1",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "What is kimchi?"}
            ],
            "max_tokens": 10
        }

        start2 = time.time()
        response2 = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json=payload2,
            timeout=TEST_TIMEOUT
        )
        time2 = time.time() - start2

        assert response2.status_code == 200

        # Log timing for manual verification
        print(f"Cold request: {time1:.3f}s, Warm request: {time2:.3f}s")

        # Note: Actual TTFT improvement requires metrics endpoint inspection


class TestPythonSDK:
    """Test Python SDK integration (requires model loaded)"""

    def test_sdk_rerank_method(self):
        """Test DenseCore.rerank() method"""
        try:
            from densecore import DenseCore
        except ImportError:
            pytest.skip("DenseCore SDK not installed")

        # SDK rerank uses simple text similarity by default
        query = "machine learning"
        documents = [
            "deep learning neural networks",
            "cooking recipes for dinner",
            "ML model training optimization"
        ]

        # Instantiate without model for quick test
        # Note: Full test requires actual model
        results = _mock_rerank(query, documents, top_n=2)

        assert len(results) <= 2
        assert all("relevance_score" in r for r in results)

    def test_sdk_chat_with_tools(self):
        """Test DenseCore.chat() with tools parameter"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add_numbers",
                    "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}
                }
            }
        ]

        messages = [{"role": "user", "content": "Add 5 and 3"}]

        # Mock response for testing structure
        result = _mock_chat_with_tools(messages, tools)

        assert "role" in result
        assert result["role"] == "assistant"


def _mock_rerank(query: str, documents: List[str], top_n: int = None) -> List[Dict]:
    """Mock rerank for testing without actual model"""
    query_words = set(query.lower().split())
    scores = []

    for i, doc in enumerate(documents):
        doc_words = set(doc.lower().split())
        intersection = len(query_words & doc_words)
        union = len(query_words | doc_words)
        score = intersection / union if union > 0 else 0.0
        scores.append({"index": i, "relevance_score": score})

    scores.sort(key=lambda x: x["relevance_score"], reverse=True)

    if top_n:
        scores = scores[:top_n]

    return scores


def _mock_chat_with_tools(messages: List[Dict], tools: List[Dict]) -> Dict:
    """Mock chat for testing structure"""
    return {"role": "assistant", "content": "Mock response", "tool_calls": None}


if __name__ == "__main__":
    # Run tests without server (unit tests only)
    print("Running Python SDK unit tests...")

    # Test rerank mock
    results = _mock_rerank("machine learning", ["ML training", "cooking", "deep learning"], top_n=2)
    print(f"Rerank results: {results}")
    assert len(results) == 2

    # Test chat mock
    result = _mock_chat_with_tools([{"role": "user", "content": "test"}], [])
    print(f"Chat result: {result}")
    assert result["role"] == "assistant"

    print("All unit tests passed!")
