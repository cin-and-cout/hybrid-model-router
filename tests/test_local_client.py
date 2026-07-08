import pytest
from unittest.mock import MagicMock
from routing_agent.local_client import LocalClient, LocalExecutionResult

def test_local_client_query_success(mocker):
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1783521425,
        "model": "qwen2.5:0.5b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "2 plus 2 is 4"
                },
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {
                            "token": "2",
                            "logprob": -0.1,
                            "top_logprobs": [
                                {"token": "2", "logprob": -0.1},
                                {"token": "two", "logprob": -2.5}
                            ]
                        },
                        {
                            "token": " plus",
                            "logprob": -0.05,
                            "top_logprobs": [
                                {"token": " plus", "logprob": -0.05},
                                {"token": " +", "logprob": -3.0}
                            ]
                        }
                    ]
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12
        }
    }
    
    mock_post = mocker.patch("httpx.Client.post")
    mock_post.return_value = MagicMock(status_code=200, json=lambda: mock_response)
    
    client = LocalClient(base_url="http://localhost:11434", model="qwen2.5:0.5b")
    result = client.query("What is 2+2?")
    
    assert isinstance(result, LocalExecutionResult)
    assert result.text == "2 plus 2 is 4"
    assert len(result.tokens) == 2
    assert result.tokens[0].token == "2"
    assert result.tokens[0].logprob == -0.1
    assert result.tokens[1].token == " plus"
    assert result.tokens[1].logprob == -0.05
    
    # Check that summary statistics are correctly calculated
    assert result.mean_logprob == pytest.approx(-0.075)
    assert result.min_logprob == -0.1
    assert result.mean_entropy > 0.0

def test_local_client_query_empty_response(mocker):
    mock_response = {
        "id": "chatcmpl-124",
        "choices": []
    }
    
    mock_post = mocker.patch("httpx.Client.post")
    mock_post.return_value = MagicMock(status_code=200, json=lambda: mock_response)
    
    client = LocalClient(base_url="http://localhost:11434", model="qwen2.5:0.5b")
    result = client.query("Empty check")
    
    assert result.text == ""
    assert len(result.tokens) == 0
    assert result.mean_logprob == 0.0
    assert result.min_logprob == 0.0
    assert result.mean_entropy == 0.0

def test_local_client_real_query():
    import os
    import httpx
    # Within the app container, OLLAMA_HOST environment variable points to http://ollama:11434
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    
    try:
        response = httpx.get(f"{ollama_host}/api/tags", timeout=2.0)
        assert response.status_code == 200
    except Exception:
        pytest.skip("Ollama server is not reachable; skipping integration test.")
        
    client = LocalClient(base_url=ollama_host, model="qwen2.5:0.5b")
    result = client.query("What is 1+1? Answer with just the digit.")
    
    assert isinstance(result, LocalExecutionResult)
    assert len(result.text.strip()) > 0
    assert "2" in result.text
    assert len(result.tokens) > 0
    assert result.mean_logprob < 0.0
    assert result.mean_entropy >= 0.0
