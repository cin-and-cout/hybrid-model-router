import os
import pytest
from unittest.mock import MagicMock
from routing_agent.cache import SemanticCache
from routing_agent.local_client import LocalClient
from routing_agent.executor import UnifiedExecutor, UnifiedExecutionResult

def test_semantic_cache_exact_hit():
    mock_client = MagicMock(spec=LocalClient)
    # Return same mock vector for both prompts
    mock_client.get_embedding.return_value = [1.0, 0.0]
    
    cache_file = "test_cache.json"
    if os.path.exists(cache_file):
        os.remove(cache_file)
        
    cache = SemanticCache(local_client=mock_client, cache_file=cache_file)
    
    res = UnifiedExecutionResult(
        text="Cached hello response",
        source="local",
        escalated=False
    )
    
    cache.set("hello", res)
    
    # Retrieve exactly the same prompt
    cached_res = cache.get("hello")
    
    assert cached_res is not None
    assert cached_res.text == "Cached hello response"
    assert cached_res.source == "cache hit"
    assert cached_res.trust_report["original_source"] == "local"
    
    if os.path.exists(cache_file):
        os.remove(cache_file)

def test_semantic_cache_similarity_hit():
    mock_client = MagicMock(spec=LocalClient)
    
    # Mock embeddings: "hello" is [1.0, 0.0], "hi" is [0.98, 0.1]
    mock_client.get_embedding.side_effect = lambda text: {
        "hello": [1.0, 0.0],
        "hi": [0.98, 0.1]
    }.get(text, [0.0, 0.0])
    
    cache_file = "test_cache.json"
    if os.path.exists(cache_file):
        os.remove(cache_file)
        
    cache = SemanticCache(local_client=mock_client, cache_file=cache_file, similarity_threshold=0.95)
    
    res = UnifiedExecutionResult(
        text="Cached response",
        source="local",
        escalated=False
    )
    
    cache.set("hello", res)
    
    # Retrieve for "hi" which is 98% similar
    cached_res = cache.get("hi")
    
    assert cached_res is not None
    assert cached_res.text == "Cached response"
    assert cached_res.source == "cache hit"
    
    if os.path.exists(cache_file):
        os.remove(cache_file)

def test_semantic_cache_miss():
    mock_client = MagicMock(spec=LocalClient)
    
    # Mock embeddings: "hello" is [1.0, 0.0], "bye" is [0.0, 1.0]
    mock_client.get_embedding.side_effect = lambda text: {
        "hello": [1.0, 0.0],
        "bye": [0.0, 1.0]
    }.get(text, [0.0, 0.0])
    
    cache_file = "test_cache.json"
    if os.path.exists(cache_file):
        os.remove(cache_file)
        
    cache = SemanticCache(local_client=mock_client, cache_file=cache_file, similarity_threshold=0.90)
    
    res = UnifiedExecutionResult(
        text="Cached response",
        source="local",
        escalated=False
    )
    
    cache.set("hello", res)
    
    # Retrieve for "bye" (0.0 similarity)
    cached_res = cache.get("bye")
    
    assert cached_res is None
    
    if os.path.exists(cache_file):
        os.remove(cache_file)
