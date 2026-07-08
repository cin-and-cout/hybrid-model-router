import pytest
from unittest.mock import MagicMock
from routing_agent.evaluator import TrustEvaluator
from routing_agent.local_client import LocalClient, LocalExecutionResult

def test_compute_self_consistency_math_majority():
    # Setup mock local client returning consistent values
    mock_client = MagicMock(spec=LocalClient)
    # 2 out of 3 runs return "7", 1 returns "8"
    mock_client.query.side_effect = [
        LocalExecutionResult(text="7", raw_response={}),
        LocalExecutionResult(text="8", raw_response={}),
        LocalExecutionResult(text="7", raw_response={})
    ]
    
    evaluator = TrustEvaluator(local_client=mock_client)
    score = evaluator.compute_self_consistency(
        prompt="Solve 3x-7=14",
        category="math",
        n=3,
        temperature=0.7
    )
    
    # 2/3 agreement
    assert score == pytest.approx(2.0 / 3.0)
    assert mock_client.query.call_count == 3

def test_compute_self_consistency_text_pairwise():
    mock_client = MagicMock(spec=LocalClient)
    # Three outputs with partial overlap
    mock_client.query.side_effect = [
        LocalExecutionResult(text="hello world", raw_response={}),
        LocalExecutionResult(text="hello", raw_response={}),
        LocalExecutionResult(text="world", raw_response={})
    ]
    
    evaluator = TrustEvaluator(local_client=mock_client)
    score = evaluator.compute_self_consistency(
        prompt="Write greeting",
        category="code",
        n=3,
        temperature=0.7
    )
    
    # Pairwise overlaps (Jaccard):
    # pair 1: "hello world" vs "hello" -> intersection {"hello"}, union {"hello", "world"} -> 0.5
    # pair 2: "hello world" vs "world" -> intersection {"world"}, union {"hello", "world"} -> 0.5
    # pair 3: "hello" vs "world" -> intersection {}, union {"hello", "world"} -> 0.0
    # Average: (0.5 + 0.5 + 0.0) / 3 = 0.3333
    assert score == pytest.approx(1.0 / 3.0)
