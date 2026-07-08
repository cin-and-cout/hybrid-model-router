import pytest
from unittest.mock import MagicMock
from routing_agent.evaluator import TrustEvaluator
from routing_agent.local_client import LocalClient, LocalExecutionResult, TokenDetail

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

def test_compute_entropy_signal():
    evaluator = TrustEvaluator()
    tokens = [
        TokenDetail(token="hello", logprob=-0.1, entropy=0.2),
        TokenDetail(token="world", logprob=-2.3, entropy=1.5),
        TokenDetail(token="!", logprob=-0.01, entropy=0.0)
    ]
    res = LocalExecutionResult(
        text="hello world!",
        tokens=tokens,
        mean_logprob=-0.8,
        min_logprob=-2.3,
        mean_entropy=0.57,
        raw_response={}
    )
    
    sig = evaluator.compute_entropy_signal(res, high_entropy_threshold=1.0)
    
    assert sig["mean_entropy"] == 0.57
    assert sig["min_logprob"] == -2.3
    assert sig["high_entropy_ratio"] == pytest.approx(1.0 / 3.0)

def test_verify_structure():
    evaluator = TrustEvaluator()
    
    # Structured output (JSON) validation
    assert evaluator.verify_structure('{"name": "Alice"}', "structured_output") is True
    assert evaluator.verify_structure('{"name": "Alice"}', "structured_output", ["age"]) is False
    assert evaluator.verify_structure('invalid json', "structured_output") is False
    
    # Code (python syntax compilation) validation
    valid_code = "def add(a, b):\n    return a + b"
    invalid_code = "def add(a, b)\n    return a + b" # SyntaxError: missing colon
    
    assert evaluator.verify_structure(valid_code, "code") is True
    assert evaluator.verify_structure(invalid_code, "code") is False
    
    # Wrapped in markdown
    wrapped_valid = "```python\nclass Test:\n    pass\n```"
    wrapped_invalid = "```python\nclass Test\n    pass\n```"
    
    assert evaluator.verify_structure(wrapped_valid, "code") is True
    assert evaluator.verify_structure(wrapped_invalid, "code") is False
    
    # Other categories should pass by default
    assert evaluator.verify_structure("plain text", "math") is True

def test_evaluate_trust_all_pass():
    mock_client = MagicMock(spec=LocalClient)
    mock_client.query.return_value = LocalExecutionResult(text="7", raw_response={})
    
    evaluator = TrustEvaluator(
        local_client=mock_client,
        consistency_threshold=0.6,
        entropy_threshold=0.8,
        consistency_n=3,
        consistency_temp=0.7
    )
    
    local_res = LocalExecutionResult(
        text="7",
        mean_entropy=0.3,
        raw_response={}
    )
    
    trust_report = evaluator.evaluate_trust(
        prompt="Solve 3x-7=14",
        local_result=local_res,
        category="math"
    )
    
    assert trust_report["escalate"] is False
    assert trust_report["signals"]["structural_valid"] is True
    assert trust_report["signals"]["mean_entropy"] == 0.3
    assert trust_report["signals"]["self_consistency"] == 1.0
    assert trust_report["failures"]["structural"] is False
    assert trust_report["failures"]["entropy"] is False
    assert trust_report["failures"]["consistency"] is False

def test_evaluate_trust_escalation_triggers():
    mock_client = MagicMock(spec=LocalClient)
    
    # 1. Structural failure triggers escalation
    mock_client.query.return_value = LocalExecutionResult(text="invalid json", raw_response={})
    evaluator = TrustEvaluator(local_client=mock_client, entropy_threshold=0.8, consistency_threshold=0.6)
    local_res = LocalExecutionResult(text="invalid json", mean_entropy=0.2, raw_response={})
    
    report = evaluator.evaluate_trust("Get user", local_res, "structured_output")
    assert report["escalate"] is True
    assert report["failures"]["structural"] is True
    assert report["failures"]["entropy"] is False
    
    # 2. High entropy triggers escalation
    mock_client.query.return_value = LocalExecutionResult(text="7", raw_response={})
    local_res = LocalExecutionResult(text="7", mean_entropy=0.9, raw_response={})
    report = evaluator.evaluate_trust("Solve math", local_res, "math")
    assert report["escalate"] is True
    assert report["failures"]["structural"] is False
    assert report["failures"]["entropy"] is True
    assert report["failures"]["consistency"] is False
    
    # 3. Low consistency triggers escalation
    mock_client.query.side_effect = [
        LocalExecutionResult(text="7", raw_response={}),
        LocalExecutionResult(text="8", raw_response={}),
        LocalExecutionResult(text="9", raw_response={})
    ]
    local_res = LocalExecutionResult(text="7", mean_entropy=0.2, raw_response={})
    report = evaluator.evaluate_trust("Solve math", local_res, "math")
    assert report["escalate"] is True
    assert report["failures"]["structural"] is False
    assert report["failures"]["entropy"] is False
    assert report["failures"]["consistency"] is True
