import pytest
from unittest.mock import MagicMock
from routing_agent.local_client import LocalClient, LocalExecutionResult
from routing_agent.remote_client import RemoteClient, RemoteExecutionResult
from routing_agent.evaluator import TrustEvaluator
from routing_agent.executor import UnifiedExecutor

def test_three_tier_cascade_success_at_tier_0():
    mock_local_1 = MagicMock(spec=LocalClient)
    mock_local_2 = MagicMock(spec=LocalClient)
    mock_remote = MagicMock(spec=RemoteClient)
    mock_evaluator = MagicMock(spec=TrustEvaluator)
    
    mock_local_1.query.return_value = LocalExecutionResult(text="tier 0 output", total_tokens=5, raw_response={})
    mock_evaluator.evaluate_trust.return_value = {
        "escalate": False,
        "signals": {},
        "failures": {},
        "consistency_tokens": 10
    }
    
    cascade = [
        {"client": mock_local_1, "type": "local", "name": "local-0.5B"},
        {"client": mock_local_2, "type": "local", "name": "local-7B"},
        {"client": mock_remote, "type": "remote", "name": "remote-70B"}
    ]
    
    executor = UnifiedExecutor(
        local_client=mock_local_1,
        remote_client=mock_remote,
        trust_evaluator=mock_evaluator,
        cascade=cascade
    )
    
    result = executor.execute(
        prompt="Solve math",
        routing_strategy="dynamic",
        category="math"
    )
    
    assert result.source == "local (local-0.5B)"
    assert result.text == "tier 0 output"
    assert result.local_tokens_used == 15
    mock_local_1.query.assert_called_once()
    mock_local_2.query.assert_not_called()
    mock_remote.query.assert_not_called()

def test_three_tier_cascade_escalates_to_tier_1():
    mock_local_1 = MagicMock(spec=LocalClient)
    mock_local_2 = MagicMock(spec=LocalClient)
    mock_remote = MagicMock(spec=RemoteClient)
    mock_evaluator = MagicMock(spec=TrustEvaluator)
    
    mock_local_1.query.return_value = LocalExecutionResult(text="tier 0 output", total_tokens=5, raw_response={})
    mock_local_2.query.return_value = LocalExecutionResult(text="tier 1 output", total_tokens=20, raw_response={})
    
    # First trust evaluation (tier 0) escalates, second (tier 1) does not
    mock_evaluator.evaluate_trust.side_effect = [
        {
            "escalate": True,
            "signals": {},
            "failures": {},
            "consistency_tokens": 10
        },
        {
            "escalate": False,
            "signals": {},
            "failures": {},
            "consistency_tokens": 0
        }
    ]
    
    cascade = [
        {"client": mock_local_1, "type": "local", "name": "local-0.5B"},
        {"client": mock_local_2, "type": "local", "name": "local-7B"},
        {"client": mock_remote, "type": "remote", "name": "remote-70B"}
    ]
    
    executor = UnifiedExecutor(
        local_client=mock_local_1,
        remote_client=mock_remote,
        trust_evaluator=mock_evaluator,
        cascade=cascade
    )
    
    result = executor.execute(
        prompt="Solve logic puzzle",
        routing_strategy="dynamic",
        category="reasoning"
    )
    
    assert result.source == "local (local-7B)"
    assert result.text == "tier 1 output"
    # Local tokens accumulated: tier 0 query (5) + tier 0 consistency (10) + tier 1 query (20) = 35
    assert result.local_tokens_used == 35
    mock_local_1.query.assert_called_once()
    mock_local_2.query.assert_called_once()
    mock_remote.query.assert_not_called()

def test_three_tier_cascade_escalates_to_remote_tier_2():
    mock_local_1 = MagicMock(spec=LocalClient)
    mock_local_2 = MagicMock(spec=LocalClient)
    mock_remote = MagicMock(spec=RemoteClient)
    mock_evaluator = MagicMock(spec=TrustEvaluator)
    
    mock_local_1.query.return_value = LocalExecutionResult(text="tier 0 output", total_tokens=5, raw_response={})
    mock_local_2.query.return_value = LocalExecutionResult(text="tier 1 output", total_tokens=20, raw_response={})
    mock_remote.query.return_value = RemoteExecutionResult(text="remote output", total_tokens=50, raw_response={})
    
    # Both local trust evaluations escalate
    mock_evaluator.evaluate_trust.side_effect = [
        {
            "escalate": True,
            "signals": {},
            "failures": {},
            "consistency_tokens": 10
        },
        {
            "escalate": True,
            "signals": {},
            "failures": {},
            "consistency_tokens": 5
        }
    ]
    
    cascade = [
        {"client": mock_local_1, "type": "local", "name": "local-0.5B"},
        {"client": mock_local_2, "type": "local", "name": "local-7B"},
        {"client": mock_remote, "type": "remote", "name": "remote-70B"}
    ]
    
    executor = UnifiedExecutor(
        local_client=mock_local_1,
        remote_client=mock_remote,
        trust_evaluator=mock_evaluator,
        cascade=cascade
    )
    
    result = executor.execute(
        prompt="Hard puzzle",
        routing_strategy="dynamic",
        category="reasoning"
    )
    
    assert result.source == "remote (remote-70B)"
    assert result.text == "remote output"
    # Local tokens accumulated: tier 0 (5+10) + tier 1 (20+5) = 40
    assert result.local_tokens_used == 40
    assert result.remote_tokens_used == 50
    mock_local_1.query.assert_called_once()
    mock_local_2.query.assert_called_once()
    mock_remote.query.assert_called_once()
