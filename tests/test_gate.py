import pytest
from unittest.mock import MagicMock
from routing_agent.gate import PredictiveRoutingGate
from routing_agent.adjuster import BudgetAwareAdjuster
from routing_agent.executor import UnifiedExecutor, UnifiedExecutionResult
from routing_agent.local_client import LocalClient, LocalExecutionResult
from routing_agent.remote_client import RemoteClient, RemoteExecutionResult

def test_predictive_gate_keywords():
    gate = PredictiveRoutingGate()
    
    # Mathematical integration keyword should trigger bypass for math or global check
    assert gate.should_bypass_local("Solve the integral of x^2", category="math") is True
    assert gate.should_bypass_local("Explain concurrency models in Go", category="code") is True
    assert gate.should_bypass_local("What is the weather today?", category="general") is False

def test_predictive_gate_length():
    gate = PredictiveRoutingGate(max_local_length=50)
    
    long_prompt = "This is a very long prompt designed to exceed the max local length limit of fifty characters."
    assert gate.should_bypass_local(long_prompt, category="general") is True

def test_predictive_gate_adjuster_reliability():
    mock_adjuster = MagicMock(spec=BudgetAwareAdjuster)
    mock_adjuster.category_stats = {
        "reasoning": {"correct": 0.0, "total": 4} # 0% reliability
    }
    
    gate = PredictiveRoutingGate(adjuster=mock_adjuster)
    
    # Should bypass local because of historical category reliability < 20%
    assert gate.should_bypass_local("Easy logic riddle", category="reasoning") is True

def test_unified_executor_predictive_bypass():
    mock_local = MagicMock(spec=LocalClient)
    mock_remote = MagicMock(spec=RemoteClient)
    mock_gate = MagicMock(spec=PredictiveRoutingGate)
    
    mock_gate.should_bypass_local.return_value = True
    mock_remote.query.return_value = RemoteExecutionResult(
        text="remote bypass response",
        total_tokens=50,
        raw_response={}
    )
    
    executor = UnifiedExecutor(
        local_client=mock_local,
        remote_client=mock_remote,
        predictive_gate=mock_gate
    )
    
    result = executor.execute(
        prompt="Solve integral of x",
        routing_strategy="dynamic",
        category="math"
    )
    
    assert result.source == "remote (predictive bypass)"
    assert result.text == "remote bypass response"
    assert result.local_tokens_used == 0
    assert result.remote_tokens_used == 50
    assert result.escalated is True
    
    mock_gate.should_bypass_local.assert_called_once_with("Solve integral of x", "math")
    mock_local.query.assert_not_called()
    mock_remote.query.assert_called_once()
