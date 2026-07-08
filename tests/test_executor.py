import pytest
from unittest.mock import MagicMock
from routing_agent.executor import UnifiedExecutor, UnifiedExecutionResult
from routing_agent.local_client import LocalClient, LocalExecutionResult
from routing_agent.remote_client import RemoteClient, RemoteExecutionResult
from routing_agent.evaluator import TrustEvaluator
from routing_agent.adjuster import BudgetAwareAdjuster

def test_unified_executor_route_local():
    mock_local = MagicMock(spec=LocalClient)
    mock_remote = MagicMock(spec=RemoteClient)
    
    mock_local.query.return_value = LocalExecutionResult(
        text="local answer",
        total_tokens=15,
        raw_response={}
    )
    
    executor = UnifiedExecutor(local_client=mock_local, remote_client=mock_remote)
    result = executor.execute("What is 1+1?", use_remote=False)
    
    assert isinstance(result, UnifiedExecutionResult)
    assert result.source == "local"
    assert result.text == "local answer"
    assert result.local_tokens_used == 15
    assert result.remote_tokens_used == 0
    assert result.local_result is not None
    assert result.remote_result is None
    
    mock_local.query.assert_called_once_with(
        prompt="What is 1+1?",
        temperature=0.0,
        max_tokens=None,
        system_prompt=None
    )
    mock_remote.query.assert_not_called()

def test_unified_executor_route_remote():
    mock_local = MagicMock(spec=LocalClient)
    mock_remote = MagicMock(spec=RemoteClient)
    
    mock_remote.query.return_value = RemoteExecutionResult(
        text="remote answer",
        total_tokens=45,
        raw_response={}
    )
    
    executor = UnifiedExecutor(local_client=mock_local, remote_client=mock_remote)
    result = executor.execute("What is 1+1?", use_remote=True, temperature=0.5, max_tokens=50)
    
    assert isinstance(result, UnifiedExecutionResult)
    assert result.source == "remote"
    assert result.text == "remote answer"
    assert result.local_tokens_used == 0
    assert result.remote_tokens_used == 45
    assert result.local_result is None
    assert result.remote_result is not None
    
    mock_remote.query.assert_called_once_with(
        prompt="What is 1+1?",
        temperature=0.5,
        max_tokens=50,
        system_prompt=None
    )
    mock_local.query.assert_not_called()

def test_unified_executor_dynamic_no_escalate():
    mock_local = MagicMock(spec=LocalClient)
    mock_remote = MagicMock(spec=RemoteClient)
    mock_evaluator = MagicMock(spec=TrustEvaluator)
    
    mock_local.query.return_value = LocalExecutionResult(
        text="local output",
        total_tokens=10,
        raw_response={}
    )
    
    mock_evaluator.evaluate_trust.return_value = {
        "escalate": False,
        "signals": {"structural_valid": True, "mean_entropy": 0.2, "self_consistency": 1.0},
        "failures": {"structural": False, "entropy": False, "consistency": False},
        "consistency_tokens": 30
    }
    
    executor = UnifiedExecutor(
        local_client=mock_local,
        remote_client=mock_remote,
        trust_evaluator=mock_evaluator
    )
    
    result = executor.execute(
        prompt="Solve math",
        routing_strategy="dynamic",
        category="math"
    )
    
    assert result.source == "local"
    assert result.text == "local output"
    assert result.local_tokens_used == 40  # 10 initial + 30 self-consistency
    assert result.remote_tokens_used == 0
    assert result.escalated is False
    
    mock_local.query.assert_called_once()
    mock_evaluator.evaluate_trust.assert_called_once()
    mock_remote.query.assert_not_called()

def test_unified_executor_dynamic_escalate():
    mock_local = MagicMock(spec=LocalClient)
    mock_remote = MagicMock(spec=RemoteClient)
    mock_evaluator = MagicMock(spec=TrustEvaluator)
    
    mock_local.query.return_value = LocalExecutionResult(
        text="local output",
        total_tokens=10,
        raw_response={}
    )
    
    mock_evaluator.evaluate_trust.return_value = {
        "escalate": True,
        "signals": {"structural_valid": True, "mean_entropy": 0.9, "self_consistency": 1.0},
        "failures": {"structural": False, "entropy": True, "consistency": False},
        "consistency_tokens": 30
    }
    
    mock_remote.query.return_value = RemoteExecutionResult(
        text="remote output",
        total_tokens=50,
        raw_response={}
    )
    
    executor = UnifiedExecutor(
        local_client=mock_local,
        remote_client=mock_remote,
        trust_evaluator=mock_evaluator
    )
    
    result = executor.execute(
        prompt="Solve complex math",
        routing_strategy="dynamic",
        category="math",
        temperature=0.0
    )
    
    assert result.source == "remote"
    assert result.text == "remote output"
    assert result.local_tokens_used == 40  # 10 initial + 30 self-consistency
    assert result.remote_tokens_used == 50  # 50 remote tokens
    assert result.escalated is True
    
    mock_local.query.assert_called_once()
    mock_evaluator.evaluate_trust.assert_called_once()
    mock_remote.query.assert_called_once()

def test_unified_executor_adaptive():
    mock_local = MagicMock(spec=LocalClient)
    mock_remote = MagicMock(spec=RemoteClient)
    mock_evaluator = MagicMock(spec=TrustEvaluator)
    mock_adjuster = MagicMock(spec=BudgetAwareAdjuster)
    
    mock_local.query.return_value = LocalExecutionResult(
        text="local answer",
        total_tokens=10,
        raw_response={}
    )
    
    mock_evaluator.evaluate_trust.return_value = {
        "escalate": False,
        "signals": {"structural_valid": True, "mean_entropy": 0.2, "self_consistency": 1.0},
        "failures": {"structural": False, "entropy": False, "consistency": False},
        "consistency_tokens": 15
    }
    
    mock_adjuster.get_adjusted_thresholds.return_value = {
        "consistency_threshold": 0.3,
        "entropy_threshold": 0.9
    }
    
    executor = UnifiedExecutor(
        local_client=mock_local,
        remote_client=mock_remote,
        trust_evaluator=mock_evaluator,
        budget_adjuster=mock_adjuster
    )
    
    result = executor.execute(
        prompt="Solve arithmetic",
        routing_strategy="adaptive",
        category="math"
    )
    
    assert result.source == "local"
    assert result.local_tokens_used == 25
    assert result.remote_tokens_used == 0
    assert result.escalated is False
    
    mock_adjuster.get_adjusted_thresholds.assert_called_once_with("math")
    mock_adjuster.register_task_completed.assert_called_once_with(
        category="math",
        local_text="local answer",
        remote_text=None,
        remote_tokens_used=0
    )
    
    assert mock_evaluator.consistency_threshold == 0.3
    assert mock_evaluator.entropy_threshold == 0.9
