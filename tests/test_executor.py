import pytest
from unittest.mock import MagicMock
from routing_agent.executor import UnifiedExecutor, UnifiedExecutionResult
from routing_agent.local_client import LocalClient, LocalExecutionResult
from routing_agent.remote_client import RemoteClient, RemoteExecutionResult

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
