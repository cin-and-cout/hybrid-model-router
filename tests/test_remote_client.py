import pytest
from unittest.mock import MagicMock, patch
from routing_agent.remote_client import RemoteClient, RemoteExecutionResult

def test_remote_client_missing_api_key():
    # If no API key is set/provided, accessing client property should raise ValueError
    with patch.dict("os.environ", {}, clear=True):
        client = RemoteClient(api_key=None)
        with pytest.raises(ValueError, match="FIREWORKS_API_KEY environment variable is not set"):
            _ = client.client

def test_remote_client_query_success(mocker):
    # Mocking completion return object
    class MockUsage:
        def __init__(self):
            self.prompt_tokens = 15
            self.completion_tokens = 25
            self.total_tokens = 40

    class MockMessage:
        def __init__(self):
            self.role = "assistant"
            self.content = "This is a response from the remote model."

    class MockChoice:
        def __init__(self):
            self.index = 0
            self.message = MockMessage()
            self.finish_reason = "stop"

    class MockCompletion:
        def __init__(self):
            self.id = "cmpl-abc123"
            self.choices = [MockChoice()]
            self.usage = MockUsage()
            self.model = "accounts/fireworks/models/llama-v3p1-70b-instruct"

        def model_dump(self):
            return {
                "id": self.id,
                "model": self.model,
                "choices": [{"index": 0, "message": {"content": "This is a response from the remote model."}}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40}
            }

    # Patch Fireworks client creation
    mock_fireworks_class = mocker.patch("routing_agent.remote_client.Fireworks")
    mock_client_instance = MagicMock()
    mock_fireworks_class.return_value = mock_client_instance
    
    # Mock the chat completions creation endpoint
    mock_client_instance.chat.completions.create.return_value = MockCompletion()
    
    client = RemoteClient(api_key="fake-api-key")
    result = client.query("Hello Remote", temperature=0.7, max_tokens=100)
    
    assert isinstance(result, RemoteExecutionResult)
    assert result.text == "This is a response from the remote model."
    assert result.prompt_tokens == 15
    assert result.completion_tokens == 25
    assert result.total_tokens == 40
    
    # Assert parameters passed correctly to client
    mock_client_instance.chat.completions.create.assert_called_once_with(
        model="accounts/fireworks/models/llama-v3p1-70b-instruct",
        messages=[{"role": "user", "content": "Hello Remote"}],
        temperature=0.7,
        max_tokens=100,
        timeout=30.0
    )
