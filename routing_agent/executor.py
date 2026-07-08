from typing import Optional
from pydantic import BaseModel, Field
from routing_agent.local_client import LocalClient, LocalExecutionResult
from routing_agent.remote_client import RemoteClient, RemoteExecutionResult

class UnifiedExecutionResult(BaseModel):
    text: str
    source: str  # "local" or "remote"
    local_result: Optional[LocalExecutionResult] = None
    remote_result: Optional[RemoteExecutionResult] = None
    local_tokens_used: int = 0
    remote_tokens_used: int = 0

class UnifiedExecutor:
    """
    Unified executor interface that routes queries to either the local client
    (Ollama) or the remote client (Fireworks AI) based on a routing instruction.
    """
    def __init__(
        self, 
        local_client: Optional[LocalClient] = None, 
        remote_client: Optional[RemoteClient] = None
    ):
        self.local_client = local_client or LocalClient()
        self.remote_client = remote_client or RemoteClient()

    def execute(
        self,
        prompt: str,
        use_remote: bool,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> UnifiedExecutionResult:
        """
        Executes the prompt by routing it to either the local or remote client.
        """
        if use_remote:
            res = self.remote_client.query(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt
            )
            return UnifiedExecutionResult(
                text=res.text,
                source="remote",
                remote_result=res,
                remote_tokens_used=res.total_tokens,
                local_tokens_used=0
            )
        else:
            res = self.local_client.query(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt
            )
            return UnifiedExecutionResult(
                text=res.text,
                source="local",
                local_result=res,
                local_tokens_used=res.total_tokens,
                remote_tokens_used=0
            )
