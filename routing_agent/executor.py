from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from routing_agent.local_client import LocalClient, LocalExecutionResult
from routing_agent.remote_client import RemoteClient, RemoteExecutionResult
from routing_agent.evaluator import TrustEvaluator

class UnifiedExecutionResult(BaseModel):
    text: str
    source: str  # "local" or "remote"
    local_result: Optional[LocalExecutionResult] = None
    remote_result: Optional[RemoteExecutionResult] = None
    local_tokens_used: int = 0
    remote_tokens_used: int = 0
    escalated: bool = False
    trust_report: Optional[Dict[str, Any]] = None

class UnifiedExecutor:
    """
    Unified executor interface that routes queries to either the local client
    (Ollama) or the remote client (Fireworks AI) based on a routing instruction.
    """
    def __init__(
        self, 
        local_client: Optional[LocalClient] = None, 
        remote_client: Optional[RemoteClient] = None,
        trust_evaluator: Optional[TrustEvaluator] = None
    ):
        self.local_client = local_client or LocalClient()
        self.remote_client = remote_client or RemoteClient()
        self.trust_evaluator = trust_evaluator or TrustEvaluator(local_client=self.local_client)

    def execute(
        self,
        prompt: str,
        use_remote: Optional[bool] = None,
        routing_strategy: str = "static",
        category: Optional[str] = None,
        required_keys: Optional[List[str]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> UnifiedExecutionResult:
        """
        Executes the prompt by routing it to either the local or remote client.
        Supports both static and dynamic routing strategies.
        """
        if routing_strategy == "static":
            is_remote = use_remote if use_remote is not None else False
            if is_remote:
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
                    local_tokens_used=0,
                    escalated=True
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
                    remote_tokens_used=0,
                    escalated=False
                )
        elif routing_strategy == "dynamic":
            if not category:
                raise ValueError("Category must be provided for dynamic routing.")
                
            # 1. Run local pass at temperature 0
            local_res = self.local_client.query(
                prompt=prompt,
                temperature=0.0,
                max_tokens=max_tokens,
                system_prompt=system_prompt
            )
            
            # 2. Evaluate trust
            trust_report = self.trust_evaluator.evaluate_trust(
                prompt=prompt,
                local_result=local_res,
                category=category,
                required_keys=required_keys,
                system_prompt=system_prompt
            )
            
            local_tokens = local_res.total_tokens + trust_report.get("consistency_tokens", 0)
            
            if trust_report["escalate"]:
                # 3. Escalate to remote
                remote_res = self.remote_client.query(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt
                )
                return UnifiedExecutionResult(
                    text=remote_res.text,
                    source="remote",
                    local_result=local_res,
                    remote_result=remote_res,
                    local_tokens_used=local_tokens,
                    remote_tokens_used=remote_res.total_tokens,
                    escalated=True,
                    trust_report=trust_report
                )
            else:
                # 4. Return local answer
                return UnifiedExecutionResult(
                    text=local_res.text,
                    source="local",
                    local_result=local_res,
                    local_tokens_used=local_tokens,
                    remote_tokens_used=0,
                    escalated=False,
                    trust_report=trust_report
                )
        else:
            raise ValueError(f"Unknown routing strategy: {routing_strategy}")
