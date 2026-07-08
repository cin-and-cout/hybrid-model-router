import json
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from routing_agent.local_client import LocalClient, LocalExecutionResult
from routing_agent.remote_client import RemoteClient, RemoteExecutionResult
from routing_agent.evaluator import TrustEvaluator
from routing_agent.adjuster import BudgetAwareAdjuster
from routing_agent.gate import PredictiveRoutingGate
from routing_agent.cache import SemanticCache

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
        trust_evaluator: Optional[TrustEvaluator] = None,
        budget_adjuster: Optional[BudgetAwareAdjuster] = None,
        predictive_gate: Optional[PredictiveRoutingGate] = None,
        semantic_cache: Optional[SemanticCache] = None,
        cascade: Optional[List[Dict[str, Any]]] = None
    ):
        self.local_client = local_client or LocalClient()
        self.remote_client = remote_client or RemoteClient()
        self.trust_evaluator = trust_evaluator or TrustEvaluator(local_client=self.local_client)
        self.budget_adjuster = budget_adjuster
        self.predictive_gate = predictive_gate or PredictiveRoutingGate(adjuster=self.budget_adjuster)
        self.semantic_cache = semantic_cache or SemanticCache(local_client=self.local_client)
        self.cascade = cascade or [
            {"client": self.local_client, "type": "local", "name": getattr(self.local_client, "model", "local-0.5B")},
            {"client": self.remote_client, "type": "remote", "name": getattr(self.remote_client, "model", "remote-70B")}
        ]

    def _log_execution(
        self,
        prompt: str,
        routing_strategy: str,
        category: Optional[str],
        result: UnifiedExecutionResult,
        latency: float
    ):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "prompt": prompt,
            "routing_strategy": routing_strategy,
            "category": category,
            "source": result.source,
            "escalated": result.escalated,
            "local_tokens_used": result.local_tokens_used,
            "remote_tokens_used": result.remote_tokens_used,
            "latency_seconds": latency,
            "trust_report": result.trust_report
        }
        try:
            with open("routing_execution.jsonl", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass

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
        Supports static, dynamic, and adaptive routing strategies.
        """
        start_time = time.time()
        
        # Check semantic cache first
        if self.semantic_cache:
            cached_res = self.semantic_cache.get(prompt)
            if cached_res:
                latency = time.time() - start_time
                self._log_execution(
                    prompt=prompt,
                    routing_strategy=routing_strategy,
                    category=category,
                    result=cached_res,
                    latency=latency
                )
                return cached_res
        
        
        if routing_strategy == "static":
            is_remote = use_remote if use_remote is not None else False
            if is_remote:
                res = self.remote_client.query(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt
                )
                result = UnifiedExecutionResult(
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
                result = UnifiedExecutionResult(
                    text=res.text,
                    source="local",
                    local_result=res,
                    local_tokens_used=res.total_tokens,
                    remote_tokens_used=0,
                    escalated=False
                )
                
        elif routing_strategy in ("dynamic", "adaptive"):
            if not category:
                raise ValueError(f"Category must be provided for {routing_strategy} routing.")
                
            # If adaptive, dynamically adjust thresholds first
            if routing_strategy == "adaptive":
                if not self.budget_adjuster:
                    raise ValueError("budget_adjuster must be initialized for adaptive strategy.")
                adjusted = self.budget_adjuster.get_adjusted_thresholds(category)
                self.trust_evaluator.consistency_threshold = adjusted["consistency_threshold"]
                self.trust_evaluator.entropy_threshold = adjusted["entropy_threshold"]
                
            # Check predictive routing gate to determine starting cascade tier
            start_index = 0
            if self.predictive_gate and self.predictive_gate.should_bypass_local(prompt, category):
                if self.cascade[0]["type"] == "local" and len(self.cascade) > 1:
                    start_index = 1
                    
            accumulated_local_tokens = 0
            last_local_res = None
            last_trust_report = None
            
            # Cascade processing loop
            for idx in range(start_index, len(self.cascade)):
                tier = self.cascade[idx]
                client = tier["client"]
                client_type = tier["type"]
                client_name = tier.get("name", f"tier_{idx}")
                
                if client_type == "remote" or idx == len(self.cascade) - 1:
                    try:
                        remote_res = client.query(
                            prompt=prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            system_prompt=system_prompt
                        )
                        
                        if routing_strategy == "adaptive" and self.budget_adjuster:
                            local_text = last_local_res.text if last_local_res else ""
                            self.budget_adjuster.register_task_completed(
                                category=category,
                                local_text=local_text,
                                remote_text=remote_res.text,
                                remote_tokens_used=remote_res.total_tokens
                            )
                            
                        trust_info = last_trust_report or {"escalate": True}
                        if start_index > 0:
                            trust_info["predictive_bypass"] = True
                        trust_info["cascade_index"] = idx
                        trust_info["cascade_name"] = client_name
                        
                        if start_index > 0:
                            source_label = "remote (predictive bypass)"
                        elif len(self.cascade) == 2:
                            source_label = "remote"
                        else:
                            source_label = f"remote ({client_name})"
                            
                        result = UnifiedExecutionResult(
                            text=remote_res.text,
                            source=source_label,
                            local_result=last_local_res,
                            remote_result=remote_res,
                            local_tokens_used=accumulated_local_tokens,
                            remote_tokens_used=remote_res.total_tokens,
                            escalated=idx > start_index or start_index > 0,
                            trust_report=trust_info
                        )
                        break
                    except Exception as e:
                        import sys
                        print(f"Warning: Cascade tier {client_name} query failed ({e}).", file=sys.stderr)
                        
                        if last_local_res:
                            if routing_strategy == "adaptive" and self.budget_adjuster:
                                self.budget_adjuster.register_task_completed(
                                    category=category,
                                    local_text=last_local_res.text,
                                    remote_text=None,
                                    remote_tokens_used=0
                                )
                            result = UnifiedExecutionResult(
                                text=last_local_res.text,
                                source="local (fallback)",
                                local_result=last_local_res,
                                local_tokens_used=accumulated_local_tokens,
                                remote_tokens_used=0,
                                escalated=False,
                                trust_report=last_trust_report or {"escalate": False, "cascade_fallback": True}
                            )
                            break
                        else:
                            raise
                else:
                    # Local model tier: query it and check trust evaluator
                    self.trust_evaluator.local_client = client
                    local_res = client.query(
                        prompt=prompt,
                        temperature=0.0,
                        max_tokens=max_tokens,
                        system_prompt=system_prompt
                    )
                    
                    trust_report = self.trust_evaluator.evaluate_trust(
                        prompt=prompt,
                        local_result=local_res,
                        category=category,
                        required_keys=required_keys,
                        system_prompt=system_prompt
                    )
                    
                    accumulated_local_tokens += local_res.total_tokens + trust_report.get("consistency_tokens", 0)
                    last_local_res = local_res
                    last_trust_report = trust_report
                    
                    if not trust_report["escalate"]:
                        if routing_strategy == "adaptive" and self.budget_adjuster:
                            self.budget_adjuster.register_task_completed(
                                category=category,
                                local_text=local_res.text,
                                remote_text=None,
                                remote_tokens_used=0
                            )
                        if len(self.cascade) == 2:
                            source_label = "local"
                        else:
                            source_label = f"local ({client_name})"
                            
                        result = UnifiedExecutionResult(
                            text=local_res.text,
                            source=source_label,
                            local_result=local_res,
                            local_tokens_used=accumulated_local_tokens,
                            remote_tokens_used=0,
                            escalated=idx > start_index or start_index > 0,
                            trust_report=trust_report
                        )
                        break
                    
                    print(f"Escalating from cascade tier {client_name} to next tier.")
        else:
            raise ValueError(f"Unknown routing strategy: {routing_strategy}")

        latency = time.time() - start_time
        self._log_execution(
            prompt=prompt,
            routing_strategy=routing_strategy,
            category=category,
            result=result,
            latency=latency
        )
        
        # Save to semantic cache on success (skip fallbacks and existing cache hits)
        if self.semantic_cache and result.source != "local (fallback)" and result.source != "cache hit":
            self.semantic_cache.set(prompt, result)
            
        return result
