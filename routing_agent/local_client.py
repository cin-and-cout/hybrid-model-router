import os
import math
import httpx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class TokenDetail(BaseModel):
    token: str
    logprob: float
    entropy: float
    top_logprobs: List[Dict[str, Any]] = Field(default_factory=list)

class LocalExecutionResult(BaseModel):
    text: str
    tokens: List[TokenDetail] = Field(default_factory=list)
    mean_logprob: float = 0.0
    min_logprob: float = 0.0
    mean_entropy: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw_response: Dict[str, Any] = Field(default_factory=dict)

class LocalClient:
    """
    Client for querying the local Ollama instance using the OpenAI-compatible v1 completions API,
    with built-in token-level logprob and entropy calculation.
    """
    def __init__(self, base_url: Optional[str] = None, model: str = "qwen2.5:0.5b"):
        if not base_url:
            # Check OLLAMA_HOST first (set inside docker-compose)
            env_host = os.environ.get("OLLAMA_HOST")
            if env_host:
                base_url = env_host
            else:
                base_url = "http://localhost:11434"
        
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=60.0)

    def query(
        self, 
        prompt: str, 
        temperature: float = 0.0, 
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> LocalExecutionResult:
        url = f"{self.base_url}/v1/chat/completions"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": 5
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
            
        response = self.client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        choices = data.get("choices", [])
        if not choices:
            return LocalExecutionResult(text="", raw_response=data)
            
        choice = choices[0]
        text = choice.get("message", {}).get("content", "")
        
        token_details = []
        logprobs_data = choice.get("logprobs", {})
        
        if logprobs_data and "content" in logprobs_data:
            content_logprobs = logprobs_data["content"] or []
            for item in content_logprobs:
                token_str = item.get("token", "")
                logprob_val = item.get("logprob", 0.0)
                top_items = item.get("top_logprobs", [])
                
                # Compute Shannon entropy for the distribution of top candidates at this position
                entropy_val = self._compute_entropy(top_items)
                
                token_details.append(
                    TokenDetail(
                        token=token_str,
                        logprob=logprob_val,
                        entropy=entropy_val,
                        top_logprobs=[
                            {"token": t.get("token", ""), "logprob": t.get("logprob", 0.0)}
                            for t in top_items
                        ]
                    )
                )
                
        if token_details:
            logprobs_list = [t.logprob for t in token_details]
            entropies_list = [t.entropy for t in token_details]
            
            mean_logprob = sum(logprobs_list) / len(logprobs_list)
            min_logprob = min(logprobs_list)
            mean_entropy = sum(entropies_list) / len(entropies_list)
        else:
            mean_logprob = 0.0
            min_logprob = 0.0
            mean_entropy = 0.0
            
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        return LocalExecutionResult(
            text=text,
            tokens=token_details,
            mean_logprob=mean_logprob,
            min_logprob=min_logprob,
            mean_entropy=mean_entropy,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_response=data
        )

    def _compute_entropy(self, top_logprobs: List[Dict[str, Any]]) -> float:
        if not top_logprobs:
            return 0.0
            
        probs = []
        for item in top_logprobs:
            lp = item.get("logprob")
            if lp is not None:
                probs.append(math.exp(lp))
                
        sum_p = sum(probs)
        if sum_p <= 0:
            return 0.0
            
        normalized = [p / sum_p for p in probs]
        return -sum(p * math.log(p) for p in normalized if p > 0.0)
