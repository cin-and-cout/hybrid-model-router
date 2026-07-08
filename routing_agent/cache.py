import os
import json
import math
from typing import List, Dict, Any, Optional
from routing_agent.local_client import LocalClient

class SemanticCache:
    """
    Local vector-cache that stores prompt embeddings and corresponding results
    in a JSON file, returning cached results for semantically similar prompts.
    """
    def __init__(
        self,
        local_client: Optional[LocalClient] = None,
        cache_file: str = "semantic_cache.json",
        similarity_threshold: float = 0.95
    ):
        self.local_client = local_client or LocalClient()
        self.cache_file = cache_file
        self.similarity_threshold = similarity_threshold
        self.cache: List[Dict[str, Any]] = []
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = []

    def _save_cache(self):
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception:
            pass

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_a = math.sqrt(sum(a * a for a in v1))
        norm_b = math.sqrt(sum(b * b for b in v2))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def get(self, prompt: str) -> Optional["UnifiedExecutionResult"]:
        """
        Attempts to retrieve a semantically similar cached response.
        """
        if not self.cache:
            return None

        try:
            prompt_emb = self.local_client.get_embedding(prompt)
            if not prompt_emb or not isinstance(prompt_emb, list):
                return None
        except Exception:
            return None

        best_score = -1.0
        best_match = None

        for entry in self.cache:
            entry_emb = entry.get("embedding")
            if entry_emb and isinstance(entry_emb, list):
                score = self._cosine_similarity(prompt_emb, entry_emb)
                if score > best_score:
                    best_score = score
                    best_match = entry

        if best_score >= self.similarity_threshold and best_match:
            res_data = best_match["result"]
            from routing_agent.executor import UnifiedExecutionResult
            # Reconstruct UnifiedExecutionResult with source as "cache hit"
            return UnifiedExecutionResult(
                text=res_data.get("text", ""),
                source="cache hit",
                local_tokens_used=0,
                remote_tokens_used=0,
                escalated=res_data.get("escalated", False),
                trust_report={
                    "cache_similarity": best_score,
                    "original_source": res_data.get("source", "")
                }
            )
        return None

    def set(self, prompt: str, result: "UnifiedExecutionResult"):
        """
        Stores a prompt and its result in the cache.
        """
        try:
            prompt_emb = self.local_client.get_embedding(prompt)
            if not prompt_emb or not isinstance(prompt_emb, list):
                return
        except Exception:
            return

        # Prepare result data dictionary
        res_data = {
            "text": result.text,
            "source": result.source,
            "escalated": result.escalated
        }

        # Check if already exists (avoid duplicates)
        for entry in self.cache:
            if entry["prompt"] == prompt:
                entry["embedding"] = prompt_emb
                entry["result"] = res_data
                self._save_cache()
                return

        self.cache.append({
            "prompt": prompt,
            "embedding": prompt_emb,
            "result": res_data
        })
        self._save_cache()
