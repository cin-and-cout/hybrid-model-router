import collections
import re
from typing import List, Dict, Any, Optional
from routing_agent.local_client import LocalClient, LocalExecutionResult
from routing_agent.scorer import clean_string, token_overlap_scorer, json_format_scorer

class TrustEvaluator:
    """
    Evaluates trust signals for local model execution to decide if a query
    needs to be escalated to a larger remote model.
    """
    def __init__(
        self, 
        local_client: Optional[LocalClient] = None,
        consistency_threshold: float = 0.6,
        entropy_threshold: float = 0.8,
        consistency_n: int = 3,
        consistency_temp: float = 0.7
    ):
        self.local_client = local_client or LocalClient()
        self.consistency_threshold = consistency_threshold
        self.entropy_threshold = entropy_threshold
        self.consistency_n = consistency_n
        self.consistency_temp = consistency_temp

    def compute_self_consistency(
        self,
        prompt: str,
        category: str,
        n: int = 3,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> float:
        """
        Queries the local model N times at a higher temperature, and measures
        the consistency/agreement among the generated outputs.
        """
        if n <= 1:
            return 1.0
            
        predictions = []
        for _ in range(n):
            res = self.local_client.query(
                prompt=prompt,
                temperature=temperature,
                system_prompt=system_prompt
            )
            predictions.append(res.text)
            
        category = category.strip().lower()
        if category in ("math", "reasoning"):
            cleaned_preds = [clean_string(p) for p in predictions]
            counts = collections.Counter(cleaned_preds)
            most_common_count = counts.most_common(1)[0][1]
            return float(most_common_count) / n
        else:
            total_similarity = 0.0
            pairs_count = 0
            for i in range(len(predictions)):
                for j in range(i + 1, len(predictions)):
                    sim = token_overlap_scorer(predictions[i], predictions[j])
                    total_similarity += sim
                    pairs_count += 1
            return total_similarity / pairs_count if pairs_count > 0 else 1.0

    def compute_entropy_signal(
        self,
        local_result: LocalExecutionResult,
        high_entropy_threshold: float = 1.0
    ) -> Dict[str, Any]:
        """
        Extracts token-level entropy and confidence metrics from the local execution result.
        """
        tokens = local_result.tokens or []
        high_entropy_ratio = 0.0
        if tokens:
            high_entropy_count = sum(1 for t in tokens if t.entropy > high_entropy_threshold)
            high_entropy_ratio = float(high_entropy_count) / len(tokens)
        
        return {
            "mean_entropy": local_result.mean_entropy or 0.0,
            "min_logprob": local_result.min_logprob or 0.0,
            "high_entropy_ratio": high_entropy_ratio
        }

    def verify_structure(
        self,
        text: str,
        category: str,
        required_keys: Optional[List[str]] = None
    ) -> bool:
        """
        Performs format and syntax validation depending on the task category.
        """
        category = category.strip().lower()
        if category == "structured_output":
            return json_format_scorer(text, required_keys) == 1.0
        elif category == "code":
            code = self._extract_python_code(text)
            if not code:
                return False
            try:
                compile(code, "<string>", "exec")
                return True
            except SyntaxError:
                return False
        return True

    def _extract_python_code(self, text: str) -> str:
        pattern = r"```(?:python)?\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return "\n".join(matches).strip()
        return text.strip()

    def evaluate_trust(
        self,
        prompt: str,
        local_result: LocalExecutionResult,
        category: str,
        required_keys: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Combines consistency, entropy, and structural signals to decide if the local execution
        should be escalated to the remote model.
        """
        structural_valid = self.verify_structure(
            text=local_result.text,
            category=category,
            required_keys=required_keys
        )
        structural_failed = not structural_valid
        
        entropy_sig = self.compute_entropy_signal(local_result)
        mean_entropy = entropy_sig["mean_entropy"]
        entropy_failed = mean_entropy > self.entropy_threshold
        
        self_consistency = self.compute_self_consistency(
            prompt=prompt,
            category=category,
            n=self.consistency_n,
            temperature=self.consistency_temp,
            system_prompt=system_prompt
        )
        consistency_failed = self_consistency < self.consistency_threshold
        
        escalate = consistency_failed or entropy_failed or structural_failed
        
        return {
            "escalate": escalate,
            "signals": {
                "structural_valid": structural_valid,
                "mean_entropy": mean_entropy,
                "self_consistency": self_consistency
            },
            "failures": {
                "structural": structural_failed,
                "entropy": entropy_failed,
                "consistency": consistency_failed
            }
        }
