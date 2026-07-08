import collections
from typing import List, Dict, Any, Optional
from routing_agent.local_client import LocalClient
from routing_agent.scorer import clean_string, token_overlap_scorer

class TrustEvaluator:
    """
    Evaluates trust signals for local model execution to decide if a query
    needs to be escalated to a larger remote model.
    """
    def __init__(self, local_client: Optional[LocalClient] = None):
        self.local_client = local_client or LocalClient()

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
