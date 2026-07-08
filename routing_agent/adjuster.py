from typing import Dict, Any, Optional
from routing_agent.scorer import evaluate_task

class BudgetAwareAdjuster:
    """
    Tracks budget state and category-specific performance to dynamically adjust
    routing thresholds in real-time.
    """
    def __init__(
        self,
        total_tasks: int,
        total_allowed_remote_tokens: int,
        base_consistency_threshold: float = 0.40,
        base_entropy_threshold: float = 0.80
    ):
        self.total_tasks = total_tasks
        self.total_allowed_remote_tokens = total_allowed_remote_tokens
        self.base_consistency_threshold = base_consistency_threshold
        self.base_entropy_threshold = base_entropy_threshold
        
        self.tasks_processed = 0
        self.remote_tokens_spent = 0
        
        # Category reliability: {category: {"correct": count, "total": count}}
        self.category_stats = {}

    def register_task_completed(self, category: str, local_text: str, remote_text: Optional[str], remote_tokens_used: int):
        """
        Updates internal state after a task execution is completed.
        If escalated, compares the local prediction against the remote response (as proxy ground truth)
        to evaluate local model reliability for the task category.
        """
        self.tasks_processed += 1
        self.remote_tokens_spent += remote_tokens_used
        
        if remote_text is not None:
            # We escalated, so we can check if the local text would have been correct.
            score = evaluate_task(
                task_type=category,
                prediction=local_text,
                target=remote_text
            )
            
            if category not in self.category_stats:
                self.category_stats[category] = {"correct": 0.0, "total": 0}
                
            self.category_stats[category]["total"] += 1
            if score >= 0.7:  # Deem local response as correct if score is high
                self.category_stats[category]["correct"] += 1.0

    def get_adjusted_thresholds(self, category: str) -> Dict[str, float]:
        """
        Computes dynamically adjusted consistency and entropy thresholds based on
        global budget pressure and category reliability.
        """
        remaining_tasks = self.total_tasks - self.tasks_processed
        remaining_tokens = self.total_allowed_remote_tokens - self.remote_tokens_spent
        
        # Calculate global budget pressure
        if remaining_tasks <= 0:
            budget_pressure = 1.0
        else:
            target_tokens_per_task = self.total_allowed_remote_tokens / self.total_tasks
            actual_tokens_per_task = max(remaining_tokens, 0) / remaining_tasks
            if actual_tokens_per_task == 0:
                budget_pressure = 999.0  # Infinite pressure: budget is fully spent
            else:
                budget_pressure = target_tokens_per_task / actual_tokens_per_task
                
        # Calculate category reliability adjustment
        reliability_multiplier = 1.0
        if category in self.category_stats:
            stats = self.category_stats[category]
            if stats["total"] >= 2:  # Need at least a couple of samples to trust stats
                reliability = stats["correct"] / stats["total"]
                if reliability > 0.75:
                    # Local model is performing well in this category -> trust it more (lower threshold)
                    reliability_multiplier = 0.8
                elif reliability < 0.40:
                    # Local model is failing in this category -> trust it less (raise threshold)
                    reliability_multiplier = 1.2
                    
        # Apply adjustments
        # Higher pressure -> lower consistency (escalate less) and higher entropy limit
        # Lower pressure -> higher consistency (escalate more) and lower entropy limit
        adjusted_consistency = self.base_consistency_threshold / budget_pressure * reliability_multiplier
        adjusted_entropy = self.base_entropy_threshold * budget_pressure / reliability_multiplier
        
        # Apply min/max bounds to avoid degenerate thresholds
        adjusted_consistency = max(0.0, min(1.0, adjusted_consistency))
        adjusted_entropy = max(0.2, min(2.5, adjusted_entropy))
        
        return {
            "consistency_threshold": adjusted_consistency,
            "entropy_threshold": adjusted_entropy
        }
