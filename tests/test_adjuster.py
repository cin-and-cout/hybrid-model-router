import pytest
from routing_agent.adjuster import BudgetAwareAdjuster

def test_adjuster_budget_pressure_low():
    # 10 tasks, 500 remote tokens (target 50 tokens per task)
    adjuster = BudgetAwareAdjuster(
        total_tasks=10,
        total_allowed_remote_tokens=500,
        base_consistency_threshold=0.4,
        base_entropy_threshold=0.8
    )
    
    # After 5 tasks, spent only 50 tokens (expected spent: 250 tokens)
    # Remaining tasks = 5, remaining tokens = 450 (actual tokens/task = 90)
    # Target tokens/task = 50. Pressure = 50 / 90 = 0.55 (surplus budget)
    # Lower pressure should raise consistency requirement (trust local less, escalate more)
    adjuster.tasks_processed = 5
    adjuster.remote_tokens_spent = 50
    
    thresholds = adjuster.get_adjusted_thresholds("math")
    
    # consistency threshold should be higher than 0.4
    assert thresholds["consistency_threshold"] > 0.4
    # entropy threshold should be lower than 0.8
    assert thresholds["entropy_threshold"] < 0.8

def test_adjuster_budget_pressure_high():
    adjuster = BudgetAwareAdjuster(
        total_tasks=10,
        total_allowed_remote_tokens=500,
        base_consistency_threshold=0.4,
        base_entropy_threshold=0.8
    )
    
    # After 5 tasks, spent 400 tokens (expected spent: 250 tokens)
    # Remaining tasks = 5, remaining tokens = 100 (actual tokens/task = 20)
    # Target tokens/task = 50. Pressure = 50 / 20 = 2.5 (deficit budget)
    # Higher pressure should lower consistency requirement (trust local more, escalate less)
    adjuster.tasks_processed = 5
    adjuster.remote_tokens_spent = 400
    
    thresholds = adjuster.get_adjusted_thresholds("math")
    
    # consistency threshold should be lower than 0.4
    assert thresholds["consistency_threshold"] < 0.4
    # entropy threshold should be higher than 0.8
    assert thresholds["entropy_threshold"] > 0.8

def test_adjuster_category_reliability():
    adjuster = BudgetAwareAdjuster(
        total_tasks=10,
        total_allowed_remote_tokens=500,
        base_consistency_threshold=0.4,
        base_entropy_threshold=0.8
    )
    
    # Register 3 tasks for math category: 3 correct
    adjuster.register_task_completed("math", "7", "7", 50)
    adjuster.register_task_completed("math", "8", "8", 50)
    adjuster.register_task_completed("math", "9", "9", 50)
    
    # Register 3 tasks for code category: 3 incorrect
    adjuster.register_task_completed("code", "print(1)", "print(2)", 50)
    adjuster.register_task_completed("code", "x=1", "y=2", 50)
    adjuster.register_task_completed("code", "a=1", "b=2", 50)
    
    # With equal budget pressure, thresholds for math should be more lenient (trust local more)
    # and thresholds for code should be stricter (trust local less)
    math_thresholds = adjuster.get_adjusted_thresholds("math")
    code_thresholds = adjuster.get_adjusted_thresholds("code")
    
    # Math has high reliability (1.0) -> lower consistency threshold than Code
    assert math_thresholds["consistency_threshold"] < code_thresholds["consistency_threshold"]
    # Math has high reliability -> higher entropy threshold than Code (more tolerant of entropy)
    assert math_thresholds["entropy_threshold"] > code_thresholds["entropy_threshold"]
