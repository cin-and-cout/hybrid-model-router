import re
from typing import Optional, Dict, List
from routing_agent.adjuster import BudgetAwareAdjuster

# Complex terminology indicating that a query is likely too difficult for a 0.5B parameter local model.
COMPLEXITY_KEYWORDS: Dict[str, List[str]] = {
    "math": [
        "integral", "derivative", "calculus", "differential", "theorem", "proof", 
        "limit", "matrix", "eigenvalue", "determinant", "trigonometry", "vector space",
        "probability density", "fourier", "laplace"
    ],
    "code": [
        "multithreading", "concurrency", "deadlock", "memory leak", "pointer", 
        "time complexity", "o(n", "o(log", "binary search tree", "recursion",
        "dijkstra", "dynamic programming", "race condition", "mutex"
    ],
    "reasoning": [
        "syllogism", "logical deduction", "paradox", "riddle", "transitive", 
        "negation", "truth table", "venn diagram"
    ]
}

class PredictiveRoutingGate:
    """
    Lightweight, low-latency gate to predict if a query is too complex for 
    the local model, bypassing local inference entirely to save time and tokens.
    """
    def __init__(
        self, 
        complexity_keywords: Optional[Dict[str, List[str]]] = None,
        max_local_length: int = 1200,
        adjuster: Optional[BudgetAwareAdjuster] = None
    ):
        self.complexity_keywords = complexity_keywords or COMPLEXITY_KEYWORDS
        self.max_local_length = max_local_length
        self.adjuster = adjuster

    def should_bypass_local(self, prompt: str, category: Optional[str] = None) -> bool:
        """
        Analyzes prompt structure, length, keywords, and history to determine
        if the local model should be bypassed.
        """
        # 1. Prompt Length check
        if len(prompt) > self.max_local_length:
            return True
            
        # 2. Heuristic Keyword check
        prompt_lower = prompt.lower()
        
        # Check global categories if category is specified, otherwise check all keywords
        categories_to_check = [category] if category in self.complexity_keywords else self.complexity_keywords.keys()
        
        for cat in categories_to_check:
            for keyword in self.complexity_keywords[cat]:
                # Word boundary check for keywords to prevent sub-string matching issues
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, prompt_lower):
                    return True
                    
        # 3. Category Reliability check (Phase 5 Feedback Loop)
        if self.adjuster and category and hasattr(self.adjuster, "category_stats") and category in self.adjuster.category_stats:
            stats = self.adjuster.category_stats[category]
            if stats["total"] >= 3:
                reliability = stats["correct"] / stats["total"]
                # If local model fails more than 80% of the time, skip it
                if reliability < 0.20:
                    return True
                    
        return False
