import re
import json
from typing import List, Optional

def clean_string(s: str) -> str:
    """
    Cleans a string by converting to lowercase, removing surrounding quotes/whitespace,
    and stripping basic punctuation to enable robust exact matching.
    """
    s = s.strip().lower()
    # Remove markdown code blocks if present (e.g., ```json ... ```)
    s = re.sub(r"^```(?:json|python)?\n", "", s)
    s = re.sub(r"\n```$", "", s)
    s = s.strip()
    # Strip common trailing punctuation
    s = re.sub(r"[.,\/#!$%\^&\*;:{}=\-_`~()\"'?]", "", s)
    return s.strip()

def exact_match_scorer(prediction: str, target: str) -> float:
    """
    Returns 1.0 if the cleaned prediction matches the cleaned target, 0.0 otherwise.
    """
    return 1.0 if clean_string(prediction) == clean_string(target) else 0.0

def json_format_scorer(prediction: str, required_keys: Optional[List[str]] = None) -> float:
    """
    Returns 1.0 if the prediction is valid JSON and contains all required_keys,
    0.0 otherwise.
    """
    # Try to extract JSON if it is wrapped in markdown blocks
    cleaned = prediction.strip()
    json_match = re.search(r"({.*})|(\[.*\])", cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(0)
        
    try:
        data = json.loads(cleaned)
        if required_keys:
            if not isinstance(data, dict):
                return 0.0
            for key in required_keys:
                if key not in data:
                    return 0.0
        return 1.0
    except (json.JSONDecodeError, TypeError):
        return 0.0

def token_overlap_scorer(prediction: str, target: str) -> float:
    """
    Calculates the Jaccard similarity (word token overlap) between prediction and target.
    Returns a score between 0.0 and 1.0.
    """
    pred_clean = clean_string(prediction)
    target_clean = clean_string(target)
    
    pred_words = set(pred_clean.split())
    target_words = set(target_clean.split())
    
    if not pred_words and not target_words:
        return 1.0
        
    intersection = pred_words.intersection(target_words)
    union = pred_words.union(target_words)
    
    return float(len(intersection)) / len(union)

def evaluate_task(
    task_type: str, 
    prediction: str, 
    target: str, 
    required_keys: Optional[List[str]] = None
) -> float:
    """
    Evaluates a prediction against a target based on the task type.
    """
    task_type = task_type.strip().lower()
    
    if task_type in ("math", "reasoning"):
        # For math and reasoning, exact match is the primary accuracy metric
        return exact_match_scorer(prediction, target)
    elif task_type == "structured_output":
        # Check both JSON validity and content overlap/exact match
        json_score = json_format_scorer(prediction, required_keys)
        if json_score == 0.0:
            return 0.0
        # If valid JSON, check token overlap against expected answer
        return token_overlap_scorer(prediction, target)
    elif task_type == "code":
        # Check if output is formatted correctly or check overlaps
        return token_overlap_scorer(prediction, target)
    else:
        # Default fallback to token overlap
        return token_overlap_scorer(prediction, target)
