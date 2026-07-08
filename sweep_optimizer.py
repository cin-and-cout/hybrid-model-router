import os
import json
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock
from routing_agent.executor import UnifiedExecutor
from routing_agent.remote_client import RemoteClient, RemoteExecutionResult
from routing_agent.local_client import LocalClient, LocalExecutionResult
from routing_agent.scorer import evaluate_task

class CachedLocalClient:
    """
    Wraps the local client to cache query responses.
    Handles multiple samples at non-zero temperatures by indexing sequential calls.
    """
    def __init__(self, real_client: LocalClient):
        self.real_client = real_client
        self.cache = {}  # key -> list of LocalExecutionResult
        self.call_counters = {}  # key -> count of calls in the current evaluation step

    def reset_counters(self):
        self.call_counters.clear()

    def query(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> LocalExecutionResult:
        key = (prompt, temperature, max_tokens, system_prompt)
        
        self.call_counters[key] = self.call_counters.get(key, 0) + 1
        call_idx = self.call_counters[key] - 1
        
        if key not in self.cache:
            self.cache[key] = []
            
        while len(self.cache[key]) <= call_idx:
            res = self.real_client.query(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt
            )
            self.cache[key].append(res)
            
        return self.cache[key][call_idx]

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        return json.load(f)

def run_sweep(
    dataset: List[Dict[str, Any]],
    consistency_thresholds: List[float],
    entropy_thresholds: List[float],
    accuracy_floor: float = 0.8
) -> List[Dict[str, Any]]:
    has_remote_key = bool(os.environ.get("FIREWORKS_API_KEY"))
    
    # Initialize executor and wrap the local client with our cache
    executor = UnifiedExecutor()
    cached_local = CachedLocalClient(executor.local_client)
    executor.local_client = cached_local
    executor.trust_evaluator.local_client = cached_local
    
    mock_remote = None
    if not has_remote_key:
        print("Note: FIREWORKS_API_KEY is not set. Simulating remote calls returning ground truth answers (50 tokens each).")
        mock_remote = MagicMock(spec=RemoteClient)
        executor.remote_client = mock_remote
        
    sweep_results = []
    
    print("\n" + "="*80)
    print(f"{'Consistency':<12} | {'Entropy':<8} | {'Accuracy':<10} | {'Local Tok':<10} | {'Remote Tok':<10} | {'Escalation':<10}")
    print("="*80)
    
    for c_thresh in consistency_thresholds:
        for e_thresh in entropy_thresholds:
            # Reconfigure thresholds
            executor.trust_evaluator.consistency_threshold = c_thresh
            executor.trust_evaluator.entropy_threshold = e_thresh
            
            total_score = 0.0
            total_local_tokens = 0
            total_remote_tokens = 0
            escalations = 0
            
            for task in dataset:
                prompt = task["prompt"]
                target = task["target"]
                category = task["category"]
                required_keys = task.get("required_keys")
                
                # Reset cache lookup index before each task run
                cached_local.reset_counters()
                
                if mock_remote:
                    mock_remote.query.return_value = RemoteExecutionResult(
                        text=target,
                        total_tokens=50,
                        raw_response={}
                    )
                
                system_prompt = (
                    "You are a strict, direct QA assistant. Output ONLY the direct answer. "
                    "Do not include conversational filler, markdown formatting (like codeblocks), or explanations. "
                    "If asked for a single number or word, reply with only that number or word. "
                    "If asked for JSON, return ONLY the raw JSON object."
                )
                
                try:
                    res = executor.execute(
                        prompt=prompt,
                        routing_strategy="dynamic",
                        category=category,
                        required_keys=required_keys,
                        system_prompt=system_prompt
                    )
                    
                    score = evaluate_task(
                        task_type=category,
                        prediction=res.text,
                        target=target,
                        required_keys=required_keys
                    )
                    
                    total_score += score
                    total_local_tokens += res.local_tokens_used
                    total_remote_tokens += res.remote_tokens_used
                    if res.escalated:
                        escalations += 1
                        
                except Exception as e:
                    print(f"Error executing task {task['id']}: {e}")
                    pass
            
            num_tasks = len(dataset)
            avg_acc = total_score / num_tasks if num_tasks > 0 else 0.0
            escalation_rate = escalations / num_tasks if num_tasks > 0 else 0.0
            
            sweep_results.append({
                "consistency_threshold": c_thresh,
                "entropy_threshold": e_thresh,
                "accuracy": avg_acc,
                "local_tokens": total_local_tokens,
                "remote_tokens": total_remote_tokens,
                "escalation_rate": escalation_rate
            })
            
            print(f"{c_thresh:<12.2f} | {e_thresh:<8.2f} | {avg_acc*100:<9.1f}% | {total_local_tokens:<10} | {total_remote_tokens:<10} | {escalation_rate*100:<9.1f}%")
            
    print("="*80)
    return sweep_results

def find_best_configuration(sweep_results: List[Dict[str, Any]], accuracy_floor: float = 0.8) -> Dict[str, Any]:
    valid_configs = [c for c in sweep_results if c["accuracy"] >= accuracy_floor]
    
    if not valid_configs:
        print(f"No configuration met the target accuracy floor of {accuracy_floor*100:.1f}%. Finding the highest accuracy config instead.")
        best_config = max(sweep_results, key=lambda x: (x["accuracy"], -x["remote_tokens"]))
    else:
        best_config = min(valid_configs, key=lambda x: (x["remote_tokens"], -x["accuracy"]))
        
    return best_config

def main():
    dataset_path = os.path.join("data", "validation_set.json")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return
        
    dataset = load_dataset(dataset_path)
    
    # Configure sweep search space
    consistency_thresholds = [0.0, 0.4, 0.6, 0.7, 0.8, 1.0]
    entropy_thresholds = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    
    accuracy_floor = 0.75  # Target accuracy floor
    
    results = run_sweep(dataset, consistency_thresholds, entropy_thresholds, accuracy_floor)
    
    best = find_best_configuration(results, accuracy_floor)
    
    print("\n" + "*"*60)
    print("              CALIBRATED OPTIMAL CONFIGURATION")
    print("*"*60)
    print(f"Target Accuracy Floor:         {accuracy_floor * 100:.1f}%")
    print(f"Optimal Consistency Threshold: {best['consistency_threshold']:.2f}")
    print(f"Optimal Entropy Threshold:     {best['entropy_threshold']:.2f}")
    print(f"Expected Accuracy:             {best['accuracy'] * 100:.2f}%")
    print(f"Expected Local Tokens:         {best['local_tokens']}")
    print(f"Expected Remote Tokens:        {best['remote_tokens']}")
    print(f"Expected Escalation Rate:      {best['escalation_rate'] * 100:.2f}%")
    print("*"*60)
    
    config_data = {
        "consistency_threshold": best["consistency_threshold"],
        "entropy_threshold": best["entropy_threshold"],
        "accuracy_floor": accuracy_floor,
        "calibrated_accuracy": best["accuracy"],
        "calibrated_remote_tokens": best["remote_tokens"]
    }
    with open("routing_config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    print("\nSaved calibrated thresholds to 'routing_config.json'")

if __name__ == "__main__":
    main()
