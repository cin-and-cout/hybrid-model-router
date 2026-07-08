import os
import json
import time
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
        self.cache = {}
        self.call_counters = {}

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

def run_evaluation(
    executor: UnifiedExecutor,
    dataset: List[Dict[str, Any]],
    routing_strategy: str = "static",
    use_remote: bool = False,
    cached_local: Optional[CachedLocalClient] = None
) -> Dict[str, Any]:
    results = []
    total_local_tokens = 0
    total_remote_tokens = 0
    total_score = 0.0
    escalated_count = 0
    
    if routing_strategy == "dynamic":
        mode_name = "Hybrid-Calibrated-Router"
    else:
        mode_name = "Remote-Only" if use_remote else "Local-Only"
        
    print(f"\nRunning evaluation for: {mode_name}...")
    
    has_remote_key = bool(os.environ.get("FIREWORKS_API_KEY"))
    mock_remote = None
    if not has_remote_key and (use_remote or routing_strategy == "dynamic"):
        mock_remote = MagicMock(spec=RemoteClient)
        executor.remote_client = mock_remote
        
    for task in dataset:
        prompt = task["prompt"]
        target = task["target"]
        category = task["category"]
        required_keys = task.get("required_keys")
        
        if cached_local:
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
            start_time = time.time()
            exec_res = executor.execute(
                prompt=prompt,
                use_remote=use_remote,
                routing_strategy=routing_strategy,
                category=category,
                required_keys=required_keys,
                temperature=0.0,
                system_prompt=system_prompt
            )
            latency = time.time() - start_time
            
            score = evaluate_task(
                task_type=category,
                prediction=exec_res.text,
                target=target,
                required_keys=required_keys
            )
            
            total_score += score
            total_local_tokens += exec_res.local_tokens_used
            total_remote_tokens += exec_res.remote_tokens_used
            if exec_res.escalated:
                escalated_count += 1
                
            result_entry = {
                "id": task["id"],
                "category": category,
                "prompt": prompt,
                "target": target,
                "prediction": exec_res.text,
                "score": score,
                "local_tokens": exec_res.local_tokens_used,
                "remote_tokens": exec_res.remote_tokens_used,
                "latency": latency,
                "escalated": exec_res.escalated
            }
            results.append(result_entry)
            print(f"  [{task['id']}] Category: {category} | Score: {score:.1f} | Tokens (L/R): {exec_res.local_tokens_used}/{exec_res.remote_tokens_used} | Escalated: {exec_res.escalated}")
            
        except Exception as e:
            print(f"  [{task['id']}] Failed to execute: {e}")
            results.append({
                "id": task["id"],
                "category": category,
                "error": str(e),
                "score": 0.0,
                "local_tokens": 0,
                "remote_tokens": 0,
                "escalated": False
            })
            
    num_tasks = len(dataset)
    avg_accuracy = (total_score / num_tasks) if num_tasks > 0 else 0.0
    
    return {
        "mode": mode_name,
        "avg_accuracy": avg_accuracy,
        "total_local_tokens": total_local_tokens,
        "total_remote_tokens": total_remote_tokens,
        "escalation_rate": escalated_count / num_tasks if num_tasks > 0 else 0.0,
        "results": results
    }

def main():
    dataset_path = os.path.join("data", "validation_set.json")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return
        
    dataset = load_dataset(dataset_path)
    
    # Initialize the Unified Executor
    executor = UnifiedExecutor()
    
    # Wrap with cache to speed up evaluation runs
    cached_local = CachedLocalClient(executor.local_client)
    executor.local_client = cached_local
    executor.trust_evaluator.local_client = cached_local
    
    # Run Local-Only Baseline
    local_baseline = run_evaluation(executor, dataset, routing_strategy="static", use_remote=False, cached_local=cached_local)
    
    # Run Hybrid Calibrated Router
    hybrid_run = run_evaluation(executor, dataset, routing_strategy="dynamic", cached_local=cached_local)
    
    # Run Remote-Only Baseline (if API key is available or in simulation mode)
    remote_baseline = run_evaluation(executor, dataset, routing_strategy="static", use_remote=True, cached_local=cached_local)
    
    # Print comparison summary
    print("\n" + "="*80)
    print("                      ROUTER PERFORMANCE EVALUATION SUMMARY")
    print("="*80)
    print(f"Total Tasks: {len(dataset)}")
    print("-"*80)
    print(f"1. Local-Only Baseline:")
    print(f"  Average Accuracy:      {local_baseline['avg_accuracy'] * 100:.2f}%")
    print(f"  Total Local Tokens:    {local_baseline['total_local_tokens']}")
    print(f"  Total Remote Tokens:   {local_baseline['total_remote_tokens']}")
    print(f"  Escalation Rate:       {local_baseline['escalation_rate'] * 100:.2f}%")
    
    print("-"*80)
    print(f"2. Hybrid Calibrated Router:")
    print(f"  Average Accuracy:      {hybrid_run['avg_accuracy'] * 100:.2f}%")
    print(f"  Total Local Tokens:    {hybrid_run['total_local_tokens']}")
    print(f"  Total Remote Tokens:   {hybrid_run['total_remote_tokens']}")
    print(f"  Escalation Rate:       {hybrid_run['escalation_rate'] * 100:.2f}%")
    
    print("-"*80)
    print(f"3. Remote-Only Baseline:")
    print(f"  Average Accuracy:      {remote_baseline['avg_accuracy'] * 100:.2f}%")
    print(f"  Total Local Tokens:    {remote_baseline['total_local_tokens']}")
    print(f"  Total Remote Tokens:   {remote_baseline['total_remote_tokens']}")
    print(f"  Escalation Rate:       {remote_baseline['escalation_rate'] * 100:.2f}%")
    
    print("="*80)
    
    # Save results to a file
    output_log = {
        "timestamp": time.time(),
        "local_baseline": local_baseline,
        "hybrid_run": hybrid_run,
        "remote_baseline": remote_baseline
    }
    with open("eval_results_log.json", "w") as f:
        json.dump(output_log, f, indent=2)
    print("\nSaved full evaluation log to 'eval_results_log.json'")

if __name__ == "__main__":
    main()
