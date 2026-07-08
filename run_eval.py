import os
import json
import time
from typing import List, Dict, Any
from routing_agent.executor import UnifiedExecutor
from routing_agent.scorer import evaluate_task

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r") as f:
        return json.load(f)

def run_evaluation(executor: UnifiedExecutor, dataset: List[Dict[str, Any]], use_remote: bool) -> Dict[str, Any]:
    results = []
    total_local_tokens = 0
    total_remote_tokens = 0
    total_score = 0.0
    
    mode_name = "Remote-Only" if use_remote else "Local-Only"
    print(f"\nRunning baseline evaluation for: {mode_name}...")
    
    for task in dataset:
        prompt = task["prompt"]
        target = task["target"]
        category = task["category"]
        required_keys = task.get("required_keys")
        
        try:
            start_time = time.time()
            exec_res = executor.execute(
                prompt=prompt,
                use_remote=use_remote,
                temperature=0.0
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
            
            result_entry = {
                "id": task["id"],
                "category": category,
                "prompt": prompt,
                "target": target,
                "prediction": exec_res.text,
                "score": score,
                "local_tokens": exec_res.local_tokens_used,
                "remote_tokens": exec_res.remote_tokens_used,
                "latency": latency
            }
            results.append(result_entry)
            print(f"  [{task['id']}] Category: {category} | Score: {score:.1f} | Tokens (L/R): {exec_res.local_tokens_used}/{exec_res.remote_tokens_used}")
            
        except Exception as e:
            print(f"  [{task['id']}] Failed to execute: {e}")
            results.append({
                "id": task["id"],
                "category": category,
                "error": str(e),
                "score": 0.0,
                "local_tokens": 0,
                "remote_tokens": 0
            })
            
    num_tasks = len(dataset)
    avg_accuracy = (total_score / num_tasks) if num_tasks > 0 else 0.0
    
    return {
        "mode": mode_name,
        "avg_accuracy": avg_accuracy,
        "total_local_tokens": total_local_tokens,
        "total_remote_tokens": total_remote_tokens,
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
    
    # Run Local-Only Baseline
    local_baseline = run_evaluation(executor, dataset, use_remote=False)
    
    # Run Remote-Only Baseline (if API key is available)
    has_remote_key = bool(os.environ.get("FIREWORKS_API_KEY"))
    remote_baseline = None
    if has_remote_key:
        try:
            remote_baseline = run_evaluation(executor, dataset, use_remote=True)
        except Exception as e:
            print(f"\nRemote evaluation failed: {e}")
    else:
        print("\nFIREWORKS_API_KEY environment variable is not set. Skipping Remote-Only baseline.")
        
    # Print comparison summary
    print("\n" + "="*60)
    print("                EVALUATION BASELINE SUMMARY")
    print("="*60)
    print(f"Total Tasks: {len(dataset)}")
    print("-"*60)
    print(f"Local-Only Baseline:")
    print(f"  Average Accuracy:      {local_baseline['avg_accuracy'] * 100:.2f}%")
    print(f"  Total Local Tokens:    {local_baseline['total_local_tokens']}")
    print(f"  Total Remote Tokens:   {local_baseline['total_remote_tokens']}")
    
    if remote_baseline:
        print("-"*60)
        print(f"Remote-Only Baseline:")
        print(f"  Average Accuracy:      {remote_baseline['avg_accuracy'] * 100:.2f}%")
        print(f"  Total Local Tokens:    {remote_baseline['total_local_tokens']}")
        print(f"  Total Remote Tokens:   {remote_baseline['total_remote_tokens']}")
        print("-"*60)
        
        acc_diff = (remote_baseline['avg_accuracy'] - local_baseline['avg_accuracy']) * 100
        token_diff = remote_baseline['total_remote_tokens'] - local_baseline['total_remote_tokens']
        print(f"Comparison (Remote vs Local):")
        print(f"  Accuracy Difference:   {acc_diff:+.2f}%")
        print(f"  Remote Token Cost Increase: {token_diff} tokens")
    else:
        print("-"*60)
        print("Note: Set FIREWORKS_API_KEY to see Remote-Only baseline comparison.")
    print("="*60)

    # Save results to a file
    output_log = {
        "timestamp": time.time(),
        "local_baseline": local_baseline,
        "remote_baseline": remote_baseline
    }
    with open("eval_results_log.json", "w") as f:
        json.dump(output_log, f, indent=2)
    print("\nSaved full evaluation log to 'eval_results_log.json'")

if __name__ == "__main__":
    main()
