import os
import json
import time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from routing_agent.executor import UnifiedExecutor

app = FastAPI(title="RouteLM: Adaptive Model Routing Engine API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RouteRequest(BaseModel):
    prompt: str
    routing_strategy: str = "dynamic"
    category: str = "general"
    required_keys: Optional[List[str]] = None
    temperature: float = 0.0

class ConfigUpdate(BaseModel):
    consistency_threshold: float
    entropy_threshold: float

@app.post("/api/route")
async def route_prompt(req: RouteRequest):
    try:
        executor = UnifiedExecutor()
        
        # Map frontend strategy strings to internal executor params
        strat = req.routing_strategy
        use_remote = None
        if strat == "static-local":
            strat = "static"
            use_remote = False
        elif strat == "static-remote":
            strat = "static"
            use_remote = True
            
        start_time = time.time()
        res = executor.execute(
            prompt=req.prompt,
            routing_strategy=strat,
            use_remote=use_remote,
            category=req.category,
            required_keys=req.required_keys,
            temperature=req.temperature
        )
        latency = time.time() - start_time
        
        # Build response payload
        return {
            "text": res.text,
            "source": res.source,
            "latency": round(latency, 3),
            "local_tokens": res.local_tokens_used,
            "remote_tokens": res.remote_tokens_used,
            "escalated": res.escalated,
            "trust_report": res.trust_report,
            "routing_strategy": req.routing_strategy
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    log_path = "routing_execution.jsonl"
    total_queries = 0
    cache_hits = 0
    local_runs = 0
    escalations = 0
    
    total_local_tokens = 0
    total_remote_tokens = 0
    
    source_counts: Dict[str, int] = {}
    latencies: List[float] = []
    
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    total_queries += 1
                    source = data.get("source", "unknown")
                    source_counts[source] = source_counts.get(source, 0) + 1
                    
                    if "cache" in source.lower():
                        cache_hits += 1
                    elif "local" in source.lower():
                        local_runs += 1
                    elif "remote" in source.lower():
                        escalations += 1
                        
                    total_local_tokens += data.get("local_tokens_used", 0)
                    total_remote_tokens += data.get("remote_tokens_used", 0)
                    latencies.append(data.get("latency_seconds", 0.0))
        except Exception:
            pass

    # Standard costs: Remote GPT-4o approx $15/1M ($0.000015/token)
    # If all queries went to remote baseline instead of our hybrid router
    # We estimate remote baseline would consume ~180 tokens average per query
    avg_tokens_per_query = 180
    remote_only_token_estimate = total_queries * avg_tokens_per_query
    
    baseline_cost = remote_only_token_estimate * 0.000015
    actual_cost = total_remote_tokens * 0.000015
    savings = max(0.0, baseline_cost - actual_cost)
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    
    return {
        "total_queries": total_queries,
        "cache_hits": cache_hits,
        "local_runs": local_runs,
        "escalations": escalations,
        "total_local_tokens": total_local_tokens,
        "total_remote_tokens": total_remote_tokens,
        "savings_dollars": round(savings, 4),
        "avg_latency": round(avg_latency, 3),
        "source_distribution": source_counts
    }

@app.get("/api/history")
async def get_history(limit: int = 15):
    log_path = "routing_execution.jsonl"
    entries = []
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    entries.append(json.loads(line))
        except Exception:
            pass
    # Return latest entries first
    return entries[-limit:][::-1]

@app.get("/api/config")
async def get_config():
    config_path = "routing_config.json"
    data = {
        "consistency_threshold": 0.4,
        "entropy_threshold": 0.8
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                loaded = json.load(f)
                data["consistency_threshold"] = loaded.get("consistency_threshold", data["consistency_threshold"])
                data["entropy_threshold"] = loaded.get("entropy_threshold", data["entropy_threshold"])
        except Exception:
            pass
    return data

@app.post("/api/config")
async def update_config(cfg: ConfigUpdate):
    config_path = "routing_config.json"
    data = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
        except Exception:
            pass
            
    data["consistency_threshold"] = cfg.consistency_threshold
    data["entropy_threshold"] = cfg.entropy_threshold
    
    try:
        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)
        return {"status": "success", "config": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
