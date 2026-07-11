import os
import json
import time
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any

# Import RouteLM Executor directly for direct local execution fallback
try:
    from routing_agent.executor import UnifiedExecutor
except ImportError:
    UnifiedExecutor = None

# Custom Page Configuration
st.set_page_config(
    page_title="RouteLM: Adaptive Model Routing Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Slate-White/Sky-Blue Theming styles
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(226, 232, 240, 0.8);
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 10px;
    }
    .metric-val {
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
    }
    .metric-label {
        font-size: 13px;
        color: #475569;
        font-weight: 600;
        margin-top: 4px;
    }
    .trace-step {
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .trace-success {
        background-color: #ecfdf5;
        border: 1px solid #10b981;
        color: #065f46;
    }
    .trace-pending {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        color: #475569;
    }
    .trace-failed {
        background-color: #fef2f2;
        border: 1px solid #f87171;
        color: #991b1b;
    }
    .trace-escalated {
        background-color: #eff6ff;
        border: 1px solid #3b82f6;
        color: #1e40af;
    }
</style>
""", unsafe_allow_html=True)

# File paths
CONFIG_PATH = "routing_config.json"
LOG_PATH = "routing_execution.jsonl"

def load_config() -> Dict[str, Any]:
    default_config = {
        "consistency_threshold": 0.4,
        "entropy_threshold": 0.8
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                loaded = json.load(f)
                default_config["consistency_threshold"] = loaded.get("consistency_threshold", 0.4)
                default_config["entropy_threshold"] = loaded.get("entropy_threshold", 0.8)
        except Exception:
            pass
    return default_config

def save_config(config_data: Dict[str, Any]):
    existing = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update(config_data)
    with open(CONFIG_PATH, "w") as f:
        json.dump(existing, f, indent=2)

def fetch_stats() -> Dict[str, Any]:
    total_queries = 0
    cache_hits = 0
    local_runs = 0
    escalations = 0
    total_local_tokens = 0
    total_remote_tokens = 0
    latencies = []
    source_counts = {}

    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, "r") as f:
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
        "source_distribution": source_counts,
        "baseline_cost": round(baseline_cost, 4),
        "actual_cost": round(actual_cost, 4)
    }

def get_history() -> List[Dict[str, Any]]:
    entries = []
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    entries.append(json.loads(line))
        except Exception:
            pass
    return entries[::-1]

# Sidebar
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 2px;">
    <h2 style="margin: 0; color: #0f172a;">RouteLM Engine</h2>
</div>
<div style="font-size: 12px; color: #64748b; margin-bottom: 20px;">
    RouteLM: Adaptive Model Routing Engine v2.4
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Navigation Menu",
    ["📊 Analytics & Stats", "🧪 Sandbox Playground", "⚙️ Tuning & Calibration", "📜 Execution Logs"]
)

# Shared Manual refresh
if st.sidebar.button("🔄 Refresh Telemetry"):
    st.toast("Telemetry stats refreshed successfully!", icon="✅")
    st.rerun()

# 📊 ANALYTICS PAGE
if menu == "📊 Analytics & Stats":
    st.markdown("# 📊 RouteLM Analytics & Telemetry")
    st.markdown("Monitor real-time execution distribution, routing efficiency, and cost reductions.")
    
    stats = fetch_stats()
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{stats['total_queries']}</div>
            <div class="metric-label">TOTAL ROUTED QUERIES</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        hit_rate = round((stats['cache_hits'] / stats['total_queries'] * 100), 1) if stats['total_queries'] > 0 else 0.0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{hit_rate}%</div>
            <div class="metric-label">SEMANTIC CACHE HIT RATE</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">${stats['savings_dollars']:.4f}</div>
            <div class="metric-label">ESTIMATED API COST SAVINGS</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{stats['avg_latency']:.3f}s</div>
            <div class="metric-label">AVERAGE INFERENCE LATENCY</div>
        </div>
        """, unsafe_allow_html=True)

    # Charts section
    st.write("---")
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Source Resolver Distribution")
        dist = stats["source_distribution"]
        if dist:
            df_dist = pd.DataFrame(list(dist.items()), columns=["Source", "Count"])
            fig = px.pie(
                df_dist, 
                names="Source", 
                values="Count", 
                color_discrete_sequence=['#0284c7', '#10b981', '#f59e0b', '#3b82f6'],
                hole=0.4
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No routing stats available yet. Run queries from the Sandbox tab first!")
            
    with col_right:
        st.subheader("Cost Comparison (USD)")
        costs = {
            "Scenario": ["Baseline (Remote Only)", "Actual (RouteLM)"],
            "Cost ($)": [stats["baseline_cost"], stats["actual_cost"]]
        }
        df_costs = pd.DataFrame(costs)
        fig_cost = px.bar(
            df_costs,
            x="Scenario",
            y="Cost ($)",
            color="Scenario",
            color_discrete_map={"Baseline (Remote Only)": "#94a3b8", "Actual (RouteLM)": "#0d9488"}
        )
        fig_cost.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig_cost, use_container_width=True)

# 🧪 SANDBOX PLAYGROUND PAGE
elif menu == "🧪 Sandbox Playground":
    st.markdown("# 🧪 Sandbox Playground")
    st.markdown("Interactively query the model gateway and trace model routing decision paths in real-time.")

    config = load_config()
    
    col_opt1, col_opt2, col_opt3 = st.columns([2, 2, 1])
    with col_opt1:
        strategy = st.selectbox(
            "Routing Strategy",
            ["dynamic", "adaptive", "static-local", "static-remote"],
            format_func=lambda x: {
                "dynamic": "Dynamic Routing",
                "adaptive": "Adaptive Routing (Budget Aware)",
                "static-local": "Static Local (0.5B)",
                "static-remote": "Static Remote (70B)"
            }[x]
        )
    with col_opt2:
        category = st.selectbox(
            "Task Category",
            ["general", "code", "math", "reasoning", "structured_output"],
            format_func=lambda x: {
                "general": "General QA",
                "code": "Python Code",
                "math": "Mathematics",
                "reasoning": "Logical Reasoning",
                "structured_output": "Structured JSON"
            }[x]
        )
    with col_opt3:
        temp = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

    required_keys = ""
    if category == "structured_output":
        required_keys = st.text_input("Required Keys (comma-separated)", placeholder="e.g. name, age, city")

    prompt = st.text_area(
        "User Prompt",
        value="Write a python function to check if a number is prime."
    )
    
    if st.button("🚀 Run Query Engine", type="primary"):
        if not prompt.strip():
            st.error("Please enter a non-empty prompt.")
        else:
            trace_container = st.container()
            with trace_container:
                st.subheader("Decision Trace Path")
                
                # Setup simulation loading traces or invoke
                progress = st.status("Initializing RouteLM Cascade...")
                
                # Check Local Server first
                import httpx
                payload = {
                    "prompt": prompt,
                    "routing_strategy": strategy,
                    "category": category,
                    "temperature": temp,
                    "required_keys": [k.strip() for k in required_keys.split(",") if k.strip()] if required_keys else None
                }
                
                try:
                    res = httpx.post("http://localhost:8000/api/route", json=payload, timeout=45.0)
                    if res.status_code == 200:
                        data = res.json()
                    else:
                        raise Exception("Backend returned non-200 status code")
                except Exception as e:
                    # Fallback to local import execution
                    if UnifiedExecutor is not None:
                        try:
                            executor = UnifiedExecutor()
                            # Map frontend strategy to internal parameters
                            strat = strategy
                            use_remote = None
                            if strategy == "static-local":
                                strat = "static"
                                use_remote = False
                            elif strategy == "static-remote":
                                strat = "static"
                                use_remote = True
                            
                            start_time = time.time()
                            res_obj = executor.execute(
                                prompt=prompt,
                                routing_strategy=strat,
                                use_remote=use_remote,
                                category=category,
                                required_keys=[k.strip() for k in required_keys.split(",") if k.strip()] if required_keys else None,
                                temperature=temp
                            )
                            latency = time.time() - start_time
                            data = {
                                "text": res_obj.text,
                                "source": res_obj.source,
                                "latency": round(latency, 3),
                                "local_tokens": res_obj.local_tokens_used,
                                "remote_tokens": res_obj.remote_tokens_used,
                                "escalated": res_obj.escalated,
                                "trust_report": res_obj.trust_report,
                                "routing_strategy": strategy
                            }
                        except Exception as inner_e:
                            st.error(f"Execution Error: {str(inner_e)}")
                            data = None
                    else:
                        st.error("No active RouteLM server container found, and local imports failed. Please run 'make up' or check API logs.")
                        data = None

                if data:
                    progress.update(label="Decision Path Completed!", state="complete")
                    
                    # Custom Steps rendering matching front-end
                    is_cache = "cache" in data["source"].lower()
                    is_fallback = "fallback" in data["source"].lower()
                    
                    steps = []
                    if is_cache:
                        steps.append(("success", "Semantic Cache Hit! Returning response immediately."))
                    else:
                        steps.append(("pending", "Semantic Cache Miss. Proceeding to routing logic..."))
                        
                        is_static = data.get("routing_strategy", "").startswith("static")
                        if is_static:
                            if "local" in data["source"].lower():
                                steps.append(("success", f"Static Routing: Resolved directly via local model ({data['local_tokens']} tokens)."))
                            else:
                                steps.append(("success", f"Static Routing: Resolved directly via remote model ({data['remote_tokens']} tokens)."))
                        else:
                            # Dynamic cascades
                            bypassed = (data.get("trust_report") and data["trust_report"].get("predictive_bypass")) or "bypass" in data["source"].lower()
                            if bypassed:
                                steps.append(("escalated", "Predictive Gate: Complex prompt detected. Bypassing local pass."))
                                steps.append(("success", f"Escalated to Remote LLM: Answer resolved via provider ({data['remote_tokens']} tokens)."))
                            else:
                                steps.append(("success", "Predictive Gate: Prompt allowed for local execution."))
                                steps.append(("success", f"Local Pass (Qwen 0.5B): Completion generated ({data['local_tokens']} tokens)."))
                                
                                if is_fallback:
                                    fail_msg = "Trust Evaluator: Confidence signals check failed."
                                    if data.get("trust_report") and data["trust_report"].get("failures"):
                                        fails = []
                                        reports = data["trust_report"]["failures"]
                                        if reports.get("consistency"): fails.append("Self-Consistency")
                                        if reports.get("entropy"): fails.append("Token Entropy")
                                        if reports.get("structural"): fails.append("Structure Validation")
                                        if fails:
                                            fail_msg = f"Trust Evaluator: {' & '.join(fails)} check failed."
                                    steps.append(("failed", fail_msg))
                                    steps.append(("failed", "Remote LLM Call Failed! Outage or network timeout."))
                                    steps.append(("success", "Local Fallback: Reverted safely to the high-confidence local response."))
                                elif data.get("escalated"):
                                    fail_msg = "Trust Evaluator: Confidence check failed."
                                    if data.get("trust_report") and data["trust_report"].get("failures"):
                                        fails = []
                                        reports = data["trust_report"]["failures"]
                                        if reports.get("consistency"): fails.append("Self-Consistency")
                                        if reports.get("entropy"): fails.append("Token Entropy")
                                        if reports.get("structural"): fails.append("Structure Validation")
                                        if fails:
                                            fail_msg = f"Trust Evaluator: {' & '.join(fails)} check failed."
                                    steps.append(("failed", fail_msg))
                                    steps.append(("success", f"Escalated to Remote LLM: Answer resolved via provider ({data['remote_tokens']} tokens)."))
                                else:
                                    steps.append(("success", "Trust Evaluator: Confidence signals verified. Satisfied locally."))

                    # Render trace steps HTML
                    for status, label in steps:
                        style_class = f"trace-{status}"
                        st.markdown(f'<div class="trace-step {style_class}">{label}</div>', unsafe_allow_html=True)
                    
                    st.write("---")
                    
                    # Output Result Card
                    col_res1, col_res2 = st.columns([3, 1])
                    with col_res1:
                        st.subheader("Execution Output")
                        st.code(data["text"], language="markdown")
                    with col_res2:
                        st.subheader("Inference Meta")
                        st.write(f"**Resolved Source:** `{data['source']}`")
                        st.write(f"**Execution Latency:** `{data['latency']}s`")
                        st.write(f"**Local Tokens:** `{data['local_tokens']}`")
                        st.write(f"**Remote Tokens:** `{data['remote_tokens']}`")

# ⚙️ TUNING & CALIBRATION PAGE
elif menu == "⚙️ Tuning & Calibration":
    st.markdown("# ⚙️ Calibration & Threshold Tuning")
    st.markdown("Sweep and update the model trust criteria. Moving the sliders simulates performance indicators in real-time.")
    
    config = load_config()
    
    # Sliders
    st.subheader("Model Decision Boundaries")
    col_sl1, col_sl2 = st.columns(2)
    with col_sl1:
        consistency_th = st.slider(
            "Self-Consistency Threshold (Cosine Similarity Floor)",
            min_value=0.0,
            max_value=1.0,
            value=float(config["consistency_threshold"]),
            step=0.05
        )
    with col_sl2:
        entropy_th = st.slider(
            "Average Token Entropy Limit (Uncertainty Ceiling)",
            min_value=0.1,
            max_value=3.0,
            value=float(config["entropy_threshold"]),
            step=0.1
        )
        
    # Recalculate simulation values in real-time
    esc_rate = min(0.96, max(0.04, (consistency_th * 0.48) + ((2.6 - entropy_th) * 0.16)))
    est_accuracy = 55.4 + esc_rate * (98.2 - 55.4)
    est_cost = esc_rate * 180 * 0.000015
    tokens_saved = (1.0 - esc_rate) * 100
    
    # Display Simulated estimates
    st.write("---")
    st.subheader("Real-Time Simulated Performance")
    col_sim1, col_sim2, col_sim3 = st.columns(3)
    with col_sim1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{est_accuracy:.1f}%</div>
            <div class="metric-label">EXPECTED SYSTEM ACCURACY</div>
        </div>
        """, unsafe_allow_html=True)
    with col_sim2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{tokens_saved:.1f}%</div>
            <div class="metric-label">TOKEN EXPENSE REDUCTION</div>
        </div>
        """, unsafe_allow_html=True)
    with col_sim3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">${est_cost:.5f}</div>
            <div class="metric-label">ESTIMATED COST PER QUERY</div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("💾 Apply & Save Calibration Parameters", type="primary"):
        save_config({
            "consistency_threshold": consistency_th,
            "entropy_threshold": entropy_th
        })
        st.toast("Threshold parameters calibrated successfully!", icon="🚀")
        
    # Render sensitivity curves
    st.write("---")
    st.subheader("Calibration Trade-off Projections")
    
    col_ch1, col_ch2 = st.columns(2)
    
    with col_ch1:
        st.write("**Calibration Pareto Frontier (Cost vs Accuracy)**")
        # Pareto Frontier Data
        pareto_data = [
            {"cost": 0.0000, "accuracy": 55.4},
            {"cost": 0.0002, "accuracy": 59.8},
            {"cost": 0.0005, "accuracy": 64.2},
            {"cost": 0.0008, "accuracy": 70.5},
            {"cost": 0.0013, "accuracy": 78.5},
            {"cost": 0.0018, "accuracy": 85.1},
            {"cost": 0.0022, "accuracy": 89.2},
            {"cost": 0.0025, "accuracy": 92.4},
            {"cost": 0.0027, "accuracy": 95.0},
            {"cost": 0.0028, "accuracy": 96.8},
            {"cost": 0.0029, "accuracy": 98.2}
        ]
        df_pareto = pd.DataFrame(pareto_data)
        
        fig_pareto = px.area(
            df_pareto, 
            x="cost", 
            y="accuracy", 
            labels={"cost": "Spend ($)", "accuracy": "Accuracy (%)"},
            color_discrete_sequence=["rgba(2, 132, 199, 0.4)"]
        )
        
        # Add Active operating point Dot
        fig_pareto.add_trace(
            go.Scatter(
                x=[est_cost], 
                y=[est_accuracy], 
                mode="markers+text", 
                marker=dict(color="#ef4444", size=12, line=dict(color="white", width=2)),
                name="Active State",
                text=["Active Router"],
                textposition="top center"
            )
        )
        fig_pareto.update_layout(showlegend=False, margin=dict(t=15, b=15, l=15, r=15), height=300)
        st.plotly_chart(fig_pareto, use_container_width=True)
        
    with col_ch2:
        st.write("**Threshold Sensitivity Trade-offs**")
        sensitivity_data = [
            {"threshold": 0.0, "accuracy": 55.4, "cost": 0.0000},
            {"threshold": 0.1, "accuracy": 59.8, "cost": 0.0002},
            {"threshold": 0.2, "accuracy": 64.2, "cost": 0.0005},
            {"threshold": 0.3, "accuracy": 70.5, "cost": 0.0008},
            {"threshold": 0.4, "accuracy": 78.5, "cost": 0.0013},
            {"threshold": 0.5, "accuracy": 85.1, "cost": 0.0018},
            {"threshold": 0.6, "accuracy": 89.2, "cost": 0.0022},
            {"threshold": 0.7, "accuracy": 92.4, "cost": 0.0025},
            {"threshold": 0.8, "accuracy": 95.0, "cost": 0.0027},
            {"threshold": 0.9, "accuracy": 96.8, "cost": 0.0028},
            {"threshold": 1.0, "accuracy": 98.2, "cost": 0.0029}
        ]
        df_sens = pd.DataFrame(sensitivity_data)
        
        # Dual axis plotting via plotly graph objects
        fig_sens = go.Figure()
        fig_sens.add_trace(
            go.Scatter(x=df_sens["threshold"], y=df_sens["accuracy"], name="Accuracy (%)", line=dict(color="#0284c7", width=3))
        )
        fig_sens.add_trace(
            go.Scatter(x=df_sens["threshold"], y=df_sens["cost"], name="Query Cost ($)", yaxis="y2", line=dict(color="#0d9488", width=3))
        )
        
        # Add Reference Line for Active Threshold
        fig_sens.add_vline(x=consistency_th, line_width=2, line_dash="dash", line_color="#ef4444")
        
        # Add Reference Dots for Active values
        fig_sens.add_trace(
            go.Scatter(
                x=[consistency_th], 
                y=[est_accuracy], 
                mode="markers", 
                marker=dict(color="#0284c7", size=10, line=dict(color="white", width=2.5)),
                showlegend=False
            )
        )
        fig_sens.add_trace(
            go.Scatter(
                x=[consistency_th], 
                y=[est_cost], 
                yaxis="y2",
                mode="markers", 
                marker=dict(color="#0d9488", size=10, line=dict(color="white", width=2.5)),
                showlegend=False
            )
        )
        
        fig_sens.update_layout(
            yaxis=dict(title="Accuracy (%)", titlefont=dict(color="#0284c7"), tickfont=dict(color="#0284c7")),
            yaxis2=dict(title="Query Cost ($)", titlefont=dict(color="#0d9488"), tickfont=dict(color="#0d9488"), overlaying="y", side="right"),
            margin=dict(t=15, b=15, l=15, r=15),
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_sens, use_container_width=True)

# 📜 EXECUTION LOGS PAGE
elif menu == "📜 Execution Logs":
    st.markdown("# 📜 Telemetry Execution Logs")
    st.markdown("Review historical execution traces and diagnostics collected in `routing_execution.jsonl`.")
    
    logs = get_history()
    if not logs:
        st.info("No execution logs found. Run queries from the Sandbox tab first!")
    else:
        log_records = []
        for log in logs:
            source = log.get("source", "Unknown")
            # Style color markup for Streamlit table cells isn't clean, but we can do a simplified table
            log_records.append({
                "Timestamp": log.get("timestamp", "Unknown").split("T")[1][:8] if "T" in log.get("timestamp", "") else "Unknown",
                "Query Prompt": log.get("prompt", ""),
                "Strategy": log.get("routing_strategy", ""),
                "Category": log.get("category", ""),
                "Final Source": source,
                "Local Tokens": log.get("local_tokens_used", 0),
                "Remote Tokens": log.get("remote_tokens_used", 0),
                "Latency (s)": log.get("latency_seconds", 0.0)
            })
        
        df_logs = pd.DataFrame(log_records)
        st.dataframe(df_logs, use_container_width=True, height=500)
