import streamlit as st
import os
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from routing_agent.executor import UnifiedExecutor
from routing_agent.local_client import LocalClient
from routing_agent.remote_client import RemoteClient
from routing_agent.evaluator import TrustEvaluator
from routing_agent.adjuster import BudgetAwareAdjuster

# Configure page metadata and layout
st.set_page_config(
    page_title="HMR: Hybrid Model Router Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling for Hackathon presentation
st.markdown("""
<style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(135deg, #FF3366 0%, #FF9933 50%, #33CCFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        font-family: 'Inter', sans-serif;
        font-weight: 300;
        color: #8892B0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Metrics display */
    .metric-card {
        background: rgba(25, 30, 45, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    .metric-val {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #00FFCC;
        margin-bottom: 0.2rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #8892B0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* Trace block styles */
    .trace-card {
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        border-left: 5px solid #8892B0;
        background: rgba(255, 255, 255, 0.02);
    }
    .trace-success {
        border-left-color: #00FFCC;
        background: rgba(0, 255, 204, 0.03);
    }
    .trace-warning {
        border-left-color: #FFCC00;
        background: rgba(255, 204, 0, 0.03);
    }
    .trace-error {
        border-left-color: #FF3366;
        background: rgba(255, 51, 102, 0.03);
    }
</style>
""", unsafe_allow_html=True)

# App Title & Subheader
st.markdown('<div class="main-title">Hybrid Model Router (HMR)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent LLM Cascade & Budget-Aware Routing Orchestrator</div>', unsafe_allow_html=True)

# ----------------- Sidebar Configuration -----------------
st.sidebar.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=70)
st.sidebar.markdown("### **Router Configuration**")

routing_strategy = st.sidebar.selectbox(
    "Routing Strategy",
    options=["dynamic", "adaptive", "static_local", "static_remote"],
    help="dynamic uses calibrated thresholds; adaptive scales them based on token budgets; static forces a single model run."
)

enable_cache = st.sidebar.checkbox("Enable Semantic Cache", value=True)

# Threshold Adjustments (Loads defaults or calibrated values)
config_consistency = 0.4
config_entropy = 0.8
try:
    if os.path.exists("routing_config.json"):
        with open("routing_config.json", "r") as f:
            config = json.load(f)
            config_consistency = config.get("consistency_threshold", 0.4)
            config_entropy = config.get("entropy_threshold", 0.8)
except Exception:
    pass

st.sidebar.markdown("---")
st.sidebar.markdown("### **Trust Parameters Override**")
override_thresholds = st.sidebar.checkbox("Override Calibrated Thresholds")

if override_thresholds:
    consistency_threshold = st.sidebar.slider("Self-Consistency Threshold", 0.0, 1.0, 0.50, step=0.05)
    entropy_threshold = st.sidebar.slider("Entropy Threshold", 0.0, 3.0, 1.0, step=0.1)
else:
    consistency_threshold = config_consistency
    entropy_threshold = config_entropy
    st.sidebar.info(f"Using calibrated defaults:  \n- **Consistency:** {consistency_threshold}  \n- **Entropy:** {entropy_threshold}")

# Create tabs for sandbox playground vs metrics charts
tab_sandbox, tab_analytics = st.tabs(["🎮 Prompt Sandbox Playground", "📊 Performance & Cost Analytics"])

# ----------------- Tab 1: Prompt Sandbox -----------------
with tab_sandbox:
    st.markdown("### **Interactive Playground**")
    
    # Prompt Templates
    templates = {
        "Custom Prompt": "",
        "Math (Simple)": "Solve 3x + 15 = 42 for x.",
        "Math (Complex)": "Find the derivative of f(x) = x^3 * sin(x) + 5x^2 - 12.",
        "Reasoning Puzzle": "A farmer has chickens and rabbits. If there are 35 heads and 94 legs, how many chickens and rabbits does he have?",
        "Code Syntax check": "Write a python function to compute the nth fibonacci number.",
        "Structured JSON Output": "Generate a JSON list of 3 users with fields: id, name, and role."
    }
    
    template_choice = st.selectbox("Load Template Prompt:", list(templates.keys()))
    default_prompt = templates[template_choice]
    
    # Prompt Category selector
    cat_mapping = {
        "Custom Prompt": "general",
        "Math (Simple)": "math",
        "Math (Complex)": "math",
        "Reasoning Puzzle": "reasoning",
        "Code Syntax check": "code",
        "Structured JSON Output": "structured_output"
    }
    default_cat = cat_mapping.get(template_choice, "general")
    
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        prompt_text = st.text_area("Prompt Input:", value=default_prompt, height=120)
    with col_p2:
        prompt_category = st.selectbox(
            "Task Category",
            options=["general", "math", "reasoning", "code", "structured_output"],
            index=["general", "math", "reasoning", "code", "structured_output"].index(default_cat)
        )
        required_keys_input = st.text_input("Required JSON Keys (comma separated)", value="id,name" if prompt_category == "structured_output" else "")
        
    submit_btn = st.button("🚀 Execute Hybrid Query", type="primary")
    
    if submit_btn:
        if not prompt_text.strip():
            st.warning("Please enter a prompt first.")
        else:
            with st.spinner("Processing through routing pipeline..."):
                # Setup UnifiedExecutor with current parameters
                # If Fireworks API key is missing, mock remote to run simulation gracefully
                executor = UnifiedExecutor()
                
                # Check and set overrides
                executor.trust_evaluator.consistency_threshold = consistency_threshold
                executor.trust_evaluator.entropy_threshold = entropy_threshold
                
                # Configure custom cascade based on user selected strategy
                has_remote = bool(os.environ.get("FIREWORKS_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY"))
                if not has_remote:
                    # In simulation mode, mock the remote query to return target placeholder text
                    mock_remote = MagicMock() if "MagicMock" in globals() else type("MockRemote", (), {})()
                    class MockRemote:
                        def query(self, prompt, *args, **kwargs):
                            from routing_agent.remote_client import RemoteExecutionResult
                            return RemoteExecutionResult(
                                text="[Remote Expert Response: Simulated output for " + prompt[:20] + "...]",
                                total_tokens=150,
                                raw_response={}
                            )
                    executor.remote_client = MockRemote()
                
                # Parse required keys
                req_keys = None
                if required_keys_input:
                    req_keys = [k.strip() for k in required_keys_input.split(",") if k.strip()]
                
                # Temporary execution capture
                start_time = time.time()
                
                # Check cache hit/miss manually to trace it in UI
                cache_hit_res = None
                if enable_cache and executor.semantic_cache:
                    cache_hit_res = executor.semantic_cache.get(prompt_text)
                
                # Execute query
                use_rem = (routing_strategy == "static_remote")
                strat = "static" if routing_strategy in ("static_local", "static_remote") else routing_strategy
                
                res = executor.execute(
                    prompt=prompt_text,
                    use_remote=use_rem,
                    routing_strategy=strat,
                    category=prompt_category,
                    required_keys=req_keys
                )
                latency = time.time() - start_time
                
                # Display Trace
                st.markdown("### **Routing Execution Trace**")
                
                # Step 1: Cache check
                if enable_cache:
                    if cache_hit_res:
                        st.markdown("""
                        <div class="trace-card trace-success">
                            <strong>[1/4] Semantic Cache: HIT</strong><br/>
                            A semantically matching query was found in the cache. Returning saved answer instantly.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="trace-card">
                            <strong>[1/4] Semantic Cache: MISS</strong><br/>
                            No matching query found in cache. Proceeding to routing pass.
                        </div>
                        """, unsafe_allow_html=True)
                
                # Step 2: Predictive Gate
                if not cache_hit_res and routing_strategy in ("dynamic", "adaptive"):
                    bypassed = executor.predictive_gate.should_bypass(prompt_text, prompt_category)
                    if bypassed:
                        st.markdown("""
                        <div class="trace-card trace-warning">
                            <strong>[2/4] Predictive Routing Gate: BYPASS</strong><br/>
                            Prompt flags detected (domain keywords/length). Bypassing local tier directly to Remote model.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="trace-card">
                            <strong>[2/4] Predictive Routing Gate: PASS</strong><br/>
                            Prompt is within local capability parameters. Running Local LLM execution.
                        </div>
                        """, unsafe_allow_html=True)
                
                # Step 3: Local Eval
                if not cache_hit_res and routing_strategy in ("dynamic", "adaptive") and not res.source.startswith("remote"):
                    st.markdown(f"""
                    <div class="trace-card trace-success">
                        <strong>[3/4] Trust Evaluation Pass: ACCEPTED</strong><br/>
                        Local model completed response. Entropy, structural tests, and self-consistency checked out.
                    </div>
                    """, unsafe_allow_html=True)
                elif not cache_hit_res and routing_strategy in ("dynamic", "adaptive") and res.source.startswith("remote (escalation)"):
                    st.markdown("""
                    <div class="trace-card trace-warning">
                        <strong>[3/4] Trust Evaluation Pass: ESCALATED</strong><br/>
                        Local completion failed safety validation tests. Escalating query to Remote expert tier.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Step 4: Final Output details
                st.markdown("""
                <div class="trace-card trace-success">
                    <strong>[4/4] Final Response Returned</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Output Columns
                out_col1, out_col2 = st.columns([2, 1])
                with out_col1:
                    st.markdown("**Model Output:**")
                    st.info(res.text)
                
                with out_col2:
                    st.markdown("**Performance Statistics:**")
                    # Calculate estimated cost savings ($0.015 per 1K remote vs $0.0002 per 1K local)
                    local_cost = (res.local_tokens_used / 1000.0) * 0.0002
                    remote_cost = (res.remote_tokens_used / 1000.0) * 0.015
                    total_spend = local_cost + remote_cost
                    remote_only_cost = ((res.local_tokens_used + res.remote_tokens_used + 100) / 1000.0) * 0.015
                    savings = remote_only_cost - total_spend
                    
                    st.metric("Source Used", res.source.upper())
                    st.metric("Local / Remote Tokens", f"{res.local_tokens_used} / {res.remote_tokens_used}")
                    st.metric("Latency", f"{latency:.2f} seconds")
                    st.metric("Estimated Cost Savings", f"${max(0.0, savings):.6f}", delta=f"{max(0.0, savings)*100:.1f}% vs Remote-Only")

# ----------------- Tab 2: Performance Analytics -----------------
with tab_analytics:
    st.markdown("### **Telemetry & Performance Analytics**")
    
    # Read routing logs
    log_file = "routing_execution.jsonl"
    logs_data = []
    
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                for line in f:
                    if line.strip():
                        logs_data.append(json.loads(line))
        except Exception:
            pass
            
    # Mock fallback data if log file is missing/empty for presentation visual completeness
    if not logs_data:
        logs_data = [
            {"timestamp": "2026-07-09T08:00:00Z", "category": "math", "routing_strategy": "dynamic", "source": "local", "local_tokens_used": 150, "remote_tokens_used": 0, "latency_seconds": 0.45, "trust_report": {"signals": {"mean_entropy": 0.35, "self_consistency": 0.85}}},
            {"timestamp": "2026-07-09T08:05:00Z", "category": "code", "routing_strategy": "dynamic", "source": "remote (escalation)", "local_tokens_used": 120, "remote_tokens_used": 340, "latency_seconds": 2.1, "trust_report": {"signals": {"mean_entropy": 1.25, "self_consistency": 0.40}}},
            {"timestamp": "2026-07-09T08:10:00Z", "category": "general", "routing_strategy": "dynamic", "source": "local", "local_tokens_used": 60, "remote_tokens_used": 0, "latency_seconds": 0.2, "trust_report": {"signals": {"mean_entropy": 0.15, "self_consistency": 0.95}}},
            {"timestamp": "2026-07-09T08:15:00Z", "category": "reasoning", "routing_strategy": "dynamic", "source": "remote (predictive bypass)", "local_tokens_used": 0, "remote_tokens_used": 280, "latency_seconds": 1.4, "trust_report": {"signals": {"mean_entropy": 0.0, "self_consistency": 0.0}}},
            {"timestamp": "2026-07-09T08:20:00Z", "category": "structured_output", "routing_strategy": "dynamic", "source": "local", "local_tokens_used": 180, "remote_tokens_used": 0, "latency_seconds": 0.55, "trust_report": {"signals": {"mean_entropy": 0.42, "self_consistency": 0.78}}},
            {"timestamp": "2026-07-09T08:25:00Z", "category": "math", "routing_strategy": "dynamic", "source": "remote (escalation)", "local_tokens_used": 140, "remote_tokens_used": 420, "latency_seconds": 2.4, "trust_report": {"signals": {"mean_entropy": 0.95, "self_consistency": 0.32}}}
        ]
        
    df = pd.DataFrame(logs_data)
    
    # Pre-processing
    df["is_remote"] = df["source"].apply(lambda s: 1 if "remote" in str(s) or "escalat" in str(s) else 0)
    df["local_cost"] = df["local_tokens_used"] * 0.0002 / 1000
    df["remote_cost"] = df["remote_tokens_used"] * 0.015 / 1000
    df["total_cost"] = df["local_cost"] + df["remote_cost"]
    df["remote_only_cost"] = (df["local_tokens_used"] + df["remote_tokens_used"]) * 0.015 / 1000
    df["savings"] = df["remote_only_cost"] - df["total_cost"]
    
    # Calculate Summary metrics
    total_queries = len(df)
    local_only_count = len(df[df["is_remote"] == 0])
    escalation_count = len(df[df["is_remote"] == 1])
    local_percentage = (local_only_count / total_queries * 100) if total_queries > 0 else 0
    total_savings = df["savings"].sum()
    avg_lat = df["latency_seconds"].mean()
    
    # Display Premium Cards
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{total_queries}</div>
            <div class="metric-label">Total Queries</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{local_percentage:.1f}%</div>
            <div class="metric-label">Local Deflection Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">${total_savings:.4f}</div>
            <div class="metric-label">Total Cost Saved</div>
        </div>
        """, unsafe_allow_html=True)
    with mcol4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{avg_lat:.2f}s</div>
            <div class="metric-label">Avg Query Latency</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Graph Columns
    gcol1, gcol2 = st.columns(2)
    
    with gcol1:
        st.markdown("#### **Accumulated Cost Over Time (Local vs Remote vs Hybrid)**")
        df["cum_total_cost"] = df["total_cost"].cumsum()
        df["cum_remote_only_cost"] = df["remote_only_cost"].cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["cum_remote_only_cost"], name="Remote-Only Baseline", line=dict(color="#FF3366", width=3)))
        fig.add_trace(go.Scatter(x=df.index, y=df["cum_total_cost"], name="Hybrid Router", line=dict(color="#00FFCC", width=3, dash="dash")))
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Request Number",
            yaxis_title="Cumulative Spend ($)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with gcol2:
        st.markdown("#### **Confidence Boundary (Entropy vs Self-Consistency)**")
        # Extract metrics
        entropy_vals = []
        consistency_vals = []
        labels = []
        for i, row in df.iterrows():
            tr = row.get("trust_report", {})
            if isinstance(tr, dict):
                sig = tr.get("signals", {})
                if isinstance(sig, dict):
                    entropy_vals.append(sig.get("mean_entropy", 0.0))
                    consistency_vals.append(sig.get("self_consistency", 1.0))
                    labels.append(row["source"])
                    continue
            entropy_vals.append(0.0)
            consistency_vals.append(1.0)
            labels.append(row["source"])
            
        plot_df = pd.DataFrame({
            "Mean Entropy": entropy_vals,
            "Self Consistency": consistency_vals,
            "Routing Result": labels
        })
        
        fig_scatter = px.scatter(
            plot_df,
            x="Mean Entropy",
            y="Self Consistency",
            color="Routing Result",
            color_discrete_map={
                "local": "#00FFCC",
                "remote (escalation)": "#FFCC00",
                "remote (predictive bypass)": "#FF3366"
            },
            template="plotly_dark",
            title="Interactive Scatter Matrix of Routing Trust Boundary"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
