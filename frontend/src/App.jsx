import React, { useState, useEffect } from 'react';
import { 
  Play, ShieldAlert, Cpu, Database, TrendingUp, CheckCircle, 
  Settings, History, BarChart3, ChevronRight, Activity, 
  HelpCircle, RefreshCw, AlertTriangle, ArrowRight, DollarSign
} from 'lucide-react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, 
  ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar, Legend,
  LineChart, Line
} from 'recharts';

export default function App() {
  const [activeTab, setActiveTab] = useState('analytics'); // Analytics as default page
  const [prompt, setPrompt] = useState('Write a python function to check if a number is prime.');
  const [strategy, setStrategy] = useState('dynamic');
  const [category, setCategory] = useState('code');
  const [temp, setTemp] = useState(0.0);
  const [requiredKeys, setRequiredKeys] = useState('');
  
  // Settings & config
  const [config, setConfig] = useState({ consistency_threshold: 0.4, entropy_threshold: 0.8 });
  const [saveLoading, setSaveLoading] = useState(false);
  const [notification, setNotification] = useState(null);

  const showNotification = (message, type = 'success') => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 4000);
  };
  
  // Execution status
  const [execLoading, setExecLoading] = useState(false);
  const [execResult, setExecResult] = useState(null);
  const [traceSteps, setTraceSteps] = useState([]);
  
  // Telemetry stats & history
  const [stats, setStats] = useState({
    total_queries: 0,
    cache_hits: 0,
    local_runs: 0,
    escalations: 0,
    total_local_tokens: 0,
    total_remote_tokens: 0,
    savings_dollars: 0.0,
    avg_latency: 0.0,
    source_distribution: {}
  });
  const [history, setHistory] = useState([]);
  const [statsLoading, setStatsLoading] = useState(false);

  // Pareto Frontier data for calibration visualization
  const paretoData = [
    { name: 'Max Local', cost: 0.0, accuracy: 55.4, threshold: '0.1' },
    { name: 'Highly Optimized', cost: 0.002, accuracy: 78.5, threshold: '0.3' },
    { name: 'Optimal Balancer', cost: 0.005, accuracy: 89.2, threshold: '0.45' },
    { name: 'Conservative Remote', cost: 0.012, accuracy: 94.6, threshold: '0.7' },
    { name: 'Max Remote', cost: 0.025, accuracy: 98.2, threshold: '1.0' }
  ];

  // Fetch telemetry and configurations
  const fetchTelemetry = async (manual = false) => {
    setStatsLoading(true);
    try {
      const statsRes = await fetch('http://localhost:8000/api/stats');
      const statsData = await statsRes.json();
      setStats(statsData);
      
      const historyRes = await fetch('http://localhost:8000/api/history');
      const historyData = await historyRes.json();
      setHistory(historyData);
      if (manual === true) {
        showNotification("Telemetry and stats refreshed!", "success");
      }
    } catch (err) {
      console.error("Error fetching telemetry:", err);
      if (manual === true) {
        showNotification("Failed to refresh stats: " + err.message, "error");
      }
    } finally {
      setStatsLoading(false);
    }
  };

  const fetchConfig = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/config');
      const data = await res.json();
      setConfig(data);
    } catch (err) {
      console.error("Error fetching config:", err);
    }
  };

  useEffect(() => {
    fetchTelemetry();
    fetchConfig();
  }, []);

  // Save updated config thresholds
  const handleSaveConfig = async () => {
    setSaveLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      if (res.ok) {
        showNotification("Threshold parameters calibrated successfully!", "success");
      } else {
        throw new Error("Server returned status " + res.status);
      }
    } catch (err) {
      showNotification("Failed to save config: " + err.message, "error");
    } finally {
      setSaveLoading(false);
    }
  };

  // Submit query for execution tracing
  const handleExecute = async () => {
    if (!prompt.trim()) return;
    setExecLoading(true);
    setExecResult(null);
    
    // Set up step-by-step visual tracing
    setTraceSteps([
      { id: 'cache', status: 'pending', label: 'Checking Semantic Cache...' }
    ]);

    try {
      // Step 1: Cache Check simulation delay
      await new Promise(r => setTimeout(r, 600));
      
      const reqPayload = {
        prompt,
        routing_strategy: strategy,
        category,
        temperature: temp,
        required_keys: requiredKeys.trim() ? requiredKeys.split(',').map(k => k.trim()) : null
      };

      const res = await fetch('http://localhost:8000/api/route', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reqPayload)
      });
      
      if (!res.ok) throw new Error("Backend execution failed");
      const data = await res.json();

      setExecResult(data);

      const steps = [];
      const isCache = data.source.toLowerCase().includes('cache');
      const isFallback = data.source.toLowerCase().includes('fallback');
      
      if (isCache) {
        steps.push({ id: 'cache', status: 'success', label: 'Semantic Cache Hit! Returning response immediately.' });
      } else {
        steps.push({ id: 'cache', status: 'miss', label: 'Semantic Cache Miss. Proceeding to routing logic...' });
        
        const isStatic = data.routing_strategy && data.routing_strategy.startsWith('static');
        
        if (isStatic) {
          if (data.source.toLowerCase().includes('local')) {
            steps.push({ id: 'local', status: 'success', label: `Static Routing: Resolved directly via local model (${data.local_tokens} tokens).` });
          } else {
            steps.push({ id: 'remote', status: 'success', label: `Static Routing: Resolved directly via remote model (${data.remote_tokens} tokens).` });
          }
        } else {
          // Dynamic or Adaptive Routing
          const bypassed = (data.trust_report && data.trust_report.predictive_bypass) || data.source.toLowerCase().includes('bypass');
          
          if (bypassed) {
            steps.push({ id: 'gate', status: 'escalated', label: 'Predictive Gate: Complex prompt detected. Bypassing local pass.' });
            steps.push({ id: 'remote', status: 'success', label: `Escalated to Remote LLM: Answer resolved via provider (${data.remote_tokens} tokens).` });
          } else {
            steps.push({ id: 'gate', status: 'success', label: 'Predictive Gate: Prompt allowed for local execution.' });
            steps.push({ id: 'local', status: 'success', label: `Local Pass (Qwen 0.5B): Completion generated (${data.local_tokens} tokens).` });
            
            if (isFallback) {
              let failMsg = "Trust Evaluator: Confidence signals check failed.";
              if (data.trust_report && data.trust_report.failures) {
                const fails = [];
                if (data.trust_report.failures.consistency) fails.push("Self-Consistency");
                if (data.trust_report.failures.entropy) fails.push("Token Entropy");
                if (data.trust_report.failures.structural) fails.push("Structure Validation");
                if (fails.length > 0) {
                  failMsg = `Trust Evaluator: ${fails.join(" & ")} check failed.`;
                }
              }
              steps.push({ id: 'trust', status: 'failed', label: failMsg });
              steps.push({ id: 'remote', status: 'failed', label: 'Remote LLM Call Failed! Outage or network timeout.' });
              steps.push({ id: 'local', status: 'success', label: 'Local Fallback: Reverted safely to the high-confidence local response.' });
            } else if (data.escalated) {
              let failMsg = "Trust Evaluator: Confidence signals check failed.";
              if (data.trust_report && data.trust_report.failures) {
                const fails = [];
                if (data.trust_report.failures.consistency) fails.push("Self-Consistency");
                if (data.trust_report.failures.entropy) fails.push("Token Entropy");
                if (data.trust_report.failures.structural) fails.push("Structure Validation");
                if (fails.length > 0) {
                  failMsg = `Trust Evaluator: ${fails.join(" & ")} check failed.`;
                }
              }
              steps.push({ id: 'trust', status: 'failed', label: failMsg });
              steps.push({ id: 'remote', status: 'success', label: `Escalated to Remote LLM: Answer resolved via provider (${data.remote_tokens} tokens).` });
            } else {
              steps.push({ id: 'trust', status: 'success', label: 'Trust Evaluator: Confidence signals verified. Satisfied locally.' });
            }
          }
        }
      }

      setTraceSteps(steps);
      fetchTelemetry(); // Refresh metrics on sandbox run
    } catch (err) {
      console.error(err);
      setTraceSteps(prev => [...prev, { id: 'error', status: 'failed', label: `Error: ${err.message}` }]);
    } finally {
      setExecLoading(false);
    }
  };

  // Pie chart data prep
  const sourcePieData = Object.entries(stats.source_distribution).map(([name, value]) => ({
    name, value
  }));

  const COLORS = ['#0284c7', '#10b981', '#f59e0b', '#db2777', '#3b82f6'];

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#f8fafc' }}>
      {/* Floating Notification Toast */}
      {notification && (
        <div style={{
          position: 'fixed',
          top: '24px',
          right: '24px',
          backgroundColor: notification.type === 'success' ? '#ecfdf5' : '#fef2f2',
          border: '1px solid ' + (notification.type === 'success' ? '#10b981' : '#f87171'),
          borderRadius: '8px',
          padding: '16px 20px',
          boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02)',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          zIndex: 9999,
          animation: 'slideDown 0.3s ease-out'
        }}>
          {notification.type === 'success' ? (
            <CheckCircle size={20} color="#10b981" />
          ) : (
            <AlertTriangle size={20} color="#ef4444" />
          )}
          <span style={{ 
            fontSize: '14px', 
            fontWeight: '600', 
            color: notification.type === 'success' ? '#065f46' : '#991b1b' 
          }}>
            {notification.message}
          </span>
        </div>
      )}
      {/* Sidebar Navigation */}
      <div style={{ 
        width: '260px', 
        minWidth: '260px',
        maxWidth: '260px',
        flexShrink: 0,
        borderRight: '1px solid rgba(226, 232, 240, 0.8)', 
        backgroundColor: '#ffffff',
        padding: '24px 16px',
        display: 'flex',
        flexDirection: 'column',
        gap: '24px'
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px' }}>
            <div style={{ 
              width: '32px', 
              height: '32px', 
              borderRadius: '8px', 
              background: 'linear-gradient(135deg, #0284c7, #0d9488)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Cpu size={18} color="#fff" />
            </div>
            <span style={{ fontSize: '18px', fontWeight: '800', tracking: '-0.025em', color: '#0f172a' }}>HMR Engine</span>
          </div>
          <span style={{ fontSize: '12px', color: '#64748b' }}>Hybrid Model Router v2.4</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', flexGrow: 1 }}>
          <button 
            onClick={() => setActiveTab('analytics')}
            style={{
              display: 'flex', alignItems: 'center', gap: '12px', padding: '12px', borderRadius: '8px',
              border: 'none', cursor: 'pointer', textAlign: 'left', width: '100%',
              backgroundColor: activeTab === 'analytics' ? 'rgba(2, 132, 199, 0.08)' : 'transparent',
              color: activeTab === 'analytics' ? '#0284c7' : '#475569',
              fontWeight: activeTab === 'analytics' ? '600' : '400',
              transition: 'all 0.2s'
            }}
          >
            <BarChart3 size={18} />
            <span>Analytics & Stats</span>
          </button>

          <button 
            onClick={() => setActiveTab('sandbox')}
            style={{
              display: 'flex', alignItems: 'center', gap: '12px', padding: '12px', borderRadius: '8px',
              border: 'none', cursor: 'pointer', textAlign: 'left', width: '100%',
              backgroundColor: activeTab === 'sandbox' ? 'rgba(2, 132, 199, 0.08)' : 'transparent',
              color: activeTab === 'sandbox' ? '#0284c7' : '#475569',
              fontWeight: activeTab === 'sandbox' ? '600' : '400',
              transition: 'all 0.2s'
            }}
          >
            <Play size={18} />
            <span>Sandbox Playground</span>
          </button>

          <button 
            onClick={() => setActiveTab('calibration')}
            style={{
              display: 'flex', alignItems: 'center', gap: '12px', padding: '12px', borderRadius: '8px',
              border: 'none', cursor: 'pointer', textAlign: 'left', width: '100%',
              backgroundColor: activeTab === 'calibration' ? 'rgba(2, 132, 199, 0.08)' : 'transparent',
              color: activeTab === 'calibration' ? '#0284c7' : '#475569',
              fontWeight: activeTab === 'calibration' ? '600' : '400',
              transition: 'all 0.2s'
            }}
          >
            <Settings size={18} />
            <span>Tuning Calibration</span>
          </button>

          <button 
            onClick={() => setActiveTab('logs')}
            style={{
              display: 'flex', alignItems: 'center', gap: '12px', padding: '12px', borderRadius: '8px',
              border: 'none', cursor: 'pointer', textAlign: 'left', width: '100%',
              backgroundColor: activeTab === 'logs' ? 'rgba(2, 132, 199, 0.08)' : 'transparent',
              color: activeTab === 'logs' ? '#0284c7' : '#475569',
              fontWeight: activeTab === 'logs' ? '600' : '400',
              transition: 'all 0.2s'
            }}
          >
            <History size={18} />
            <span>Telemetry Logs</span>
          </button>
        </div>

        <div style={{ 
          borderTop: '1px solid rgba(226, 232, 240, 0.8)', 
          paddingTop: '16px',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px'
        }}>
          <button 
            onClick={() => fetchTelemetry(true)}
            style={{
              display: 'flex', alignItems: 'center', gap: '8px', backgroundColor: 'transparent',
              border: '1px solid rgba(71, 85, 105, 0.15)', padding: '8px 12px', borderRadius: '6px',
              color: '#475569', cursor: 'pointer', fontSize: '13px', justifyContent: 'center'
            }}
          >
            <RefreshCw size={14} className={statsLoading ? 'animate-spin' : ''} />
            <span>Refresh Stats</span>
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div style={{ flexGrow: 1, padding: '40px', overflowY: 'auto' }}>
        
        {/* Play Sandbox Tab */}
        {activeTab === 'sandbox' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
            <div>
              <h1 style={{ margin: '0 0 8px 0', fontSize: '28px', fontWeight: '800', color: '#0f172a' }}>Sandbox Playground</h1>
              <p style={{ color: '#475569', margin: 0 }}>Interact with the Unified Executor and trace execution decisions in real time.</p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '3fr 1.3fr', gap: '30px' }}>
              {/* Prompt Settings and input */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  <label style={{ fontSize: '14px', fontWeight: '600', color: '#334155' }}>User Query Prompt</label>
                  <textarea 
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    style={{
                      width: '100%', height: '120px', backgroundColor: '#ffffff', border: '1px solid rgba(226, 232, 240, 0.8)',
                      borderRadius: '8px', padding: '12px', color: '#0f172a', fontSize: '15px', fontFamily: 'inherit',
                      resize: 'vertical', boxSizing: 'border-box'
                    }}
                  />
                  <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', flexGrow: 1 }}>
                      <label style={{ fontSize: '13px', color: '#475569' }}>Routing Strategy</label>
                      <select 
                        value={strategy} 
                        onChange={(e) => setStrategy(e.target.value)}
                        style={{
                          backgroundColor: '#ffffff', border: '1px solid rgba(226, 232, 240, 0.8)',
                          padding: '10px', borderRadius: '6px', color: '#0f172a'
                        }}
                      >
                        <option value="dynamic">Dynamic Routing</option>
                        <option value="adaptive">Adaptive Routing (Budget Aware)</option>
                        <option value="static-local">Static Local (0.5B)</option>
                        <option value="static-remote">Static Remote (70B)</option>
                      </select>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', flexGrow: 1 }}>
                      <label style={{ fontSize: '13px', color: '#475569' }}>Task Category</label>
                      <select 
                        value={category} 
                        onChange={(e) => setCategory(e.target.value)}
                        style={{
                          backgroundColor: '#ffffff', border: '1px solid rgba(226, 232, 240, 0.8)',
                          padding: '10px', borderRadius: '6px', color: '#0f172a'
                        }}
                      >
                        <option value="general">General QA</option>
                        <option value="code">Python Code</option>
                        <option value="math">Mathematics</option>
                        <option value="reasoning">Logical Reasoning</option>
                        <option value="structured_output">Structured JSON</option>
                      </select>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', width: '120px' }}>
                      <label style={{ fontSize: '13px', color: '#475569' }}>Temperature</label>
                      <input 
                        type="number" 
                        min="0" max="1" step="0.1" 
                        value={temp}
                        onChange={(e) => setTemp(parseFloat(e.target.value))}
                        style={{
                          backgroundColor: '#ffffff', border: '1px solid rgba(226, 232, 240, 0.8)',
                          padding: '10px', borderRadius: '6px', color: '#0f172a'
                        }}
                      />
                    </div>
                  </div>

                  {category === 'structured_output' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }} className="animate-slide-down">
                      <label style={{ fontSize: '13px', color: '#475569' }}>Required Keys (comma-separated)</label>
                      <input 
                        type="text" 
                        placeholder="e.g. name, age, city" 
                        value={requiredKeys}
                        onChange={(e) => setRequiredKeys(e.target.value)}
                        style={{
                          backgroundColor: '#ffffff', border: '1px solid rgba(226, 232, 240, 0.8)',
                          padding: '10px', borderRadius: '6px', color: '#0f172a'
                        }}
                      />
                    </div>
                  )}

                  <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '10px' }}>
                    <button 
                      onClick={handleExecute}
                      disabled={execLoading}
                      className="btn-primary"
                    >
                      {execLoading ? <RefreshCw size={16} className="animate-spin" /> : <Play size={16} />}
                      <span>{execLoading ? 'Executing Traces...' : 'Run Query Engine'}</span>
                    </button>
                  </div>
                </div>

                {/* Final Completion Output Result */}
                {execResult && (
                  <div className="glass-panel animate-slide-down" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(226,232,240,0.8)', paddingBottom: '12px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <CheckCircle size={18} color="#10b981" />
                        <span style={{ fontWeight: '700', fontSize: '16px', color: '#0f172a' }}>Execution Output</span>
                      </div>
                      <div style={{ display: 'flex', gap: '10px' }}>
                        <span style={{ padding: '4px 8px', borderRadius: '4px', backgroundColor: 'rgba(2, 132, 199, 0.08)', fontSize: '12px', color: '#0284c7', fontWeight: '600' }}>
                          Resolved: {execResult.source}
                        </span>
                        <span style={{ padding: '4px 8px', borderRadius: '4px', backgroundColor: 'rgba(16, 185, 129, 0.08)', fontSize: '12px', color: '#059669', fontWeight: '600' }}>
                          Latency: {execResult.latency}s
                        </span>
                      </div>
                    </div>
                    
                    <div style={{ 
                      backgroundColor: '#f1f5f9', border: '1px solid rgba(226,232,240,0.8)',
                      borderRadius: '8px', padding: '16px', fontSize: '14px', lineHeight: '1.6',
                      fontFamily: category === 'code' ? 'Courier, monospace' : 'inherit',
                      whiteSpace: 'pre-wrap', color: '#0f172a', maxHeight: '350px', overflowY: 'auto'
                    }}>
                      {execResult.text}
                    </div>
                  </div>
                )}
              </div>

              {/* Live Trace waterfall flowchart */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div className="glass-panel" style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: '18px' }}>
                  <h3 style={{ margin: 0, fontSize: '16px', fontWeight: '700', color: '#0f172a' }}>Decision Trace Path</h3>
                  
                  {execLoading ? (
                    <div style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#0284c7', textAlign: 'center', gap: '16px', minHeight: '180px' }}>
                      <RefreshCw size={32} className="animate-spin" />
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        <span style={{ fontSize: '14px', fontWeight: '600', color: '#0f172a' }}>Tracing Execution Path...</span>
                        <span style={{ fontSize: '12px', color: '#64748b' }}>Running cache check and model evaluation</span>
                      </div>
                    </div>
                  ) : traceSteps.length === 0 ? (
                    <div style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#64748b', textAlign: 'center', gap: '12px' }}>
                      <Activity size={32} />
                      <span style={{ fontSize: '14px' }}>No active runs.<br/>Click "Run Query Engine" to view decision mapping.</span>
                    </div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', position: 'relative' }}>
                      {traceSteps.map((step, idx) => (
                        <div key={idx} style={{ display: 'flex', gap: '16px', position: 'relative' }}>
                          {/* Connector Line */}
                          {idx < traceSteps.length - 1 && (
                            <div style={{
                              position: 'absolute', left: '15px', top: '30px', bottom: '-20px', width: '2px',
                              backgroundColor: 'rgba(2, 132, 199, 0.2)'
                            }} />
                          )}
                          
                          {/* Step icon */}
                          <div style={{
                            width: '32px', height: '32px', borderRadius: '50%',
                            display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1,
                            backgroundColor: step.status === 'success' ? 'rgba(16, 185, 129, 0.1)' :
                                             step.status === 'failed' ? 'rgba(239, 68, 68, 0.1)' :
                                             step.status === 'escalated' ? 'rgba(245, 158, 11, 0.1)' :
                                             step.status === 'miss' ? 'rgba(148, 163, 184, 0.1)' : 'rgba(2, 132, 199, 0.1)',
                            color: step.status === 'success' ? '#10b981' :
                                   step.status === 'failed' ? '#ef4444' :
                                   step.status === 'escalated' ? '#f59e0b' : '#0284c7',
                            border: '1px solid ' + (
                              step.status === 'success' ? 'rgba(16, 185, 129, 0.3)' :
                              step.status === 'failed' ? 'rgba(239, 68, 68, 0.3)' :
                              step.status === 'escalated' ? 'rgba(245, 158, 11, 0.3)' : 'rgba(2, 132, 199, 0.3)'
                            )
                          }}>
                            {step.id === 'cache' && <Database size={14} />}
                            {step.id === 'gate' && <ShieldAlert size={14} />}
                            {step.id === 'local' && <Cpu size={14} />}
                            {step.id === 'trust' && <Activity size={14} />}
                            {step.id === 'remote' && <ArrowRight size={14} />}
                            {step.id === 'error' && <AlertTriangle size={14} />}
                          </div>

                          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', flexGrow: 1, justifyContent: 'center' }}>
                            <span style={{ fontSize: '13px', fontWeight: '600', color: '#0f172a' }}>
                              {step.id.toUpperCase() === 'CACHE' ? 'Cache Checker' :
                               step.id.toUpperCase() === 'GATE' ? 'Predictive Gate' :
                               step.id.toUpperCase() === 'LOCAL' ? 'Local LLM (0.5B)' :
                               step.id.toUpperCase() === 'TRUST' ? 'Signals Check' : 'Remote Expert'}
                            </span>
                            <span style={{ fontSize: '12px', color: '#475569', lineHeight: '1.4' }}>{step.label}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Analytics & Charts Tab */}
        {activeTab === 'analytics' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
            <div>
              <h1 style={{ margin: '0 0 8px 0', fontSize: '28px', fontWeight: '800', color: '#0f172a' }}>Engine Analytics & Metrics</h1>
              <p style={{ color: '#475569', margin: 0 }}>View cost, latency, and query distributions saved by our hybrid routing framework.</p>
            </div>

            {/* KPI Cards row */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '20px' }}>
              <div className="glass-panel" style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                <div style={{ width: '48px', height: '48px', borderRadius: '12px', backgroundColor: 'rgba(16, 185, 129, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#059669' }}>
                  <DollarSign size={24} />
                </div>
                <div>
                  <span style={{ display: 'block', fontSize: '12px', color: '#475569' }}>Total Dollars Saved</span>
                  <span style={{ fontSize: '24px', fontWeight: '800', color: '#059669', textShadow: '0 0 10px rgba(16,185,129,0.1)' }}>
                    ${stats.savings_dollars.toFixed(3)}
                  </span>
                </div>
              </div>

              <div className="glass-panel" style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                <div style={{ width: '48px', height: '48px', borderRadius: '12px', backgroundColor: 'rgba(2, 132, 199, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#0284c7' }}>
                  <Activity size={24} />
                </div>
                <div>
                  <span style={{ display: 'block', fontSize: '12px', color: '#475569' }}>Total Queries Ran</span>
                  <span style={{ fontSize: '24px', fontWeight: '800', color: '#0f172a' }}>{stats.total_queries}</span>
                </div>
              </div>

              <div className="glass-panel" style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                <div style={{ width: '48px', height: '48px', borderRadius: '12px', backgroundColor: 'rgba(219, 39, 119, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#db2777' }}>
                  <Database size={24} />
                </div>
                <div>
                  <span style={{ display: 'block', fontSize: '12px', color: '#475569' }}>Cache Hits</span>
                  <span style={{ fontSize: '24px', fontWeight: '800', color: '#0f172a' }}>
                    {stats.cache_hits} ({stats.total_queries > 0 ? Math.round((stats.cache_hits/stats.total_queries)*100) : 0}%)
                  </span>
                </div>
              </div>

              <div className="glass-panel" style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                <div style={{ width: '48px', height: '48px', borderRadius: '12px', backgroundColor: 'rgba(245, 158, 11, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#d97706' }}>
                  <TrendingUp size={24} />
                </div>
                <div>
                  <span style={{ display: 'block', fontSize: '12px', color: '#475569' }}>Escalation Rate</span>
                  <span style={{ fontSize: '24px', fontWeight: '800', color: '#0f172a' }}>
                    {stats.total_queries > 0 ? Math.round((stats.escalations/stats.total_queries)*100) : 0}%
                  </span>
                </div>
              </div>
            </div>

            {/* Graphs Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
              {/* Cost comparison chart */}
              <div className="glass-panel" style={{ height: '350px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <h3 style={{ margin: 0, fontSize: '16px', fontWeight: '700', color: '#0f172a' }}>Cost Savings ($): Router vs Remote Baseline</h3>
                <div style={{ flexGrow: 1, width: '100%', height: '80%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={[
                        { name: 'Remote-Only Baseline', cost: stats.total_queries * 180 * 0.000015 },
                        { name: 'Our Hybrid Router', cost: stats.total_remote_tokens * 0.000015 }
                      ]}
                      margin={{ top: 20, right: 30, left: 10, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.05)" />
                      <XAxis dataKey="name" stroke="#475569" />
                      <YAxis stroke="#475569" unit="$" />
                      <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '1px solid rgba(226,232,240,0.8)', color: '#0f172a' }} />
                      <Bar dataKey="cost" name="Total Spend" fill="#0284c7" radius={[8, 8, 0, 0]}>
                        <Cell fill="#e11d48" />
                        <Cell fill="#10b981" />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Source Distribution donut */}
              <div className="glass-panel" style={{ height: '350px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <h3 style={{ margin: 0, fontSize: '16px', fontWeight: '700', color: '#0f172a' }}>Routing Decisions Distribution</h3>
                <div style={{ flexGrow: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {sourcePieData.length === 0 ? (
                    <span style={{ color: '#64748b' }}>No data logs found.</span>
                  ) : (
                    <div style={{ display: 'flex', gap: '24px', alignItems: 'center' }}>
                      <div style={{ width: '200px', height: '200px' }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie
                              data={sourcePieData}
                              cx="50%"
                              cy="50%"
                              innerRadius={60}
                              outerRadius={80}
                              paddingAngle={5}
                              dataKey="value"
                            >
                              {sourcePieData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                              ))}
                            </Pie>
                            <Tooltip />
                          </PieChart>
                        </ResponsiveContainer>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        {sourcePieData.map((entry, idx) => (
                          <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px' }}>
                            <div style={{ width: '12px', height: '12px', borderRadius: '3px', backgroundColor: COLORS[idx % COLORS.length] }} />
                            <span style={{ color: '#475569' }}>{entry.name}:</span>
                            <span style={{ fontWeight: '700', color: '#0f172a' }}>{entry.value}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Threshold Tuning Tab */}
        {activeTab === 'calibration' && (() => {
          const ct = config.consistency_threshold;
          const et = config.entropy_threshold;
          const escRate = Math.min(0.96, Math.max(0.04, (ct * 0.48) + ((2.6 - et) * 0.16)));
          const estAccuracy = 55.4 + escRate * (98.2 - 55.4);
          const estCost = escRate * 180 * 0.000015;
          const estSavings = (1 - escRate) * 100;

          return (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
              <div>
                <h1 style={{ margin: '0 0 8px 0', fontSize: '28px', fontWeight: '800', color: '#0f172a' }}>Threshold Tuning & Calibration</h1>
                <p style={{ color: '#475569', margin: 0 }}>Configure and balance routing parameters based on Pareto-frontier cost vs accuracy trade-offs.</p>
              </div>

              {/* Top Row: Config Form & Live Sim Cards */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
                {/* Sliders Form */}
                <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                  <h3 style={{ margin: 0, fontSize: '18px', fontWeight: '700', borderBottom: '1px solid rgba(226,232,240,0.8)', paddingBottom: '12px', color: '#0f172a' }}>
                    Threshold Configurations
                  </h3>
                  
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontWeight: '600', fontSize: '14px', color: '#0f172a' }}>Self-Consistency Threshold</span>
                      <span style={{ fontSize: '14px', fontWeight: '700', color: '#0284c7' }}>{config.consistency_threshold.toFixed(2)}</span>
                    </div>
                    <input 
                      type="range" min="0" max="1" step="0.05"
                      value={config.consistency_threshold}
                      onChange={(e) => setConfig(prev => ({ ...prev, consistency_threshold: parseFloat(e.target.value) }))}
                      style={{ width: '100%', accentColor: '#0284c7' }}
                    />
                    <span style={{ fontSize: '11px', color: '#64748b' }}>Minimum cosine-similarity score required between local temperature runs to trust local output.</span>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontWeight: '600', fontSize: '14px', color: '#0f172a' }}>Token Entropy Limit</span>
                      <span style={{ fontSize: '14px', fontWeight: '700', color: '#0284c7' }}>{config.entropy_threshold.toFixed(2)}</span>
                    </div>
                    <input 
                      type="range" min="0.1" max="3" step="0.1"
                      value={config.entropy_threshold}
                      onChange={(e) => setConfig(prev => ({ ...prev, entropy_threshold: parseFloat(e.target.value) }))}
                      style={{ width: '100%', accentColor: '#0284c7' }}
                    />
                    <span style={{ fontSize: '11px', color: '#64748b' }}>Maximum average transition token entropy allowed before escalating. Lower means more strict.</span>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '10px' }}>
                    <button 
                      onClick={handleSaveConfig}
                      disabled={saveLoading}
                      className="btn-primary"
                    >
                      {saveLoading ? <RefreshCw size={16} className="animate-spin" /> : <Settings size={16} />}
                      <span>{saveLoading ? 'Calibrating...' : 'Apply Calibrated Settings'}</span>
                    </button>
                  </div>
                </div>

                {/* Live Sim Card Panel */}
                <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                  <h3 style={{ margin: 0, fontSize: '18px', fontWeight: '700', borderBottom: '1px solid rgba(226,232,240,0.8)', paddingBottom: '12px', color: '#0f172a' }}>
                    Simulated System Estimates
                  </h3>
                  
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', flexGrow: 1, justifyContent: 'center' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(2, 132, 199, 0.04)', border: '1px solid rgba(2, 132, 199, 0.1)' }}>
                      <span style={{ fontSize: '14px', color: '#475569', fontWeight: '500' }}>Expected System Accuracy</span>
                      <span style={{ fontSize: '18px', fontWeight: '800', color: '#0284c7' }}>{estAccuracy.toFixed(1)}%</span>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(13, 148, 136, 0.04)', border: '1px solid rgba(13, 148, 136, 0.1)' }}>
                      <span style={{ fontSize: '14px', color: '#475569', fontWeight: '500' }}>Token Expense Reduction</span>
                      <span style={{ fontSize: '18px', fontWeight: '800', color: '#0d9488' }}>{estSavings.toFixed(1)}%</span>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px', borderRadius: '8px', backgroundColor: 'rgba(245, 158, 11, 0.04)', border: '1px solid rgba(245, 158, 11, 0.1)' }}>
                      <span style={{ fontSize: '14px', color: '#475569', fontWeight: '500' }}>Estimated Cost per Query</span>
                      <span style={{ fontSize: '18px', fontWeight: '800', color: '#d97706' }}>${estCost.toFixed(5)}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Bottom Row: Two Charts */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
                {/* Pareto Frontier */}
                <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '16px', height: '350px' }}>
                  <h3 style={{ margin: 0, fontSize: '16px', fontWeight: '700', color: '#0f172a' }}>Calibration Pareto Frontier (Cost vs Accuracy)</h3>
                  <div style={{ flexGrow: 1, width: '100%', height: '80%' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart
                        data={paretoData}
                        margin={{ top: 20, right: 30, left: 45, bottom: 25 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.05)" />
                        <XAxis dataKey="cost" stroke="#475569" tickFormatter={(v) => `$${v.toFixed(3)}`} />
                        <YAxis domain={[50, 100]} stroke="#475569" tickFormatter={(v) => `${v}%`} />
                        <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '1px solid rgba(226,232,240,0.8)', color: '#0f172a' }} />
                        <Area type="monotone" dataKey="accuracy" name="Router Accuracy (%)" stroke="#0284c7" fill="rgba(2, 132, 199, 0.1)" strokeWidth={2} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Threshold Sensitivity */}
                <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '16px', height: '350px' }}>
                  <h3 style={{ margin: 0, fontSize: '16px', fontWeight: '700', color: '#0f172a' }}>Threshold Sensitivity Trade-offs</h3>
                  <div style={{ flexGrow: 1, width: '100%', height: '80%' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={[
                          { threshold: 0.0, accuracy: 55.4, cost: 0.0000 },
                          { threshold: 0.1, accuracy: 59.8, cost: 0.0002 },
                          { threshold: 0.2, accuracy: 64.2, cost: 0.0005 },
                          { threshold: 0.3, accuracy: 70.5, cost: 0.0008 },
                          { threshold: 0.4, accuracy: 78.5, cost: 0.0013 },
                          { threshold: 0.5, accuracy: 85.1, cost: 0.0018 },
                          { threshold: 0.6, accuracy: 89.2, cost: 0.0022 },
                          { threshold: 0.7, accuracy: 92.4, cost: 0.0025 },
                          { threshold: 0.8, accuracy: 95.0, cost: 0.0027 },
                          { threshold: 0.9, accuracy: 96.8, cost: 0.0028 },
                          { threshold: 1.0, accuracy: 98.2, cost: 0.0029 }
                        ]}
                        margin={{ top: 20, right: 55, left: 45, bottom: 25 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.05)" />
                        <XAxis dataKey="threshold" stroke="#475569" />
                        <YAxis yAxisId="left" stroke="#0284c7" domain={[50, 100]} tickFormatter={(v) => `${v}%`} />
                        <YAxis yAxisId="right" orientation="right" stroke="#0d9488" tickFormatter={(v) => `$${v.toFixed(4)}`} />
                        <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '1px solid rgba(226,232,240,0.8)', color: '#0f172a' }} />
                        <Legend />
                        <Line yAxisId="left" type="monotone" dataKey="accuracy" name="Accuracy (%)" stroke="#0284c7" strokeWidth={2.5} activeDot={{ r: 8 }} />
                        <Line yAxisId="right" type="monotone" dataKey="cost" name="Query Cost ($)" stroke="#0d9488" strokeWidth={2.5} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>
          );
        })()}

        {/* Telemetry Logs history Tab */}
        {activeTab === 'logs' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
            <div>
              <h1 style={{ margin: '0 0 8px 0', fontSize: '28px', fontWeight: '800', color: '#0f172a' }}>Telemetry Execution Logs</h1>
              <p style={{ color: '#475569', margin: 0 }}>Review historical execution traces and diagnostics collected in `routing_execution.jsonl`.</p>
            </div>

            <div className="glass-panel" style={{ padding: '0px', overflow: 'hidden' }}>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                  <thead>
                    <tr style={{ backgroundColor: 'rgba(0,0,0,0.02)', borderBottom: '1px solid rgba(226,232,240,0.8)' }}>
                      <th style={{ padding: '16px', fontSize: '13px', fontWeight: '700', color: '#475569' }}>Timestamp</th>
                      <th style={{ padding: '16px', fontSize: '13px', fontWeight: '700', color: '#475569' }}>Query Prompt</th>
                      <th style={{ padding: '16px', fontSize: '13px', fontWeight: '700', color: '#475569' }}>Strategy</th>
                      <th style={{ padding: '16px', fontSize: '13px', fontWeight: '700', color: '#475569' }}>Category</th>
                      <th style={{ padding: '16px', fontSize: '13px', fontWeight: '700', color: '#475569' }}>Final Source</th>
                      <th style={{ padding: '16px', fontSize: '13px', fontWeight: '700', color: '#475569' }}>Tokens (L/R)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.length === 0 ? (
                      <tr>
                        <td colSpan={6} style={{ padding: '30px', textAlign: 'center', color: '#64748b' }}>
                          No execution logs found. Run queries from the Sandbox tab first!
                        </td>
                      </tr>
                    ) : (
                      history.map((log, idx) => (
                        <tr key={idx} style={{ 
                          borderBottom: '1px solid rgba(226,232,240,0.4)',
                          backgroundColor: idx % 2 === 0 ? 'transparent' : 'rgba(0,0,0,0.01)',
                          transition: 'background-color 0.2s'
                        }}
                        >
                          <td style={{ padding: '14px 16px', fontSize: '12px', color: '#64748b' }}>
                            {log.timestamp ? log.timestamp.split('T')[1].substring(0, 8) : 'Unknown'}
                          </td>
                          <td style={{ 
                            padding: '14px 16px', fontSize: '13px', color: '#0f172a', 
                            maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'
                          }}>
                            {log.prompt}
                          </td>
                          <td style={{ padding: '14px 16px', fontSize: '13px', color: '#475569' }}>
                            {log.routing_strategy}
                          </td>
                          <td style={{ padding: '14px 16px', fontSize: '12px' }}>
                            <span style={{
                              padding: '2px 6px', borderRadius: '4px', backgroundColor: 'rgba(0,0,0,0.05)',
                              color: '#475569', fontSize: '11px', fontWeight: '600'
                            }}>
                              {log.category}
                            </span>
                          </td>
                          <td style={{ padding: '14px 16px', fontSize: '13px', fontWeight: '600' }}>
                            <span style={{
                              color: log.source.toLowerCase().includes('cache') ? '#10b981' :
                                     log.source.toLowerCase().includes('local') ? '#0284c7' : '#d97706'
                            }}>
                              {log.source}
                            </span>
                          </td>
                          <td style={{ padding: '14px 16px', fontSize: '13px', color: '#475569' }}>
                            {log.local_tokens_used || 0} / {log.remote_tokens_used || 0}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
