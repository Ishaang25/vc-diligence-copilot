"""
Advanced observability and metrics tracking.
Tracks API latency, token usage, P50/P95 percentiles, and estimated costs.
"""
import time
import streamlit as st
from typing import Any
import numpy as np

PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
}

def init_observability():
    if "api_calls" not in st.session_state:
        st.session_state.api_calls = []
    if "cache_hits" not in st.session_state:
        st.session_state.cache_hits = 0

def log_api_call(model: str, prompt_tokens: int, completion_tokens: int, latency: float, agent_name: str = "Unknown"):
    cost = 0.0
    if model in PRICING:
        in_cost = (prompt_tokens / 1_000_000) * PRICING[model]["input"]
        out_cost = (completion_tokens / 1_000_000) * PRICING[model]["output"]
        cost = in_cost + out_cost
    
    st.session_state.api_calls.append({
        "model": model,
        "agent": agent_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency": latency,
        "cost": cost,
        "timestamp": time.time()
    })

def log_cache_hit():
    st.session_state.cache_hits += 1

def get_session_stats() -> dict[str, Any]:
    calls = st.session_state.get("api_calls", [])
    cache_hits = st.session_state.get("cache_hits", 0)
    
    if not calls:
        return {
            "total_calls": 0, "total_tokens": 0, "total_cost": 0.0,
            "avg_latency": 0.0, "p50_latency": 0.0, "p95_latency": 0.0,
            "cache_hits": cache_hits, "cache_hit_ratio": 0.0
        }
    
    latencies = [c["latency"] for c in calls]
    total_calls = len(calls)
    total_api_requests = total_calls + cache_hits
    
    return {
        "total_calls": total_calls,
        "total_tokens": sum(c["prompt_tokens"] + c["completion_tokens"] for c in calls),
        "total_cost": sum(c["cost"] for c in calls),
        "avg_latency": sum(latencies) / total_calls,
        "p50_latency": float(np.percentile(latencies, 50)),
        "p95_latency": float(np.percentile(latencies, 95)),
        "cache_hits": cache_hits,
        "cache_hit_ratio": (cache_hits / total_api_requests) if total_api_requests > 0 else 0.0
    }
