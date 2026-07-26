"""
DD Copilot v4.0 — Enterprise Grade AI Due Diligence.
Async pipeline, RAG, strict limits, advanced observability.
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime

import streamlit as st
from openai import AsyncOpenAI

from analyzer import get_agent_display_name, run_all_agents
from cache import clear_cache, get_cache_entry_count, get_cache_size, init_cache
from extractor import extract_company, run_sanity_checks
from models import AgentResult, Company, ICMemo, SanityCheck, SlideAnalysis
from observability import get_session_stats, init_observability
from parser import parse_pdf, render_all_pages
from report import SCORE_WEIGHTS, format_memo_as_html, format_memo_as_markdown, generate_ic_memo
from utils import get_async_openai_client, hash_content
from vision import analyze_all_slides, embed_slides

logger = logging.getLogger("dd_copilot.app")
st.set_page_config(page_title="DD Copilot Enterprise", page_icon="📊", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0e1117; color: #e6edf3; }
section[data-testid="stSidebar"] { background-color: #0a0d12; border-right: 1px solid #21262d; }
h1, h2, h3 { color: #e6edf3 !important; }
.metric-card { background: #161b22; border-radius: 12px; padding: 20px; border: 1px solid #30363d; text-align: center; }
.metric-card-title { font-size: 12px; color: #8b949e; text-transform: uppercase; }
.metric-card-value { font-size: 28px; font-weight: 700; color: #00d4aa; }
.stButton > button { background: #00d4aa; color: #0e1117; border: none; border-radius: 8px; font-weight: 600; }
.stButton > button:hover { background: #00f0bf; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def init_state():
    defaults = {"api_key": "", "analysis_complete": False, "company": None, "ic_memo": None, "agent_results": None, "file_hash": None}
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_state()
init_cache()
init_observability()

def get_api_key():
    if st.session_state.api_key:
        return st.session_state.api_key
    try:
        if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
            return st.secrets["openai"]["api_key"]
    except Exception as e:
        logger.warning("Could not read API key from secrets: %s", e)
    return ""

def render_sidebar():
    with st.sidebar:
        st.markdown("## 📊 DD Copilot Enterprise")
        st.caption("v4.0 Async / RAG Architecture")
        st.divider()
        st.markdown("### 🔑 Configuration")
        st.text_input("OpenAI API Key", type="password", key="api_key", placeholder="sk-proj-...")
        st.divider()
        st.markdown("### 📈 Live Observability")
        stats = get_session_stats()
        st.metric("Est. Cost (USD)", f"${stats['total_cost']:.4f}")
        st.metric("API Calls", stats["total_calls"])
        st.metric("Cache Hit Ratio", f"{stats['cache_hit_ratio']:.0%}")
        st.metric("P95 Latency", f"{stats['p95_latency']:.2f}s")
        st.caption(f"Cache Size: {get_cache_size()/(1024*1024):.1f} MB")
        if st.button("🗑️ Clear Cache"): clear_cache(); st.rerun()

def render_header():
    st.markdown('<h1 style="color:#e6edf3">DD Copilot Enterprise</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e">Upload a pitch deck. Async AI agents will analyze it in seconds.</p>', unsafe_allow_html=True)

def render_upload():
    f = st.file_uploader("Upload Pitch Deck (PDF, Max 50MB / 100 pages)", type=["pdf"])
    if f:
        size_mb = len(f.getvalue()) / (1024*1024)
        if size_mb > 50:
            st.error("File exceeds 50MB limit.")
            return
            
        file_hash = hash_content(f.getvalue())
        if st.session_state.file_hash != file_hash:
            st.session_state.file_hash = file_hash
            st.session_state.analysis_complete = False
            
        st.success(f"✅ {f.name} ({size_mb:.1f} MB)")
        if st.button("🚀 Run Due Diligence", type="primary"):
            asyncio.run(run_pipeline(f.getvalue(), file_hash, get_api_key()))

async def run_pipeline(file_bytes: bytes, file_hash: str, api_key: str):
    client = get_async_openai_client(api_key)
    
    try:
        with st.status("📄 Parsing PDF & Detecting Limits...", expanded=True) as status:
            pdf_doc = parse_pdf(file_bytes, max_pages=100)
            junk = sum(1 for p in pdf_doc.pages if p.is_junk)
            st.write(f"✓ {pdf_doc.page_count} pages. Skipping {junk} junk slides.")
            status.update(label="PDF parsed", state="complete")

        with st.status("🖼️ Rendering slides...", expanded=True) as status:
            images = render_all_pages(file_bytes)
            status.update(label="Rendered", state="complete")

        with st.status("🔍 Async Slide Analysis (Rate Limited)...", expanded=True) as status:
            container = st.container()
            def slide_prog(curr, tot, res):
                if res.is_skipped: container.write(f"⏭️ Slide {curr}/{tot} skipped")
                else: container.write(f"✓ Slide {curr}/{tot} analyzed")
            
            analyses = await analyze_all_slides(client, images, file_hash, pdf_doc.pages, pdf_doc.is_scanned, slide_prog)
            st.session_state.slide_analyses = analyses
            
            # Generate embeddings for RAG
            await embed_slides(client, analyses)
            status.update(label="Slides analyzed & embedded", state="complete")

        with st.status("🏗️ Extracting Company Data...", expanded=True) as status:
            company = await extract_company(analyses, file_hash, client)
            st.session_state.company = company
            status.update(label="Company extracted", state="complete")

        with st.status("🤖 Running 7 Agents Concurrently (RAG)...", expanded=True) as status:
            container = st.container()
            def agent_prog(name, res):
                icon = "⚠️" if res.failed else "✓"
                container.write(f"{icon} {get_agent_display_name(name)} — Score: {res.score:.0f}")
            
            sanity = run_sanity_checks(company)
            agents = await run_all_agents(company, analyses, file_hash, client, agent_prog)
            st.session_state.agent_results = agents
            st.session_state.sanity_checks = sanity
            status.update(label="Agents complete", state="complete")

        with st.status("📝 Generating IC Memo...", expanded=True) as status:
            memo = await generate_ic_memo(company, agents, sanity, analyses, file_hash, client)
            st.session_state.ic_memo = memo
            status.update(label="Memo ready", state="complete")

        st.session_state.analysis_complete = True
        st.balloons()
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Pipeline Error: {e}")
        logger.exception("Pipeline failed")

def render_results():
    company: Company = st.session_state.company
    memo: ICMemo = st.session_state.ic_memo
    agents: dict[str, AgentResult] = st.session_state.agent_results

    st.markdown(f"## 📊 {company.name}")
    st.markdown(
        f"**Recommendation:** {memo.recommendation}  |  **Overall Score:** {memo.overall_score:.1f}/100  |  "
        f"**Confidence:** {memo.confidence:.0%}  |  **Prob. of Failure:** {memo.probability_of_failure:.0%}"
    )

    cols = st.columns(len(SCORE_WEIGHTS))
    for i, (dim, w) in enumerate(SCORE_WEIGHTS.items()):
        score = memo.scores.get(dim, 0.0)
        cols[i].metric(dim.title(), f"{score:.0f}")

    st.markdown("---")
    st.markdown("### Executive Summary")
    st.write(memo.executive_summary or "_Not provided in the deck._")
    st.markdown("### Investment Thesis")
    st.write(memo.investment_thesis or "_Not provided in the deck._")
    st.markdown("### Why Now?")
    st.write(memo.why_now or "_Not provided in the deck._")

    tabs = st.tabs([
        "Business & Product", "Market", "Competition & Moat", "Traction & Economics",
        "Technology", "Founders", "Risks & Diligence"
    ])
    with tabs[0]:
        st.write(memo.business_model or "_Not provided in the deck._")
        if memo.product:
            st.markdown("**Product**")
            st.write(memo.product)
    with tabs[1]:
        st.write(memo.market or "_Not provided in the deck._")
    with tabs[2]:
        st.write(memo.competition or "_Not provided in the deck._")
        if company.competitors:
            st.markdown("**Competitor Detail**")
            st.table([{"Name": c.name, "Type": c.competitor_type, "Threat": c.threat_level} for c in company.competitors])
        if memo.moat:
            st.markdown("**Moat Assessment**")
            st.write(memo.moat)
    with tabs[3]:
        st.markdown("**Traction**")
        st.write(memo.traction or "_Not provided in the deck._")
        st.markdown("**Unit Economics**")
        st.write(memo.unit_economics or "_Not provided in the deck._")
        st.markdown("**Financial Health**")
        st.write(memo.financial_health or "_Not provided in the deck._")
    with tabs[4]:
        st.write(memo.technology or "_Not provided in the deck._")
    with tabs[5]:
        st.write(memo.founder_assessment or "_Not provided in the deck._")
        if company.founders:
            st.markdown("**Founders on File**")
            for f in company.founders:
                st.markdown(f"- **{f.name or 'Unnamed'}** ({f.role or 'Role not stated'})"
                             + (f" — {f.education}" if f.education else "")
                             + (f" — prev: {', '.join(f.previous_companies)}" if f.previous_companies else ""))
    with tabs[6]:
        st.markdown("**Risks**")
        st.write(memo.risks or "_Not provided in the deck._")
        if memo.open_questions:
            st.markdown("**Open Questions**")
            for q in memo.open_questions: st.write(f"- {q}")
        if memo.required_follow_up:
            st.markdown("**Required Diligence**")
            for item in memo.required_follow_up: st.write(f"- {item}")
        st.markdown("**Deterministic Sanity Checks**")
        for c in st.session_state.sanity_checks:
            icon = "✅" if c.passed else ("🛑" if c.severity == "critical" else "⚠️")
            st.write(f"{icon} **{c.name}**: {c.message}")

    st.markdown("---")
    st.markdown("### Agent Analyses")
    for name, res in agents.items():
        with st.expander(f"🔍 {get_agent_display_name(name)} — Score: {res.score:.0f}"):
            if res.failed: st.warning("Agent failed.")
            else: st.write(res.analysis)

    st.markdown("---")
    st.markdown("### Download")
    md_report = format_memo_as_markdown(memo, company, agents, st.session_state.sanity_checks)
    st.download_button("📄 Download MD", md_report, "ic_memo.md")

def main():
    render_sidebar()
    render_header()
    if st.session_state.analysis_complete:
        render_results()
    else:
        render_upload()

if __name__ == "__main__":
    main()
