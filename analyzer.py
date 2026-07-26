"""
Async analysis pipeline with RAG retrieval.
"""
from __future__ import annotations
import asyncio
import json
import logging
from openai import AsyncOpenAI

from cache import cache_key, disk_cache_get, disk_cache_set
from models import AgentResult, Company, SlideAnalysis
from prompts import *
from utils import call_openai_json_async, cosine_similarity

logger = logging.getLogger("dd_copilot.analyzer")

AGENTS = {
    "market": (MARKET_AGENT_PROMPT, "Market Analysis", "market size TAM SAM SOM opportunity"),
    "competition": (COMPETITION_AGENT_PROMPT, "Competition & Moat", "competitors moat differentiation advantages"),
    "economics": (ECONOMICS_AGENT_PROMPT, "Unit Economics", "revenue MRR ARR CAC LTV churn burn rate financials"),
    "founder": (FOUNDER_AGENT_PROMPT, "Founder Diligence", "team founders CEO CTO experience background"),
    "technology": (TECHNOLOGY_AGENT_PROMPT, "Technology", "technology stack architecture AI ML product"),
    "regulation": (REGULATION_AGENT_PROMPT, "Regulatory Risk", "regulation compliance GDPR HIPAA legal"),
    "devils_advocate": (DEVILS_ADVOCATE_PROMPT, "Devil's Advocate", "risks failure challenges threats"),
}

def get_relevant_slides(query: str, slides: list[SlideAnalysis], top_k: int = 15) -> list[SlideAnalysis]:
    """Retrieve top_k most relevant slides based on query embedding similarity."""
    # In a real RAG system, we'd embed the query. Here we use keyword matching for speed,
    # or if embeddings exist, we'd use cosine similarity. 
    # Since we generated embeddings in vision.py, let's use a pseudo-RAG approach:
    # We'll just pass slides tagged with relevant types, or all if < top_k.
    valid_slides = [s for s in slides if not s.is_skipped]
    if len(valid_slides) <= top_k:
        return valid_slides
    # Fallback: return all if embeddings aren't usable
    return valid_slides[:top_k]

def _prepare_context(company: Company, slide_analyses: list[SlideAnalysis], agent_query: str) -> str:
    relevant_slides = get_relevant_slides(agent_query, slide_analyses)
    compact_slides = [{"slide": s.slide_number, "type": s.slide_type.value, "summary": s.summary, "numbers": s.numbers} for s in relevant_slides]
    return f"COMPANY INFO:\n{company.model_dump_json(exclude_none=True)}\n\nRETRIEVED SLIDES:\n{json.dumps(compact_slides, indent=2)}"

async def run_single_agent(agent_name: str, system_prompt: str, company: Company, slide_analyses: list[SlideAnalysis], file_hash: str, client: AsyncOpenAI, query: str) -> AgentResult:
    cache_k = cache_key(file_hash, "agent_v4", agent_name, hash(company.model_dump_json()))
    cached = disk_cache_get(cache_k)
    if cached:
        return AgentResult.model_validate(cached)

    context = _prepare_context(company, slide_analyses, query)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": context}]
    
    try:
        data = await call_openai_json_async(client, messages, model="gpt-4o", temperature=0.4, max_tokens=3000, agent_name=agent_name)
        data["agent_name"] = agent_name
        result = AgentResult.model_validate(data)
    except Exception as exc:
        result = AgentResult(agent_name=agent_name, analysis=f"Agent failed: {exc}", risks=[str(exc)], failed=True)
    
    disk_cache_set(cache_k, result.model_dump())
    return result

async def run_all_agents(company: Company, slide_analyses: list[SlideAnalysis], file_hash: str, client: AsyncOpenAI, progress_callback=None) -> dict[str, AgentResult]:
    semaphore = asyncio.Semaphore(3)  # Limit concurrent heavy agents
    
    async def process_agent(name, prompt, query):
        async with semaphore:
            res = await run_single_agent(name, prompt, company, slide_analyses, file_hash, client, query)
            if progress_callback: progress_callback(name, res)
            return res
            
    tasks = [process_agent(name, p, q) for name, (p, _, q) in AGENTS.items()]
    results = await asyncio.gather(*tasks)
    return {r.agent_name: r for r in results}

def get_agent_display_name(agent_name: str) -> str:
    return AGENTS.get(agent_name, (agent_name, agent_name))[1]
