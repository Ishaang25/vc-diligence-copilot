"""
Async analysis pipeline with real RAG retrieval (slide-type routing + embedding similarity).
"""
from __future__ import annotations
import asyncio
import json
import logging
from openai import AsyncOpenAI

from cache import cache_key, disk_cache_get, disk_cache_set
from models import AgentResult, Company, SlideAnalysis, SlideType
from prompts import *
from utils import call_openai_json_async, cosine_similarity, get_embeddings

logger = logging.getLogger("dd_copilot.analyzer")

AGENTS = {
    "market": (MARKET_AGENT_PROMPT, "Market Analysis", "market size TAM SAM SOM opportunity growth rate"),
    "competition": (COMPETITION_AGENT_PROMPT, "Competition & Moat", "competitors positioning moat differentiation advantages"),
    "economics": (ECONOMICS_AGENT_PROMPT, "Unit Economics", "revenue MRR ARR CAC LTV churn burn rate runway financials"),
    "founder": (FOUNDER_AGENT_PROMPT, "Founder Diligence", "founders CEO CTO team education previous companies leadership experience"),
    "technology": (TECHNOLOGY_AGENT_PROMPT, "Technology", "technology stack architecture AI ML product infrastructure defensibility moat"),
    "regulation": (REGULATION_AGENT_PROMPT, "Regulatory Risk", "regulation compliance GDPR HIPAA legal data privacy"),
    "devils_advocate": (DEVILS_ADVOCATE_PROMPT, "Devil's Advocate", "risks failure challenges threats weaknesses"),
}

# Maps each agent to the slide types it should ALWAYS see, regardless of embedding similarity.
# This is the deterministic half of retrieval: e.g. the founder agent must never miss a slide
# that was actually classified as TEAM, even if its summary text happens to score low on
# similarity to the query string. Slide-type classification only works now that fix #1
# (classify_slide_type) no longer silently defaults every slide to OTHER.
AGENT_SLIDE_TYPES: dict[str, list[SlideType]] = {
    "market": [SlideType.MARKET, SlideType.PROBLEM],
    "competition": [SlideType.COMPETITION, SlideType.PRODUCT, SlideType.SOLUTION],
    "economics": [SlideType.TRACTION, SlideType.FINANCIALS],
    "founder": [SlideType.TEAM, SlideType.COVER],
    "technology": [SlideType.PRODUCT, SlideType.SOLUTION],
    "regulation": [SlideType.MARKET, SlideType.PRODUCT, SlideType.TRACTION, SlideType.FINANCIALS],
    "devils_advocate": [],  # empty = sees the whole deck; it needs the full picture to attack it
}

# Minimum cosine similarity for a non-type-matched slide to be pulled in as additional context.
MIN_SIMILARITY = 0.15


async def get_relevant_slides(
    client: AsyncOpenAI,
    agent_name: str,
    query: str,
    slides: list[SlideAnalysis],
    top_k: int = 15,
) -> list[SlideAnalysis]:
    """Hybrid RAG retrieval for a single agent.

    1. Always include slides whose classified slide_type is in this agent's domain
       (e.g. founder agent always gets TEAM slides) — this is the reliable, deterministic path.
    2. Rank the remaining slides by embedding cosine similarity to the agent's focus query and
       backfill up to top_k, since relevant facts often appear outside their "expected" slide
       (a churn number quoted on a traction slide's footnote, a competitor named on a product slide).
    3. Devil's Advocate (and any agent with no configured type filter) gets the entire valid deck,
       since cross-cutting risk-hunting needs full context, not a narrow slice.
    4. If embeddings are missing (e.g. embed_slides failed) or nothing scores above the similarity
       floor, we still return the type-matched slides rather than an empty context.
    """
    valid_slides = [s for s in slides if not s.is_skipped]
    if not valid_slides:
        return []

    target_types = AGENT_SLIDE_TYPES.get(agent_name, [])
    if not target_types:
        return valid_slides

    type_matched = [s for s in valid_slides if s.slide_type in target_types]
    remaining = [s for s in valid_slides if s.slide_type not in target_types]

    if len(type_matched) >= top_k or not remaining:
        return type_matched or valid_slides[:top_k]

    scored: list[tuple[float, SlideAnalysis]] = []
    embeddable = [s for s in remaining if s.embedding]
    if embeddable:
        try:
            query_embeddings = await get_embeddings(client, [query])
            query_embedding = query_embeddings[0]
            for s in embeddable:
                sim = cosine_similarity(query_embedding, s.embedding)
                if sim >= MIN_SIMILARITY:
                    scored.append((sim, s))
            scored.sort(key=lambda pair: pair[0], reverse=True)
        except Exception as exc:
            logger.warning("Embedding-based retrieval failed for agent '%s': %s", agent_name, exc)

    slots_left = max(0, top_k - len(type_matched))
    similarity_matched = [s for _, s in scored[:slots_left]]

    result = type_matched + similarity_matched
    if not result:
        # Nothing matched type or similarity — better to give the agent something than nothing.
        result = valid_slides[:top_k]
    return result


def _compact_slide(s: SlideAnalysis) -> dict:
    return {
        "slide": s.slide_number,
        "type": s.slide_type.value,
        "summary": s.summary,
        "numbers": s.numbers,
        "claims": s.claims,
        "risks": s.risks,
        "evidence": [
            {"claim": e.claim, "quote": e.quote, "confidence": e.confidence}
            for e in s.evidence
        ],
    }


async def _prepare_context(
    client: AsyncOpenAI, agent_name: str, company: Company, slide_analyses: list[SlideAnalysis], agent_query: str
) -> tuple[str, list[int]]:
    relevant_slides = await get_relevant_slides(client, agent_name, agent_query, slide_analyses)
    compact_slides = [_compact_slide(s) for s in relevant_slides]
    retrieved_slide_numbers = [s.slide_number for s in relevant_slides]
    context = (
        f"COMPANY INFO:\n{company.model_dump_json(exclude_none=True)}\n\n"
        f"RETRIEVED SLIDES (only cite slide numbers that appear below — {len(relevant_slides)} of "
        f"{len([s for s in slide_analyses if not s.is_skipped])} total slides were retrieved for this agent):\n"
        f"{json.dumps(compact_slides, indent=2)}"
    )
    return context, retrieved_slide_numbers


async def run_single_agent(
    agent_name: str,
    system_prompt: str,
    company: Company,
    slide_analyses: list[SlideAnalysis],
    file_hash: str,
    client: AsyncOpenAI,
    query: str,
) -> AgentResult:
    cache_k = cache_key(file_hash, "agent_v5", agent_name, hash(company.model_dump_json()))
    cached = disk_cache_get(cache_k)
    if cached:
        return AgentResult.model_validate(cached)

    context, retrieved_slides = await _prepare_context(client, agent_name, company, slide_analyses, query)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": context}]

    try:
        data = await call_openai_json_async(
            client, messages, model="gpt-4o", temperature=0.4, max_tokens=3000, agent_name=agent_name
        )
        data["agent_name"] = agent_name
        result = AgentResult.model_validate(data)
    except Exception as exc:
        logger.error("Agent '%s' failed: %s", agent_name, exc)
        result = AgentResult(agent_name=agent_name, analysis=f"Agent failed to complete: {exc}", risks=[str(exc)], failed=True)

    disk_cache_set(cache_k, result.model_dump())
    return result


async def run_all_agents(
    company: Company, slide_analyses: list[SlideAnalysis], file_hash: str, client: AsyncOpenAI, progress_callback=None
) -> dict[str, AgentResult]:
    semaphore = asyncio.Semaphore(3)  # Limit concurrent heavy agents

    async def process_agent(name, prompt, query):
        async with semaphore:
            res = await run_single_agent(name, prompt, company, slide_analyses, file_hash, client, query)
            if progress_callback:
                progress_callback(name, res)
            return res

    tasks = [process_agent(name, p, q) for name, (p, _, q) in AGENTS.items()]
    results = await asyncio.gather(*tasks)
    return {r.agent_name: r for r in results}


def get_agent_display_name(agent_name: str) -> str:
    return AGENTS.get(agent_name, (agent_name, agent_name))[1]
