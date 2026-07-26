"""
Report generation.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
import markdown as md
from openai import AsyncOpenAI

from cache import cache_key, disk_cache_get, disk_cache_set
from models import AgentResult, Company, ICMemo, SanityCheck, SlideAnalysis
from prompts import IC_MEMO_PROMPT
from utils import call_openai_json_async, format_currency, format_percentage

logger = logging.getLogger("dd_copilot.report")
SCORE_WEIGHTS = {"market": 0.20, "team": 0.20, "product": 0.20, "traction": 0.15, "economics": 0.15, "moat": 0.10}
AGENT_SCORE_MAP = {"market": "market", "competition": "moat", "economics": "economics", "founder": "team", "technology": "product"}

async def generate_ic_memo(company: Company, agent_results: dict[str, AgentResult], sanity_checks: list[SanityCheck], slide_analyses: list[SlideAnalysis], file_hash: str, client: AsyncOpenAI) -> ICMemo:
    cache_k = cache_key(file_hash, "ic_memo_v4", hash(company.model_dump_json()))
    cached = disk_cache_get(cache_k)
    if cached: return ICMemo.model_validate(cached)

    parts = ["=== COMPANY ===", company.model_dump_json(exclude_none=True), "=== AGENTS ==="]
    for name, res in agent_results.items():
        parts.append(f"--- {name} ---\nFailed: {res.failed}\nScore: {res.score}\n{res.analysis}")
    parts.append("=== SANITY CHECKS ===")
    for c in sanity_checks: parts.append(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.message}")
    
    messages = [{"role": "system", "content": IC_MEMO_PROMPT}, {"role": "user", "content": "\n".join(parts)}]
    data = await call_openai_json_async(client, messages, model="gpt-4o", temperature=0.4, max_tokens=4000, agent_name="IC Memo")
    
    memo = ICMemo.model_validate(data)
    memo.scores = _merge_scores(memo.scores, agent_results, company)
    memo.overall_score = _calculate_overall_score(memo.scores)
    disk_cache_set(cache_k, memo.model_dump())
    return memo

def _merge_scores(scores, agents, company):
    from extractor import calculate_traction_score
    s = {k: float(scores.get(k, 50.0)) for k in SCORE_WEIGHTS}
    for a, dim in AGENT_SCORE_MAP.items():
        if a in agents and not agents[a].failed: s[dim] = agents[a].score
    s["traction"] = calculate_traction_score(company.metrics)
    return s

def _calculate_overall_score(scores):
    return round(sum(scores.get(d, 50.0) * w for d, w in SCORE_WEIGHTS.items()), 1)

def format_memo_as_markdown(memo: ICMemo, company: Company, agent_results: dict[str, AgentResult], sanity_checks: list[SanityCheck]) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    md_text = f"# IC Memo: {company.name}\n**Date:** {date_str} | **Recommendation:** {memo.recommendation}\n\n"
    md_text += f"**Overall Score:** {memo.overall_score:.1f}/100 | **Prob of Failure:** {memo.probability_of_failure:.0%}\n\n"
    md_text += f"## Executive Summary\n{memo.executive_summary}\n\n"
    md_text += f"## Investment Decision\n{memo.investment_decision}\n\n"
    
    if company.benchmarks:
        md_text += "## Benchmarks\n| Metric | Company | Industry | Var |\n|---|---|---|---|\n"
        for b in company.benchmarks:
            md_text += f"| {b.metric} | {b.company_value} | {b.benchmark_value} | {b.variance_percentage}% ({b.status}) |\n"
            
    md_text += "\n## Detailed Agent Analyses\n"
    for name, res in agent_results.items():
        md_text += f"\n### {name.replace('_', ' ').title()}\n"
        if res.failed: md_text += "⚠️ *Agent failed.*\n"; continue
        md_text += f"**Score:** {res.score:.0f}/100\n\n{res.analysis}\n"
        
    return md_text

def format_memo_as_html(memo: ICMemo, company: Company, agent_results: dict[str, AgentResult], sanity_checks: list[SanityCheck]) -> str:
    md_text = format_memo_as_markdown(memo, company, agent_results, sanity_checks)
    html_body = md.markdown(md_text, extensions=["tables", "fenced_code"])
    return f"<!DOCTYPE html><html><head><style>body{{font-family:sans-serif;max-width:800px;margin:auto;padding:20px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px}}</style></head><body>{html_body}</body></html>"
