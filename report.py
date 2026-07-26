"""
Report generation. Renders the FULL IC memo (previously only executive_summary and
investment_decision were rendered; every other section the LLM generated was discarded).
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

RECOMMENDATION_VALUES = ("STRONG YES", "YES", "HOLD", "PASS")


def normalize_recommendation(value: str) -> str:
    """Maps model output onto the canonical 4-tier recommendation scale, defensively —
    the prompt asks for an exact match, but this guards against drift/synonyms so the
    UI never shows a nonsense or out-of-schema recommendation string."""
    v = (value or "").strip().upper()
    if v in RECOMMENDATION_VALUES:
        return v
    if "STRONG" in v and any(w in v for w in ("YES", "INVEST", "BUY")):
        return "STRONG YES"
    if any(w in v for w in ("HOLD", "MORE DILIGENCE", "WAIT")):
        return "HOLD"
    if any(w in v for w in ("PASS", "DECLINE", "NO ")) or v == "NO":
        return "PASS"
    if any(w in v for w in ("YES", "INVEST", "BUY")):
        return "YES"
    return "HOLD"


async def generate_ic_memo(company: Company, agent_results: dict[str, AgentResult], sanity_checks: list[SanityCheck], slide_analyses: list[SlideAnalysis], file_hash: str, client: AsyncOpenAI) -> ICMemo:
    cache_k = cache_key(file_hash, "ic_memo_v5", hash(company.model_dump_json()))
    cached = disk_cache_get(cache_k)
    if cached:
        return ICMemo.model_validate(cached)

    parts = ["=== COMPANY ===", company.model_dump_json(exclude_none=True), "=== AGENTS ==="]
    for name, res in agent_results.items():
        parts.append(f"--- {name} ---\nFailed: {res.failed}\nScore: {res.score}\n{res.analysis}")
    parts.append("=== SANITY CHECKS ===")
    for c in sanity_checks:
        parts.append(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.message}")

    messages = [{"role": "system", "content": IC_MEMO_PROMPT}, {"role": "user", "content": "\n".join(parts)}]
    data = await call_openai_json_async(client, messages, model="gpt-4o", temperature=0.4, max_tokens=4000, agent_name="IC Memo")

    if isinstance(data.get("risks"), list):
        data["risks"] = "\n".join(f"- {r}" for r in data["risks"])
    if "recommendation" in data:
        data["recommendation"] = normalize_recommendation(data["recommendation"])

    memo = ICMemo.model_validate(data)
    memo.scores = _merge_scores(memo.scores, agent_results, company)
    memo.overall_score = _calculate_overall_score(memo.scores)
    disk_cache_set(cache_k, memo.model_dump())
    return memo


def _merge_scores(scores, agents, company):
    from extractor import calculate_traction_score
    s = {k: float(scores.get(k, 50.0)) for k in SCORE_WEIGHTS}
    for a, dim in AGENT_SCORE_MAP.items():
        if a in agents and not agents[a].failed:
            s[dim] = agents[a].score
    s["traction"] = calculate_traction_score(company.metrics)
    return s


def _calculate_overall_score(scores):
    return round(sum(scores.get(d, 50.0) * w for d, w in SCORE_WEIGHTS.items()), 1)


def _section(title: str, body: str) -> str:
    return f"\n## {title}\n{body.strip() if body and body.strip() else '_Not provided in the deck._'}\n"


def format_memo_as_markdown(memo: ICMemo, company: Company, agent_results: dict[str, AgentResult], sanity_checks: list[SanityCheck]) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    lines: list[str] = []

    lines.append(f"# Investment Committee Memo: {company.name}")
    if company.tagline:
        lines.append(f"*{company.tagline}*")
    lines.append(
        f"\n**Date:** {date_str}  |  **Recommendation:** {memo.recommendation}  |  "
        f"**Overall Score:** {memo.overall_score:.1f}/100"
    )
    lines.append(
        f"**Confidence:** {memo.confidence:.0%}  |  **Probability of Failure:** {memo.probability_of_failure:.0%}  |  "
        f"**Probability of Outsized (Unicorn) Outcome:** {memo.probability_of_unicorn:.0%}"
    )

    lines.append(_section("Executive Summary", memo.executive_summary))
    lines.append(_section("Investment Thesis", memo.investment_thesis))
    lines.append(_section("Why Now?", memo.why_now))
    lines.append(_section("Business Model", memo.business_model))
    if memo.product:
        lines.append(_section("Product", memo.product))
    lines.append(_section("Market", memo.market))
    lines.append(_section("Competition", memo.competition))

    if company.competitors:
        lines.append("\n**Competitor Detail:**\n")
        lines.append("| Competitor | Type | Threat | Description |\n|---|---|---|---|")
        for c in company.competitors:
            lines.append(f"| {c.name} | {c.competitor_type} | {c.threat_level} | {c.description} |")

    lines.append(_section("Moat Assessment", memo.moat))
    lines.append(_section("Traction", memo.traction))
    lines.append(_section("Unit Economics", memo.unit_economics))
    lines.append(_section("Financial Health", memo.financial_health))
    lines.append(_section("Technology", memo.technology))
    lines.append(_section("Founder Assessment", memo.founder_assessment))

    if company.founders:
        lines.append("\n**Founders on File:**\n")
        for f in company.founders:
            details = ", ".join(filter(None, [f.education, ", ".join(f.previous_companies) if f.previous_companies else None]))
            lines.append(f"- **{f.name or 'Unnamed'}** ({f.role or 'Role not stated'})" + (f" — {details}" if details else ""))

    lines.append(_section("Risks", memo.risks))

    if memo.open_questions:
        lines.append("\n## Open Questions")
        for q in memo.open_questions:
            lines.append(f"- {q}")

    if memo.required_follow_up:
        lines.append("\n## Required Diligence")
        for item in memo.required_follow_up:
            lines.append(f"- {item}")

    lines.append(_section("Investment Recommendation", memo.investment_decision))

    lines.append("\n## Deterministic Sanity Checks")
    for c in sanity_checks:
        icon = "PASS" if c.passed else ("CRITICAL" if c.severity == "critical" else "WARNING")
        rec = f" _Recommendation: {c.recommendation}_" if c.recommendation else ""
        lines.append(f"- **[{icon}] {c.name}:** {c.message}{rec}")

    if company.benchmarks:
        lines.append("\n## Benchmarks vs. Sector Medians")
        lines.append("| Metric | Company | Benchmark | Variance | Status | Source |\n|---|---|---|---|---|---|")
        for b in company.benchmarks:
            lines.append(f"| {b.metric} | {b.company_value} | {b.benchmark_value} | {b.variance_percentage}% | {b.status} | {b.benchmark_source} |")

    lines.append("\n## Scoring Breakdown")
    lines.append("| Dimension | Score |\n|---|---|")
    for dim, score in memo.scores.items():
        lines.append(f"| {dim.title()} | {score:.0f}/100 |")

    lines.append("\n## Detailed Agent Analyses")
    for name, res in agent_results.items():
        lines.append(f"\n### {name.replace('_', ' ').title()}")
        if res.failed:
            lines.append("*Agent failed to complete.*")
            continue
        lines.append(f"**Score:** {res.score:.0f}/100  |  **Confidence:** {res.confidence:.0%}\n")
        lines.append(res.analysis)
        if res.key_findings:
            lines.append("\n**Key Findings:**")
            lines.extend(f"- {kf}" for kf in res.key_findings)
        if res.risks:
            lines.append("\n**Risks Identified:**")
            lines.extend(f"- {r}" for r in res.risks)
        if res.recommendations:
            lines.append("\n**Recommendations:**")
            lines.extend(f"- {r}" for r in res.recommendations)

    return "\n".join(lines)


def format_memo_as_html(memo: ICMemo, company: Company, agent_results: dict[str, AgentResult], sanity_checks: list[SanityCheck]) -> str:
    md_text = format_memo_as_markdown(memo, company, agent_results, sanity_checks)
    html_body = md.markdown(md_text, extensions=["tables", "fenced_code"])
    return f"<!DOCTYPE html><html><head><style>body{{font-family:sans-serif;max-width:800px;margin:auto;padding:20px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px}}</style></head><body>{html_body}</body></html>"
