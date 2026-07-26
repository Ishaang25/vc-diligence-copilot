"""
Structured extraction, deterministic math, and external benchmarking.
"""
from __future__ import annotations
import json
import logging
from openai import AsyncOpenAI
from pydantic import ValidationError

from cache import cache_key, disk_cache_get, disk_cache_set
from models import BenchmarkComparison, Company, Metrics, SanityCheck, SlideAnalysis
from prompts import EXTRACTION_PROMPT
from utils import BENCHMARKS, call_openai_json_async, safe_float, safe_int

logger = logging.getLogger("dd_copilot.extractor")

async def extract_company(slide_analyses: list[SlideAnalysis], file_hash: str, client: AsyncOpenAI) -> Company:
    slides_hash = json.dumps([s.model_dump() for s in slide_analyses if not s.is_skipped], sort_keys=True)
    cache_k = cache_key(file_hash, "company_v4", slides_hash)
    cached = disk_cache_get(cache_k)
    cached = None
    
    valid_slides = [s for s in slide_analyses if not s.is_skipped]
    slides_text = _format_slides_for_prompt(valid_slides)
    messages = [{"role": "system", "content": EXTRACTION_PROMPT}, {"role": "user", "content": slides_text}]

    data = await call_openai_json_async(client, messages, model="gpt-4o-mini", temperature=0.2, max_tokens=4000, agent_name="Company Extractor")

    # Sanitize GPT output before validation
    if not isinstance(data, dict):
        data = {}
    
    # Required string fields
    if data.get("name") is None:
        data["name"] = "Unknown"
    
    if data.get("description") is None:
        data["description"] = ""
    
    # Nested objects
    if data.get("metrics") is None:
        data["metrics"] = {}
    
    if data.get("pricing") is None:
        data["pricing"] = {}
    
    if data.get("fundraising") is None:
        data["fundraising"] = {}
    
    # Lists
    for field in [
        "competitors",
        "risks",
        "regulatory_risks",
        "benchmarks",
    ]:
        if data.get(field) is None:
            data[field] = []
    
    # Optional complex objects
    if data.get("moat") is None:
        data["moat"] = {}
    
    if data.get("market") is None:
        data["market"] = None
    
    print("\n" + "="*80)
    print("RAW COMPANY EXTRACTION")
    print("="*80)
    
    print(json.dumps(data, indent=2))
    
    print("="*80)
    
    company = Company.model_validate(data)

    disk_cache_set(cache_k, company.model_dump())
    company.metrics = calculate_derived_metrics(company.metrics)
    company.benchmarks = generate_benchmarks(company)
    return company

def _format_slides_for_prompt(slide_analyses):
    parts = []

    for s in slide_analyses:
        part = f"""
==============================
SLIDE {s.slide_number}
TYPE: {s.slide_type.value}

SUMMARY:
{s.summary}

NUMBERS:
{json.dumps(s.numbers, indent=2)}

CLAIMS:
{json.dumps(s.claims, indent=2)}

RISKS:
{json.dumps(s.risks, indent=2)}

EVIDENCE:
{json.dumps([e.model_dump() for e in s.evidence], indent=2)}
==============================
"""
        parts.append(part)

    return "\n".join(parts)
    
def calculate_derived_metrics(metrics: Metrics) -> Metrics:
    if metrics.ltv and metrics.cac and metrics.cac > 0:
        metrics.ltv_cac_ratio = round(metrics.ltv / metrics.cac, 2)
    if metrics.cac and metrics.revenue_per_customer and metrics.gross_margin and metrics.gross_margin > 0:
        monthly_gp = metrics.revenue_per_customer * (metrics.gross_margin / 100)
        if monthly_gp > 0: metrics.cac_payback_months = round(metrics.cac / monthly_gp, 1)
    if metrics.growth_rate and metrics.gross_margin:
        metrics.rule_of_40 = round(metrics.growth_rate + metrics.gross_margin, 1)
    if metrics.churn_rate:
        metrics.annualized_churn = round((1 - (1 - metrics.churn_rate / 100) ** 12) * 100, 1)
    return metrics

def generate_benchmarks(company: Company) -> list[BenchmarkComparison]:
    sector = company.sector or "Default"
    sector_key = "SaaS" if "saas" in sector.lower() else "FinTech" if "fintech" in sector.lower() else "Healthcare" if "health" in sector.lower() else "Default"
    bench = BENCHMARKS.get(sector_key, BENCHMARKS["Default"])
    
    comparisons = []
    m = company.metrics
    
    def add_comp(metric_name, val, bench_val):
        if val is not None and bench_val is not None:
            var = ((val - bench_val) / bench_val) * 100
            status = "Better" if (metric_name == "Churn Rate" and var < -5) or (metric_name != "Churn Rate" and var > 5) else "Worse" if (metric_name == "Churn Rate" and var > 5) or (metric_name != "Churn Rate" and var < -5) else "Neutral"
            comparisons.append(BenchmarkComparison(
                metric=metric_name, company_value=val, benchmark_value=bench_val,
                variance_percentage=round(var, 1), status=status, benchmark_source=bench.get("source", "")
            ))
            
    add_comp("Churn Rate (Monthly)", m.churn_rate, bench["median_churn"])
    add_comp("Gross Margin", m.gross_margin, bench["median_gross_margin"])
    add_comp("LTV:CAC Ratio", m.ltv_cac_ratio, bench["median_ltv_cac"])
    add_comp("Rule of 40", m.rule_of_40, bench["median_rule_of_40"])
    return comparisons

def run_sanity_checks(company: Company) -> list[SanityCheck]:
    checks = []
    m = company.metrics

    if m.ltv_cac_ratio is not None and m.ltv_cac_ratio < 1.0:
        checks.append(
            SanityCheck(
                name="CAC exceeds LTV",
                severity="critical",
                message=f"LTV:CAC ratio is {m.ltv_cac_ratio:.1f}:1.",
                recommendation="Do not invest.",
                passed=False,
            )
        )

    if m.runway_months is not None and m.runway_months < 6:
        checks.append(
            SanityCheck(
                name="Runway < 6 months",
                severity="critical",
                message=f"Runway is {m.runway_months:.0f} months.",
                passed=False,
            )
        )

    if m.gross_margin is not None and m.gross_margin < 40:
        checks.append(
            SanityCheck(
                name="Low Gross Margin",
                severity="warning",
                message=f"Gross margin is {m.gross_margin:.1f}%.",
                passed=False,
            )
        )

    if m.churn_rate is not None and m.churn_rate > 5:
        checks.append(
            SanityCheck(
                name="High Churn",
                severity="critical" if m.churn_rate > 10 else "warning",
                message=f"Monthly churn is {m.churn_rate:.1f}%.",
                passed=False,
            )
        )

    if not checks:
        checks.append(
            SanityCheck(
                name="No red flags",
                severity="info",
                message="No deterministic red flags.",
                passed=True,
            )
        )

    return checks

def calculate_traction_score(metrics: Metrics) -> float:
    score = 50.0
    if metrics.arr:
        if metrics.arr >= 10_000_000: score += 25
        elif metrics.arr >= 1_000_000: score += 15
    if metrics.churn_rate and metrics.churn_rate > 5: score -= 20
    return max(0.0, min(100.0, score))
