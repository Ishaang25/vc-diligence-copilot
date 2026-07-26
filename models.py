"""
Pydantic models for structured data.
Includes SlideType classification for advanced routing.
"""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field, ConfigDict
from dataclasses import dataclass, field
from enum import Enum

class SlideType(str, Enum):
    COVER = "cover"
    PROBLEM = "problem"
    SOLUTION = "solution"
    MARKET = "market"
    PRODUCT = "product"
    TRACTION = "traction"
    TEAM = "team"
    FINANCIALS = "financials"
    COMPETITION = "competition"
    APPENDIX = "appendix"
    OTHER = "other"

class Evidence(BaseModel):
    model_config = ConfigDict(extra="ignore")
    claim: str = ""
    quote: str | None = None
    slide_number: int | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

class SlideAnalysis(BaseModel):
    model_config = ConfigDict(extra="ignore")
    slide_number: int = 0
    slide_type: SlideType = SlideType.OTHER
    is_skipped: bool = False
    skip_reason: str | None = None
    summary: str = ""
    numbers: dict[str, Any] = Field(default_factory=dict)
    claims: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    embedding: list[float] | None = None  # For RAG

class BenchmarkComparison(BaseModel):
    model_config = ConfigDict(extra="ignore")
    metric: str = ""
    company_value: float | None = None
    benchmark_value: float | None = None
    benchmark_source: str = ""
    variance_percentage: float | None = None
    status: str = "Neutral"

class Metrics(BaseModel):
    model_config = ConfigDict(extra="ignore")
    revenue: float | None = None
    mrr: float | None = None
    arr: float | None = None
    burn_rate: float | None = None
    runway_months: float | None = None
    cac: float | None = None
    ltv: float | None = None
    churn_rate: float | None = None
    gross_margin: float | None = None
    customer_count: int | None = None
    user_count: int | None = None
    growth_rate: float | None = None
    revenue_per_customer: float | None = None
    headcount: int | None = None
    ltv_cac_ratio: float | None = None
    cac_payback_months: float | None = None
    rule_of_40: float | None = None
    annualized_churn: float | None = None

class Pricing(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str | None = None
    starting_price: float | None = None
    currency: str = "USD"
    details: str | None = None

class Fundraising(BaseModel):
    model_config = ConfigDict(extra="ignore")
    round: str | None = None
    amount_raised: float | None = None
    amount_raising: float | None = None
    valuation: float | None = None
    pre_money: float | None = None
    use_of_funds: list[str] = Field(default_factory=list)
    investors: list[str] = Field(default_factory=list)

class MarketAnalysis(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tam: float | None = None
    sam: float | None = None
    som: float | None = None
    tam_source: str | None = None
    tam_methodology: str | None = None
    recalculated_tam: float | None = None
    recalculated_som: float | None = None
    growth_rate: float | None = None
    critique: str | None = None

class Competitor(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = ""
    competitor_type: str = "direct"
    description: str = ""
    market_share: float | None = None
    threat_level: str = "medium"

class MoatAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    network_effects: str = "Unknown"
    switching_costs: str = "Unknown"
    distribution: str = "Unknown"
    data: str = "Unknown"
    technology: str = "Unknown"
    brand: str = "Unknown"
    regulatory: str = "Unknown"
    overall_strength: str = "Unknown"

class Risk(BaseModel):
    model_config = ConfigDict(extra="ignore")
    category: str = ""
    description: str = ""
    severity: str = "medium"
    mitigation: str | None = None

class SanityCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = ""
    severity: str = "info"
    message: str = ""
    recommendation: str | None = None
    passed: bool = True

class Company(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = "Unknown"
    tagline: str | None = None
    description: str = ""
    stage: str | None = None
    sector: str | None = None
    headquarters: str | None = None
    founded: int | None = None
    website: str | None = None
    metrics: Metrics = Field(default_factory=Metrics)
    pricing: Pricing = Field(default_factory=Pricing)
    competitors: list[Competitor] = Field(default_factory=list)
    moat: MoatAssessment = Field(default_factory=MoatAssessment)
    market: MarketAnalysis | None = None
    risks: list[Risk] = Field(default_factory=list)
    technology: str | None = None
    regulatory_risks: list[str] = Field(default_factory=list)
    fundraising: Fundraising = Field(default_factory=Fundraising)
    benchmarks: list[BenchmarkComparison] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

class AgentResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    agent_name: str = ""
    analysis: str = ""
    key_findings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    score: float = Field(default=50.0, ge=0.0, le=100.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    failed: bool = False

class ICMemo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    recommendation: str = "HOLD"
    executive_summary: str = ""
    business_model: str = ""
    product: str = ""
    technology: str = ""
    market: str = ""
    competition: str = ""
    traction: str = ""
    unit_economics: str = ""
    financial_health: str = ""
    risks: str = ""
    moat: str = ""
    founder_assessment: str = ""
    required_follow_up: list[str] = Field(default_factory=list)
    investment_decision: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    probability_of_failure: float = Field(default=0.5, ge=0.0, le=1.0)
    probability_of_unicorn: float = Field(default=0.05, ge=0.0, le=1.0)
    overall_score: float = Field(default=50.0, ge=0.0, le=100.0)
    scores: dict[str, float] = Field(default_factory=dict)

@dataclass
class PDFPage:
    number: int
    text: str
    image_bytes: bytes | None = None
    is_junk: bool = False

@dataclass
class PDFDocument:
    file_hash: str
    page_count: int
    pages: list[PDFPage] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    total_text: str = ""
    is_scanned: bool = False
