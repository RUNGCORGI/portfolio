"""
quantamental/models.py — Data contracts cho phân tích Quantamental AI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class NewsItem:
    title: str
    link: str
    pub_date: str
    source: str
    snippet: str = ""


@dataclass
class SentimentResult:
    ticker: str
    score: float             # 0.0 -> 1.0
    flag: int                # 1: Tích cực / Sôi động, 0: Bình thường / Tiêu cực
    label: str               # "Tích cực", "Trung tính", "Tiêu cực"
    news_items: List[NewsItem] = field(default_factory=list)
    key_themes: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class FinancialMetrics:
    revenue: float = 0.0          # Triệu USD hoặc Tỷ VND
    revenue_growth_yoy: float = 0.0  # %
    net_income: float = 0.0
    net_margin: float = 0.0       # %
    ocf: float = 0.0              # Operating Cash Flow
    debt_to_equity: float = 0.0
    pe_ratio: float = 0.0
    eps: float = 0.0
    currency: str = "USD"


@dataclass
class FundamentalResult:
    ticker: str
    score: float             # 0.0 -> 1.0
    flag: int                # 1: Nội tại vững chắc, 0: Bất ổn / Rỗng ruột
    label: str               # "Vững chắc", "Bất ổn", "Rỗng ruột"
    metrics: FinancialMetrics = field(default_factory=FinancialMetrics)
    criteria_passed: List[str] = field(default_factory=list)
    criteria_failed: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class QuantamentalSignal:
    ticker: str
    T: int                   # Tín hiệu Kỹ thuật (0 hoặc 1)
    S: int                   # Cảm xúc Báo chí (0 hoặc 1)
    F: int                   # Nội tại BCTC (0 hoặc 1)
    pattern_code: str        # "[1, 0, 0]"
    pattern_name: str        # "Bẫy tăng giá (Bull Trap)"
    badge_color: str         # "red", "green", "orange", "blue"
    sentiment: SentimentResult
    fundamental: FundamentalResult
    description: str
    recommendation: str      # "CẢNH BÁO / LOẠI BỎ", "MUA MẠNH", v.v.
    weight_action: str       # "w = 0%", "Tối đa w", v.v.


@dataclass
class QuantamentalReport:
    per_ticker: Dict[str, QuantamentalSignal]
    summary_df: pd.DataFrame
    patterns_count: Dict[str, int]

