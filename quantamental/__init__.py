"""
Package quantamental — Mô hình Quantamental AI Dual-RAG:
Kết hợp Phân tích Kỹ thuật (T) + Cảm xúc Báo chí thời gian thực (S) + Sức khỏe Nội tại BCTC (F).
"""
from quantamental.financials_engine import fetch_financial_metrics
from quantamental.models import (
    FinancialMetrics,
    FundamentalResult,
    NewsItem,
    QuantamentalReport,
    QuantamentalSignal,
    SentimentResult,
)
from quantamental.news_engine import fetch_realtime_news
from quantamental.rag_fundamental import analyze_fundamentals
from quantamental.rag_sentiment import analyze_sentiment
from quantamental.tab_ui import generate_quantamental_report, render_quantamental_tab
from quantamental.truth_table import TRUTH_TABLE_RULES, classify_quantamental_signal

__all__ = [
    "render_quantamental_tab",
    "generate_quantamental_report",
    "classify_quantamental_signal",
    "TRUTH_TABLE_RULES",
    "analyze_sentiment",
    "analyze_fundamentals",
    "fetch_realtime_news",
    "fetch_financial_metrics",
    "QuantamentalSignal",
    "QuantamentalReport",
    "SentimentResult",
    "FundamentalResult",
    "FinancialMetrics",
    "NewsItem",
]

