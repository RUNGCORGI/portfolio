"""
tests/test_quantamental.py — Bộ kiểm thử tự động cho Quantamental AI Engine & Dual-RAG.
Kiểm tra:
- Mô hình dữ liệu & hợp đồng contracts
- Ma trận Chân lý 8 trạng thái [T, S, F]
- Phân tích RAG Cảm xúc Báo chí (Flag S)
- Phân tích RAG Nội tại BCTC (Flag F)
- Cache & dữ liệu mẫu offline (fixtures)
- Báo cáo Quantamental tổng hợp & Thử nghiệm đối chứng A/B Testing
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.contracts import MarketData
from core.portfolio import build_portfolio, ma_crossover_signals
from quantamental.financials_engine import fetch_financial_metrics
from quantamental.fixtures_cache import MOCK_FINANCIALS, MOCK_NEWS
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
from quantamental.tab_ui import generate_quantamental_report, simulate_ab_testing
from quantamental.truth_table import TRUTH_TABLE_RULES, classify_quantamental_signal
from tabs import tab_quantamental


# ==============================================================================
# 1. KIỂM THỬ MA TRẬN CHÂN LÝ (TRUTH TABLE 8 TRẠNG THÁI)
# ==============================================================================

def test_truth_table_all_8_states_defined():
    """Đảm bảo đầy đủ 8 trạng thái nhị phân 2^3 từ (0,0,0) đến (1,1,1)."""
    assert len(TRUTH_TABLE_RULES) == 8
    expected_tuples = {
        (1, 0, 0), (1, 1, 1), (1, 1, 0), (1, 0, 1),
        (0, 1, 1), (0, 1, 0), (0, 0, 1), (0, 0, 0)
    }
    assert set(TRUTH_TABLE_RULES.keys()) == expected_tuples


def test_classify_bull_trap():
    """T=1, S=0, F=0 -> Bẫy tăng giá (Bull Trap)."""
    sent = SentimentResult(ticker="TRAP", score=0.40, flag=0, label="Tiêu cực")
    fund = FundamentalResult(ticker="TRAP", score=0.25, flag=0, label="Bất ổn")
    sig = classify_quantamental_signal("TRAP", technical_flag=1, sentiment=sent, fundamental=fund)

    assert sig.T == 1 and sig.S == 0 and sig.F == 0
    assert sig.pattern_code == "[1, 0, 0]"
    assert "Bull Trap" in sig.pattern_name
    assert "Blacklist" in sig.weight_action or "0%" in sig.weight_action


def test_classify_quality_momentum():
    """T=1, S=1, F=1 -> Tín hiệu vàng / Siêu cổ phiếu."""
    sent = SentimentResult(ticker="GOLD", score=0.85, flag=1, label="Sôi động")
    fund = FundamentalResult(ticker="GOLD", score=0.90, flag=1, label="Vững chắc")
    sig = classify_quantamental_signal("GOLD", technical_flag=1, sentiment=sent, fundamental=fund)

    assert sig.T == 1 and sig.S == 1 and sig.F == 1
    assert sig.pattern_code == "[1, 1, 1]"
    assert "Tín Hiệu Vàng" in sig.pattern_name


def test_classify_fomo():
    """T=1, S=1, F=0 -> Hùa theo đám đông (FOMO)."""
    sent = SentimentResult(ticker="MEME", score=0.80, flag=1, label="Sôi động")
    fund = FundamentalResult(ticker="MEME", score=0.30, flag=0, label="Bất ổn")
    sig = classify_quantamental_signal("MEME", technical_flag=1, sentiment=sent, fundamental=fund)

    assert sig.pattern_code == "[1, 1, 0]"
    assert "FOMO" in sig.pattern_name


def test_classify_hidden_gem():
    """T=1, S=0, F=1 -> Ngọc trong đá (Hidden Gem)."""
    sent = SentimentResult(ticker="GEM", score=0.50, flag=0, label="Bình thường")
    fund = FundamentalResult(ticker="GEM", score=0.85, flag=1, label="Vững chắc")
    sig = classify_quantamental_signal("GEM", technical_flag=1, sentiment=sent, fundamental=fund)

    assert sig.pattern_code == "[1, 0, 1]"
    assert "Ngọc Trong Đá" in sig.pattern_name


# ==============================================================================
# 2. KIỂM THỬ RAG CẢM XÚC BÁO CHÍ (FLAG S)
# ==============================================================================

def test_rag_sentiment_positive():
    """Tin tức tích cực và tăng trưởng -> Flag S = 1."""
    news = [
        NewsItem(
            title="Lợi nhuận quý kỷ lục, tăng trưởng vượt bậc và doanh thu bứt phá",
            link="http://example.com/1",
            pub_date="2026-10-30",
            source="Reuters",
            snippet="Cổ phiếu được nâng xếp hạng khuyến nghị mua mạnh mẽ, bùng nổ tiềm năng."
        ),
        NewsItem(
            title="Đột phá công nghệ mới mang lại hợp đồng tỷ đô mở rộng thị phần",
            link="http://example.com/2",
            pub_date="2026-10-29",
            source="Bloomberg",
            snippet="Biên lợi nhuận gộp mở rộng đáng kể."
        ),
    ]
    res = analyze_sentiment("TEST_POS", news)
    assert res.score >= 0.58
    assert res.flag == 1
    assert len(res.news_items) == 2


def test_rag_sentiment_negative():
    """Tin tức tiêu cực về thua lỗ, nợ xấu, điều tra -> Flag S = 0."""
    news = [
        NewsItem(
            title="Doanh nghiệp ghi nhận thua lỗ nặng nề, sụt giảm doanh thu nghiêm trọng",
            link="http://example.com/3",
            pub_date="2026-10-28",
            source="News",
            snippet="Bị cơ quan chức năng điều tra vì vi phạm công bố thông tin, nguy cơ hủy niêm yết."
        ),
    ]
    res = analyze_sentiment("TEST_NEG", news)
    assert res.score <= 0.50
    assert res.flag == 0


def test_rag_sentiment_empty():
    """Danh sách tin tức rỗng -> Không crash, trả về điểm trung tính."""
    res = analyze_sentiment("EMPTY", [])
    assert res.score == 0.50
    assert res.flag == 0
    assert "Thiếu tin" in res.label or "Trung tính" in res.label


# ==============================================================================
# 3. KIỂM THỬ RAG NỘI TẠI BÁO CÁO TÀI CHÍNH (FLAG F)
# ==============================================================================

def test_rag_fundamental_solid():
    """Doanh nghiệp tài chính mạnh: OCF dương, tăng trưởng tốt, biên cao, nợ thấp -> Flag F = 1."""
    fin = FinancialMetrics(
        revenue=1000.0,
        revenue_growth_yoy=15.0,
        net_income=250.0,
        net_margin=25.0,
        ocf=300.0,
        debt_to_equity=0.35,
        pe_ratio=18.0,
        eps=5.0
    )
    res = analyze_fundamentals("SOLID", fin)
    assert res.flag == 1
    assert res.score >= 0.60
    assert len(res.criteria_passed) >= 3


def test_rag_fundamental_weak():
    """Doanh nghiệp thua lỗ, OCF âm, nợ cao -> Flag F = 0."""
    fin = FinancialMetrics(
        revenue=100.0,
        revenue_growth_yoy=-20.0,
        net_income=-80.0,
        net_margin=-80.0,
        ocf=-150.0,
        debt_to_equity=5.0,
        pe_ratio=-1.0,
        eps=-2.0
    )
    res = analyze_fundamentals("WEAK", fin)
    assert res.flag == 0
    assert res.score < 0.50
    assert len(res.criteria_failed) >= 3


# ==============================================================================
# 4. KIỂM THỬ FIXTURES & FALLBACK CHỐNG LỖI MẠNG
# ==============================================================================

def test_fixtures_cache_contents():
    """Kiểm tra kho dữ liệu mẫu sẵn có cho các mã mặc định."""
    for tk in ["AAPL", "MSFT", "NVDA", "VCB", "FPT", "HPG"]:
        assert tk in MOCK_FINANCIALS or tk in MOCK_NEWS


def test_fetch_financial_metrics_fallback():
    """Hàm fetch_financial_metrics trả về object hợp lệ ngay cả khi mã không tồn tại."""
    metrics = fetch_financial_metrics("XYZ_UNKNOWN_TICKER")
    assert isinstance(metrics, FinancialMetrics)
    assert metrics.revenue is not None


# ==============================================================================
# 5. KIỂM THỬ PIPELINE TỔNG HỢP & A/B TESTING
# ==============================================================================

@pytest.fixture
def mock_market_data():
    dates = pd.date_range("2023-01-01", periods=100)
    prices = pd.DataFrame({
        "AAPL": np.linspace(150, 190, 100),
        "MSFT": np.linspace(250, 320, 100),
    }, index=dates)
    benchmark = pd.Series(np.linspace(400, 450, 100), index=dates, name="SPY")
    return MarketData(
        prices=prices,
        benchmark=benchmark,
        factors=prices.copy(),
        tickers=["AAPL", "MSFT"],
        source="test_fixture",
        start=pd.Timestamp("2023-01-01")
    )


def test_generate_quantamental_report(mock_market_data):
    """Kiểm tra chạy toàn bộ quy trình sinh báo cáo Quantamental."""
    md = mock_market_data
    sigs = ma_crossover_signals(md, 20, 50)
    report = generate_quantamental_report(md, sigs)

    assert isinstance(report, QuantamentalReport)
    assert "AAPL" in report.per_ticker
    assert "MSFT" in report.per_ticker
    assert isinstance(report.summary_df, pd.DataFrame)
    assert len(report.summary_df) == 2


def test_simulate_ab_testing(mock_market_data):
    """Kiểm tra thuật toán A/B Testing giữa danh mục truyền thống và Quantamental."""
    md = mock_market_data
    pf = build_portfolio(md, [0.5, 0.5])
    sigs = ma_crossover_signals(md, 20, 50)
    report = generate_quantamental_report(md, sigs)

    ab = simulate_ab_testing(md, pf, report)
    assert "cum_trad" in ab
    assert "cum_quant" in ab
    assert "m_trad" in ab
    assert "m_quant" in ab
    assert len(ab["w_quant"]) == 2
    assert np.isclose(ab["w_quant"].sum(), 1.0) or np.isclose(ab["w_quant"].sum(), 0.0)


def test_tab_quantamental_import():
    """Kiểm tra nạp module tab_quantamental an toàn."""
    assert hasattr(tab_quantamental, "render")


def test_api_quantamental_endpoint():
    """Kiểm tra API endpoint /api/quantamental trả về đúng cấu trúc JSON."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    resp = client.post(
        "/api/quantamental",
        json={"tickers": ["AAPL", "MSFT"], "start": "2023-01-01", "market": "us", "use_sample": True}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "tickers" in data
    assert "signals" in data
    assert "ab_testing" in data
    assert len(data["signals"]) == 2
    assert "pattern_code" in data["signals"][0]

