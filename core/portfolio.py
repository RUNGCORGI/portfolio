"""
core/portfolio.py  —  P2 (SỞ HỮU)
Khớp bài gốc: khối PORTFOLIO ENGINE + SPY BENCHMARK + ROI METRICS
(myapp2.py -> myapp5.py).

Bổ sung so với bài gốc: chuẩn hóa weight an toàn, align benchmark một lần
duy nhất, và tùy chọn tính phí giao dịch (core/costs.py).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from core.contracts import TRADING_DAYS, MarketData, PortfolioResult, SignalReport
from core.costs import apply_costs, cost_drag
# IMPORT HÀM TÍNH BETA TỪ RISK.PY
from core.risk import calculate_market_beta 


def normalize_weights(raw: List[float] | np.ndarray, n: int) -> np.ndarray:
    w = np.asarray(raw, dtype=float)
    if w.size != n or not np.isfinite(w).all() or w.sum() <= 0:
        return np.ones(n) / n
    return w / w.sum()


def build_portfolio(
    md: MarketData,
    weights: List[float] | np.ndarray,
    cost_bps: float = 0.0,
    rebalance: str = "Không",
) -> PortfolioResult:
    rets = md.returns
    w = normalize_weights(weights, rets.shape[1])

    # Lưu ý: tái cân bằng và chi phí là HAI hiệu ứng khác nhau.
    # Chạy apply_costs khi có bất kỳ hiệu ứng nào, để so sánh công bằng.
    if cost_bps > 0 or rebalance != "Không":
        net, gross, turnover = apply_costs(rets, w, cost_bps, rebalance)
        drag = cost_drag(gross, net)
    else:
        net = rets.fillna(0).dot(w)
        gross, turnover, drag = None, None, 0.0

    # Align với benchmark MỘT LẦN — bài gốc làm đi làm lại ở nhiều chỗ
    bm = md.benchmark_returns
    joined = pd.concat([net, bm], axis=1, join="inner").dropna()
    joined.columns = ["portfolio", "benchmark"]

    pr = joined["portfolio"]
    br = joined["benchmark"]
    cum = (1 + pr).cumprod()
    bcum = (1 + br).cumprod()

    sd = float(pr.std())
    metrics = {
        "total_return": float((cum.iloc[-1] - 1) * 100),
        "benchmark_total_return": float((bcum.iloc[-1] - 1) * 100),
        "volatility": float(sd * np.sqrt(TRADING_DAYS) * 100),
        "sharpe": 0.0 if sd == 0 else float(pr.mean() / sd * np.sqrt(TRADING_DAYS)),
        "cagr": float((cum.iloc[-1] ** (TRADING_DAYS / len(cum)) - 1) * 100),
    }
    metrics["alpha"] = metrics["total_return"] - metrics["benchmark_total_return"]

    # ==========================================
    # THÊM TÍNH TOÁN BETA VÀO METRICS
    # ==========================================
    # 1. Market Beta (Toàn thời gian)
    metrics["market_beta"] = calculate_market_beta(pr, br)

    # 2. Rolling Beta (Lấy giá trị của ngày gần nhất - cửa sổ 20 ngày)
    cov = pr.rolling(20).cov(br)
    var = br.rolling(20).var()
    rolling_beta = (cov / var).dropna()
    metrics["rolling_beta_latest"] = float(rolling_beta.iloc[-1]) if not rolling_beta.empty else 0.0
    # ==========================================

    return PortfolioResult(
        weights=w,
        tickers=list(md.prices.columns),
        returns=pr,
        cumulative=cum,
        benchmark_returns=br,
        benchmark_cumulative=bcum,
        metrics=metrics,
        gross_returns=gross.reindex(pr.index) if gross is not None else None,
        turnover=turnover.reindex(pr.index) if turnover is not None else None,
        cost_drag_pct=drag,
    )


def ma_crossover_signals(md: MarketData, short: int = 20, long: int = 50) -> SignalReport:
    """Khối STRATEGY ENGINE của bài gốc — nhưng KHÔNG tải lại dữ liệu từng mã."""
    px = md.prices
    sig = (px.rolling(short).mean() > px.rolling(long).mean()).astype(int).dropna()
    port = sig.mean(axis=1)
    return SignalReport(
        per_ticker=sig,
        portfolio_signal=port,
        current="BUY" if (len(port) and port.iloc[-1] > 0.5) else "SELL",
    )