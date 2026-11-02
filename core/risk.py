"""
core/risk.py  —  P3 (SỞ HỮU)
Khớp bài gốc: RISK ENGINE + ROLLING BETA ENGINE (myapp5 -> myapp7).

PHÁT TRIỂN THÊM #1 của P3: bộ metric đuôi rủi ro mà bài gốc hoàn toàn không có
    VaR 95% (historical + parametric), CVaR 95%, Sortino, Calmar,
    Max Drawdown và ĐỘ DÀI giai đoạn drawdown (tính bằng ngày).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from core.contracts import TRADING_DAYS, MarketData, PortfolioResult, RiskReport


def drawdown_series(cum: pd.Series) -> pd.Series:
    return cum / cum.cummax() - 1


def max_dd_duration(cum: pd.Series) -> int:
    """Số ngày dài nhất danh mục chưa lấy lại đỉnh cũ."""
    peak = cum.cummax()
    under = cum < peak
    best = run = 0
    for flag in under:
        run = run + 1 if flag else 0
        best = max(best, run)
    return int(best)


def tail_metrics(r: pd.Series, cum: pd.Series, mar: float = 0.0) -> dict:
    r = r.dropna()
    var_hist = float(np.percentile(r, 5))
    var_param = float(stats.norm.ppf(0.05, r.mean(), r.std()))
    cvar = float(r[r <= var_hist].mean()) if (r <= var_hist).any() else var_hist

    downside = r[r < mar]
    dsd = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = 0.0 if dsd == 0 else float(r.mean() / dsd * np.sqrt(TRADING_DAYS))

    mdd = float(drawdown_series(cum).min())
    cagr = float(cum.iloc[-1] ** (TRADING_DAYS / len(cum)) - 1)
    calmar = 0.0 if mdd == 0 else float(cagr / abs(mdd))

    return {
        "var_95_hist": var_hist * 100,
        "var_95_param": var_param * 100,
        "cvar_95": cvar * 100,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": mdd * 100,
        "max_dd_duration_days": max_dd_duration(cum),
        "skew": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
    }


def calculate_market_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Tính Market Beta (Beta toàn thời gian) so với benchmark."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.empty:
        return 0.0
    cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    var = aligned.iloc[:, 1].var()
    return float(cov / var) if var != 0 else 0.0


def build_risk_report(
    md: MarketData, pf: PortfolioResult, window: int = 20
) -> RiskReport:
    pr, br = pf.returns, pf.benchmark_returns
    bcum = pf.benchmark_cumulative

    cov = pr.rolling(window).cov(br)
    var = br.rolling(window).var()
    beta = (cov / var).dropna()
    alpha = (pr.rolling(window).mean() - beta * br.rolling(window).mean()).dropna()

    return RiskReport(
        corr=md.returns.corr(),
        drawdown=drawdown_series(pf.cumulative),
        benchmark_drawdown=drawdown_series(bcum),
        rolling_vol=(pr.rolling(window).std() * np.sqrt(TRADING_DAYS)).dropna(),
        benchmark_rolling_vol=(br.rolling(window).std() * np.sqrt(TRADING_DAYS)).dropna(),
        rolling_beta=beta,
        rolling_alpha=alpha,
        tail=tail_metrics(pr, pf.cumulative),
    )


def stress_test(pf: PortfolioResult, shock_pct: float = -20.0) -> dict:
    """PHÁT TRIỂN THÊM #3 (dự phòng): cú sốc tức thời theo beta."""
    b = calculate_market_beta(pf.returns, pf.benchmark_returns)
    return {"beta": b, "shock_market_pct": shock_pct,
            "expected_portfolio_pct": b * shock_pct}