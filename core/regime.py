"""
core/regime.py  —  P4 (SỞ HỮU)
Khớp bài gốc: REGIME DETECTION + REGIME SWITCHING (myapp10 -> myapp13).

Điểm sửa quan trọng so với bài gốc: myapp10 tính regime_score trên
spy_aligned là chuỗi LỢI SUẤT, rồi lại so sánh MA50 vs MA200 của lợi suất —
sai về mặt kinh tế. Ở đây dùng đúng chuỗi GIÁ cho tín hiệu xu hướng.

PHÁT TRIỂN THÊM #2 của P4: walk_forward() — tối ưu trên cửa sổ quá khứ rồi
áp dụng cho giai đoạn kế tiếp, chứng minh Sharpe "đẹp" của bài gốc phần lớn
đến từ look-ahead bias vì tối ưu và đo trên CÙNG một tập dữ liệu.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.contracts import TRADING_DAYS, MarketData, PortfolioResult, RegimeReport
from core.optimizer import optimize


def detect_regime(md: MarketData, index: Optional[pd.Index] = None) -> RegimeReport:
    px = md.benchmark                       # GIÁ, không phải lợi suất
    ma50, ma200 = px.rolling(50).mean(), px.rolling(200).mean()
    mom = px.pct_change(12)
    vol = md.benchmark_returns.rolling(20).std() * np.sqrt(TRADING_DAYS)

    score = ((px > ma50).astype(int)
             + (ma50 > ma200).astype(int)
             + (mom > 0).astype(int)
             - (vol > vol.rolling(50).mean()).astype(int))
    score = score.dropna()
    if index is not None:
        score = score.reindex(index).dropna()

    state = pd.Series("NEUTRAL", index=score.index, dtype=object)
    state[score >= 2] = "BULL"
    state[score <= 0] = "BEAR"

    # Điều kiện Macro Regime của bài gốc (myapp13 trang 56)
    qqq = md.factors["QQQ"] if ("QQQ" in md.factors.columns) else md.benchmark.copy()
    qqq_ma200 = qqq.rolling(200).mean()
    vix = md.factors["^VIX"] if ("^VIX" in md.factors.columns) else pd.Series(18.0, index=px.index)
    risk_on = (px > ma200) & (qqq > qqq_ma200) & (vix < 20)
    last_risk = risk_on.dropna()
    macro_regime = "RISK ON" if (len(last_risk) > 0 and bool(last_risk.iloc[-1])) else "RISK OFF"

    return RegimeReport(score=score, state=state,
                        current=str(state.iloc[-1]) if len(state) else "N/A",
                        macro_regime=macro_regime)


def switching_strategy(md: MarketData, pf: PortfolioResult,
                       reg: RegimeReport, max_weight: float = 0.5) -> RegimeReport:
    """BULL -> max Sharpe | NEUTRAL -> equal weight | BEAR -> tiền mặt."""
    rets = md.returns.reindex(reg.state.index).dropna()
    state = reg.state.reindex(rets.index).ffill()
    n = rets.shape[1]

    bull = optimize(rets, "max_sharpe", max_weight).weights
    neutral = np.ones(n) / n
    bear = np.zeros(n)
    table = {"BULL": bull, "NEUTRAL": neutral, "BEAR": bear}

    # shift(1): dùng tín hiệu NGÀY HÔM TRƯỚC -> loại look-ahead
    W = np.vstack([table[s] for s in state.shift(1).fillna("NEUTRAL")])
    sr = pd.Series((rets.values * W).sum(axis=1), index=rets.index)

    return RegimeReport(score=reg.score, state=reg.state, current=reg.current,
                        strategy_returns=sr, strategy_cumulative=(1 + sr).cumprod(),
                        macro_regime=reg.macro_regime)


def walk_forward(md: MarketData, train_days: int = 252, test_days: int = 63,
                 method: str = "max_sharpe", max_weight: float = 0.5) -> dict:
    """
    Tối ưu trên `train_days`, áp dụng cho `test_days` kế tiếp, lăn liên tục.
    Trả về chuỗi out-of-sample + so sánh với bản in-sample của bài gốc.
    """
    rets = md.returns.dropna()
    n = len(rets)
    if n < train_days + test_days:
        return {"error": "Không đủ dữ liệu cho walk-forward"}

    oos, marks = [], []
    for s in range(train_days, n - test_days + 1, test_days):
        w = optimize(rets.iloc[s - train_days:s], method, max_weight).weights
        blk = rets.iloc[s:s + test_days]
        oos.append(pd.Series(blk.values @ w, index=blk.index))
        marks.append(blk.index[0])

    oos_r = pd.concat(oos)
    ins_w = optimize(rets, method, max_weight).weights
    ins_r = pd.Series(rets.values @ ins_w, index=rets.index).loc[oos_r.index]

    def stat(r):
        sd = float(r.std())
        c = float((1 + r).prod() - 1) * 100
        return {"total_return": c,
                "sharpe": 0.0 if sd == 0 else float(r.mean() / sd * np.sqrt(TRADING_DAYS)),
                "volatility": sd * np.sqrt(TRADING_DAYS) * 100}

    return {
        "oos_returns": oos_r,
        "oos_cumulative": (1 + oos_r).cumprod(),
        "ins_returns": ins_r,
        "ins_cumulative": (1 + ins_r).cumprod(),
        "oos": stat(oos_r),
        "in_sample": stat(ins_r),
        "n_rebalances": len(marks),
        "rebalance_dates": marks,
    }
