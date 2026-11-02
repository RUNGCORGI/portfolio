"""
core/contracts.py  —  BẢN HỢP ĐỒNG DỮ LIỆU GIỮA 4 THÀNH VIÊN
================================================================
*** FILE NÀY BỊ KHÓA. KHÔNG AI ĐƯỢC SỬA MỘT MÌNH. ***

LUẬT CHƠI
---------
1. Được THÊM field mới (phải có default) -> không vỡ code người khác.
2. CẤM đổi tên / đổi kiểu / xóa field đã có.
3. Muốn phá luật 2 -> họp nhóm, cả 4 người approve PR.

LUỒNG PHỤ THUỘC
---------------
    P1 (data.py)      -> MarketData
    P2 (portfolio.py) -> PortfolioResult      (cần MarketData)
    P3 (risk/factors) -> RiskReport, FactorReport   (chỉ ĐỌC, không ai phụ thuộc)
    P4 (optimizer/regime) -> OptimizeResult, RegimeReport (chỉ ĐỌC)

=> P3 và P4 chạy song song hoàn toàn, không đụng nhau.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# --------------------------------------------------------------------------
# P1  —  TẦNG DỮ LIỆU
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketData:
    """Đầu ra duy nhất của P1. Mọi người khác chỉ được nhận object này."""

    prices: pd.DataFrame          # index=Date, columns=tickers, giá đã adjust
    benchmark: pd.Series          # SPY close
    factors: pd.DataFrame         # columns: SPY, QQQ, IWM, ^VIX
    tickers: List[str]
    source: str = "yfinance"      # "yfinance" | "cache" | "sample"
    start: Optional[pd.Timestamp] = None

    @property
    def returns(self) -> pd.DataFrame:
        return self.prices.pct_change().dropna(how="all")

    @property
    def benchmark_returns(self) -> pd.Series:
        return self.benchmark.pct_change().dropna()

    @property
    def factor_returns(self) -> pd.DataFrame:
        return self.factors.pct_change().dropna(how="all")


# --------------------------------------------------------------------------
# P2  —  TẦNG DANH MỤC
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioResult:
    """Đầu ra của P2. P3 và P4 đều tiêu thụ object này."""

    weights: np.ndarray
    tickers: List[str]
    returns: pd.Series            # lợi suất ngày của danh mục (đã align benchmark)
    cumulative: pd.Series         # growth of $1
    benchmark_returns: pd.Series  # đã align cùng index với `returns`
    benchmark_cumulative: pd.Series
    metrics: Dict[str, float] = field(default_factory=dict)
    # metrics keys tối thiểu: total_return, volatility, sharpe,
    #                         benchmark_total_return, alpha
    gross_returns: Optional[pd.Series] = None   # trước phí (P2 - phát triển thêm)
    turnover: Optional[pd.Series] = None        # P2 - phát triển thêm
    cost_drag_pct: float = 0.0                  # P2 - phát triển thêm

    @property
    def allocation(self) -> pd.DataFrame:
        return pd.DataFrame({"Ticker": self.tickers, "Weight": self.weights})


# --------------------------------------------------------------------------
# P3  —  TẦNG RỦI RO & NHÂN TỐ  (read-only, không ai phụ thuộc)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskReport:
    corr: pd.DataFrame
    drawdown: pd.Series
    benchmark_drawdown: pd.Series
    rolling_vol: pd.Series
    benchmark_rolling_vol: pd.Series
    rolling_beta: pd.Series
    rolling_alpha: pd.Series
    tail: Dict[str, float] = field(default_factory=dict)
    # tail keys: var_95_hist, var_95_param, cvar_95, sortino,
    #            calmar, max_drawdown, max_dd_duration_days


@dataclass(frozen=True)
class FactorReport:
    betas: Dict[str, float]                 # SPY / QQQ / IWM / ^VIX
    capm_alpha: float
    rolling_momentum_corr: pd.Series
    ff_params: Optional[pd.DataFrame] = None   # P3 - phát triển thêm (Fama-French)
    ff_r2: Optional[float] = None
    treynor: float = 0.0
    beta_matrix: Optional[pd.DataFrame] = None


# --------------------------------------------------------------------------
# P4  —  TẦNG TỐI ƯU & REGIME  (read-only, không ai phụ thuộc)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizeResult:
    method: str                 # max_sharpe | min_variance | risk_parity | hrp
    weights: np.ndarray
    tickers: List[str]
    exp_return: float
    volatility: float
    sharpe: float
    success: bool = True

    @property
    def allocation(self) -> pd.DataFrame:
        return pd.DataFrame({"Ticker": self.tickers, "Weight": self.weights})


@dataclass(frozen=True)
class RegimeReport:
    score: pd.Series
    state: pd.Series            # BULL / NEUTRAL / BEAR
    current: str
    strategy_returns: Optional[pd.Series] = None
    strategy_cumulative: Optional[pd.Series] = None
    macro_regime: str = "RISK ON"


@dataclass(frozen=True)
class SignalReport:
    """Tín hiệu MA crossover — P2 sở hữu (đơn giản, bám tab Signals)."""

    per_ticker: pd.DataFrame
    portfolio_signal: pd.Series
    current: str                # BUY / SELL
