"""
core/fixtures.py  —  P1
Dữ liệu mẫu offline. Commit ngay NGÀY 1 để P2/P3/P4 code được mà không phải
chờ P1 làm xong tầng cache, và không cần internet.

    from core.fixtures import load_sample
    md = load_sample()          # -> MarketData
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from core.contracts import MarketData

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_PRICES = DATA_DIR / "sample_prices.csv"
SAMPLE_FACTORS = DATA_DIR / "sample_factors.csv"

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA"]
FACTOR_TICKERS = ["SPY", "QQQ", "IWM", "^VIX"]


def _gbm(n: int, mu: float, sigma: float, s0: float, rng) -> np.ndarray:
    """Geometric Brownian Motion — tạo chuỗi giá tổng hợp."""
    dt = 1 / 252
    shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n)
    return s0 * np.exp(np.cumsum(shocks))


def generate_sample(
    start: str = "2023-01-01",
    end: str = "2026-10-28",
    seed: int = 2024,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sinh dữ liệu tổng hợp có cấu trúc giống thật: tech beta cao, VIX nghịch SPY."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end, name="Date")
    n = len(idx)

    # Nhân tố thị trường chung -> tạo tương quan giữa các mã
    market = rng.normal(0.08 / 252, 0.16 / np.sqrt(252), n)

    specs = {
        "AAPL": (1.05, 0.22, 130.0),
        "MSFT": (1.00, 0.21, 240.0),
        "NVDA": (1.70, 0.45, 15.0),
        "SPY": (1.00, 0.00, 380.0),
        "QQQ": (1.15, 0.10, 265.0),
        "IWM": (1.05, 0.12, 175.0),
    }

    cols = {}
    for tkr, (beta, idio_vol, s0) in specs.items():
        idio = rng.normal(0.0, idio_vol / np.sqrt(252), n) if idio_vol > 0 else 0.0
        r = beta * market + idio
        cols[tkr] = s0 * np.exp(np.cumsum(r))

    # VIX: nghịch biến với thị trường, mean-reverting quanh 18
    vix = np.zeros(n)
    vix[0] = 18.0
    for i in range(1, n):
        vix[i] = max(9.0, vix[i - 1] + 0.12 * (18.0 - vix[i - 1])
                     - 180.0 * market[i] + rng.normal(0, 0.8))
    cols["^VIX"] = vix

    df = pd.DataFrame(cols, index=idx)
    prices = df[DEFAULT_TICKERS].copy()
    factors = df[FACTOR_TICKERS].copy()
    return prices, factors


def write_sample() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices, factors = generate_sample()
    prices.to_csv(SAMPLE_PRICES)
    factors.to_csv(SAMPLE_FACTORS)


def load_sample(tickers: List[str] | None = None) -> MarketData:
    """Trả về MarketData offline. Tự sinh file nếu chưa có."""
    if not SAMPLE_PRICES.exists() or not SAMPLE_FACTORS.exists():
        write_sample()

    prices = pd.read_csv(SAMPLE_PRICES, index_col=0, parse_dates=True)
    factors = pd.read_csv(SAMPLE_FACTORS, index_col=0, parse_dates=True)

    if tickers:
        keep = [t for t in tickers if t in prices.columns]
        if keep:
            prices = prices[keep]

    return MarketData(
        prices=prices,
        benchmark=factors["SPY"],
        factors=factors,
        tickers=list(prices.columns),
        source="sample",
        start=prices.index[0],
    )


if __name__ == "__main__":
    write_sample()
    print(f"Đã ghi dữ liệu mẫu vào {DATA_DIR}")
