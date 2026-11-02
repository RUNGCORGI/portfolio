"""Lớp nối (adapter) giữa code sẵn có của bạn và gói nâng cấp.

TRIẾT LÝ THIẾT KẾ
-----------------
Gói nâng cấp này **không đọc, không sửa, không phụ thuộc** vào chữ ký hàm
trong ``core/data.py``, ``core/portfolio.py``... của bạn. Mọi module mới chỉ
nhận vào hai thứ phổ quát:

* ``prices``  – ``pd.DataFrame``, index là ngày, mỗi cột là một mã.
* ``returns`` – ``pd.DataFrame`` hoặc ``pd.Series`` lợi suất ngày.

Nhờ vậy nó ghép được vào bất kỳ cài đặt nào, kể cả khi hàm của bạn tên khác.

File này làm hai việc:

1. Cố gắng "mượn" hàm có sẵn của bạn nếu tìm thấy (``try_import``).
2. Nếu không tìm thấy thì dùng bản dự phòng nội bộ, nên gói luôn chạy được.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Mượn hàm từ code sẵn có (nếu có)
# --------------------------------------------------------------------------


def try_import(module: str, *names: str) -> Callable | None:
    """Thử lấy hàm đầu tiên tìm được trong một module. Không có thì trả None."""
    try:
        mod = importlib.import_module(module)
    except Exception:
        return None
    for name in names:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    return None


def _generate_synthetic_prices(
    tickers: Sequence[str], start: Any, end: Any = None, seed: int = 20260101
) -> pd.DataFrame:
    """Sinh chuỗi giá tổng hợp Geometric Brownian Motion có tương quan thực tế khi mất mạng."""
    rng = np.random.default_rng(seed)
    end_dt = pd.Timestamp.now() if end is None else pd.Timestamp(end)
    start_dt = pd.Timestamp(start) if start else pd.Timestamp("2020-01-01")
    idx = pd.bdate_range(start_dt, end_dt, name="Date")
    if len(idx) < 10:
        idx = pd.bdate_range(end_dt - pd.Timedelta(days=365), end_dt, name="Date")
    n = len(idx)
    dt = 1 / 252

    # Yếu tố thị trường chung
    market = rng.normal(0.08 * dt, 0.16 * np.sqrt(dt), n)
    cols = {}
    known_specs = {
        "AAPL": (1.05, 0.22, 130.0),
        "MSFT": (1.00, 0.21, 240.0),
        "NVDA": (1.70, 0.45, 15.0),
        "JNJ": (0.65, 0.14, 150.0),
        "XOM": (0.85, 0.25, 80.0),
        "KO": (0.60, 0.13, 55.0),
        "SPY": (1.00, 0.00, 380.0),
        "QQQ": (1.15, 0.10, 265.0),
        "IWM": (1.05, 0.12, 175.0),
        "GLD": (0.10, 0.15, 170.0),
        "TLT": (-0.25, 0.16, 100.0),
    }

    for tkr in tickers:
        up = str(tkr).upper().strip()
        if up == "^VIX":
            vix = np.zeros(n)
            vix[0] = 18.0
            for i in range(1, n):
                vix[i] = max(9.0, vix[i - 1] + 0.12 * (18.0 - vix[i - 1]) - 180.0 * market[i] + rng.normal(0, 0.8))
            cols[tkr] = vix
            continue

        beta, idio_vol, s0 = known_specs.get(up, (1.0, 0.20, 100.0))
        idio = rng.normal(0.0, idio_vol * np.sqrt(dt), n) if idio_vol > 0 else 0.0
        r = beta * market + idio
        cols[tkr] = s0 * np.exp(np.cumsum(r))

    df = pd.DataFrame(cols, index=idx)
    return clean_prices(df, tickers)


def load_prices(
    tickers: Sequence[str], start: Any, end: Any = None, fallback_seed: int = 20260101
) -> pd.DataFrame:
    """Lấy giá: ưu tiên dữ liệu thật trực tiếp từ Yahoo Finance REST API."""
    # 1. Thử lấy giá thật trực tiếp từ Yahoo Finance REST API v8 (cực nhanh, song song, không lỗi DNS crumb)
    try:
        from core.data import _raw_download
        clean_tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
        df = _raw_download(clean_tickers, str(start))
        if isinstance(df, pd.DataFrame) and not df.empty and len(df.columns) >= min(len(clean_tickers), 2):
            return clean_prices(df, tickers)
    except Exception:
        pass

    # 2. Thử các provider khác nếu có
    for fn_name in ("get_prices", "load_prices", "download_prices", "fetch_prices"):
        fn = try_import("core.data", fn_name)
        if fn is None:
            continue
        try:
            df = fn(tickers, start, end)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return clean_prices(df, tickers)
        except Exception:
            continue

    get_provider = try_import("core.data", "get_provider", "provider")
    if get_provider is not None:
        try:
            df = get_provider().get_prices(tickers, start, end)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return clean_prices(df, tickers)
        except Exception:
            pass

    # 3. Dự phòng dữ liệu tổng hợp GBM nếu mất hoàn toàn kết nối mạng
    try:
        return _generate_synthetic_prices(tickers, start, end, seed=fallback_seed)
    except Exception:
        pass

    raise ValueError(f"Không thể tải dữ liệu cho các mã: {tickers}.")

def clean_prices(df: pd.DataFrame, tickers: Sequence[str] | None = None) -> pd.DataFrame:
    """Chuẩn hoá khung giá: bỏ MultiIndex, sắp xếp, điền tiến, ép kiểu float."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if tickers:
        keep = [c for c in df.columns if str(c).upper() in {t.upper() for t in tickers}]
        if keep:
            df = df.loc[:, keep]
    df = df.sort_index().dropna(how="all").ffill().dropna(how="any")
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df.astype(float)


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Giá → lợi suất ngày. Dùng chung cho toàn bộ gói nâng cấp."""
    return prices.pct_change().dropna(how="all").fillna(0.0)


# --------------------------------------------------------------------------
# Chỉ số cơ bản (bản dự phòng, dùng khi không mượn được của bạn)
# --------------------------------------------------------------------------


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    r = pd.Series(returns).dropna()
    sd = r.std(ddof=1)
    if not np.isfinite(sd) or sd <= 1e-15:
        return 0.0
    return float((r.mean() - rf / TRADING_DAYS) / sd * np.sqrt(TRADING_DAYS))


def cagr(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return 0.0
    total = float((1.0 + r).prod())
    years = len(r) / TRADING_DAYS
    return total ** (1.0 / years) - 1.0 if total > 0 and years > 0 else -1.0


def max_drawdown(returns: pd.Series) -> float:
    eq = (1.0 + pd.Series(returns).dropna()).cumprod()
    return float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0


def ann_vol(returns: pd.Series) -> float:
    return float(pd.Series(returns).dropna().std(ddof=1) * np.sqrt(TRADING_DAYS))


def normalize(weights) -> np.ndarray:
    w = np.asarray(weights, dtype=float).ravel()
    s = w.sum()
    return w / s if np.isfinite(s) and abs(s) > 1e-15 else np.ones_like(w) / len(w)


# --------------------------------------------------------------------------
# Dữ liệu mô phỏng: cho phép demo và chạy test khi không có mạng
# --------------------------------------------------------------------------

_PROFILES = {
    "SPY": (1.00, 0.000, 0.02), "QQQ": (1.15, 0.020, 0.06), "IWM": (1.05, -0.010, 0.08),
    "^VIX": (-4.00, 0.000, 0.55), "AAPL": (1.10, 0.040, 0.14), "MSFT": (1.05, 0.050, 0.13),
    "NVDA": (1.60, 0.120, 0.30), "JNJ": (0.55, 0.005, 0.10), "XOM": (0.75, 0.020, 0.16),
    "GLD": (0.10, 0.015, 0.09), "TLT": (-0.25, -0.005, 0.10), "KO": (0.50, 0.010, 0.09),
}
_REGIMES = {0: (0.00075, 0.0065), 1: (0.00015, 0.0100), 2: (-0.00120, 0.0220)}
_TRANS = np.array([[0.985, 0.013, 0.002], [0.030, 0.945, 0.025], [0.010, 0.060, 0.930]])


def synthetic_prices(
    tickers: Sequence[str], start: Any = "2018-01-01", end: Any = None, seed: int = 20260101
) -> pd.DataFrame:
    """Sinh giá giả lập **có cấu trúc kinh tế thật**.

    Một nhân tố thị trường chung chuyển đổi giữa ba chế độ theo chuỗi Markov,
    cộng beta riêng và nhiễu riêng từng mã. Nhờ vậy dữ liệu có tương quan
    hợp lý, có drawdown thật, và mô hình HMM có cái để học — khác hẳn
    ``np.random.randn`` thuần tuý.
    """
    wanted = list(dict.fromkeys(str(t).upper() for t in tickers))
    idx = pd.bdate_range("2015-01-02", str(end) if end else "2026-06-30")
    rng = np.random.default_rng(seed)
    n = len(idx)

    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(3, p=_TRANS[states[t - 1]])
    market = rng.normal([_REGIMES[s][0] for s in states], [_REGIMES[s][1] for s in states])

    data = {}
    for tk in wanted:
        if tk in _PROFILES:
            beta, alpha, idio = _PROFILES[tk]
        else:
            h = sum(ord(c) for c in tk) % 1000
            beta, alpha, idio = 0.7 + h % 90 / 100, (h % 60 - 20) / 1000, 0.10 + h % 25 / 100
        r = alpha / TRADING_DAYS + beta * market + rng.normal(0, idio / np.sqrt(TRADING_DAYS), n)
        px0 = 350.0 if tk == "SPY" else (18.0 if tk == "^VIX" else 100.0)
        data[tk] = px0 * np.exp(np.cumsum(r))

    df = pd.DataFrame(data, index=idx)
    df = df.loc[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df.loc[df.index <= pd.Timestamp(end)]
    return df
