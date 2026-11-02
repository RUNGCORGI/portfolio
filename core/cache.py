"""
core/cache.py  —  P1  (PHÁT TRIỂN THÊM #1)

Vấn đề của bài gốc: myapp13.py gọi yf.download() tới 4-5 lần cho cùng một
khoảng thời gian (1 lần cho danh mục, 1 lần cho SPY, 1 lần cho factor,
1 lần/mã trong vòng lặp signal). Mỗi lần rerun Streamlit là tải lại từ đầu.

Giải pháp: cache 2 tầng
    Tầng 1 — st.cache_data (RAM, trong 1 phiên)
    Tầng 2 — file trên disk, gắn docker volume ./data -> /app/data (xuyên phiên,
             xuyên lần restart container, và là PHAO CỨU SINH khi mất wifi)
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
DEFAULT_TTL = 24 * 3600  # 1 ngày


def _key(tickers: List[str], start: str, tag: str) -> str:
    raw = f"{tag}|{','.join(sorted(tickers))}|{start}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.csv"


def _read(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, index_col=0, parse_dates=True)


def _write(df: pd.DataFrame, p: Path) -> None:
    df.to_csv(p)


def is_fresh(p: Path, ttl: int = DEFAULT_TTL) -> bool:
    return p.exists() and (time.time() - p.stat().st_mtime) < ttl


def cached_download(
    tickers: List[str],
    start: str,
    fetch_fn: Callable[[List[str], str], pd.DataFrame],
    tag: str = "px",
    ttl: int = DEFAULT_TTL,
    force_offline: bool = False,
) -> tuple[pd.DataFrame, str]:
    """
    Trả về (dataframe, nguồn). nguồn ∈ {"cache", "yfinance", "cache-stale"}.

    Thứ tự ưu tiên:
        1. cache còn hạn          -> dùng luôn, 0 request mạng
        2. gọi fetch_fn           -> ghi cache
        3. mạng chết + có cache cũ -> dùng cache cũ, cảnh báo (DEMO KHÔNG CHẾT)
    """
    p = _path(_key(tickers, start, tag))

    if is_fresh(p, ttl) or (force_offline and p.exists()):
        return _read(p), "cache"

    try:
        df = fetch_fn(tickers, start)
        if df is None or df.empty:
            raise ValueError("fetch trả về rỗng")
        _write(df, p)
        return df, "yfinance"
    except Exception:
        if p.exists():
            return _read(p), "cache-stale"
        raise


def clear_cache() -> int:
    if not CACHE_DIR.exists():
        return 0
    files = list(CACHE_DIR.glob("*.csv"))
    for f in files:
        f.unlink()
    return len(files)
