"""
tabs/tab_quantamental.py — Giao diện Tab Quantamental AI & Dual-RAG.
Kết hợp Phân tích Kỹ thuật (T) + Cảm xúc Báo chí thời gian thực (S) + Sức khỏe Nội tại BCTC (F).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.contracts import MarketData
from core.data import load_market_data
from core.portfolio import build_portfolio, ma_crossover_signals
from core.vn_market import VNStockProvider
from quantamental.tab_ui import render_quantamental_tab


@st.cache_data(ttl=3600, show_spinner=False)
def _load_standalone_data(tickers: tuple[str, ...], start_str: str, market: str) -> MarketData | None:
    if market == "vn":
        provider = VNStockProvider()
        full_tickers = list(set(tickers) | {"VNINDEX"})
        prices = provider.get_prices(full_tickers, start_str)
        if prices is None or prices.empty:
            return None
        if "VNINDEX" not in prices.columns:
            prices["VNINDEX"] = prices.iloc[:, 0]
        return MarketData(
            prices=prices[[t for t in tickers if t in prices.columns]],
            benchmark=prices["VNINDEX"],
            factors=prices.copy(),
            tickers=[t for t in tickers if t in prices.columns],
            source=provider.source_in_use(),
            start=pd.Timestamp(start_str)
        )
    else:
        try:
            return load_market_data(list(tickers), start_str, use_sample=False)
        except Exception:
            return load_market_data(list(tickers), start_str, use_sample=True)


def render(md: MarketData | None = None, pf=None, sig_report=None, cfg: dict | None = None) -> None:
    """
    Render giao diện Tab Quantamental AI.
    Hỗ trợ cả chế độ tích hợp (nhận md, pf từ app chính) lẫn chế độ standalone (tự load dữ liệu).
    """
    cfg = cfg or {}

    # Nếu được gọi trong luồng Terminal chính (đã có sẵn md và pf)
    if md is not None and pf is not None:
        if sig_report is None:
            short_w = int(cfg.get("ma_short", 20))
            long_w = int(cfg.get("ma_long", 50))
            sig_report = ma_crossover_signals(md, short_w, long_w)
        render_quantamental_tab(md, pf, sig_report, cfg)
        return

    # Chế độ Standalone (khi truy cập từ Các Module Nâng Cao hoặc mục riêng)
    st.subheader("🧠 Quantamental AI & Dual-RAG Lab (Độc lập)")
    st.caption("Khám phá mô hình định lượng đa trụ cột: Kỹ thuật (T) + Cảm xúc Báo chí (S) + Nội tại BCTC (F).")

    with st.sidebar:
        st.markdown("### Thiết lập Quantamental AI")
        market_choice = st.radio("Thị trường", ["Mỹ (US)", "Việt Nam (VN)"], index=0, key="q_market_choice")
        market_code = "vn" if "Việt Nam" in market_choice else "us"
        default_tickers = "VCB, FPT, HPG" if market_code == "vn" else "AAPL, MSFT, NVDA"
        raw_tickers = st.text_input("Danh sách mã cổ phiếu", default_tickers, key="q_tickers_input")
        start_date = st.date_input("Ngày bắt đầu dữ liệu", pd.Timestamp("2023-01-01"), key="q_start_date")

    tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
    if not tickers:
        st.warning("Vui lòng nhập ít nhất một mã cổ phiếu.")
        return

    with st.spinner("Đang tải dữ liệu giá cổ phiếu..."):
        loaded_md = _load_standalone_data(tuple(tickers), str(start_date), market_code)

    if loaded_md is None or loaded_md.prices.empty:
        st.error("Không thể tải dữ liệu thị trường cho các mã đã chọn.")
        return

    n_tickers = len(loaded_md.tickers)
    equal_weights = [1.0 / max(n_tickers, 1)] * n_tickers
    standalone_pf = build_portfolio(loaded_md, equal_weights)
    standalone_sigs = ma_crossover_signals(loaded_md, 20, 50)
    standalone_cfg = {"market": market_code, "ma_short": 20, "ma_long": 50}

    render_quantamental_tab(loaded_md, standalone_pf, standalone_sigs, standalone_cfg)
