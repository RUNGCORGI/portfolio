"""
app.py — Bộ điều phối giao diện. CẢ NHÓM sở hữu, sửa ít nhất có thể.
"""

from __future__ import annotations

import time
import pandas as pd
import streamlit as st

from core.data import health_check, load_market_data
from core.factors import build_factor_report
from core.portfolio import build_portfolio, ma_crossover_signals
from core.regime import detect_regime
from core.risk import build_risk_report
from core.vn_market import VNStockProvider
from core.contracts import MarketData

# Xử lý import an toàn
from tabs import (tab_backtest, tab_factors, tab_optimize, tab_portfolio,
                  tab_risk, tab_signals, tab_lab, tab_xray, tab_monitor)

try:
    from tabs import tab_quantamental
except ImportError:
    tab_quantamental = None
try:
    from tabs import tab_regime
except ImportError:
    tab_regime = None
try:
    from tabs import tab_scenario
except ImportError:
    tab_scenario = None
try:
    from tabs import tab_report
except ImportError:
    tab_report = None

st.set_page_config(page_title="Portfolio Terminal · Nhóm 4",
                   page_icon="📊", layout="wide")

MODES = {
    "Toàn bộ (bản tích hợp)": ["portfolio", "risk", "signals", "optimize",
                               "backtest", "factors", "quantamental", "regime", "scenario", "report"],
    "P1 · Dữ liệu & Hạ tầng": [],
    "P2 · Portfolio Engine": ["portfolio", "signals"],
    "P3 · Risk & Factor": ["risk", "factors"],
    "P4 · Optimizer, Regime & Scenario": ["optimize", "backtest", "regime", "scenario"],
    "P5 · Báo cáo": ["report"],
    "🧠 Quantamental AI": ["quantamental"],
}


# ===================================================================
# CACHE DỮ LIỆU MỸ
# ===================================================================
@st.cache_data(ttl=3600, show_spinner="Đang tải dữ liệu thị trường Mỹ...")
def get_data(tickers, start, use_sample: bool = False):
    return load_market_data(list(tickers), str(start), use_sample=use_sample, market="us")


# ===================================================================
# CACHE DỮ LIỆU VIỆT NAM
# ===================================================================
@st.cache_data(ttl=3600, show_spinner="Đang tải dữ liệu thị trường Việt Nam...")
def get_vn_data(tickers, start):
    """Hàm tạo MarketData cho thị trường Việt Nam để dùng chung cho mọi tab."""
    provider = VNStockProvider()
    # Thêm VNINDEX làm benchmark
    full_tickers = list(set(tickers) | {"VNINDEX"})
    prices = provider.get_prices(full_tickers, str(start))

    if prices is None or prices.empty:
        return None  # Trả về None để main() xử lý fallback

    # Đảm bảo có cột VNINDEX
    if "VNINDEX" not in prices.columns:
        prices["VNINDEX"] = prices.iloc[:, 0]

    # Tạo factors giả lập (dùng chính các mã làm factors để tránh lỗi)
    factors = prices.copy()

    return MarketData(
        prices=prices[[t for t in tickers if t in prices.columns]],
        benchmark=prices["VNINDEX"],
        factors=factors,
        tickers=[t for t in tickers if t in prices.columns],
        source=provider.source_in_use(),
        start=pd.Timestamp(start)
    )


def sidebar():
    st.sidebar.title("Portfolio Terminal")

    app_section = st.sidebar.radio(
        "Chọn khu vực làm việc",
        ["📊 Portfolio Terminal", "🔬 Các Module Nâng Cao", "🇻🇳 Giám sát (Monitor)"],
        index=0
    )
    st.sidebar.divider()

    if app_section == "🔬 Các Module Nâng Cao":
        return {"app_section": app_section}

    if app_section == "🇻🇳 Giám sát (Monitor)":
        return {"app_section": app_section}

    mode = st.sidebar.selectbox("Chế độ trình bày", list(MODES.keys()), key="main_mode")
    st.sidebar.divider()

    st.sidebar.subheader("Thị trường")
    market_choice = st.sidebar.radio("Chọn thị trường", ["Mỹ", "Việt Nam"], index=0)

    if market_choice == "Mỹ":
        default_tickers = "AAPL, MSFT, NVDA"
        market_code = "us"
    else:
        default_tickers = "VCB, FPT, HPG"
        market_code = "vn"

    st.sidebar.header("Danh mục")
    raw = st.sidebar.text_input("Mã cổ phiếu", default_tickers)
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    start = st.sidebar.date_input("Ngày bắt đầu", pd.Timestamp("2023-01-01"))

    st.sidebar.subheader("Tỷ trọng (%)")
    auto_normalize = st.sidebar.checkbox("Tự động chuẩn hóa về 100%", value=True)

    raw_weights = []
    default_w = int(100 / max(len(tickers), 1))
    for t in tickers:
        w = st.sidebar.slider(f"{t} (%)", min_value=0, max_value=100, value=default_w, step=1)
        raw_weights.append(w)

    total_w = sum(raw_weights)

    if auto_normalize:
        if total_w == 0:
            weights = [1.0 / len(tickers)] * len(tickers)
        else:
            weights = [w / total_w for w in raw_weights]
        st.sidebar.caption(f"Tổng: {total_w}% (Đã chuẩn hóa về 100%)")
    else:
        weights = [w / 100.0 for w in raw_weights]
        if total_w > 100:
            st.sidebar.warning(f"Tổng: {total_w}% (Đang dùng đòn bẩy)")
        elif total_w < 100:
            st.sidebar.info(f"Tổng: {total_w}% (Có tiền mặt chờ)")
        else:
            st.sidebar.success("Tổng: 100%")

    st.sidebar.subheader("Chi phí giao dịch · P2")
    cost_bps = st.sidebar.slider("Phí mỗi chiều (bps)", 0, 50, 10)
    rebalance = st.sidebar.selectbox("Tần suất tái cân bằng",
                                     ["Không", "Tháng", "Quý", "Năm"], index=2)

    st.sidebar.subheader("Tham số rủi ro · P3")
    window = st.sidebar.slider("Cửa sổ trượt (phiên)", 10, 120, 20, 5)

    st.sidebar.subheader("Tối ưu hóa · P4")
    opt_method = st.sidebar.selectbox(
        "Phương pháp", ["max_sharpe", "min_variance", "risk_parity",
                        "hrp", "equal_weight"])
    max_weight = st.sidebar.slider("Trần tỷ trọng 1 mã (%)", 20, 100, 50, 5) / 100.0
    n_mc = st.sidebar.select_slider("Số danh mục Monte Carlo",
                                    [1000, 2000, 5000, 10000], value=5000)
    train_days = st.sidebar.slider("Walk-forward · train (phiên)", 126, 504, 252, 21)
    test_days = st.sidebar.slider("Walk-forward · test (phiên)", 21, 126, 63, 21)

    return dict(app_section=app_section, mode=mode, tickers=tickers, start=start,
                weights=weights, cost_bps=cost_bps, rebalance=rebalance,
                window=window, opt_method=opt_method, max_weight=max_weight,
                n_mc=n_mc, train_days=train_days, test_days=test_days,
                ma_short=20, ma_long=50, market=market_code)


def render_data_panel(md, elapsed):
    st.subheader("Data & Infrastructure Engine · P1")
    hc = health_check(md)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số phiên", f"{hc['rows']:,}")
    c2.metric("Số mã", hc["tickers"])
    c3.metric("Thiếu dữ liệu", f"{hc['nan_pct']:.2f}%")
    c4.metric("Thời gian tải", f"{elapsed * 1000:.0f} ms")
    st.dataframe(md.prices.tail(10), use_container_width=True)
    st.line_chart(md.prices / md.prices.iloc[0])


def main():
    cfg = sidebar()

    # ===========================================================
    # 1. XỬ LÝ MODULE NÂNG CAO
    # ===========================================================
    if cfg.get("app_section") == "🔬 Các Module Nâng Cao":
        st.title("🔬 Các Module Nâng Cao")
        tabs_advanced = st.tabs(["Quant Lab", "X-Ray", "🧠 Quantamental AI"])
        with tabs_advanced[0]: tab_lab.render()
        with tabs_advanced[1]: tab_xray.render()
        with tabs_advanced[2]:
            if tab_quantamental is not None:
                tab_quantamental.render()
            else:
                st.info("Chưa nạp được module Quantamental AI.")
        return

    # ===========================================================
    # 2. XỬ LÝ GIÁM SÁT (TAB MONITOR)
    # ===========================================================
    if cfg.get("app_section") == "🇻🇳 Giám sát (Monitor)":
        st.title("Giám sát & Cảnh báo")
        tab_monitor.render()
        return

    # ===========================================================
    # 3. XỬ LÝ PHÂN TÍCH (MỸ HOẶC VIỆT NAM)
    # ===========================================================
    if not cfg.get("tickers"):
        st.warning("Hãy nhập ít nhất một mã cổ phiếu.")
        return

    t0 = time.perf_counter()
    md = None
    data_source_msg = ""

    # --- Lấy dữ liệu theo thị trường ---
    if cfg["market"] == "vn":
        md = get_vn_data(cfg["tickers"], cfg["start"])
        if md is None:
            st.error("❌ Không lấy được dữ liệu Việt Nam. "
                     "Vui lòng kiểm tra kết nối mạng hoặc thư viện vnstock.")
            st.stop()
        data_source_msg = f"Dữ liệu Việt Nam · Nguồn: {md.source}"
    else:
        try:
            md = get_data(tuple(cfg["tickers"]), cfg["start"], use_sample=False)
            data_source_msg = f"Dữ liệu Mỹ · Nguồn: {md.source}"
        except Exception as e:
            st.warning(f"⚠️ Không thể kết nối Yahoo Finance ({e}). Đang dùng dữ liệu dự phòng offline.")
            md = get_data(tuple(cfg["tickers"]), cfg["start"], use_sample=True)
            data_source_msg = f"Dữ liệu Mỹ · Nguồn: {md.source}"

    elapsed = time.perf_counter() - t0

    # --- Hiển thị tiêu đề ---
    st.title("Bloomberg-Style Portfolio Terminal")
    bench_name = "SPY" if cfg["market"] == "us" else "VNINDEX"
    st.caption(f"Chế độ: **{cfg['mode']}**  ·  Nhóm 4  ·  Benchmark: **{bench_name}**  ·  {data_source_msg}")

    # --- Xây dựng danh mục ---
    pf = build_portfolio(md, cfg["weights"], cfg["cost_bps"], cfg["rebalance"])
    wanted = MODES[cfg["mode"]]

    # --- Hiển thị P1 panel (nếu có) ---
    if cfg["mode"].startswith("P1") or cfg["mode"].startswith("Toàn bộ"):
        with st.expander("Tầng dữ liệu · P1", expanded=cfg["mode"].startswith("P1")):
            render_data_panel(md, elapsed)

    if not wanted:
        return

    # --- Lọc các tab hợp lệ ---
    names = {"portfolio": "Portfolio", "risk": "Risk", "signals": "Signals",
             "optimize": "Optimize", "backtest": "Backtest", "factors": "Factors",
             "quantamental": "Quantamental AI",
             "regime": "Regime", "scenario": "Scenario", "report": "Report"}

    valid_tabs = []
    for k in wanted:
        if k == "regime" and tab_regime is None: continue
        if k == "scenario" and tab_scenario is None: continue
        if k == "report" and tab_report is None: continue
        if k == "quantamental" and tab_quantamental is None: continue
        valid_tabs.append(k)

    tabs = st.tabs([names[k] for k in valid_tabs])

    # --- Tạo cfg cho các tab ---
    tab_cfg = dict(cfg)
    tab_cfg["market"] = cfg["market"]
    tab_cfg["benchmark_name"] = bench_name

    # --- Render từng tab ---
    for key, tab in zip(valid_tabs, tabs):
        with tab:
            if key == "portfolio":
                tab_portfolio.render(md, pf, tab_cfg)
            elif key == "signals":
                tab_signals.render(ma_crossover_signals(md, cfg["ma_short"], cfg["ma_long"]), tab_cfg)
            elif key == "risk":
                tab_risk.render(md, pf, build_risk_report(md, pf, cfg["window"]), tab_cfg)
            elif key == "factors":
                # Chỉ chạy Factor Analytics nếu là thị trường Mỹ
                if cfg["market"] == "us":
                    tab_factors.render(md, pf, build_factor_report(pf, md, cfg["window"]), tab_cfg)
                else:
                    st.info("💡 **Phân tích Nhân tố (Factors)** chỉ khả dụng cho thị trường Mỹ. "
                            "Dữ liệu hiện tại là Việt Nam nên không có các chỉ số như SPY, VIX để làm nhân tố.")
            elif key == "optimize":
                tab_optimize.render(md, pf, tab_cfg)
            elif key == "backtest":
                tab_backtest.render(md, pf, detect_regime(md), tab_cfg)
            elif key == "quantamental":
                sig_report = ma_crossover_signals(md, cfg["ma_short"], cfg["ma_long"])
                tab_quantamental.render(md, pf, sig_report, tab_cfg)
            elif key == "regime":
                tab_regime.render(md, pf, tab_cfg)
            elif key == "scenario":
                tab_scenario.render(md, pf, tab_cfg)
            elif key == "report":
                tab_report.render(md, pf, tab_cfg)


if __name__ == "__main__":
    main()