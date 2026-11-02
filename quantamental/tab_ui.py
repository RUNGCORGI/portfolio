"""
quantamental/tab_ui.py — Giao diện Tab 7: 🧠 Quantamental AI & Dual-RAG.
Trình bày phân tích đa chiều: Tín hiệu kỹ thuật (T), Cảm xúc báo chí (S) và Nội tại BCTC (F).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from quantamental.financials_engine import fetch_financial_metrics
from quantamental.models import QuantamentalReport, QuantamentalSignal
from quantamental.news_engine import fetch_realtime_news
from quantamental.rag_fundamental import analyze_fundamentals
from quantamental.rag_sentiment import analyze_sentiment
from quantamental.truth_table import classify_quantamental_signal


def generate_quantamental_report(md, sig_report=None, market: str | None = None) -> QuantamentalReport:
    """
    Chạy toàn bộ pipeline Quantamental cho các mã trong danh mục song song đa luồng.
    """
    import concurrent.futures

    signals: dict[str, QuantamentalSignal] = {}
    patterns_count: dict[str, int] = {}
    summary_rows = []

    tickers = list(md.prices.columns)
    # Lấy tín hiệu kỹ thuật T từ sig_report (hoặc tính nhanh từ MA20 vs MA50)
    per_ticker_sig = getattr(sig_report, "per_ticker", None)

    is_vn_market = (market == "vn")

    def _process_single(ticker: str):
        # 1. Tín hiệu Kỹ thuật T (0 hoặc 1)
        px_series = md.prices[ticker].dropna() if ticker in md.prices.columns else pd.Series()
        if len(px_series) >= 50:
            ma20 = float(px_series.rolling(20).mean().iloc[-1])
            ma50 = float(px_series.rolling(50).mean().iloc[-1])
            last_px = float(px_series.iloc[-1])
            # Tín hiệu Kỹ thuật T=1: Hoặc MA20 cắt lên MA50 (Golden Cross), hoặc Giá bứt phá vượt trên cả MA20 và MA50
            t_flag = 1 if ((ma20 > ma50) or (last_px > ma20 and last_px > ma50)) else 0
        elif len(px_series) >= 5:
            t_flag = 1 if px_series.iloc[-1] > px_series.iloc[0] else 0
        else:
            t_flag = 1

        # 2. RAG Cảm xúc Báo chí S (0 hoặc 1)
        news_items = fetch_realtime_news(ticker, max_items=100, is_vn=is_vn_market)
        sentiment_res = analyze_sentiment(ticker, news_items)

        # 3. RAG Nội tại BCTC F (0 hoặc 1)
        fin_metrics = fetch_financial_metrics(ticker, is_vn=is_vn_market)
        fundamental_res = analyze_fundamentals(ticker, fin_metrics)

        # 4. Ánh xạ vào Ma trận Chân lý [T, S, F]
        q_sig = classify_quantamental_signal(ticker, t_flag, sentiment_res, fundamental_res)
        return ticker, t_flag, sentiment_res, fundamental_res, q_sig

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max(len(tickers), 1), 8)) as executor:
        results = list(executor.map(_process_single, tickers))

    for ticker, t_flag, sentiment_res, fundamental_res, q_sig in results:
        signals[ticker] = q_sig
        p_name = q_sig.pattern_name
        patterns_count[p_name] = patterns_count.get(p_name, 0) + 1

        summary_rows.append({
            "Mã CP": ticker,
            "Trạng thái [T, S, F]": q_sig.pattern_code,
            "Phân loại Mẫu hình": q_sig.pattern_name,
            "Kỹ thuật (T)": "✅ Uptrend (1)" if t_flag == 1 else "❌ Downtrend (0)",
            "Cảm xúc (S)": f"{'🔥 Sôi động (1)' if sentiment_res.flag == 1 else '❄️ Bình thường (0)'} ({sentiment_res.score:.2f})",
            "Nội tại BCTC (F)": f"{'💎 Vững chắc (1)' if fundamental_res.flag == 1 else '⚠️ Bất ổn (0)'} ({fundamental_res.score:.2f})",
            "Khuyến nghị Hành động": q_sig.recommendation,
            "Tác động Tỷ trọng": q_sig.weight_action,
        })

    summary_df = pd.DataFrame(summary_rows)
    return QuantamentalReport(
        per_ticker=signals,
        summary_df=summary_df,
        patterns_count=patterns_count
    )


def simulate_ab_testing(md, pf, report):
    """
    Mô phỏng thực nghiệm Đối chứng (A/B Testing):
    - Danh mục A: Truyền thống (chưa lọc bẫy, dùng tỷ trọng hiện tại của người dùng).
    - Danh mục B: Quantamental AI (tự động loại bỏ mã [1,0,0], [0,0,0], dồn vốn cho [1,1,1], [1,0,1]).
    """
    tickers = list(md.prices.columns)
    rets = md.returns.fillna(0.0)
    w_trad = np.array(pf.weights, dtype=float) if pf is not None and hasattr(pf, "weights") else np.ones(len(tickers))
    if len(w_trad) == len(tickers) and w_trad.sum() > 0:
        w_trad = w_trad / w_trad.sum()
    else:
        w_trad = np.ones(len(tickers)) / len(tickers)

    w_quant = w_trad.copy()
    blacklisted = []
    boosted = []
    for i, t in enumerate(tickers):
        sig = report.per_ticker.get(t)
        if sig and sig.pattern_code in ["[1, 0, 0]", "[0, 0, 0]", "[0, 1, 0]"]:
            w_quant[i] = 0.0
            blacklisted.append(t)
        elif sig and sig.pattern_code in ["[1, 1, 1]", "[1, 0, 1]"]:
            boosted.append(t)

    if w_quant.sum() > 0:
        w_quant = w_quant / w_quant.sum()
    else:
        w_quant = w_trad.copy()

    r_trad = rets.dot(w_trad)
    r_quant = rets.dot(w_quant)

    cum_trad = (1 + r_trad).cumprod()
    cum_quant = (1 + r_quant).cumprod()

    def _calc_metrics(r, cum):
        if len(cum) == 0:
            return {"total_return": 0.0, "volatility": 0.0, "sharpe": 0.0, "max_dd": 0.0}
        sd = float(r.std())
        tot = float((cum.iloc[-1] - 1) * 100) if not cum.empty else 0.0
        vol = float(sd * np.sqrt(252) * 100)
        sharpe = 0.0 if sd == 0 else float(r.mean() / sd * np.sqrt(252))
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = float(dd.min() * 100) if not dd.empty else 0.0
        return {"total_return": tot, "volatility": vol, "sharpe": sharpe, "max_dd": max_dd}

    m_trad = _calc_metrics(r_trad, cum_trad)
    m_quant = _calc_metrics(r_quant, cum_quant)

    return {
        "cum_trad": cum_trad,
        "cum_quant": cum_quant,
        "m_trad": m_trad,
        "m_quant": m_quant,
        "w_trad": w_trad,
        "w_quant": w_quant,
        "blacklisted": blacklisted,
        "boosted": boosted,
        "tickers": tickers
    }


def render_quantamental_tab(md, pf, sig_report=None, cfg=None):
    """
    Hàm hiển thị chính của Tab 7.
    """
    st.subheader("🧠 Quantamental AI Engine · Phân Tích Đa Trụ Cột [T, S, F]")
    st.caption(
        "Kết hợp **Tín hiệu Kỹ thuật (T)** + **RAG Chấm điểm Cảm xúc Báo chí thời gian thực (S)** + "
        "**RAG Phân tích Dòng tiền & Sức khỏe BCTC (F)** để loại bỏ bẫy kéo xả / thao túng giá."
    )

    market = (cfg or {}).get("market", "us")
    with st.spinner("Đang truy vấn API tin tức thời gian thực và phân tích số liệu BCTC..."):
        report = generate_quantamental_report(md, sig_report, market=market)

    # 1. Các thẻ Metric tổng quan
    c1, c2, c3, c4 = st.columns(4)
    bull_traps = sum(1 for s in report.per_ticker.values() if s.pattern_code == "[1, 0, 0]")
    gold_signals = sum(1 for s in report.per_ticker.values() if s.pattern_code == "[1, 1, 1]")
    hidden_gems = sum(1 for s in report.per_ticker.values() if s.pattern_code == "[1, 0, 1]")
    fomo_signals = sum(1 for s in report.per_ticker.values() if s.pattern_code == "[1, 1, 0]")

    c1.metric("⭐ Tín hiệu Vàng [1,1,1]", f"{gold_signals} mã",
              help="Đồng thuận Kỹ thuật + Truyền thông + Nội tại vững chắc")
    c2.metric("💎 Ngọc Trong Đá [1,0,1]", f"{hidden_gems} mã",
              help="Cơ bản tốt, giá chớm tăng, thị trường chưa FOMO")
    c3.metric("🚨 Bẫy Tăng Giá [1,0,0]", f"{bull_traps} mã",
              delta="- Nguy hiểm" if bull_traps > 0 else "An toàn",
              delta_color="inverse", help="Kéo xả rỗng ruột / Thao túng kỹ thuật")
    c4.metric("⚠️ FOMO Đầu Cơ [1,1,0]", f"{fomo_signals} mã",
              delta="Thận trọng" if fomo_signals > 0 else "0",
              delta_color="off", help="Báo chí thổi phồng nhưng BCTC không có nền tảng")

    st.divider()

    # 2. Biểu đồ Radar So sánh Điểm Đa Trụ Cột (T, S, F)
    st.markdown("#### 📡 Bản đồ Radar Năng lực Cổ phiếu (Quantamental Triad)")
    fig_radar = go.Figure()
    categories = ["Kỹ thuật (T)", "Cảm xúc Báo chí (S)", "Nội tại BCTC (F)", "Kỹ thuật (T)"]

    for ticker, sig in report.per_ticker.items():
        t_val = 1.0 if sig.T == 1 else 0.2
        s_val = sig.sentiment.score
        f_val = sig.fundamental.score
        fig_radar.add_trace(go.Scatterpolar(
            r=[t_val, s_val, f_val, t_val],
            theta=categories,
            fill="toself",
            name=f"{ticker} ({sig.pattern_code})",
            line=dict(color=sig.badge_color)
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.0])
        ),
        showlegend=True,
        title="So sánh Điểm Định lượng Kỹ thuật vs Cảm xúc vs Nội tại",
        height=420
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # 3. Bảng Ma trận Chân lý Tổng hợp
    st.markdown("#### 📋 Bảng Phân Loại Ma Trận Quyết Định [T, S, F]")
    st.dataframe(report.summary_df, use_container_width=True)

    st.divider()

    # 4. Chi tiết RAG Drill-down cho từng mã
    st.markdown("#### 🔍 Chi Tiết Bằng Chứng Phân Tích RAG Từng Cổ Phiếu")
    for ticker, sig in report.per_ticker.items():
        with st.expander(f"**{ticker}** · {sig.pattern_name} · Mã trạng thái: `{sig.pattern_code}`", expanded=True):
            col_left, col_mid, col_right = st.columns([1.2, 1.4, 1.4])

            # Cột 1: Đánh giá tổng quan & Kỹ thuật
            with col_left:
                st.markdown(f"##### 🎯 Tín hiệu & Hành động")
                st.markdown(f"**Trạng thái:** `{sig.pattern_code}`")
                st.markdown(f"**Khuyến nghị:** {sig.recommendation}")
                st.markdown(f"**Đề xuất tỷ trọng:** `{sig.weight_action}`")
                st.info(sig.description)

            # Cột 2: RAG Cảm xúc Báo chí
            with col_mid:
                st.markdown(f"##### 📰 RAG Cảm xúc Báo chí (Flag S = {sig.S})")
                st.write(f"**Điểm cảm xúc:** `{sig.sentiment.score:.2f} / 1.00` — {sig.sentiment.label}")
                st.caption(sig.sentiment.reasoning)
                if sig.sentiment.key_themes:
                    for th in sig.sentiment.key_themes:
                        st.markdown(f"- 📌 *{th}*")
                n_items = len(sig.sentiment.news_items)
                st.markdown(f"**Bài báo trích xuất ({n_items} bài phân tích):**")
                for news in sig.sentiment.news_items[:5]:
                    st.markdown(f"- [{news.title}]({news.link}) *({news.source} - {news.pub_date})*")
                if n_items > 5:
                    with st.expander(f"📖 Xem toàn bộ {n_items} bài báo trích xuất..."):
                        for news in sig.sentiment.news_items[5:]:
                            st.markdown(f"- [{news.title}]({news.link}) *({news.source} - {news.pub_date})*")

            # Cột 3: RAG Nội tại BCTC
            with col_right:
                m = sig.fundamental.metrics
                st.markdown(f"##### 📊 RAG Nội tại BCTC (Flag F = {sig.F})")
                st.write(f"**Điểm nội tại:** `{sig.fundamental.score:.2f} / 1.00` — {sig.fundamental.label}")
                st.caption(sig.fundamental.reasoning)
                
                # Bảng chỉ số tài chính
                st.markdown(
                    f"- **Doanh thu:** `{m.revenue:,.1f}` {m.currency} *(YoY: `{m.revenue_growth_yoy:+.1f}%`)*\n"
                    f"- **Lợi nhuận ròng:** `{m.net_income:,.1f}` {m.currency} *(Biên LN: `{m.net_margin:.1f}%`)*\n"
                    f"- **Dòng tiền HĐKD (OCF):** `{m.ocf:,.1f}` {m.currency}\n"
                    f"- **Định giá:** P/E = `{m.pe_ratio:.1f}x` | EPS = `{m.eps:,.2f}` | Nợ/VCSH = `{m.debt_to_equity:.2f}`"
                )

                if sig.fundamental.criteria_passed:
                    st.markdown("**Điểm mạnh tài chính:**")
                    for p in sig.fundamental.criteria_passed[:2]:
                        st.markdown(f"- ✅ <small>{p}</small>", unsafe_allow_html=True)
                if sig.fundamental.criteria_failed:
                    st.markdown("**Điểm lưu ý:**")
                    for f in sig.fundamental.criteria_failed[:2]:
                        st.markdown(f"- ⚠️ <small>{f}</small>", unsafe_allow_html=True)

    # 5. Thực nghiệm Đối chứng Danh mục (A/B Testing: Traditional vs Quantamental Overlay)
    st.divider()
    st.markdown("#### ⚡ Thử Nghiệm Đối Chứng Danh Mục (A/B Testing)")
    st.caption(
        "So sánh hiệu quả giữa **Danh mục Truyền thống** (chỉ dùng PTKT như bài báo gốc, dễ dính bẫy Bull Trap) "
        "và **Danh mục Quantamental AI** (tự động loại bỏ các mã [1,0,0], [0,0,0] và dồn vốn sang các mã chất lượng)."
    )

    ab = simulate_ab_testing(md, pf, report)

    # Hiển thị metrics so sánh
    m1, m2, m3, m4 = st.columns(4)
    c_trad, c_quant = ab["m_trad"], ab["m_quant"]

    delta_ret = c_quant["total_return"] - c_trad["total_return"]
    delta_sharpe = c_quant["sharpe"] - c_trad["sharpe"]
    delta_vol = c_quant["volatility"] - c_trad["volatility"]
    delta_dd = c_quant["max_dd"] - c_trad["max_dd"]

    m1.metric("Lợi nhuận Tích lũy (Quant vs Trad)", f"{c_quant['total_return']:.1f}%",
              delta=f"{delta_ret:+.1f}% vs Truyền thống")
    m2.metric("Tỷ số Sharpe", f"{c_quant['sharpe']:.2f}",
              delta=f"{delta_sharpe:+.2f} vs Truyền thống")
    m3.metric("Biến động danh mục (Vol)", f"{c_quant['volatility']:.1f}%",
              delta=f"{delta_vol:+.1f}%", delta_color="inverse")
    m4.metric("Sụt giảm tối đa (Max DD)", f"{c_quant['max_dd']:.1f}%",
              delta=f"{delta_dd:+.1f}%", delta_color="normal" if delta_dd >= 0 else "inverse")

    # Vẽ biểu đồ Plotly so sánh tăng trưởng tài sản
    fig_ab = go.Figure()
    fig_ab.add_trace(go.Scatter(
        x=ab["cum_quant"].index, y=ab["cum_quant"],
        name="🟢 Danh mục Quantamental AI (Đã lọc bẫy)",
        line=dict(color="#00CC96", width=2.5)
    ))
    fig_ab.add_trace(go.Scatter(
        x=ab["cum_trad"].index, y=ab["cum_trad"],
        name="🔴 Danh mục Truyền thống (Chưa lọc bẫy)",
        line=dict(color="#EF553B", width=1.8, dash="dot")
    ))
    fig_ab.update_layout(
        title="Tăng Trưởng Tài Sản: Danh mục Quantamental AI vs. Danh mục Truyền Thống (Growth of $1)",
        xaxis_title="Thời gian",
        yaxis_title="Giá trị Tích lũy ($1)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=450
    )
    st.plotly_chart(fig_ab, use_container_width=True)

    # Bảng phân bổ tỷ trọng trước và sau
    st.markdown("##### ⚖️ So Sánh Tỷ Trọng Phân Bổ Vốn")
    w_df = pd.DataFrame({
        "Mã CP": ab["tickers"],
        "Tỷ trọng Truyền thống": [f"{w*100:.1f}%" for w in ab["w_trad"]],
        "Tỷ trọng Quantamental AI": [f"{w*100:.1f}%" for w in ab["w_quant"]],
        "Tác động Can thiệp": [
            "🚨 Blacklist (Ép về 0%)" if t in ab["blacklisted"]
            else ("⭐ Ưu tiên Tăng vốn" if t in ab["boosted"] else "Giữ nguyên")
            for t in ab["tickers"]
        ]
    })
    st.dataframe(w_df, use_container_width=True)

