"""tabs/tab_portfolio.py — P2 trình bày tab này."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.sizing import build_orders, summarize, to_csv


def render(md, pf, cfg):
    st.subheader("Portfolio Engine · P2")

    # Lấy tên Benchmark động (SPY hoặc VNINDEX)
    bench_name = cfg.get("benchmark_name", "SPY")

    m = pf.metrics
    
    # HÀNG 1: Lợi nhuận & Trạng thái
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Lợi nhuận danh mục", f"{m['total_return']:.2f}%")
    c2.metric(f"Lợi nhuận {bench_name}", f"{m['benchmark_total_return']:.2f}%")
    c3.metric(f"Alpha vs {bench_name}", f"{m['alpha']:.2f}%")
    c4.metric("Sharpe", f"{m['sharpe']:.2f}")
    
    # Phần Macro Regime chỉ hiển thị cho Mỹ
    if cfg.get("market") == "us":
        spy = md.benchmark
        spy_ma200 = spy.rolling(200).mean()
        qqq = md.factors["QQQ"] if ("QQQ" in md.factors.columns) else md.benchmark.copy()
        qqq_ma200 = qqq.rolling(200).mean()
        vix = md.factors["^VIX"] if ("^VIX" in md.factors.columns) else pd.Series(18.0, index=spy.index)
        risk_on = (spy > spy_ma200) & (qqq > qqq_ma200) & (vix < 20)
        last_risk = risk_on.dropna()
        macro_regime = "RISK ON" if (len(last_risk) > 0 and bool(last_risk.iloc[-1])) else "RISK OFF"
        c5.metric("Trạng thái Vĩ mô", macro_regime, delta="Tăng trưởng" if macro_regime == "RISK ON" else "Phòng thủ")
    else:
        c5.metric("Trạng thái Vĩ mô", "N/A")

    # HÀNG 2: Các chỉ số rủi ro
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", f"{m['cagr']:.2f}%")
    c2.metric("Biến động năm", f"{m['volatility']:.2f}%")
    c3.metric("Hao hụt do phí", f"-{pf.cost_drag_pct:.2f}%",
              help="Chênh lệch tổng lợi nhuận trước và sau chi phí giao dịch")
    
    c4.metric(f"Rolling Beta ({bench_name})", f"{m.get('rolling_beta_latest', 1.0):.2f}", 
              help=f"Beta cuộn 20 ngày (mức độ nhạy cảm hiện tại so với {bench_name})")
    c5.metric("Market Beta", f"{m.get('market_beta', 1.0):.2f}",
              help=f"Beta toàn thời gian của danh mục so với {bench_name}")

    left, right = st.columns([1, 2])
    with left:
        st.plotly_chart(
            px.pie(pf.allocation, names="Ticker", values="Weight",
                   title="Phân bổ hiện tại", hole=0.45),
            use_container_width=True)
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pf.cumulative.index, y=pf.cumulative,
                                 name="Danh mục"))
        fig.add_trace(go.Scatter(x=pf.benchmark_cumulative.index,
                                 y=pf.benchmark_cumulative, name=bench_name))
        if pf.gross_returns is not None:
            gc = (1 + pf.gross_returns).cumprod()
            fig.add_trace(go.Scatter(x=gc.index, y=gc, name="Trước phí",
                                     line=dict(dash="dot")))
        fig.update_layout(title="Tăng trưởng của $1", yaxis_title="Growth of $1")
        st.plotly_chart(fig, use_container_width=True)

    # ---- PHÁT TRIỂN THÊM CỦA P2 ----
    if pf.turnover is not None and pf.turnover.sum() > 0:
        with st.expander("Phát triển thêm (P2) · Chi phí giao dịch & tái cân bằng",
                         expanded=False):
            st.caption(
                "Bài gốc bỏ qua hoàn toàn phí giao dịch và turnover. "
                "Đường chấm ở biểu đồ trên là hiệu suất trước phí."
            )
            a, b = st.columns(2)
            a.metric("Số lần tái cân bằng", int((pf.turnover > 0).sum()))
            b.metric("Turnover trung bình/lần",
                     f"{pf.turnover[pf.turnover > 0].mean() * 100:.1f}%"
                     if (pf.turnover > 0).any() else "0%")
            st.bar_chart(pf.turnover[pf.turnover > 0])

    # ---- PHÁT TRIỂN THÊM #2 CỦA P2: tỷ trọng -> phiếu lệnh thực thi ----
    with st.expander("Phát triển thêm (P2) · Từ tỷ trọng % sang phiếu lệnh thực tế",
                     expanded=False):
        st.caption(
            "Bài gốc dừng ở tỷ trọng %. Nhà đầu tư thật không mua được "
            "'33,3% NVDA' — họ mua số cổ phiếu nguyên, theo lô, với vốn cụ thể. "
            "Làm tròn lô tạo sai lệch so với tỷ trọng lý thuyết."
        )
        c1, c2 = st.columns(2)
        capital = c1.number_input("Vốn đầu tư ($)", 1_000, 10_000_000,
                                  100_000, step=1_000)
        market = c2.selectbox("Thị trường (lô giao dịch)",
                              ["US", "HOSE", "HNX"],
                              help="Mỹ: lô 1 CP · HOSE/HNX: lô 100 CP")

        orders = build_orders(md.prices.iloc[-1], pf.weights, capital,
                              market, cfg["cost_bps"])
        s = summarize(orders, capital)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Đã giải ngân", f"${s['invested']:,.0f}")
        m2.metric("Tiền mặt dư", f"${s['cash_left']:,.0f}",
                  f"{s['cash_pct']:.2f}%")
        m3.metric("Phí mua vào", f"${s['fee']:,.0f}")
        m4.metric("Sai lệch tỷ trọng lớn nhất", f"{s['max_drift_pct']:.2f}%",
                  help="Do làm tròn theo lô giao dịch")

        st.dataframe(orders.style.format({
            "Giá": "{:.2f}", "Tỷ trọng mục tiêu %": "{:.2f}",
            "Giá trị": "{:,.0f}", "Phí": "{:,.0f}",
            "Tỷ trọng thực tế %": "{:.2f}", "Sai lệch %": "{:+.2f}",
        }), use_container_width=True)

        st.download_button("Tải phiếu lệnh (CSV)", to_csv(orders),
                           file_name="phieu-lenh.csv", mime="text/csv")