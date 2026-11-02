"""tabs/tab_risk.py — P3 trình bày tab này."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render(md, pf, rr, cfg):
    st.subheader("Risk Engine · P3")
    
    # Lấy tên Benchmark động (SPY hoặc VNINDEX)
    bench_name = cfg.get("benchmark_name", "SPY")

    t = rr.tail
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VaR 95% (ngày)", f"{t['var_95_hist']:.2f}%",
              help="Lịch sử: 5% số phiên tệ nhất lỗ sâu hơn mức này")
    c2.metric("CVaR 95%", f"{t['cvar_95']:.2f}%",
              help="Lỗ trung bình trong 5% phiên tệ nhất")
    c3.metric("Sortino", f"{t['sortino']:.2f}")
    c4.metric("Calmar", f"{t['calmar']:.2f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max Drawdown", f"{t['max_drawdown']:.2f}%")
    c2.metric("Thời gian dưới đỉnh", f"{t['max_dd_duration_days']} ngày")
    c3.metric("Skew", f"{t['skew']:.2f}")
    c4.metric("Kurtosis", f"{t['kurtosis']:.2f}")

    st.caption("Khối 8 chỉ số trên là PHÁT TRIỂN THÊM của P3 — bài gốc chỉ có "
               "volatility và drawdown.")

    st.plotly_chart(
        px.imshow(rr.corr, text_auto=".2f", aspect="auto",
                  color_continuous_scale="Blues", title="Ma trận tương quan"),
        use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rr.drawdown.index, y=rr.drawdown,
                             name="Danh mục", fill="tozeroy"))
    fig.add_trace(go.Scatter(x=rr.benchmark_drawdown.index,
                             y=rr.benchmark_drawdown, name=bench_name))
    fig.update_layout(title=f"So sánh Drawdown (vs {bench_name})", yaxis_title="Drawdown")
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rr.rolling_vol.index, y=rr.rolling_vol,
                             name="Danh mục"))
    fig.add_trace(go.Scatter(x=rr.benchmark_rolling_vol.index,
                             y=rr.benchmark_rolling_vol, name=bench_name))
    fig.update_layout(title=f"Biến động trượt {cfg['window']}D (quy năm)")
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rr.rolling_beta.index, y=rr.rolling_beta,
                             name=f"Rolling Beta ({cfg['window']}D)"))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig.update_layout(title=f"Rolling Beta vs {bench_name}", yaxis_title="Beta")
    st.plotly_chart(fig, use_container_width=True)