"""tabs/tab_factors.py — P3 trình bày tab này."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render(md, pf, fr, cfg):
    st.subheader("Factor Analytics · P3")

    # Metrics nhân tố: Beta từng nhân tố, CAPM Alpha và Tỷ số Treynor (myapp13)
    cols = st.columns(len(fr.betas) + 2)
    for col, (k, v) in zip(cols, fr.betas.items()):
        col.metric(f"{k} Beta", f"{v:.2f}")
    cols[-2].metric("CAPM Alpha (năm)", f"{fr.capm_alpha * 100:.2f}%")
    cols[-1].metric("Tỷ số Treynor", f"{fr.treynor:.2f}")

    st.plotly_chart(
        px.bar(x=list(fr.betas.keys()), y=list(fr.betas.values()),
               labels={"x": "Nhân tố", "y": "Beta"},
               title="Phơi nhiễm nhân tố của danh mục"),
        use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fr.rolling_momentum_corr.index,
                             y=fr.rolling_momentum_corr,
                             name="Momentum Exposure"))
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray")
    fig.update_layout(title="Phơi nhiễm nhân tố Momentum (trượt)",
                      yaxis_title="Tương quan")
    st.plotly_chart(fig, use_container_width=True)

    # Khớp bài gốc (myapp13): Rolling Beta Heatmap 60 phiên & Xếp hạng Beta
    if fr.beta_matrix is not None and not fr.beta_matrix.empty:
        clean_b = fr.beta_matrix.dropna(how="all")
        if not clean_b.empty:
            st.subheader("Bản đồ Nhiệt Rolling Beta Heatmap (Cửa sổ 60 Phiên Giao dịch)")
            st.plotly_chart(
                px.imshow(clean_b.T, aspect="auto",
                          title="Biến động Hệ số Beta 60 Phiên Từng Cổ phiếu so với SPY",
                          color_continuous_scale="Viridis"),
                use_container_width=True)

            st.subheader("Bảng Tổng kết Nhân tố và Xếp hạng Beta Tài sản")
            latest_b = clean_b.iloc[-1].sort_values(ascending=False)
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                st.write("**Top Tài sản Có Beta Cao Nhất (Nhạy cảm nhất):**")
                st.dataframe(latest_b.head(3).to_frame(name="Beta").round(2), use_container_width=True)
            with b_col2:
                st.write("**Top Tài sản Có Beta Thấp Nhất (Phòng thủ nhất):**")
                st.dataframe(latest_b.tail(3).to_frame(name="Beta").round(2), use_container_width=True)

    # ---- PHÁT TRIỂN THÊM CỦA P3 ----
    if fr.ff_params is not None:
        with st.expander("Phát triển thêm (P3) · Hồi quy đa nhân tố có thống kê",
                         expanded=True):
            st.caption(
                "Bài gốc chỉ lấy TƯƠNG QUAN trượt với một momentum giả lập từ SPY. "
                "Ở đây chạy OLS thật: hệ số, sai số chuẩn, t-stat và R². "
                "|t| > 2 nghĩa là phơi nhiễm có ý nghĩa thống kê."
            )
            df = fr.ff_params.copy()
            df["Có ý nghĩa"] = df["t_stat"].abs().gt(2).map({True: "Có", False: "Không"})
            st.dataframe(df.round(4), use_container_width=True)
            st.metric("R² của mô hình", f"{fr.ff_r2:.3f}",
                      help="Tỷ lệ biến động danh mục được giải thích bởi các nhân tố")

    st.subheader("Phân tích Lợi nhuận Chủ động (Active Return vs Benchmark)")
    active_diff = pf.returns - pf.benchmark_returns
    act_c1, act_c2 = st.columns(2)
    with act_c1:
        active_cum = active_diff.cumsum()
        st.plotly_chart(
            px.line(x=active_cum.index, y=active_cum, title="Active Return tích lũy"),
            use_container_width=True)
    with act_c2:
        st.plotly_chart(
            px.histogram(active_diff, nbins=50, title="Phân phối Tần suất Lợi nhuận Alpha Vượt trội"),
            use_container_width=True)
