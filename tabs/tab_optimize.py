"""tabs/tab_optimize.py — P4 trình bày tab này."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.optimizer import compare_methods, monte_carlo, optimize


def render(md, pf, cfg):
    st.subheader("Optimization Engine · P4")

    rets = md.returns
    opt = optimize(rets, cfg["opt_method"], cfg["max_weight"])

    # --- HÀNG 1: CÁC CHỈ SỐ TỐI ƯU ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe tối ưu", f"{opt.sharpe:.2f}",
              delta=f"{opt.sharpe - pf.metrics['sharpe']:.2f} vs hiện tại")
    c2.metric("Lợi nhuận kỳ vọng", f"{opt.exp_return * 100:.2f}%")
    c3.metric("Biến động", f"{opt.volatility * 100:.2f}%")
    c4.metric("Hội tụ", "Thành công" if opt.success else "Thất bại")

    # --- HÀNG 2: BIỂU ĐỒ PHÂN BỔ & MONTE CARLO ---
    # Chỉnh lại tỷ lệ cột: Cột trái 1 phần, Cột phải 1.5 phần để biểu đồ MC rộng rãi hơn
    left, right = st.columns([1, 1.5])
    
    with left:
        fig_pie = px.pie(opt.allocation, names="Ticker", values="Weight",
                         title=f"Phân bổ tối ưu · {opt.method}", hole=0.45)
        fig_pie.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with right:
        mc = monte_carlo(rets, cfg["n_mc"])
        
        # Khởi tạo biểu đồ bằng go.Figure để dễ tinh chỉnh layout hơn px.scatter
        fig = go.Figure()

        # 1. Đám mây điểm Monte Carlo
        fig.add_trace(go.Scatter(
            x=mc["Volatility"], y=mc["Return"],
            mode="markers",
            marker=dict(
                size=4,
                color=mc["Sharpe"],
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="Sharpe", thickness=15, len=0.8)
            ),
            name="Danh mục ngẫu nhiên"
        ))

        # 2. Điểm tối ưu (Ngôi sao đỏ)
        fig.add_trace(go.Scatter(
            x=[opt.volatility], y=[opt.exp_return],
            mode="markers", name="Tối ưu (Max Sharpe)",
            marker=dict(size=18, symbol="star", color="red")
        ))

        # 3. Các mã đơn lẻ (Hình thoi)
        mu = rets.mean() * 252
        sd = rets.std() * np.sqrt(252)
        fig.add_trace(go.Scatter(
            x=sd.values, y=mu.values, 
            mode="markers+text",
            text=list(rets.columns), 
            textposition="top center",
            name="Từng mã",
            marker=dict(size=12, symbol="diamond", color="red")
        ))

        # Tinh chỉnh Layout để không bị díu chữ
        fig.update_layout(
            title=f"Đường biên hiệu quả Monte Carlo ({cfg['n_mc']:,} danh mục)",
            xaxis_title="Volatility (Rủi ro)",
            yaxis_title="Return (Lợi nhuận)",
            legend=dict(
                orientation="h", # Đưa legend nằm ngang
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=60, b=40, l=40, r=20), # Tăng margin trên để chứa legend
            height=500, # Tăng chiều cao
            plot_bgcolor='white'
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        st.plotly_chart(fig, use_container_width=True)

    # --- HÀNG 3: BACKTEST ENGINE (BẢNG SO SÁNH) ---
    st.markdown("---")
    st.markdown("### Backtest Engine: So sánh Danh mục Tối ưu vs SPY")
    
    # Tính toán đường vốn của danh mục tối ưu dựa trên tỷ trọng tối ưu
    opt_weights = opt.allocation["Weight"].values
    opt_returns = rets.dot(opt_weights)
    opt_cum = (1 + opt_returns).cumprod()
    
    # Đường vốn của SPY (lấy từ biến pf đã tính trước đó)
    spy_cum = pf.benchmark_cumulative
    
    # Gộp thành DataFrame để hiển thị
    backtest_df = pd.DataFrame({
        "Portfolio": opt_cum,
        "SPY": spy_cum
    }).dropna()
    
    # Hiển thị bảng 10 dòng cuối
    st.dataframe(
        backtest_df.tail(10).style.format("{:.4f}"),
        use_container_width=True
    )
    
    # Vẽ thêm biểu đồ đường để trực quan hóa
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["Portfolio"], 
                                name="Portfolio (Max Sharpe)", line=dict(width=2)))
    fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["SPY"], 
                                name="SPY Benchmark", line=dict(dash='dash')))
    fig_bt.update_layout(
        title="Tăng trưởng của $1 (Danh mục tối ưu vs SPY)",
        yaxis_title="Giá trị",
        height=400,
        margin=dict(t=40, b=40, l=40, r=20)
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # ---- PHÁT TRIỂN THÊM CỦA P4 ----
    with st.expander("Phát triển thêm (P4) · So sánh 5 phương pháp phân bổ",
                     expanded=False):
        st.caption(
            "Tác giả bài gốc tự nhận: Sharpe optimization không ổn định và "
            "thường cho tỷ trọng vô lý (dồn vào 1-2 mã, chạm trần 50%). "
            "Bảng dưới đặt Max Sharpe cạnh Min-Variance, Risk Parity, HRP "
            "và Equal Weight để thấy rõ."
        )
        df = compare_methods(rets, cfg["max_weight"])
        st.dataframe(
            df.style.format(precision=2).background_gradient(
                subset=["Sharpe"], cmap="RdYlGn"),
            use_container_width=True)
        st.caption("Cột **Max w %** là tỷ trọng lớn nhất — càng cao càng tập trung "
                   "rủi ro vào một mã.")