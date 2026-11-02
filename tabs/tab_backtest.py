"""tabs/tab_backtest.py — P4 trình bày tab này."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.regime import switching_strategy, walk_forward


def render(md, pf, reg, cfg):
    st.subheader("Backtest Engine · P4")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pf.cumulative.index, y=pf.cumulative,
                             name="Danh mục"))
    fig.add_trace(go.Scatter(x=pf.benchmark_cumulative.index,
                             y=pf.benchmark_cumulative, name="SPY"))

    ss = switching_strategy(md, pf, reg, cfg["max_weight"])
    if ss.strategy_cumulative is not None:
        fig.add_trace(go.Scatter(x=ss.strategy_cumulative.index,
                                 y=ss.strategy_cumulative,
                                 name="Regime Switching"))
    fig.update_layout(title="Danh mục vs SPY vs Chiến lược Regime",
                      yaxis_title="Growth of $1")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Trạng thái hiện tại", reg.current)
    c2.metric("Phân bố trạng thái",
              " · ".join(f"{k}:{v}" for k, v in reg.state.value_counts().items()))
    st.line_chart(reg.score)

    # Khớp bài gốc (myapp13 trang 62 - Hình 09-08): SPY Price Regime Map
    reg_df = pd.DataFrame({"SPY Price": md.benchmark, "Regime": reg.state}).dropna()
    if not reg_df.empty:
        st.subheader("Bản đồ Phân loại Trạng thái Giá SPY (SPY Price Regime Map)")
        fig_map = px.scatter(
            reg_df, x=reg_df.index, y="SPY Price", color="Regime",
            color_discrete_map={"BULL": "#00CC96", "NEUTRAL": "#FFA15A", "BEAR": "#EF553B"},
            title="Trạng thái Thị trường theo Giá SPY (Bull, Neutral, Bear)")
        fig_map.update_traces(marker=dict(size=4))
        st.plotly_chart(fig_map, use_container_width=True)

    st.plotly_chart(px.histogram(pf.returns, nbins=60,
                                 title="Phân phối lợi suất ngày"),
                    use_container_width=True)

    # ---- PHÁT TRIỂN THÊM CỦA P4 ----
    with st.expander("Phát triển thêm (P4) · Walk-forward: bóc tách look-ahead bias",
                     expanded=True):
        st.caption(
            "Bài gốc tối ưu và đo hiệu suất trên CÙNG một tập dữ liệu — "
            "Sharpe 1.87 và lợi nhuận 576% vì thế là in-sample. "
            f"Ở đây: tối ưu trên {cfg['train_days']} phiên, áp dụng cho "
            f"{cfg['test_days']} phiên kế tiếp, lăn liên tục."
        )
        wf = walk_forward(md, cfg["train_days"], cfg["test_days"],
                          cfg["opt_method"], cfg["max_weight"])
        if "error" in wf:
            st.warning(wf["error"])
            return

        tbl = pd.DataFrame({
            "In-sample (như bài gốc)": wf["in_sample"],
            "Out-of-sample (walk-forward)": wf["oos"],
        }).T
        st.dataframe(tbl.round(2), use_container_width=True)

        gap = wf["in_sample"]["sharpe"] - wf["oos"]["sharpe"]
        st.metric("Sharpe bị thổi phồng do look-ahead", f"{gap:.2f}",
                  delta=f"{-gap / max(wf['in_sample']['sharpe'], 1e-9) * 100:.0f}%",
                  delta_color="inverse")

        f = go.Figure()
        f.add_trace(go.Scatter(x=wf["ins_cumulative"].index, y=wf["ins_cumulative"],
                               name="In-sample"))
        f.add_trace(go.Scatter(x=wf["oos_cumulative"].index, y=wf["oos_cumulative"],
                               name="Out-of-sample"))
        f.update_layout(title=f"In-sample vs Out-of-sample "
                              f"({wf['n_rebalances']} lần tái tối ưu)")
        st.plotly_chart(f, use_container_width=True)
