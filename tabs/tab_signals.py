"""tabs/tab_signals.py — P2 trình bày tab này."""
from __future__ import annotations

import streamlit as st


def render(sig, cfg):
    st.subheader("Signal Engine · P2")
    c1, c2 = st.columns([1, 3])
    c1.metric("Tín hiệu danh mục", sig.current)
    c2.caption(f"MA{cfg['ma_short']} vs MA{cfg['ma_long']} cho từng mã, "
               "tín hiệu danh mục = trung bình các tín hiệu thành phần. "
               "Khác bài gốc: KHÔNG tải lại dữ liệu từng mã, dùng chung "
               "MarketData của P1.")
    st.line_chart(sig.portfolio_signal)
    st.dataframe(sig.per_ticker.tail(20), use_container_width=True)
