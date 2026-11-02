"""Ứng dụng demo độc lập cho gói nâng cấp Quant Lab.

File này CHỈ để chạy thử mà không đụng vào ``app.py`` thật của bạn.
Sau khi hài lòng, tích hợp theo ``docs/huong-dan-nang-cap.md``.

Chạy:  streamlit run app_demo.py
"""

import streamlit as st

st.set_page_config(page_title="Quant Lab — Gói nâng cấp", layout="wide")

from tabs import tab_lab, tab_monitor, tab_xray

t1, t2, t3 = st.tabs(["Quant Lab", "X-Ray & Báo cáo", "Giám sát"])

with t1:
    tab_lab.render()
with t2:
    tab_xray.render()
with t3:
    tab_monitor.render()
