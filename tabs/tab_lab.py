"""Tab "Quant Lab" — giao diện cho toàn bộ gói nâng cấp.


CÁCH TÍCH HỢP (chỉ thêm 3 dòng vào ``app.py``, không sửa gì khác)::


    from tabs import tab_lab
    ...
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([..., "Quant Lab"])
    with tab6:
        tab_lab.render()


Tab này **tự lo dữ liệu của chính nó**, không phụ thuộc biến toàn cục hay
session state của bạn, nên không thể làm hỏng các tab sẵn có. Nếu muốn dùng
chung dữ liệu, truyền vào: ``tab_lab.render(prices=df_gia_cua_ban)``.
"""


from __future__ import annotations


import numpy as np
import pandas as pd
import streamlit as st


try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:  # pragma: no cover
    HAS_PLOTLY = False


from core import alloc_lab, hmm_regime, stress, validation
from core import upgrade_api as api
from core import wf_backtest as wf


DEFAULT_TICKERS = "AAPL, MSFT, NVDA, JNJ, XOM, GLD, TLT"
FACTORS = ["SPY", "QQQ", "IWM", "^VIX"]




# --------------------------------------------------------------------------
# Nạp dữ liệu (có cache của Streamlit)
# --------------------------------------------------------------------------




@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _load(tickers: tuple[str, ...], start: str, end: str | None) -> pd.DataFrame:
    return api.load_prices(list(tickers), start, end)




def _pct(x: float) -> str:
    return f"{x * 100:,.2f}%"




# --------------------------------------------------------------------------
# Điểm vào
# --------------------------------------------------------------------------




def render(prices: pd.DataFrame | None = None) -> None:
    """Vẽ toàn bộ tab Quant Lab."""
    st.header("Quant Lab — kiểm định, phân bổ nâng cao và kịch bản")
    st.caption(
        "Khu vực này bổ sung những phần mà một backtest cơ bản còn thiếu: "
        "quy trình walk-forward trung thực, chi phí giao dịch, kiểm định thống kê, "
        "nhận diện chế độ thị trường bằng mô hình xác suất và mô phỏng kịch bản."
    )


    with st.sidebar:
        st.markdown("### Quant Lab")
        raw = st.text_input("Danh mục mã", DEFAULT_TICKERS, key="lab_tickers")
        start = st.date_input("Từ ngày", pd.Timestamp("2018-01-01"), key="lab_start")
        robust = st.checkbox(
            "Dùng ước lượng co ngót", True,
            help="Ledoit–Wolf cho hiệp phương sai và James–Stein cho kỳ vọng lợi suất.",
            key="lab_robust",
        )
        max_w = st.slider("Trần trọng số mỗi mã", 0.1, 1.0, 0.4, 0.05, key="lab_maxw")
        cost_bps = st.slider(
            "Chi phí giao dịch (bps/chiều)", 0, 100, 10, 5,
            help="Gồm phí môi giới, spread và trượt giá.", key="lab_cost",
        )


    tickers = tuple(t.strip().upper() for t in raw.split(",") if t.strip())
    if len(tickers) < 2:
        st.warning("Cần ít nhất 2 mã để phân tích danh mục.")
        return


    with st.spinner("Đang chuẩn bị dữ liệu..."):
        if prices is None:
            prices = _load(tuple(sorted(set(tickers) | set(FACTORS))), str(start), None)
        prices = api.clean_prices(prices)


    cols = [t for t in tickers if t in prices.columns]
    if len(cols) < 2:
        st.error("Không lấy được đủ dữ liệu cho các mã đã nhập.")
        return


    rets = api.to_returns(prices[cols])
    bench = api.to_returns(prices[["SPY"]])["SPY"] if "SPY" in prices.columns else None
    factor_rets = api.to_returns(prices[[f for f in FACTORS if f in prices.columns]])


    cfg = wf.WalkForwardConfig(
        robust=robust,
        max_weight=max_w,
        costs=wf.CostModel(commission_bps=cost_bps * 0.5, spread_bps=cost_bps * 0.2,
                           slippage_bps=cost_bps * 0.3),
    )


    s1, s2, s3, s4, s5 = st.tabs([
        "Phân bổ vốn", "Walk-forward & chi phí", "Kiểm định",
        "Chế độ thị trường", "Kịch bản & sức chịu đựng",
    ])


    with s1:
        _section_alloc(rets, robust, max_w)
    with s2:
        _section_walkforward(rets, cfg, bench)
    with s3:
        _section_validation(rets, cfg)
    with s4:
        _section_regime(rets, bench, prices)
    with s5:
        _section_stress(rets, factor_rets, bench)




# --------------------------------------------------------------------------
# 1. Phân bổ vốn
# --------------------------------------------------------------------------




def _section_alloc(rets: pd.DataFrame, robust: bool, max_w: float) -> None:
    st.subheader("So sánh sáu phương pháp phân bổ")
    st.caption(
        "Max Sharpe không phải chân lý. Khi kỳ vọng lợi suất được ước lượng từ "
        "quá khứ, nó rất nhiễu, nên lời giải thường dồn vốn vào mã may mắn nhất."
    )


    views_raw = st.text_input(
        "Quan điểm Black–Litterman (tuỳ chọn)", "",
        placeholder="NVDA=0.25, JNJ=0.05",
        help="Kỳ vọng lợi suất năm của bạn cho từng mã. Để trống nếu không có quan điểm.",
    )
    views = _parse_views(views_raw)
    conf = st.slider("Mức độ tự tin vào quan điểm", 0.1, 0.95, 0.6, 0.05) if views else 0.6


    mu, cov = alloc_lab.estimate(rets, robust=robust, views=views, confidence=conf)
    table, results = alloc_lab.compare_methods(mu, cov, max_weight=max_w)


    st.dataframe(
        table.style.format({
            "LN kỳ vọng": "{:.2%}", "Biến động": "{:.2%}", "Sharpe": "{:.2f}",
            "Số vị thế hiệu dụng": "{:.2f}", "Trọng số lớn nhất": "{:.1%}",
        }),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "**Số vị thế hiệu dụng** là 1 chia tổng bình phương trọng số. Với 7 mã, "
        "giá trị gần 7 nghĩa là phân bổ đều; gần 1 nghĩa là danh mục chỉ thực sự "
        "nắm một mã dù trên giấy vẫn đủ tên."
    )


    wdf = pd.DataFrame({k: v.weights for k, v in results.items()})
    if HAS_PLOTLY:
        fig = px.bar(wdf.T, barmode="stack", title="Trọng số theo phương pháp",
                     labels={"value": "Trọng số", "index": "Phương pháp"})
        fig.update_layout(height=420, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(wdf.style.format("{:.1%}"), use_container_width=True)


    st.markdown("#### Kiểm định độ ổn định của trọng số")
    st.caption(
        "Tối ưu lại nhiều lần trên các đoạn lịch sử con. Trọng số càng dao động "
        "mạnh thì phương pháp càng kém tin cậy khi đem ra dùng thật."
    )
    if st.button("Chạy kiểm định ổn định", key="lab_stab"):
        with st.spinner("Đang tối ưu lặp lại trên nhiều đoạn dữ liệu..."):
            rep = alloc_lab.stability_report(rets, n_trials=24, max_weight=max_w)
        st.dataframe(
            rep.style.format({
                "Độ lệch chuẩn TB của trọng số": "{:.4f}",
                "Biên độ trọng số TB": "{:.4f}",
            }),
            use_container_width=True, hide_index=True,
        )
        worst = rep.loc[rep["Độ lệch chuẩn TB của trọng số"].idxmax(), "Phương pháp"]
        best = rep.loc[rep["Độ lệch chuẩn TB của trọng số"].idxmin(), "Phương pháp"]
        st.info(f"Bất ổn nhất: **{worst}**. Ổn định nhất: **{best}**.")




# --------------------------------------------------------------------------
# 2. Walk-forward
# --------------------------------------------------------------------------




def _section_walkforward(rets, cfg, bench) -> None:
    st.subheader("Walk-forward: backtest theo đúng quy trình thật")
    st.caption(
        "Trọng số của mỗi kỳ chỉ được ước lượng từ dữ liệu *trước* kỳ đó, "
        "có thêm khoảng cấm giữa hai đoạn để chặn rò rỉ, và mỗi lần tái cân bằng "
        "đều bị tính phí."
    )


    c1, c2, c3 = st.columns(3)
    cfg.lookback = c1.number_input("Cửa sổ ước lượng (phiên)", 60, 756, cfg.lookback, 21)
    cfg.step = c2.number_input("Chu kỳ tái cân bằng (phiên)", 5, 126, cfg.step, 1)
    cfg.embargo = c3.number_input("Khoảng cấm (phiên)", 0, 21, cfg.embargo, 1)


    if len(rets) <= cfg.lookback + cfg.embargo + cfg.step:
        st.warning("Lịch sử chưa đủ dài cho cấu hình này. Hãy giảm cửa sổ ước lượng.")
        return


    with st.spinner("Đang chạy walk-forward cho từng phương pháp..."):
        table, results = wf.compare_walk_forward(rets, config=cfg)
    if table.empty:
        st.error("Không chạy được walk-forward với cấu hình hiện tại.")
        return


    st.dataframe(
        table.style.format({
            "CAGR (sau phí)": "{:.2%}", "Sharpe (sau phí)": "{:.2f}",
            "Sharpe (trước phí)": "{:.2f}", "Phí bào mòn/năm": "{:.2%}",
            "Sụt giảm tối đa": "{:.2%}", "Vòng quay/năm": "{:.2f}",
        }),
        use_container_width=True, hide_index=True,
    )


    if HAS_PLOTLY:
        fig = go.Figure()
        for name, res in results.items():
            fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity, name=name))
        if bench is not None:
            b = (1 + bench.reindex(next(iter(results.values())).net.index).fillna(0)).cumprod()
            fig.add_trace(go.Scatter(x=b.index, y=b, name="SPY", line=dict(dash="dash")))
        fig.update_layout(title="Tăng trưởng của 1 đồng vốn (sau phí)",
                          yaxis_title="Giá trị", height=430)
        st.plotly_chart(fig, use_container_width=True)


    st.markdown("#### Sharpe bốc hơi bao nhiêu khi làm đúng quy trình?")
    method = st.selectbox("Phương pháp", list(alloc_lab.METHODS), index=2, key="lab_isoos")
    try:
        cmp = wf.in_sample_vs_out_of_sample(rets, method, cfg)
        st.dataframe(
            cmp.style.format({
                "Sharpe in-sample": "{:.2f}", "Sharpe out-of-sample (sau phí)": "{:.2f}",
                "Mức suy giảm": "{:.2f}", "Tỷ lệ còn lại": "{:.1%}",
            }),
            use_container_width=True, hide_index=True,
        )
        keep = float(cmp["Tỷ lệ còn lại"].iloc[0])
        st.info(
            f"Chỉ **{keep:.0%}** Sharpe còn sống sót sau khi loại bỏ nhìn trộm "
            "tương lai và trừ chi phí giao dịch. Đây là khoảng cách giữa một "
            "backtest đẹp và một chiến lược dùng được."
        )
    except Exception as exc:
        st.warning(f"Chưa so sánh được: {exc}")


    st.markdown("#### Chiến lược chịu được mức phí tới đâu?")
    pick = st.selectbox("Chọn phương pháp để phân tích phí", list(results), key="lab_cost_m")
    res = results[pick]
    sens = wf.cost_sensitivity(res)
    be = wf.breakeven_cost_bps(res, bench)
    c1, c2 = st.columns([2, 1])
    with c1:
        if HAS_PLOTLY:
            fig = px.line(sens, x="Phí (bps/chiều)", y="Sharpe", markers=True,
                          title="Sharpe suy giảm theo mức phí")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(sens, use_container_width=True, hide_index=True)
    with c2:
        st.metric("Phí hoà vốn", f"{be:,.0f} bps" if np.isfinite(be) else "Không giới hạn")
        st.metric("Vòng quay/năm", f"{res.summary()['Vòng quay/năm']:.2f}")
        st.caption("Vượt mức phí hoà vốn thì phần vượt trội so với benchmark biến mất.")




# --------------------------------------------------------------------------
# 3. Kiểm định
# --------------------------------------------------------------------------




def _section_validation(rets, cfg) -> None:
    st.subheader("Kết quả này là kỹ năng hay may mắn?")
    st.caption(
        "Nếu thử đủ nhiều cấu hình, kiểu gì cũng có một cái cho Sharpe đẹp. "
        "Các chỉ số dưới đây phạt đúng vào hiện tượng đó."
    )


    method = st.selectbox("Chiến lược cần kiểm định", list(alloc_lab.METHODS),
                          index=4, key="lab_val_m")
    trials = st.slider(
        "Số cấu hình đã thử", 1, 100, 20,
        help="Khai báo trung thực số phương án bạn đã chạy trước khi chọn cái này.",
    )


    try:
        res = wf.run_walk_forward(rets, method, cfg)
    except Exception as exc:
        st.warning(f"Chưa chạy được walk-forward: {exc}")
        return


    rep = validation.validation_report(res.net, n_trials=trials, asset_returns=rets)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(rep.style.format({"Giá trị": "{:.3f}"}),
                     use_container_width=True, hide_index=True)
    with c2:
        dsr = float(rep.loc[rep["Chỉ tiêu"].str.startswith("DSR"), "Giá trị"].iloc[0])
        st.metric("Deflated Sharpe Ratio", f"{dsr:.3f}")
        if dsr > 0.95:
            st.success("Vượt ngưỡng 0,95: kết quả khó quy cho may mắn thuần tuý.")
        else:
            st.warning(
                "Dưới ngưỡng 0,95: chưa đủ bằng chứng thống kê để khẳng định "
                "chiến lược có alpha thật. Cần thêm dữ liệu hoặc giảm số lần thử."
            )


    st.markdown("#### Khoảng tin cậy bootstrap theo khối")
    st.caption(
        "Lấy mẫu theo từng đoạn liên tiếp để giữ tự tương quan và cụm biến động. "
        "Khoảng tin cậy rộng nghĩa là con số điểm không nên được tin tuyệt đối."
    )
    if st.button("Chạy bootstrap", key="lab_boot"):
        with st.spinner("Đang lấy mẫu lặp..."):
            ci = validation.confidence_intervals(res.net, n_boot=800)
            boot = validation.block_bootstrap(res.net, n_boot=800)
        st.dataframe(ci.style.format({
            "Giá trị thực tế": "{:.3f}", "Cận dưới": "{:.3f}",
            "Cận trên": "{:.3f}", "Độ lệch chuẩn": "{:.3f}",
        }), use_container_width=True, hide_index=True)
        if HAS_PLOTLY and not boot.empty:
            fig = px.histogram(boot, x="sharpe", nbins=50,
                               title="Phân phối bootstrap của Sharpe")
            fig.add_vline(x=api.sharpe(res.net), line_dash="dash", line_color="red")
            fig.add_vline(x=0, line_dash="dot", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)




# --------------------------------------------------------------------------
# 4. Chế độ thị trường
# --------------------------------------------------------------------------




def _section_regime(rets, bench, prices) -> None:
    st.subheader("Chế độ thị trường bằng mô hình Markov ẩn")
    st.caption(
        "Khác với cách cộng điểm tín hiệu, mô hình xác suất cho biết **độ chắc "
        "chắn** của chế độ hiện tại và khả năng chuyển sang chế độ khác."
    )
    if bench is None:
        st.warning("Cần dữ liệu SPY để nhận diện chế độ thị trường.")
        return


    with st.spinner("Đang khớp mô hình HMM..."):
        model = hmm_regime.fit_regimes(bench)


    c1, c2, c3 = st.columns(3)
    c1.metric("Chế độ hiện tại", model.current)
    c2.metric("Độ chắc chắn", _pct(model.confidence))
    c3.metric("Hội tụ", "Có" if model.converged else "Chưa")


    if HAS_PLOTLY:
        fig = go.Figure()
        for col in model.probabilities.columns:
            fig.add_trace(go.Scatter(
                x=model.probabilities.index, y=model.probabilities[col],
                name=col, stackgroup="one",
            ))
        fig.update_layout(title="Xác suất thuộc từng chế độ theo thời gian",
                          yaxis_title="Xác suất", height=380)
        st.plotly_chart(fig, use_container_width=True)


    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Ma trận chuyển trạng thái**")
        st.dataframe(model.transition.style.format("{:.3f}"), use_container_width=True)
        st.caption("Đường chéo lớn nghĩa là chế độ có tính dai dẳng, không nhảy loạn.")
    with c2:
        st.markdown("**Đặc tính từng chế độ**")
        info = pd.DataFrame({
            "Số phiên kỳ vọng": model.expected_duration(),
            "Tỷ trọng thời gian": model.stationary(),
        })
        st.dataframe(info.style.format({
            "Số phiên kỳ vọng": "{:.1f}", "Tỷ trọng thời gian": "{:.1%}",
        }), use_container_width=True)


    k = rets.shape[1]
    eq = pd.Series(rets.to_numpy() @ np.repeat(1 / k, k), index=rets.index)
    st.markdown("**Hiệu quả danh mục chia đều theo từng chế độ**")
    st.dataframe(
        hmm_regime.regime_statistics(eq, model.states).style.format({
            "Tỷ trọng thời gian": "{:.1%}", "LN năm hoá": "{:.2%}",
            "Biến động năm hoá": "{:.2%}", "Tỷ lệ phiên lãi": "{:.1%}",
        }),
        use_container_width=True, hide_index=True,
    )


    st.markdown("#### Chiến lược chuyển chế độ")
    st.caption(
        "Chế độ của hôm nay chỉ dùng để giao dịch từ hôm sau — nếu thiếu độ trễ "
        "này thì toàn bộ kết quả là nhìn trộm tương lai."
    )
    strat = hmm_regime.regime_switching_backtest(rets, model.states, np.repeat(1 / k, k))
    c1, c2 = st.columns(2)
    c1.metric("Sharpe khi chuyển chế độ", f"{api.sharpe(strat):.2f}")
    c2.metric("Sharpe khi mua và giữ", f"{api.sharpe(eq):.2f}")
    if HAS_PLOTLY:
        df = pd.DataFrame({
            "Chuyển chế độ": (1 + strat).cumprod(),
            "Mua và giữ": (1 + eq.reindex(strat.index).fillna(0)).cumprod(),
        })
        st.plotly_chart(px.line(df, title="Chuyển chế độ so với mua và giữ"),
                        use_container_width=True)




# --------------------------------------------------------------------------
# 5. Kịch bản
# --------------------------------------------------------------------------




def _section_stress(rets, factor_rets, bench) -> None:
    st.subheader("Phòng kịch bản")
    st.caption(
        "Biểu đồ lịch sử không trả lời được câu hỏi quan trọng nhất của nhà đầu tư: "
        "nếu thị trường sập thì danh mục mất bao nhiêu."
    )


    k = rets.shape[1]
    pr = pd.Series(rets.to_numpy() @ np.repeat(1 / k, k), index=rets.index)


    if not factor_rets.empty:
        st.markdown("#### Cú sốc tuỳ chỉnh")
        c1, c2, c3 = st.columns(3)
        s_spy = c1.slider("Thị trường (SPY)", -0.40, 0.40, -0.20, 0.05)
        s_vix = c2.slider("Biến động (VIX)", -0.50, 2.00, 0.80, 0.10)
        s_qqq = c3.slider("Công nghệ (QQQ)", -0.40, 0.40, -0.25, 0.05)


        shocks = {"SPY": s_spy, "^VIX": s_vix, "QQQ": s_qqq}
        shocks = {kk: v for kk, v in shocks.items() if kk in factor_rets.columns}
        out = stress.scenario_shock(pr, factor_rets, shocks)


        c1, c2 = st.columns([1, 2])
        c1.metric("Tác động ước tính lên danh mục", _pct(out["Tác động tổng"]))
        detail = pd.DataFrame([
            {"Nhân tố": f, "Cú sốc": shocks[f], "Beta": out["Beta sử dụng"].get(f, 0.0),
             "Đóng góp": v}
            for f, v in out["Theo từng nhân tố"].items()
        ])
        c2.dataframe(detail.style.format({
            "Cú sốc": "{:.0%}", "Beta": "{:.2f}", "Đóng góp": "{:.2%}",
        }), use_container_width=True, hide_index=True)


        grid = stress.scenario_grid(pr, factor_rets, "SPY")
        if HAS_PLOTLY:
            fig = px.line(grid, x="Cú sốc nhân tố", y="Tác động danh mục (ước tính)",
                          markers=True, title="Đường cong tổn thất theo cú sốc thị trường")
            fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)


    st.markdown("#### Phát lại các giai đoạn khủng hoảng")
    hist = stress.historical_stress(pr, bench)
    if hist.empty:
        st.info("Dữ liệu không phủ các giai đoạn khủng hoảng định sẵn. "
                "Dưới đây là các cửa sổ tệ nhất do chính dữ liệu chỉ ra.")
        st.dataframe(stress.worst_windows(pr), use_container_width=True, hide_index=True)
    else:
        fmt = {c: "{:.2%}" for c in hist.columns if hist[c].dtype.kind == "f"}
        st.dataframe(hist.style.format(fmt), use_container_width=True, hide_index=True)


    st.markdown("#### Mô phỏng Monte Carlo một năm tới")
    c1, c2 = st.columns(2)
    fat = c1.checkbox("Dùng phân phối đuôi dày (Student-t)", True,
                      help="Giả định phân phối chuẩn đánh giá thấp xác suất sụp đổ.")
    n_paths = c2.select_slider("Số đường mô phỏng", [1000, 2000, 5000, 10000], 2000)


    with st.spinner("Đang mô phỏng..."):
        paths = stress.monte_carlo_paths(pr, n_paths=int(n_paths), fat_tails=fat)
        summ = stress.monte_carlo_summary(paths)


    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(summ.style.format({"Giá trị": "{:.2%}"}),
                     use_container_width=True, hide_index=True)
    with c2:
        if HAS_PLOTLY:
            fig = px.histogram(paths, x="Lợi nhuận", nbins=60,
                               title="Phân phối lợi nhuận một năm tới")
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(xaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)




def _parse_views(raw: str) -> dict[str, float] | None:
    """Đọc chuỗi kiểu ``NVDA=0.25, JNJ=0.05`` thành dict quan điểm."""
    if not raw.strip():
        return None
    out: dict[str, float] = {}
    for part in raw.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[k.strip().upper()] = float(v)
        except ValueError:
            continue
    return out or None