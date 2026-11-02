"""Tab "Giám sát" — cảnh báo rủi ro, nhật ký giao dịch và thị trường Việt Nam.

TÍCH HỢP::

    from tabs import tab_monitor
    with tabs[i]:
        tab_monitor.render()
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:  # pragma: no cover
    HAS_PLOTLY = False

from core import alerts, alloc_lab, hmm_regime, journal
from core import upgrade_api as api
from core import vn_market as vn

US_DEFAULT = "AAPL, MSFT, NVDA, JNJ, XOM, KO"
VN_DEFAULT = "VCB, FPT, HPG, VNM, SSI, GAS"


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _load_us(tickers: tuple[str, ...], start: str) -> pd.DataFrame:
    return api.load_prices(list(tickers), start, None)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _load_vn(tickers: tuple[str, ...], start: str) -> pd.DataFrame:
    return vn.VNStockProvider().get_prices(list(tickers), start, None)


def render() -> None:
    st.header("Giám sát danh mục")
    st.caption(
        "Một dashboard chỉ hữu ích khi có người mở ra xem, còn rủi ro thì không chờ ai. "
        "Khu vực này biến phân tích thụ động thành công cụ theo dõi chủ động."
    )

    with st.sidebar:
        st.markdown("### Giám sát")
        market = st.radio("Thị trường", ["Mỹ", "Việt Nam"], horizontal=True, key="mn_mkt")
        default = VN_DEFAULT if market == "Việt Nam" else US_DEFAULT
        raw = st.text_input("Danh mục mã", default, key=f"mn_tk_{market}")
        start = st.date_input("Từ ngày", pd.Timestamp("2020-01-01"), key="mn_start")

    tickers = tuple(t.strip().upper() for t in raw.split(",") if t.strip())
    if len(tickers) < 2:
        st.warning("Cần ít nhất 2 mã.")
        return

    is_vn = market == "Việt Nam"
    bench_name = vn.VN_BENCHMARK if is_vn else "SPY"

    with st.spinner("Đang tải dữ liệu..."):
        full = tuple(sorted(set(tickers) | {bench_name}))
        prices = _load_vn(full, str(start)) if is_vn else _load_us(full, str(start))
        prices = api.clean_prices(prices)

    cols = [t for t in tickers if t in prices.columns]
    if len(cols) < 2:
        st.error("Không lấy đủ dữ liệu cho các mã đã nhập.")
        return

    rets = api.to_returns(prices[cols])
    bench = api.to_returns(prices[[bench_name]])[bench_name] if bench_name in prices else None

    s1, s2, s3 = st.tabs(["Cảnh báo rủi ro", "Nhật ký giao dịch", "Thị trường Việt Nam"])
    with s1:
        _alerts(rets, bench, cols, market)
    with s2:
        _journal(prices, cols)
    with s3:
        _vn_panel(is_vn, prices, cols)


# --------------------------------------------------------------------------
# Cảnh báo
# --------------------------------------------------------------------------


def _alerts(rets, bench, cols, market) -> None:
    st.subheader("Động cơ cảnh báo")

    with st.expander("Thiết lập ngưỡng", expanded=False):
        c1, c2, c3 = st.columns(3)
        dd = c1.slider("Ngưỡng sụt giảm", -0.50, -0.05, -0.15, 0.01, key="mn_dd")
        vol = c2.slider("Ngưỡng biến động năm", 0.10, 0.80, 0.30, 0.05, key="mn_vol")
        beta = c3.slider("Ngưỡng beta", 0.8, 2.5, 1.3, 0.1, key="mn_beta")
        c4, c5 = st.columns(2)
        conc = c4.slider("Ngưỡng tập trung", 0.20, 0.90, 0.40, 0.05, key="mn_conc")
        var = c5.slider("Ngưỡng VaR 95%", -0.08, -0.01, -0.03, 0.005, key="mn_var")

    method = st.selectbox("Danh mục đang theo dõi", list(alloc_lab.METHODS), index=4, key="mn_m")
    mu, cov = alloc_lab.estimate(rets)
    w = alloc_lab.allocate(method, mu, cov, max_weight=0.4).weights
    pr = pd.Series(rets.to_numpy() @ w.to_numpy(), index=rets.index)

    regime = regime_prev = None
    conf = 0.0
    if bench is not None and len(bench) > 300:
        try:
            m = hmm_regime.fit_regimes(bench)
            regime, conf = m.current, m.confidence
            regime_prev = str(m.states.iloc[-22]) if len(m.states) > 22 else regime
        except Exception:
            pass

    ctx = alerts.build_context(pr, w, rets, bench, regime, regime_prev, conf)
    rules = alerts.build_default_rules(dd, vol, beta, conc, var)
    fired = alerts.run_checks(ctx, rules)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sụt giảm hiện tại", f"{ctx['drawdown']:.1%}")
    c2.metric("Biến động 20 phiên", f"{ctx['vol_20d']:.1%}")
    c3.metric("Beta", f"{ctx['beta']:.2f}")
    c4.metric("Tương quan TB", f"{ctx['avg_corr']:.2f}")

    if not fired:
        st.success("Không có cảnh báo nào. Mọi chỉ số đều nằm trong ngưỡng đã thiết lập.")
    else:
        crit = sum(1 for a in fired if a.severity == alerts.Severity.CRITICAL)
        st.error(f"Có **{len(fired)}** cảnh báo đang hoạt động, trong đó {crit} nghiêm trọng.")
        for a in fired:
            icon = "🔴" if a.severity == alerts.Severity.CRITICAL else "🟠"
            with st.container(border=True):
                st.markdown(f"{icon} **{a.severity.value} · {a.title}**")
                st.write(a.message)

    st.markdown("#### Gửi thông báo")
    c1, c2 = st.columns([3, 1])
    url = c1.text_input(
        "Webhook URL (Teams, Slack, Telegram...)", "",
        placeholder="https://...", key="mn_hook",
    )
    with c2:
        st.write("")
        st.write("")
        preview = st.toggle("Chỉ xem trước", True, key="mn_dry")

    c1, c2 = st.columns(2)
    if c1.button("Gửi cảnh báo", disabled=not fired, key="mn_send"):
        if preview:
            st.code(alerts.dry_run_webhook(fired, f"Danh mục {method} ({market})"), language="markdown")
            st.caption("Chế độ xem trước: nội dung chưa được gửi đi thật.")
        else:
            res = alerts.send_webhook(fired, url, f"Danh mục {method}")
            st.success("Đã gửi thành công.") if res.get("sent") else st.warning(
                f"Chưa gửi được: {res.get('reason', res)}"
            )
    if c2.button("Ghi vào nhật ký", disabled=not fired, key="mn_log"):
        n = alerts.log_alerts(fired, f"{method} ({market})")
        st.success(f"Đã lưu {n} cảnh báo vào nhật ký.")

    with st.expander("Lịch sử cảnh báo"):
        hist = alerts.alert_history(limit=50)
        if hist.empty:
            st.info("Chưa có cảnh báo nào được ghi lại.")
        else:
            st.dataframe(hist, use_container_width=True, hide_index=True)
            if st.button("Xoá lịch sử", key="mn_clear"):
                st.warning(f"Đã xoá {alerts.clear_history()} bản ghi.")


# --------------------------------------------------------------------------
# Nhật ký giao dịch
# --------------------------------------------------------------------------


def _journal(prices, cols) -> None:
    st.subheader("Nhật ký giao dịch mô phỏng")
    st.caption(
        "Backtest giả định bạn luôn thực hiện đúng tín hiệu. Thực tế con người do dự, "
        "vào lệnh trễ hoặc bỏ qua tín hiệu. Sổ này ghi lại quyết định thật để đo "
        "**khoảng cách thực thi** — con số không backtest nào tính được."
    )

    with st.form("mn_trade_form"):
        c1, c2, c3 = st.columns(3)
        d = c1.date_input("Ngày", date.today())
        ticker = c2.selectbox("Mã", cols)
        side = c3.selectbox("Lệnh", ["MUA", "BAN"])

        c1, c2, c3 = st.columns(3)
        qty = c1.number_input("Khối lượng", 1.0, 1e9, 100.0, 10.0)
        last_px = float(prices[ticker].iloc[-1]) if ticker in prices.columns else 100.0
        price = c2.number_input("Giá khớp", 0.01, 1e9, round(last_px, 2))
        signal = c3.selectbox("Tín hiệu mô hình lúc đó", ["", "BUY", "HOLD", "SELL"])

        c1, c2 = st.columns([3, 1])
        note = c1.text_input("Ghi chú", "")
        followed = c2.checkbox("Làm theo mô hình", True)

        if st.form_submit_button("Ghi giao dịch", type="primary"):
            journal.record_trade(
                journal.Trade(str(d), ticker, side, qty, price, note, signal, followed)
            )
            st.success(f"Đã ghi: {side} {qty:,.0f} {ticker} tại giá {price:,.2f}")

    trades = journal.all_trades()
    if trades.empty:
        st.info("Chưa có giao dịch nào. Hãy ghi lệnh đầu tiên ở biểu mẫu phía trên.")
        return

    summary = journal.portfolio_summary(prices)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số vị thế", summary.get("Số vị thế", 0))
    c2.metric("Giá trị thị trường", f"{summary.get('Giá trị thị trường', 0):,.0f}")
    c3.metric("Tổng lãi lỗ", f"{summary.get('Tổng lãi lỗ', 0):,.0f}")
    c4.metric("Tỷ suất", f"{summary.get('Tỷ suất tổng', 0):.1%}")

    st.markdown("**Vị thế hiện tại**")
    pos = journal.mark_to_market(prices)
    if not pos.empty:
        st.dataframe(
            pos.style.format({
                "Khối lượng": "{:,.0f}", "Giá vốn BQ": "{:,.2f}",
                "Giá trị vốn": "{:,.0f}", "Giá thị trường": "{:,.2f}",
                "Giá trị thị trường": "{:,.0f}", "Lãi lỗ chưa thực hiện": "{:,.0f}",
                "Lãi lỗ đã thực hiện": "{:,.0f}", "Tỷ suất": "{:.1%}",
            }),
            use_container_width=True, hide_index=True,
        )

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Lịch sử lệnh**")
        st.dataframe(
            trades[["ID", "Ngày", "Mã", "Lệnh", "Khối lượng", "Giá", "Tín hiệu mô hình",
                    "Làm theo mô hình", "Ghi chú"]],
            use_container_width=True, hide_index=True,
        )
    with c2:
        st.markdown("**Mức độ tuân thủ mô hình**")
        disc = journal.discipline_report()
        st.dataframe(
            disc.style.format({"Giá trị": "{:,.2f}"}),
            use_container_width=True, hide_index=True,
        )
        rate = disc.loc[disc["Chỉ tiêu"] == "Tỷ lệ tuân thủ", "Giá trị"]
        if len(rate) and float(rate.iloc[0]) < 0.7:
            st.warning(
                "Tỷ lệ tuân thủ dưới 70%. Nếu kết quả thực tế kém hơn backtest, "
                "nguyên nhân có thể nằm ở khâu thực thi chứ không phải ở mô hình."
            )

    with st.expander("Theo dõi NAV so với mô hình"):
        c1, c2, c3 = st.columns(3)
        nav = c1.number_input("NAV thực tế", 0.0, 1e12, 100_000.0, 1000.0)
        mnav = c2.number_input("NAV theo mô hình", 0.0, 1e12, 100_000.0, 1000.0)
        snap_d = c3.date_input("Ngày ghi nhận", date.today(), key="mn_snap")
        if st.button("Lưu mốc NAV", key="mn_save_nav"):
            journal.save_snapshot(nav, mnav, str(snap_d))
            st.success("Đã lưu mốc NAV.")

        hist = journal.nav_history()
        if not hist.empty:
            if HAS_PLOTLY:
                st.plotly_chart(
                    px.line(hist[["NAV thực tế", "NAV mô hình"]],
                            title="NAV thực tế so với mô hình"),
                    use_container_width=True,
                )
            gap = journal.execution_gap()
            if gap.get("Có đủ dữ liệu"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Lợi nhuận thực tế", f"{gap['Lợi nhuận thực tế']:.2%}")
                c2.metric("Lợi nhuận mô hình", f"{gap['Lợi nhuận mô hình']:.2%}")
                c3.metric("Khoảng cách thực thi", f"{gap['Khoảng cách thực thi']:+.2%}")

    if st.button("Xoá toàn bộ nhật ký", key="mn_clear_j"):
        n = journal.clear_journal()
        st.warning(f"Đã xoá {n['Giao dịch']} lệnh và {n['Ảnh chụp NAV']} mốc NAV.")


# --------------------------------------------------------------------------
# Thị trường Việt Nam
# --------------------------------------------------------------------------


def _vn_panel(is_vn, prices, cols) -> None:
    st.subheader("Chế độ thị trường Việt Nam")
    provider = vn.VNStockProvider()
    st.info(f"Nguồn dữ liệu đang dùng: **{provider.source_in_use()}**")
    st.caption(
        "Toàn bộ module phân tích không cần sửa một dòng nào khi đổi sang thị trường "
        "Việt Nam — đó chính là giá trị của việc tách lớp dữ liệu khỏi lớp tính toán."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Rổ cổ phiếu hỗ trợ sẵn**")
        st.dataframe(
            vn.universe_table().style.format({"Biên độ/phiên": "{:.0%}"}),
            use_container_width=True, hide_index=True, height=300,
        )
    with c2:
        st.markdown("**Chi phí giao dịch tại Việt Nam**")
        st.dataframe(
            vn.VNCostModel().describe().style.format({"Mức": "{:.3%}"}),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Khác biệt quan trọng so với thị trường Mỹ: thuế thu nhập cá nhân 0,1% "
            "chỉ đánh **một chiều khi bán**, nên chi phí mua và bán không đối xứng. "
            "Mô hình chi phí chuẩn Mỹ sẽ tính sai điều này."
        )

    if not is_vn:
        st.warning(
            "Hãy chuyển sang **Thị trường: Việt Nam** ở thanh bên để xem phân tích "
            "trên dữ liệu cổ phiếu Việt Nam."
        )
        return

    st.markdown("#### Thống kê chạm biên độ")
    st.caption(
        "Biên độ dao động theo sàn (HOSE ±7%, HNX ±10%, UPCoM ±15%) làm phân phối lợi "
        "suất bị cắt đuôi, ảnh hưởng trực tiếp tới VaR và CVaR. Chạm biên nhiều cũng "
        "cảnh báo backtest có thể lạc quan: khi cổ phiếu nằm sàn thì thường không "
        "khớp được lệnh bán."
    )
    st.dataframe(
        vn.price_limit_hits(prices[cols]).style.format({"Tỷ lệ chạm biên": "{:.2%}"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("#### Phân bổ theo ngành")
    rets = api.to_returns(prices[cols])
    mu, cov = alloc_lab.estimate(rets)
    w = alloc_lab.allocate("HRP", mu, cov).weights
    sec = vn.sector_exposure(w)

    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(
            sec.style.format({"Trọng số": "{:.1%}"}),
            use_container_width=True, hide_index=True,
        )
    with c2:
        if HAS_PLOTLY:
            st.plotly_chart(
                px.pie(sec, names="Ngành", values="Trọng số", title="Tỷ trọng theo ngành"),
                use_container_width=True,
            )
    top = sec.iloc[0]
    if float(top["Trọng số"]) > 0.4:
        st.warning(
            f"Ngành **{top['Ngành']}** chiếm {top['Trọng số']:.0%} danh mục. Ở thị "
            "trường Việt Nam, rủi ro tập trung ngành đáng lo hơn nhiều so với Mỹ vì "
            "nhóm ngân hàng và bất động sản chi phối phần lớn chỉ số."
        )
