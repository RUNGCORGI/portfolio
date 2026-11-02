"""Tab "X-Ray & Báo cáo" — soi danh mục và xuất báo cáo tái lập được.

TÍCH HỢP::

    from tabs import tab_xray
    with tabs[i]:
        tab_xray.render()

Tab tự quản lý dữ liệu riêng, không đụng session state của bạn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:  # pragma: no cover
    HAS_PLOTLY = False

from core import alloc_lab, factor_lab, report, xray
from core import upgrade_api as api
from core import wf_backtest as wf

DEFAULT = "AAPL, MSFT, NVDA, JNJ, XOM, KO"
PROXIES = ["SPY", "QQQ", "IWM", "^VIX", "GLD", "TLT"]
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _load(tickers: tuple[str, ...], start: str) -> pd.DataFrame:
    return api.load_prices(list(tickers), start, None)


def render(prices: pd.DataFrame | None = None) -> None:
    st.header("Portfolio X-Ray và báo cáo")
    st.caption(
        "So sánh nhiều danh mục trên cùng bộ chỉ số, phân rã nguồn gốc rủi ro và "
        "lợi nhuận, rồi xuất thành báo cáo có mã tái lập."
    )

    with st.sidebar:
        st.markdown("### X-Ray")
        raw = st.text_input("Danh mục mã", DEFAULT, key="xr_tk")
        start = st.date_input("Từ ngày", pd.Timestamp("2018-01-01"), key="xr_start")

    tickers = tuple(t.strip().upper() for t in raw.split(",") if t.strip())
    if len(tickers) < 3:
        st.warning("Cần ít nhất 3 mã để phân tích có ý nghĩa.")
        return

    with st.spinner("Đang chuẩn bị dữ liệu..."):
        if prices is None:
            prices = _load(tuple(sorted(set(tickers) | set(PROXIES))), str(start))
        prices = api.clean_prices(prices)

    cols = [t for t in tickers if t in prices.columns]
    if len(cols) < 3:
        st.error("Không lấy đủ dữ liệu cho các mã đã nhập.")
        return

    rets = api.to_returns(prices[cols])
    bench = api.to_returns(prices[["SPY"]])["SPY"] if "SPY" in prices.columns else None

    s1, s2, s3, s4 = st.tabs(["So sánh danh mục", "Phân rã rủi ro", "Nhân tố", "Xuất báo cáo"])

    with s1:
        _compare(rets, bench)
    with s2:
        _risk(rets)
    with s3:
        _factors(rets, prices)
    with s4:
        _export(rets, bench, prices, cols, str(start))


# --------------------------------------------------------------------------


def _build_portfolios(rets, bench, methods) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for m in methods:
        try:
            out[m] = wf.run_walk_forward(rets, m).net
        except Exception:
            continue
    out.update(xray.benchmark_suite(rets, bench))
    return out


def _compare(rets, bench) -> None:
    st.subheader("Tearsheet đối chiếu")
    picks = st.multiselect(
        "Chiến lược cần so sánh", list(alloc_lab.METHODS),
        default=["HRP", "Max Sharpe"], key="xr_methods",
    )
    if not picks:
        st.info("Hãy chọn ít nhất một chiến lược.")
        return

    with st.spinner("Đang chạy walk-forward..."):
        ports = _build_portfolios(rets, bench, picks)
    if not ports:
        st.error("Không chạy được chiến lược nào với dữ liệu hiện tại.")
        return

    table = xray.compare(ports, bench)
    pct_rows = [
        "Tổng lợi nhuận", "CAGR", "Biến động năm", "Sụt giảm tối đa", "VaR 95%",
        "CVaR 95%", "Tỷ lệ phiên lãi", "Phiên tệ nhất", "Sai số theo dõi",
        "Lợi nhuận vượt trội",
    ]

    fmt_map = {row: ("{:.2%}" if row in pct_rows else "{:,.2f}") for row in table.index}
    st.dataframe(table.style.format(fmt_map, na_rep=""), use_container_width=True)

    st.caption(
        "**Chỉ số Ulcer** đo cả độ sâu lẫn độ dài của sụt giảm. Hai danh mục cùng "
        "giảm 30% nhưng một cái hồi phục sau hai tháng, cái kia mất hai năm — "
        "sụt giảm tối đa không phân biệt được, Ulcer thì có."
    )

    if HAS_PLOTLY:
        eq = pd.DataFrame({k: (1 + v).cumprod() for k, v in ports.items()}).dropna()
        st.plotly_chart(
            px.line(eq, title="Tăng trưởng của 1 đồng vốn"), use_container_width=True
        )

        yr = xray.yearly_returns(ports)
        fig = px.bar(yr, barmode="group", title="Lợi nhuận theo từng năm")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Biểu đồ này hay phơi bày một sự thật khó chịu: nhiều chiến lược chỉ "
            "thắng nhờ đúng một năm bùng nổ."
        )

    with st.expander("Các đợt sụt giảm sâu nhất"):
        pick = st.selectbox("Danh mục", list(ports), key="xr_dd")
        ep = xray.drawdown_episodes(ports[pick])
        if ep.empty:
            st.info("Không ghi nhận đợt sụt giảm đáng kể.")
        else:
            st.dataframe(
                ep.style.format({"Mức giảm": "{:.2%}"}),
                use_container_width=True, hide_index=True,
            )


def _risk(rets) -> None:
    st.subheader("Vốn nằm ở đâu, rủi ro nằm ở đâu")
    st.caption(
        "Biểu đồ tròn phân bổ vốn không bao giờ cho thấy điều này: một mã chiếm "
        "20% vốn hoàn toàn có thể đang gánh 45% rủi ro danh mục."
    )

    method = st.selectbox("Phương pháp phân bổ", list(alloc_lab.METHODS), index=4, key="xr_rc")
    mu, cov = alloc_lab.estimate(rets)
    w = alloc_lab.allocate(method, mu, cov, max_weight=0.4).weights

    rc = xray.risk_contribution(w, cov)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(
            rc.style.format({
                "Trọng số vốn": "{:.1%}", "Tỷ trọng rủi ro": "{:.1%}",
                "Đóng góp biến động": "{:.2%}", "Chênh lệch": "{:+.1%}",
            }),
            use_container_width=True, hide_index=True,
        )
        worst = rc.iloc[(rc["Chênh lệch"]).argmax()]
        if worst["Chênh lệch"] > 0.02:
            st.warning(
                f"**{worst['Mã']}** chỉ chiếm {worst['Trọng số vốn']:.1%} vốn "
                f"nhưng gánh {worst['Tỷ trọng rủi ro']:.1%} rủi ro."
            )
    with c2:
        st.markdown("**Mức độ tập trung**")
        conc = xray.concentration_metrics(w)
        for k, v in conc.items():
            st.metric(k, f"{v:.1%}" if "Trọng số" in k or "Ba mã" in k else f"{v:,.2f}")

    if HAS_PLOTLY:
        melt = rc.melt(
            id_vars="Mã", value_vars=["Trọng số vốn", "Tỷ trọng rủi ro"],
            var_name="Loại", value_name="Tỷ lệ",
        )
        fig = px.bar(melt, x="Mã", y="Tỷ lệ", color="Loại", barmode="group",
                     title="Vốn so với rủi ro theo từng mã")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Hai danh mục có thực sự khác nhau?")
    c1, c2 = st.columns(2)
    m_a = c1.selectbox("Danh mục A", list(alloc_lab.METHODS), index=4, key="xr_a")
    m_b = c2.selectbox("Danh mục B", list(alloc_lab.METHODS), index=2, key="xr_b")
    w_a = alloc_lab.allocate(m_a, mu, cov, max_weight=0.4).weights
    w_b = alloc_lab.allocate(m_b, mu, cov, max_weight=0.4).weights
    ov = xray.overlap_analysis(w_a, w_b)

    cols = st.columns(len(ov))
    for col, (k, v) in zip(cols, ov.items()):
        col.metric(k, f"{v:.1%}" if isinstance(v, float) and v <= 1 else f"{v}")
    st.caption(
        "**Active share** là chỉ số quỹ dùng để chứng minh họ không phải quỹ chỉ số "
        "trá hình. Dưới 20% nghĩa là hai danh mục thực chất giống nhau."
    )


def _factors(rets, prices) -> None:
    st.subheader("Lợi nhuận đến từ kỹ năng hay từ việc gánh rủi ro?")
    avail = [p for p in PROXIES if p in prices.columns]
    if "SPY" not in avail:
        st.warning("Cần dữ liệu SPY để dựng mô hình nhân tố.")
        return

    try:
        factors = factor_lab.assemble_factors(prices[avail], prices[list(rets.columns)])
    except Exception as exc:
        st.error(f"Không dựng được nhân tố: {exc}")
        return

    k = rets.shape[1]
    pr = pd.Series(rets.to_numpy() @ np.repeat(1 / k, k), index=rets.index)
    model, tbl, attr = factor_lab.factor_report(pr, factors)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alpha năm hoá", f"{model.alpha_annual:.2%}")
    c2.metric("Thống kê t của alpha", f"{model.alpha_tstat:.2f}")
    c3.metric("R² hiệu chỉnh", f"{model.adj_r_squared:.1%}")
    c4.metric("Alpha có ý nghĩa", "Có" if model.alpha_is_significant else "Không")

    st.dataframe(
        tbl.style.format({"Hệ số beta": "{:.3f}", "Thống kê t": "{:.2f}", "p-value": "{:.3f}"}),
        use_container_width=True,
    )
    st.caption(
        f"Sai số chuẩn dùng hiệu chỉnh Newey–West với {model.hac_lags} độ trễ, "
        "xử lý đồng thời phương sai thay đổi và tự tương quan. Bỏ qua bước này, "
        "thống kê t thường bị thổi phồng và alpha trông có ý nghĩa trong khi không phải."
    )

    st.markdown("#### Phân rã lợi nhuận")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(
            attr.style.format({"Đóng góp": "{:.2%}", "Tỷ trọng": "{:.1%}"}),
            use_container_width=True, hide_index=True,
        )
    with c2:
        if HAS_PLOTLY:
            st.plotly_chart(
                px.bar(attr, x="Nguồn", y="Đóng góp", title="Nguồn gốc lợi nhuận"),
                use_container_width=True,
            )

    with st.expander("Vì sao phải dựng nhân tố long/short?"):
        st.caption(
            "VIF đo mức chồng lấn giữa các nhân tố. Trên 10 nghĩa là nhân tố đó gần "
            "như được giải thích hoàn toàn bởi những nhân tố còn lại, khiến hệ số của "
            "nó mất ý nghĩa. Bảng dưới so sánh cách hồi quy thẳng các ETF với cách "
            "trừ đi SPY để tạo nhân tố long/short."
        )
        st.dataframe(
            factor_lab.naive_vs_orthogonal(prices[avail]).style.format({"VIF": "{:.2f}"}),
            use_container_width=True, hide_index=True,
        )

    with st.expander("Danh mục có bị trôi phong cách không?"):
        try:
            rb = factor_lab.rolling_exposures(pr, factors)
            if HAS_PLOTLY:
                st.plotly_chart(
                    px.line(rb, title="Beta nhân tố trượt 126 phiên"),
                    use_container_width=True,
                )
            st.dataframe(
                factor_lab.style_drift_score(rb).style.format("{:.3f}"),
                use_container_width=True,
            )
        except Exception as exc:
            st.info(f"Chưa đủ dữ liệu để tính beta trượt: {exc}")


def _export(rets, bench, prices, cols, start) -> None:
    st.subheader("Xuất báo cáo")
    st.caption(
        "Mỗi báo cáo mang một **mã tái lập** băm từ toàn bộ cấu hình và dữ liệu đầu "
        "vào. Hai báo cáo cùng mã chắc chắn đến từ cùng một lần chạy."
    )

    c1, c2 = st.columns(2)
    title = c1.text_input("Tiêu đề báo cáo", "Báo cáo phân tích danh mục đầu tư")
    method = c2.selectbox("Chiến lược chính", list(alloc_lab.METHODS), index=4, key="xr_rep_m")

    if not st.button("Tạo báo cáo", type="primary", key="xr_gen"):
        return

    with st.spinner("Đang chạy phân tích và dựng báo cáo..."):
        res = wf.run_walk_forward(rets, method)
        ports = {method: res.net, **xray.benchmark_suite(rets, bench)}
        cmp_tbl = xray.compare(ports, bench).reset_index().rename(columns={"index": "Chỉ tiêu"})

        mu, cov = alloc_lab.estimate(rets)
        w = alloc_lab.allocate(method, mu, cov, max_weight=0.4).weights

        manifest = report.RunManifest(
            tickers=list(cols), start=start,
            end=str(rets.index[-1].date()), method=method,
            config={"lookback": 252, "step": 21, "embargo": 5, "max_weight": 0.4},
            data_rows=len(rets), data_last_date=str(rets.index[-1].date()),
        )

        eq_png = report.series_to_png(
            pd.DataFrame({k: (1 + v).cumprod() for k, v in ports.items()}).dropna(),
            "Tăng trưởng của 1 đồng vốn",
        )

        sections = [
            report.Section(
                "Hiệu quả đối chiếu",
                commentary=(
                    f"Chiến lược <b>{method}</b> được chạy theo quy trình walk-forward: "
                    "trọng số mỗi kỳ chỉ ước lượng từ dữ liệu trước kỳ đó, có khoảng cấm "
                    "chống rò rỉ, và mọi lần tái cân bằng đều bị trừ phí giao dịch."
                ),
                table=cmp_tbl, figure_png=eq_png,
                note="Các mốc so sánh gồm chia đều, thị trường và 60/40 xấp xỉ.",
            ),
            report.Section(
                "Phân rã rủi ro",
                commentary=(
                    "Cột chênh lệch cho thấy mã nào đang gánh rủi ro vượt xa phần vốn "
                    "được cấp — thông tin mà biểu đồ phân bổ vốn không thể hiện."
                ),
                table=xray.risk_contribution(w, cov),
            ),
            report.Section(
                "Lợi nhuận theo từng năm",
                commentary="Kiểm tra xem kết quả có phụ thuộc vào một năm duy nhất không.",
                table=xray.yearly_returns(ports).reset_index().rename(
                    columns={"index": "Năm"}
                ),
            ),
            report.Section(
                "Thống kê vòng quay và chi phí",
                table=pd.DataFrame([res.summary()]),
                note="Vòng quay cao đồng nghĩa chi phí giao dịch bào mòn lợi nhuận nhiều hơn.",
            ),
        ]

        out = report.export_report(title, manifest, sections, out_dir=REPORT_DIR)

    st.success(f"Đã tạo báo cáo. Mã tái lập: `{manifest.run_hash}`")
    c1, c2 = st.columns(2)
    if out.get("html") and Path(out["html"]).exists():
        c1.download_button(
            "Tải báo cáo HTML", Path(out["html"]).read_bytes(),
            file_name=Path(out["html"]).name, mime="text/html",
        )
    if out.get("pdf") and Path(out["pdf"]).exists():
        c2.download_button(
            "Tải báo cáo PDF", Path(out["pdf"]).read_bytes(),
            file_name=Path(out["pdf"]).name, mime="application/pdf",
        )
    else:
        c2.info("Không sinh được PDF trong môi trường này; bản HTML vẫn đầy đủ nội dung.")

    st.dataframe(manifest.to_table(), use_container_width=True, hide_index=True)