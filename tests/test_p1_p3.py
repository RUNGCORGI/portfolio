"""Kiểm thử cho Phần 1 (mô hình nhân tố) và Phần 3 (tính năng demo).

Chạy riêng::

    pytest tests/test_p1_p3.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import alerts, alloc_lab, factor_lab, journal, report, stress, vn_market, xray
from core import upgrade_api as api
from core import wf_backtest as wf

TICKERS = ["AAPL", "MSFT", "NVDA", "JNJ", "XOM", "KO"]
PROXIES = ["SPY", "QQQ", "IWM", "^VIX", "GLD", "TLT"]


@pytest.fixture(scope="module")
def prices():
    return api.synthetic_prices(TICKERS + PROXIES, "2018-01-01", "2026-06-30")


@pytest.fixture(scope="module")
def rets(prices):
    return api.to_returns(prices[TICKERS])


@pytest.fixture(scope="module")
def port(rets):
    k = rets.shape[1]
    return pd.Series(rets.to_numpy() @ np.repeat(1 / k, k), index=rets.index)


@pytest.fixture(scope="module")
def factors(prices):
    return factor_lab.assemble_factors(prices[PROXIES], prices[TICKERS])


@pytest.fixture
def tmpdb():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "test.db"


# ==========================================================================
# PHẦN 1 — Mô hình nhân tố
# ==========================================================================


def test_factors_built(factors):
    assert "MKT" in factors.columns
    assert "GROWTH" in factors.columns
    assert "SIZE" in factors.columns
    assert len(factors) > 500


def test_longshort_factors_less_collinear(prices):
    """Trừ SPY phải làm giảm đa cộng tuyến so với dùng ETF thô."""
    cmp = factor_lab.naive_vs_orthogonal(prices[PROXIES])
    naive_max = cmp[cmp["Cách dựng"].str.contains("thô")]["VIF"].max()
    ortho = cmp[cmp["Cách dựng"].str.contains("Long/short")]
    style_max = ortho[ortho["Nhân tố"] != "MKT"]["VIF"].max()
    assert style_max < naive_max


def test_momentum_factor_is_long_short(prices):
    mom = factor_lab.build_momentum_factor(prices[TICKERS])
    assert len(mom) > 200
    # nhân tố long/short phải có trung bình nhỏ hơn nhiều so với thị trường
    assert abs(float(mom.mean())) < 0.01


def test_factor_model_fits(port, factors):
    m = factor_lab.fit_factor_model(port, factors)
    assert 0.0 <= m.r_squared <= 1.0
    assert m.n_obs > 500
    assert len(m.betas) == factors.shape[1]
    assert m.hac_lags >= 0


def test_market_beta_positive_for_long_portfolio(port, factors):
    m = factor_lab.fit_factor_model(port, factors)
    assert m.betas["MKT"] > 0.3
    assert abs(m.tstats["MKT"]) > 2  # beta thị trường phải có ý nghĩa rõ rệt


def test_newey_west_differs_from_ols(port, factors):
    cmp = factor_lab.compare_ols_vs_hac(port, factors)
    assert len(cmp) == factors.shape[1] + 1
    # ít nhất một hệ số phải thay đổi đáng kể khi hiệu chỉnh HAC
    assert (cmp["Mức thổi phồng"] - 1.0).abs().max() > 0.05


def test_attribution_sums_to_total(port, factors):
    m = factor_lab.fit_factor_model(port, factors)
    attr = factor_lab.return_attribution(m)
    y = port.reindex(factors.index).dropna()
    assert abs(float(attr["Đóng góp"].sum()) - float(y.sum())) < 1e-6
    assert abs(float(attr["Tỷ trọng"].sum()) - 1.0) < 1e-6


def test_vif_detects_collinearity(prices):
    naive = prices[PROXIES].pct_change().dropna()
    vif = factor_lab.variance_inflation_factors(naive)
    assert (vif["VIF"] >= 1.0).all()
    assert vif["VIF"].max() > 5  # SPY và QQQ chắc chắn chồng lấn


def test_rolling_exposures_shape(port, factors):
    rb = factor_lab.rolling_exposures(port, factors, window=126)
    assert rb.shape[1] == factors.shape[1]
    assert not rb.isna().any().any()
    drift = factor_lab.style_drift_score(rb)
    assert "Độ dao động" in drift.columns


def test_asset_level_betas(rets, factors):
    tbl = factor_lab.asset_factor_betas(rets, factors)
    assert len(tbl) == len(TICKERS)
    assert "R²" in tbl.columns
    assert (tbl["R²"] <= 1.0).all()


# ==========================================================================
# PHẦN 3 — X-Ray
# ==========================================================================


def test_tearsheet_complete(port):
    ts = xray.tearsheet(port)
    for key in ["Sharpe", "Sortino", "Calmar", "Chỉ số Ulcer", "CVaR 95%", "Tỷ lệ đuôi"]:
        assert key in ts
    assert ts["Sụt giảm tối đa"] <= 0


def test_cvar_worse_than_var(port):
    assert xray.conditional_var(port) <= xray.value_at_risk(port)


def test_ulcer_index_positive(port):
    assert xray.ulcer_index(port) > 0


def test_risk_contribution_sums_to_one(rets):
    mu, cov = alloc_lab.estimate(rets)
    w = alloc_lab.allocate("HRP", mu, cov).weights
    rc = xray.risk_contribution(w, cov)
    assert abs(float(rc["Tỷ trọng rủi ro"].sum()) - 1.0) < 1e-6
    assert abs(float(rc["Trọng số vốn"].sum()) - 1.0) < 1e-6


def test_overlap_identical_portfolios(rets):
    mu, cov = alloc_lab.estimate(rets)
    w = alloc_lab.allocate("HRP", mu, cov).weights
    ov = xray.overlap_analysis(w, w)
    assert abs(ov["Độ chồng lấn"] - 1.0) < 1e-9
    assert abs(ov["Active share"]) < 1e-9


def test_concentration_equal_weight(rets):
    k = rets.shape[1]
    w = pd.Series(np.repeat(1 / k, k), index=rets.columns)
    c = xray.concentration_metrics(w)
    assert abs(c["Số vị thế hiệu dụng"] - k) < 1e-6


def test_benchmark_suite_includes_equal_weight(rets, prices):
    spy = api.to_returns(prices[["SPY"]])["SPY"]
    suite = xray.benchmark_suite(rets, spy)
    assert "Chia đều" in suite
    assert "60/40 xấp xỉ" in suite


def test_yearly_returns_covers_period(port):
    yr = xray.yearly_returns({"Danh mục": port})
    assert len(yr) >= 5


def test_drawdown_episodes_sorted(port):
    ep = xray.drawdown_episodes(port, top=3)
    if not ep.empty:
        assert ep["Mức giảm"].is_monotonic_increasing or len(ep) == 1


# ==========================================================================
# PHẦN 3 — Báo cáo
# ==========================================================================


def test_manifest_hash_stable():
    kw = dict(tickers=["AAPL", "MSFT"], start="2020-01-01", end="2026-01-01", method="HRP")
    a = report.RunManifest(**kw)
    b = report.RunManifest(**kw)
    assert a.run_hash == b.run_hash
    assert len(a.run_hash) == 12


def test_manifest_hash_order_independent():
    a = report.RunManifest(tickers=["AAPL", "MSFT"], start="2020-01-01", end="2026-01-01", method="HRP")
    b = report.RunManifest(tickers=["MSFT", "AAPL"], start="2020-01-01", end="2026-01-01", method="HRP")
    assert a.run_hash == b.run_hash


def test_manifest_hash_changes_with_config():
    base = dict(tickers=["AAPL"], start="2020-01-01", end="2026-01-01", method="HRP")
    a = report.RunManifest(**base, config={"lookback": 252})
    b = report.RunManifest(**base, config={"lookback": 126})
    assert a.run_hash != b.run_hash


def test_html_report_generated(port):
    mf = report.RunManifest(tickers=TICKERS, start="2018-01-01", end="2026-06-30", method="HRP")
    secs = [report.Section("Thử nghiệm", "Diễn giải", pd.DataFrame({"A": [1.0, 2.0]}))]
    html = report.build_html("Báo cáo thử", mf, secs)
    assert mf.run_hash in html
    assert "Báo cáo thử" in html
    assert "<table" in html


def test_export_creates_files(port):
    with tempfile.TemporaryDirectory() as d:
        mf = report.RunManifest(tickers=TICKERS, start="2018-01-01", end="2026-06-30", method="HRP")
        secs = [report.Section("Hiệu quả", table=pd.DataFrame([xray.tearsheet(port)]))]
        out = report.export_report("Báo cáo", mf, secs, out_dir=d)
        assert out["html"] is not None and out["html"].exists()
        assert out["html"].stat().st_size > 1000


def test_png_generation(port):
    png = report.series_to_png((1 + port).cumprod(), "Đường vốn")
    assert png is None or png[:4] == b"\x89PNG"


# ==========================================================================
# PHẦN 3 — Cảnh báo
# ==========================================================================


def test_context_built(port, rets, prices):
    spy = api.to_returns(prices[["SPY"]])["SPY"]
    w = pd.Series(np.repeat(1 / len(TICKERS), len(TICKERS)), index=TICKERS)
    ctx = alerts.build_context(port, w, rets, spy)
    assert ctx["drawdown"] <= 0
    assert ctx["vol_20d"] >= 0
    assert -1 <= ctx["avg_corr"] <= 1


def test_drawdown_alert_fires():
    ctx = {"drawdown": -0.35, "vol_20d": 0.2, "beta": 1.0, "max_weight": 0.2,
           "var_95": -0.01, "avg_corr": 0.3}
    fired = alerts.run_checks(ctx)
    assert any(a.code == "DRAWDOWN" for a in fired)


def test_no_alerts_when_calm():
    ctx = {"drawdown": -0.02, "vol_20d": 0.12, "beta": 0.9, "max_weight": 0.2,
           "var_95": -0.01, "avg_corr": 0.3}
    assert alerts.run_checks(ctx) == []


def test_regime_alert_only_on_transition():
    base = {"drawdown": -0.02, "vol_20d": 0.12, "beta": 0.9, "max_weight": 0.2,
            "var_95": -0.01, "avg_corr": 0.3, "regime_confidence": 0.9}
    on_change = alerts.run_checks({**base, "regime": "BEAR", "regime_prev": "BULL"})
    steady = alerts.run_checks({**base, "regime": "BEAR", "regime_prev": "BEAR"})
    assert any(a.code == "REGIME" for a in on_change)
    assert not any(a.code == "REGIME" for a in steady)


def test_alerts_sorted_by_severity():
    ctx = {"drawdown": -0.40, "vol_20d": 0.50, "beta": 2.0, "max_weight": 0.9,
           "var_95": -0.08, "avg_corr": 0.95}
    fired = alerts.run_checks(ctx)
    assert len(fired) >= 4
    assert fired[0].severity == alerts.Severity.CRITICAL


def test_alert_logging_roundtrip(tmpdb):
    ctx = {"drawdown": -0.40, "vol_20d": 0.2, "beta": 1.0, "max_weight": 0.2,
           "var_95": -0.01, "avg_corr": 0.3}
    fired = alerts.run_checks(ctx)
    assert alerts.log_alerts(fired, "test", tmpdb) == len(fired)
    hist = alerts.alert_history(db_path=tmpdb)
    assert len(hist) == len(fired)
    assert alerts.clear_history(tmpdb) == len(fired)


def test_webhook_never_raises():
    ctx = {"drawdown": -0.40, "vol_20d": 0.2, "beta": 1.0, "max_weight": 0.2,
           "var_95": -0.01, "avg_corr": 0.3}
    fired = alerts.run_checks(ctx)
    for url in ["", "not-a-url", "http://localhost:1/nothing"]:
        res = alerts.send_webhook(fired, url)
        assert res["sent"] is False


def test_dry_run_contains_messages():
    ctx = {"drawdown": -0.40, "vol_20d": 0.2, "beta": 1.0, "max_weight": 0.2,
           "var_95": -0.01, "avg_corr": 0.3}
    text = alerts.dry_run_webhook(alerts.run_checks(ctx))
    assert "Sụt giảm" in text


# ==========================================================================
# PHẦN 3 — Thị trường Việt Nam
# ==========================================================================


def test_vn_prices_respect_limits():
    px = vn_market.simulate_vn_prices(["VCB", "FPT", "VND"], "2020-01-01", "2026-01-01")
    rets = px.pct_change().dropna()
    assert rets["VCB"].abs().max() <= 0.0701   # HOSE ±7%
    assert rets["VND"].abs().max() <= 0.1001   # HNX ±10%


def test_vn_provider_always_returns_data():
    p = vn_market.VNStockProvider()
    df = p.get_prices(["VNINDEX", "VCB", "FPT"], "2021-01-01", "2026-01-01")
    assert not df.empty
    assert (df > 0).all().all()
    assert "Dữ liệu" in p.source_in_use() or "vnstock" in p.source_in_use()


def test_vn_cost_asymmetric():
    c = vn_market.VNCostModel()
    assert c.sell_rate > c.buy_rate   # thuế TNCN chỉ đánh khi bán
    assert abs(c.round_trip_rate - (c.buy_rate + c.sell_rate)) < 1e-12


def test_sector_exposure_sums_to_weights():
    w = pd.Series([0.3, 0.3, 0.2, 0.2], index=["VCB", "TCB", "FPT", "HPG"])
    sec = vn_market.sector_exposure(w)
    assert abs(float(sec["Trọng số"].sum()) - 1.0) < 1e-9
    bank = sec[sec["Ngành"] == "Ngân hàng"]["Trọng số"].iloc[0]
    assert abs(bank - 0.6) < 1e-9


def test_price_limit_hits_counted():
    px = vn_market.simulate_vn_prices(["SSI", "VCB"], "2020-01-01", "2026-01-01")
    hits = vn_market.price_limit_hits(px)
    assert len(hits) == 2
    assert (hits["Tỷ lệ chạm biên"] >= 0).all()


def test_vn_universe_table():
    t = vn_market.universe_table()
    assert len(t) > 10
    assert set(t["Sàn"]).issubset({"HOSE", "HNX", "UPCOM"})


# ==========================================================================
# PHẦN 3 — Nhật ký giao dịch
# ==========================================================================


def test_trade_roundtrip(tmpdb):
    tid = journal.record_trade(journal.Trade("2026-01-05", "AAPL", "MUA", 100, 150), tmpdb)
    assert tid > 0
    df = journal.all_trades(tmpdb)
    assert len(df) == 1
    assert df["Mã"].iloc[0] == "AAPL"


def test_weighted_average_cost(tmpdb):
    journal.record_trade(journal.Trade("2026-01-05", "AAPL", "MUA", 100, 100), tmpdb)
    journal.record_trade(journal.Trade("2026-02-05", "AAPL", "MUA", 100, 200), tmpdb)
    pos = journal.current_positions(tmpdb)
    assert abs(float(pos["Giá vốn BQ"].iloc[0]) - 150.0) < 1e-9


def test_realized_pnl(tmpdb):
    journal.record_trade(journal.Trade("2026-01-05", "AAPL", "MUA", 100, 100), tmpdb)
    journal.record_trade(journal.Trade("2026-03-05", "AAPL", "BAN", 50, 150), tmpdb)
    pos = journal.current_positions(tmpdb)
    assert abs(float(pos["Lãi lỗ đã thực hiện"].iloc[0]) - 2500.0) < 1e-6
    assert abs(float(pos["Khối lượng"].iloc[0]) - 50.0) < 1e-9


def test_mark_to_market(tmpdb, prices):
    journal.record_trade(journal.Trade("2026-01-05", "AAPL", "MUA", 10, 100), tmpdb)
    mtm = journal.mark_to_market(prices, tmpdb)
    assert "Lãi lỗ chưa thực hiện" in mtm.columns
    assert not mtm["Giá thị trường"].isna().all()


def test_discipline_rate(tmpdb):
    journal.record_trade(journal.Trade("2026-01-05", "AAPL", "MUA", 10, 100, followed=True), tmpdb)
    journal.record_trade(journal.Trade("2026-01-06", "MSFT", "MUA", 10, 100, followed=False), tmpdb)
    d = journal.discipline_report(tmpdb).set_index("Chỉ tiêu")
    assert abs(float(d.loc["Tỷ lệ tuân thủ", "Giá trị"]) - 0.5) < 1e-9


def test_execution_gap(tmpdb):
    journal.save_snapshot(100_000, 100_000, "2026-01-01", db_path=tmpdb)
    journal.save_snapshot(105_000, 110_000, "2026-06-01", db_path=tmpdb)
    gap = journal.execution_gap(tmpdb)
    assert gap["Có đủ dữ liệu"] is True
    assert gap["Khoảng cách thực thi"] < 0   # thực tế kém hơn mô hình


def test_snapshot_upsert(tmpdb):
    journal.save_snapshot(100, 100, "2026-01-01", db_path=tmpdb)
    journal.save_snapshot(200, 200, "2026-01-01", db_path=tmpdb)
    hist = journal.nav_history(tmpdb)
    assert len(hist) == 1
    assert float(hist["NAV thực tế"].iloc[0]) == 200.0


def test_clear_journal(tmpdb):
    journal.record_trade(journal.Trade("2026-01-05", "AAPL", "MUA", 10, 100), tmpdb)
    journal.save_snapshot(100, 100, "2026-01-01", db_path=tmpdb)
    n = journal.clear_journal(tmpdb)
    assert n["Giao dịch"] == 1 and n["Ảnh chụp NAV"] == 1
    assert journal.all_trades(tmpdb).empty
