"""Kiểm thử cho gói nâng cấp.

Các test này độc lập hoàn toàn với ``tests/`` sẵn có của bạn: không import
module cũ, không đụng fixture cũ, tên file cũng không trùng. Chạy riêng bằng::

    pytest tests/test_upgrade_pack.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import alloc_lab, hmm_regime, stress, validation
from core import upgrade_api as api
from core import wf_backtest as wf

TICKERS = ["AAPL", "MSFT", "NVDA", "JNJ", "XOM", "GLD", "TLT"]


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return api.synthetic_prices(TICKERS + ["SPY", "QQQ", "IWM", "^VIX"], "2018-01-01", "2026-06-30")


@pytest.fixture(scope="module")
def rets(prices) -> pd.DataFrame:
    return api.to_returns(prices[TICKERS])


@pytest.fixture(scope="module")
def spy(prices) -> pd.Series:
    return api.to_returns(prices[["SPY"]])["SPY"]


# --------------------------------------------------------------------------
# Tầng dữ liệu
# --------------------------------------------------------------------------


def test_synthetic_prices_deterministic():
    a = api.synthetic_prices(["AAPL", "SPY"], "2020-01-01", "2022-01-01")
    b = api.synthetic_prices(["AAPL", "SPY"], "2020-01-01", "2022-01-01")
    pd.testing.assert_frame_equal(a, b)


def test_prices_are_positive_and_clean(prices):
    assert (prices > 0).all().all()
    assert not prices.isna().any().any()
    assert prices.index.is_monotonic_increasing


def test_returns_bounded(rets):
    assert rets.abs().max().max() < 1.0  # không có lợi suất ngày phi lý


# --------------------------------------------------------------------------
# Ước lượng tham số
# --------------------------------------------------------------------------


def test_ledoit_wolf_is_positive_semidefinite(rets):
    cov = alloc_lab.ledoit_wolf_cov(rets)
    eig = np.linalg.eigvalsh(cov.to_numpy())
    assert eig.min() > -1e-10
    assert cov.shape == (rets.shape[1], rets.shape[1])


def test_shrunk_mean_pulls_toward_grand_mean(rets):
    raw = rets.mean() * 252
    shrunk = alloc_lab.shrunk_mean(rets)
    # phương sai giữa các mã phải giảm sau khi co ngót
    assert shrunk.var() <= raw.var() + 1e-12


def test_black_litterman_respects_views(rets):
    cov = alloc_lab.ledoit_wolf_cov(rets)
    n = rets.shape[1]
    base = alloc_lab.black_litterman(cov, np.repeat(1 / n, n))
    tilt = alloc_lab.black_litterman(cov, np.repeat(1 / n, n), {"NVDA": 0.50}, confidence=0.9)
    assert tilt["NVDA"] > base["NVDA"]


# --------------------------------------------------------------------------
# Thuật toán phân bổ
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", list(alloc_lab.METHODS))
def test_weights_valid(rets, method):
    mu, cov = alloc_lab.estimate(rets)
    res = alloc_lab.allocate(method, mu, cov, max_weight=0.5)
    w = res.weights.to_numpy()
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert (w >= -1e-9).all()
    assert len(w) == rets.shape[1]


def test_min_variance_has_lowest_volatility(rets):
    mu, cov = alloc_lab.estimate(rets)
    table, res = alloc_lab.compare_methods(mu, cov, max_weight=0.5)
    mv = res["Phương sai tối thiểu"].volatility
    assert mv <= min(r.volatility for r in res.values()) + 1e-8


def test_hrp_more_diversified_than_max_sharpe(rets):
    mu, cov = alloc_lab.estimate(rets)
    hrp_n = alloc_lab.effective_n(alloc_lab.allocate("HRP", mu, cov).weights)
    ms_n = alloc_lab.effective_n(
        alloc_lab.allocate("Max Sharpe", mu, cov, max_weight=1.0).weights
    )
    assert hrp_n >= ms_n


def test_max_weight_constraint(rets):
    mu, cov = alloc_lab.estimate(rets)
    res = alloc_lab.allocate("Max Sharpe", mu, cov, max_weight=0.25)
    assert res.weights.max() <= 0.25 + 1e-6


# --------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------


def test_walk_forward_runs(rets):
    res = wf.run_walk_forward(rets, "HRP")
    assert len(res.net) > 100
    assert res.weights.shape[1] == rets.shape[1]
    assert np.allclose(res.weights.sum(axis=1), 1.0, atol=1e-6)


def test_costs_reduce_returns(rets):
    res = wf.run_walk_forward(rets, "Max Sharpe")
    assert res.net.sum() <= res.gross.sum() + 1e-12
    assert (res.turnover >= -1e-12).all()


def test_walk_forward_rejects_short_history():
    short = api.to_returns(api.synthetic_prices(["AAPL", "MSFT"], "2025-01-01", "2025-06-30"))
    with pytest.raises(ValueError):
        wf.run_walk_forward(short, "HRP")


def test_out_of_sample_sharpe_below_in_sample(rets):
    cmp = wf.in_sample_vs_out_of_sample(rets, "Max Sharpe")
    assert cmp["Sharpe in-sample"].iloc[0] > cmp["Sharpe out-of-sample (sau phí)"].iloc[0]


def test_cost_sensitivity_monotone(rets):
    res = wf.run_walk_forward(rets, "Max Sharpe")
    sens = wf.cost_sensitivity(res)
    assert sens["Sharpe"].is_monotonic_decreasing


# --------------------------------------------------------------------------
# Kiểm định thống kê
# --------------------------------------------------------------------------


def test_psr_in_unit_interval(rets):
    res = wf.run_walk_forward(rets, "HRP")
    psr = validation.probabilistic_sharpe(res.net)
    assert 0.0 <= psr <= 1.0


def test_deflated_sharpe_penalises_more_trials(rets):
    res = wf.run_walk_forward(rets, "HRP")
    assert validation.deflated_sharpe(res.net, 50) <= validation.deflated_sharpe(res.net, 1)


def test_bootstrap_ci_brackets_point_estimate(rets):
    res = wf.run_walk_forward(rets, "HRP")
    ci = validation.confidence_intervals(res.net, n_boot=200)
    row = ci[ci["Chỉ số"] == "Sharpe"].iloc[0]
    assert row["Cận dưới"] <= row["Giá trị thực tế"] <= row["Cận trên"]


# --------------------------------------------------------------------------
# Chế độ thị trường
# --------------------------------------------------------------------------


def test_hmm_fits_three_regimes(spy):
    m = hmm_regime.fit_regimes(spy)
    assert set(m.transition.index) == {"BEAR", "NEUTRAL", "BULL"}
    assert np.allclose(m.transition.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(m.probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_regimes_are_persistent(spy):
    m = hmm_regime.fit_regimes(spy)
    # chế độ thị trường phải dai dẳng, không nhảy mỗi ngày
    assert (np.diag(m.transition.to_numpy()) > 0.5).all()


def test_bear_regime_has_lowest_return(spy):
    m = hmm_regime.fit_regimes(spy)
    stats_df = hmm_regime.regime_statistics(spy, m.states).set_index("Chế độ")
    assert stats_df.loc["BEAR", "LN năm hoá"] < stats_df.loc["BULL", "LN năm hoá"]


def test_regime_backtest_uses_lag(rets, spy):
    m = hmm_regime.fit_regimes(spy)
    k = rets.shape[1]
    out = hmm_regime.regime_switching_backtest(rets, m.states, np.repeat(1 / k, k))
    assert len(out) > 0
    assert out.index[0] > m.states.index[0]  # có độ trễ, không dùng tín hiệu cùng phiên


# --------------------------------------------------------------------------
# Kịch bản
# --------------------------------------------------------------------------


def test_market_crash_hurts_long_portfolio(rets, prices):
    k = rets.shape[1]
    pr = pd.Series(rets.to_numpy() @ np.repeat(1 / k, k), index=rets.index)
    fr = api.to_returns(prices[["SPY", "QQQ", "IWM", "^VIX"]])
    out = stress.scenario_shock(pr, fr, {"SPY": -0.20})
    assert out["Tác động tổng"] < 0


def test_monte_carlo_probabilities_valid(rets):
    k = rets.shape[1]
    pr = pd.Series(rets.to_numpy() @ np.repeat(1 / k, k), index=rets.index)
    summ = stress.monte_carlo_summary(stress.monte_carlo_paths(pr, n_paths=500)).set_index(
        "Chỉ tiêu"
    )
    assert 0.0 <= summ.loc["Xác suất thua lỗ", "Giá trị"] <= 1.0
    assert summ.loc["Sụt giảm trung vị", "Giá trị"] <= 0.0


def test_fat_tails_produce_wider_distribution(rets):
    k = rets.shape[1]
    pr = pd.Series(rets.to_numpy() @ np.repeat(1 / k, k), index=rets.index)
    normal = stress.monte_carlo_paths(pr, n_paths=2000, fat_tails=False)
    fat = stress.monte_carlo_paths(pr, n_paths=2000, fat_tails=True)
    assert fat["Lợi nhuận"].quantile(0.01) < normal["Lợi nhuận"].quantile(0.01)
