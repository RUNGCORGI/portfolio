"""Test Optimizer & Regime — P4 sở hữu."""
import numpy as np
import pytest

from core.optimizer import METHODS, compare_methods, monte_carlo, optimize
from core.regime import detect_regime, switching_strategy, walk_forward


@pytest.mark.parametrize("method", METHODS)
def test_moi_phuong_phap_cho_trong_so_hop_le(md, method):
    o = optimize(md.returns, method, max_weight=0.5)
    assert abs(o.weights.sum() - 1) < 1e-6
    assert (o.weights >= -1e-9).all(), "không được bán khống"


def test_ton_trong_tran_ty_trong(md):
    o = optimize(md.returns, "max_sharpe", max_weight=0.4)
    assert o.weights.max() <= 0.4 + 1e-6


def test_max_sharpe_tot_hon_equal_weight(md):
    a = optimize(md.returns, "max_sharpe", 1.0).sharpe
    b = optimize(md.returns, "equal_weight").sharpe
    assert a >= b - 1e-9


def test_min_variance_co_vol_thap_nhat(md):
    mv = optimize(md.returns, "min_variance", 1.0).volatility
    for m in ["max_sharpe", "equal_weight", "risk_parity", "hrp"]:
        assert mv <= optimize(md.returns, m, 1.0).volatility + 1e-6


def test_risk_parity_phan_tan_hon_max_sharpe(md):
    rp = optimize(md.returns, "risk_parity", 1.0).weights
    ms = optimize(md.returns, "max_sharpe", 1.0).weights
    assert rp.max() <= ms.max() + 1e-9


def test_hrp_khong_tap_trung_het_vao_1_ma(md):
    assert optimize(md.returns, "hrp").weights.max() < 0.95


def test_monte_carlo_sinh_du_so_luong(md):
    mc = monte_carlo(md.returns, 500)
    assert len(mc) == 500
    assert (mc["Volatility"] > 0).all()


def test_monte_carlo_khong_vuot_qua_toi_uu(md):
    mc = monte_carlo(md.returns, 3000)
    opt = optimize(md.returns, "max_sharpe", 1.0)
    assert mc["Sharpe"].max() <= opt.sharpe + 1e-3


def test_bang_so_sanh_du_dong(md):
    assert len(compare_methods(md.returns)) == len(METHODS)


def test_regime_chi_co_3_trang_thai(md):
    r = detect_regime(md)
    assert set(r.state.unique()) <= {"BULL", "NEUTRAL", "BEAR"}
    assert r.current in {"BULL", "NEUTRAL", "BEAR"}


def test_chien_luoc_regime_sinh_chuoi_hop_le(md, pf):
    ss = switching_strategy(md, pf, detect_regime(md))
    assert ss.strategy_returns is not None
    assert ss.strategy_cumulative.iloc[0] > 0


def test_walk_forward_la_out_of_sample_that(md):
    wf = walk_forward(md, 252, 63)
    assert "error" not in wf
    assert wf["n_rebalances"] >= 2
    assert len(wf["oos_returns"]) == len(wf["ins_returns"])


def test_in_sample_thuong_dep_hon_oos(md):
    """Bằng chứng look-ahead bias của bài gốc."""
    wf = walk_forward(md, 252, 63)
    assert wf["in_sample"]["sharpe"] >= wf["oos"]["sharpe"]
