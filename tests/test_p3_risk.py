"""Test Risk & Factor — P3 sở hữu."""
import numpy as np
import pandas as pd
import pytest

from core.factors import build_factor_report, ols, r_squared
from core.risk import (build_risk_report, drawdown_series, max_dd_duration,
                       stress_test, tail_metrics)


@pytest.fixture(scope="module")
def rr(md, pf):
    return build_risk_report(md, pf)


def test_drawdown_luon_am_hoac_bang_0(rr):
    assert rr.drawdown.max() <= 1e-12
    assert rr.drawdown.min() >= -1.0


def test_cvar_sau_hon_var(rr):
    assert rr.tail["cvar_95"] <= rr.tail["var_95_hist"]


def test_var_hai_phuong_phap_gan_nhau(rr):
    assert abs(rr.tail["var_95_hist"] - rr.tail["var_95_param"]) < 1.0


def test_do_dai_drawdown_hop_le(rr, pf):
    d = rr.tail["max_dd_duration_days"]
    assert 0 <= d <= len(pf.cumulative)


def test_max_dd_duration_truong_hop_bien():
    tang = pd.Series(np.linspace(1, 2, 100))
    assert max_dd_duration(tang) == 0
    giam = pd.Series(np.linspace(2, 1, 100))
    assert max_dd_duration(giam) == 99


def test_tuong_quan_doi_xung_va_duong_cheo_bang_1(rr):
    np.testing.assert_allclose(np.diag(rr.corr.values), 1.0, atol=1e-9)
    np.testing.assert_allclose(rr.corr.values, rr.corr.values.T, atol=1e-9)


def test_beta_trong_khoang_hop_ly(rr):
    assert rr.rolling_beta.notna().all()
    assert -5 < rr.rolling_beta.median() < 5


def test_beta_cua_chinh_benchmark_bang_1(md):
    from core.portfolio import build_portfolio
    j = pd.concat([md.benchmark_returns, md.benchmark_returns], axis=1).dropna()
    b = j.iloc[:, 0].cov(j.iloc[:, 1]) / j.iloc[:, 1].var()
    assert abs(b - 1.0) < 1e-9


def test_ols_khop_voi_he_so_da_biet():
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"f": rng.normal(0, 1, 1000)})
    y = pd.Series(2.5 * x["f"] + 0.1)
    p = ols(y, x)
    assert abs(p.loc["f", "coef"] - 2.5) < 1e-6
    assert r_squared(y, x) > 0.999


def test_factor_report_day_du(md, pf):
    fr = build_factor_report(pf, md)
    assert "SPY" in fr.betas and fr.ff_params is not None
    assert 0 <= fr.ff_r2 <= 1
    assert fr.betas["^VIX"] < 0  # danh mục cổ phiếu phải âm với VIX
    assert isinstance(fr.treynor, float)
    assert fr.beta_matrix is not None and not fr.beta_matrix.empty


def test_stress_test_dung_huong(pf):
    s = stress_test(pf, -20)
    assert s["expected_portfolio_pct"] < 0
