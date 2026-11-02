"""Test tầng dữ liệu — P1 sở hữu."""
import pandas as pd
import pytest

from core.cache import _key, clear_cache
from core.contracts import MarketData
from core.data import health_check
from core.fixtures import FACTOR_TICKERS, generate_sample, load_sample


def test_sample_tra_ve_marketdata(md):
    assert isinstance(md, MarketData)
    assert md.source == "sample"
    assert len(md.tickers) == 3


def test_du_lieu_khong_rong_va_khong_nan(md):
    assert len(md.prices) > 500
    assert md.prices.isna().sum().sum() == 0
    assert list(md.factors.columns) == FACTOR_TICKERS


def test_returns_tinh_dung(md):
    r = md.returns
    assert len(r) == len(md.prices) - 1
    assert r.abs().max().max() < 0.5  # không có lợi suất ngày phi lý


def test_vix_nghich_bien_voi_spy(md):
    c = md.factor_returns["^VIX"].corr(md.factor_returns["SPY"])
    assert c < -0.3, "VIX phải nghịch biến với SPY"


def test_cache_key_on_dinh_va_khac_nhau():
    assert _key(["A", "B"], "2023-01-01", "px") == _key(["B", "A"], "2023-01-01", "px")
    assert _key(["A"], "2023-01-01", "px") != _key(["A"], "2024-01-01", "px")


def test_health_check_du_truong(md):
    hc = health_check(md)
    assert set(hc) >= {"rows", "tickers", "nan_pct", "from", "to", "source"}


def test_sinh_du_lieu_co_tai_lap(tmp_path):
    a, _ = generate_sample(seed=99)
    b, _ = generate_sample(seed=99)
    pd.testing.assert_frame_equal(a, b)
