"""Test Portfolio Engine — P2 sở hữu."""
import numpy as np
import pytest

from core.costs import apply_costs, rebalance_dates
from core.portfolio import build_portfolio, ma_crossover_signals, normalize_weights


def test_chuan_hoa_trong_so():
    np.testing.assert_allclose(normalize_weights([1, 1, 1], 3), [1/3]*3)
    np.testing.assert_allclose(normalize_weights([2, 2], 2), [0.5, 0.5])
    np.testing.assert_allclose(normalize_weights([0, 0, 0], 3), [1/3]*3)  # fallback
    np.testing.assert_allclose(normalize_weights([1, 2], 3), [1/3]*3)     # sai size


def test_trong_so_luon_tong_bang_1(pf):
    assert abs(pf.weights.sum() - 1.0) < 1e-9


def test_metrics_du_khoa_bat_buoc(pf):
    assert {"total_return", "volatility", "sharpe", "alpha",
            "benchmark_total_return", "cagr"} <= set(pf.metrics)


def test_alpha_dung_dinh_nghia(pf):
    m = pf.metrics
    assert abs(m["alpha"] - (m["total_return"] - m["benchmark_total_return"])) < 1e-6


def test_benchmark_align_cung_index(pf):
    assert pf.returns.index.equals(pf.benchmark_returns.index)


def test_cumulative_khop_total_return(pf):
    assert abs((pf.cumulative.iloc[-1] - 1) * 100 - pf.metrics["total_return"]) < 1e-6


def test_phi_giao_dich_lam_giam_loi_nhuan(md):
    """So sánh CÔNG BẰNG: cùng tần suất tái cân bằng, chỉ khác mức phí.
    (Nếu so buy&hold không phí với rebalance có phí thì trộn lẫn hai hiệu ứng.)"""
    free = build_portfolio(md, [1, 1, 1], cost_bps=0, rebalance="Tháng")
    paid = build_portfolio(md, [1, 1, 1], cost_bps=30, rebalance="Tháng")
    assert paid.metrics["total_return"] < free.metrics["total_return"]
    assert paid.cost_drag_pct > 0
    assert free.cost_drag_pct == pytest.approx(0.0, abs=1e-9)


def test_phi_cang_cao_hao_hut_cang_lon(md):
    drags = [build_portfolio(md, [1, 1, 1], cost_bps=b,
                             rebalance="Tháng").cost_drag_pct
             for b in (5, 15, 40)]
    assert drags[0] < drags[1] < drags[2]


def test_khong_tai_can_bang_thi_khong_ton_phi(md):
    p = build_portfolio(md, [1, 1, 1], cost_bps=30, rebalance="Không")
    assert p.turnover.sum() == 0


def test_turnover_khong_am(md):
    _, _, turn = apply_costs(md.returns, np.array([1/3]*3), 10, "Quý")
    assert (turn >= 0).all()


def test_tin_hieu_chi_nhan_buy_hoac_sell(md):
    assert ma_crossover_signals(md).current in {"BUY", "SELL"}


# ---------------------------------------------------------------------------
# PHÁT TRIỂN THÊM #2 của P2 — biến tỷ trọng thành phiếu lệnh thực thi
# ---------------------------------------------------------------------------
from core.sizing import LOT_SIZE, build_orders, summarize, to_csv


def test_khong_bao_gio_vuot_von(md):
    px = md.prices.iloc[-1]
    o = build_orders(px, np.array([1/3]*3), capital=50_000)
    assert o["Giá trị"].sum() <= 50_000


def test_so_cp_la_boi_so_cua_lo(md):
    px = md.prices.iloc[-1]
    o = build_orders(px, np.array([1/3]*3), capital=500_000, market="HOSE")
    assert (o["Số CP"] % LOT_SIZE["HOSE"] == 0).all()


def test_so_cp_khong_am(md):
    px = md.prices.iloc[-1]
    o = build_orders(px, np.array([0.5, 0.3, 0.2]), capital=100_000)
    assert (o["Số CP"] >= 0).all()


def test_ty_trong_thuc_te_tong_bang_100(md):
    px = md.prices.iloc[-1]
    o = build_orders(px, np.array([1/3]*3), capital=200_000)
    assert abs(o["Tỷ trọng thực tế %"].sum() - 100) < 1e-6


def test_von_cang_lon_sai_lech_cang_nho(md):
    """Làm tròn lô gây sai lệch — vốn nhỏ thì lệch nhiều."""
    px = md.prices.iloc[-1]
    w = np.array([1/3]*3)
    nho = summarize(build_orders(px, w, 20_000), 20_000)["max_drift_pct"]
    lon = summarize(build_orders(px, w, 5_000_000), 5_000_000)["max_drift_pct"]
    assert lon < nho


def test_lo_100_lech_nhieu_hon_lo_1(md):
    """Lô 100 của HOSE gây sai lệch lớn hơn lô 1 của Mỹ, cùng mức vốn."""
    px = md.prices.iloc[-1]
    w = np.array([1/3]*3)
    us = summarize(build_orders(px, w, 100_000, "US"), 100_000)
    vn = summarize(build_orders(px, w, 100_000, "HOSE"), 100_000)
    assert vn["max_drift_pct"] >= us["max_drift_pct"]


def test_tien_mat_con_lai_khong_am(md):
    px = md.prices.iloc[-1]
    s = summarize(build_orders(px, np.array([1/3]*3), 75_000), 75_000)
    assert s["cash_left"] >= 0


def test_xuat_csv_co_header(md):
    px = md.prices.iloc[-1]
    csv = to_csv(build_orders(px, np.array([1/3]*3), 100_000))
    assert csv.startswith("Mã,Giá,Số CP")
