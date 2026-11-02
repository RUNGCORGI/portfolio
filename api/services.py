"""Tầng dịch vụ — cầu nối giữa ``core/`` và lớp HTTP.

Toàn bộ logic định lượng nằm nguyên trong ``core/``. Tầng này chỉ làm ba việc:

1. Nạp dữ liệu và **cache theo chữ ký tham số**, để nhiều lời gọi API liên
   tiếp không phải tính lại từ đầu.
2. Chuyển DataFrame/Series của pandas sang cấu trúc JSON gọn nhẹ.
3. Bắt lỗi và biến thành thông điệp tiếng Việt dễ hiểu.

Nhờ tách lớp như vậy, cùng một bộ ``core/`` phục vụ được cả giao diện
Streamlit lẫn giao diện web mới, không phải viết lại dòng nào.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from datetime import datetime
from typing import Any, Sequence

import numpy as np
import pandas as pd

from core import (
    alerts,
    alloc_lab,
    factor_lab,
    hmm_regime,
    stress,
    validation,
    xray,
)
from core import upgrade_api as api
from core import vn_market as vnm
from core import wf_backtest as wf

PROXIES = ["SPY", "QQQ", "IWM", "^VIX", "GLD", "TLT"]
BENCHMARK = "SPY"

# --------------------------------------------------------------------------
# Cache trong bộ nhớ, an toàn đa luồng
# --------------------------------------------------------------------------

_CACHE: dict[str, Any] = {}
_LOCK = threading.Lock()
_MAX_ENTRIES = 48


def _key(*parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _cached(key: str, builder):
    """Lấy từ cache hoặc tính mới. Cache có giới hạn để không phình bộ nhớ."""
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
    value = builder()
    with _LOCK:
        if len(_CACHE) >= _MAX_ENTRIES:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = value
    return value


def clear_cache() -> int:
    with _LOCK:
        n = len(_CACHE)
        _CACHE.clear()
    return n


# --------------------------------------------------------------------------
# Tiện ích chuyển đổi sang JSON
# --------------------------------------------------------------------------


def _clean(v: Any) -> Any:
    """Biến NaN/Inf thành None để JSON hợp lệ."""
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.strftime("%Y-%m-%d")
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v


def series_to_points(s: pd.Series, max_points: int = 900) -> dict:
    """Chuỗi thời gian → {dates, values}, có giảm mẫu nếu quá dài.

    Giảm mẫu giữ nguyên điểm đầu và cuối, nên hình dạng đường vốn không bị
    méo — quan trọng khi người xem đọc mức sụt giảm tối đa trên biểu đồ.
    """
    s = pd.Series(s).dropna()
    if len(s) > max_points:
        step = max(1, len(s) // max_points)
        idx = list(range(0, len(s), step))
        if idx[-1] != len(s) - 1:
            idx.append(len(s) - 1)
        s = s.iloc[idx]
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in s.index],
        "values": [_clean(v) for v in s.to_numpy()],
    }


def frame_to_table(df: pd.DataFrame, index_name: str | None = None) -> dict:
    """DataFrame → {columns, rows} để frontend dựng bảng."""
    if df is None or df.empty:
        return {"columns": [], "rows": []}
    d = df.reset_index() if index_name else df.copy()
    if index_name:
        d = d.rename(columns={d.columns[0]: index_name})
    return {
        "columns": [str(c) for c in d.columns],
        "rows": [[_clean(v) for v in row] for row in d.to_numpy()],
    }


def dict_clean(d: dict) -> dict:
    return {str(k): _clean(v) for k, v in d.items()}


# --------------------------------------------------------------------------
# Nạp dữ liệu
# --------------------------------------------------------------------------


class DataError(Exception):
    """Lỗi có thể hiển thị trực tiếp cho người dùng."""


def load_market(tickers: Sequence[str], start: str, market: str = "us") -> dict:
    """Nạp giá cho danh mục và các mã đại diện nhân tố."""
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if len(tickers) < 2:
        raise DataError("Cần ít nhất 2 mã để phân tích danh mục.")

    def build():
        if market == "vn":
            # KIỂM TRA CHÉO: Nếu người dùng nhập mã Mỹ vào thị trường VN
            valid_vn = [t for t in tickers if t in vnm.VN_UNIVERSE]
            if not valid_vn:
                raise DataError(
                    "Không tìm thấy mã Việt Nam hợp lệ trong danh sách bạn nhập. "
                    "Vui lòng nhập đúng mã VN (ví dụ: VCB, FPT, HPG...)."
                )
            wanted = sorted(set(valid_vn) | {vnm.VN_BENCHMARK})
            prices = vnm.VNStockProvider().get_prices(wanted, start, None)
            bench_name = vnm.VN_BENCHMARK
        else:
            # KIỂM TRA CHÉO: Nếu người dùng nhập mã VN vào thị trường Mỹ
            valid_us = [t for t in tickers if t not in vnm.VN_UNIVERSE]
            if not valid_us:
                raise DataError(
                    "Không tìm thấy mã Mỹ hợp lệ trong danh sách bạn nhập. "
                    "Vui lòng chuyển sang thị trường Việt Nam nếu muốn dùng các mã này."
                )
            wanted = sorted(set(valid_us) | set(PROXIES))
            try:
                prices = api.load_prices(wanted, start, None)
            except Exception as e:
                raise DataError(f"Lỗi tải dữ liệu: {str(e)}")
                
            cols = [t for t in tickers if t in prices.columns]
            if len(cols) < len(tickers):
                missing = set(tickers) - set(cols)
                raise DataError(f"Không tìm thấy dữ liệu cho các mã: {', '.join(missing)}. Vui lòng kiểm tra lại mã.")
            bench_name = BENCHMARK

        prices = api.clean_prices(prices)
        cols = [t for t in tickers if t in prices.columns]
        if len(cols) < 2:
            raise DataError("Không lấy được đủ dữ liệu cho các mã đã nhập.")

        rets = api.to_returns(prices[cols])
        bench = (
            api.to_returns(prices[[bench_name]])[bench_name]
            if bench_name in prices.columns
            else None
        )
        return {
            "prices": prices,
            "returns": rets,
            "benchmark": bench,
            "benchmark_name": bench_name,
            "tickers": cols,
            "market": market,
        }

    return _cached(_key("market", tickers, start, market), build)


# --------------------------------------------------------------------------
# Tổng quan danh mục
# --------------------------------------------------------------------------


def overview(tickers, start, method="HRP", market="us", robust=True, max_weight=0.4) -> dict:
    """Số liệu cho màn hình chính: chỉ số chính, đường vốn, phân bổ."""

    def build():
        ctx = load_market(tickers, start, market)
        prices, rets, bench = ctx["prices"], ctx["returns"], ctx["benchmark"]

        mu, cov = alloc_lab.estimate(rets, robust=robust)
        alloc = alloc_lab.allocate(method, mu, cov, max_weight=max_weight)
        pr = pd.Series(rets.to_numpy() @ alloc.weights.to_numpy(), index=rets.index)

        ts = xray.tearsheet(pr, bench)
        equity = (1 + pr).cumprod()
        dd = equity / equity.cummax() - 1

        curves = {"Danh mục": series_to_points(equity)}
        if bench is not None:
            b = pd.Series(bench).reindex(pr.index).fillna(0.0)
            curves["Thị trường"] = series_to_points((1 + b).cumprod())

        rc = xray.risk_contribution(alloc.weights, cov)
        conc = xray.concentration_metrics(alloc.weights)

        # --- TÍNH TOÁN DỮ LIỆU CHO CÁC BIỂU ĐỒ MỚI ---
        # 1. Price Chart & MA
        price_data = None
        first_ticker = ctx["tickers"][0]
        if first_ticker in prices.columns:
            px = prices[first_ticker].dropna()
            ma20 = px.rolling(20).mean()
            ma50 = px.rolling(50).mean()
            
            # Histogram lợi nhuận
            ret_series = pr.dropna()
            hist_counts, hist_edges = np.histogram(ret_series, bins=30)

            price_data = {
                "dates": [d.strftime("%Y-%m-%d") for d in px.index],
                "prices": [_clean(v) for v in px.values],
                "ma20": [_clean(v) for v in ma20.values],
                "ma50": [_clean(v) for v in ma50.values],
                "hist_counts": [int(c) for c in hist_counts],
                "hist_edges": [_clean(e) for e in hist_edges]
            }

        # 2. Macro Regime (SPY > MA200 & QQQ > MA200 & VIX < 20)
        macro_regime = "N/A"
        spy = prices.get(BENCHMARK)
        qqq = prices.get("QQQ")
        vix = prices.get("^VIX")
        if spy is not None and qqq is not None and vix is not None:
            spy_ma200 = spy.rolling(200).mean()
            qqq_ma200 = qqq.rolling(200).mean()
            risk_on = (spy > spy_ma200) & (qqq > qqq_ma200) & (vix < 20)
            last_risk = risk_on.dropna()
            if len(last_risk) > 0:
                macro_regime = "RISK ON" if bool(last_risk.iloc[-1]) else "RISK OFF"

        return {
            "metrics": dict_clean(ts),
            "concentration": dict_clean(conc),
            "curves": curves,
            "drawdown": series_to_points(dd),
            "price_data": price_data,
            "macro_regime": macro_regime,
            "allocation": [
                {"ticker": str(t), "weight": _clean(w)}
                for t, w in alloc.weights.sort_values(ascending=False).items()
            ],
            "risk_contribution": [
                {
                    "ticker": str(r["Mã"]),
                    "capital": _clean(r["Trọng số vốn"]),
                    "risk": _clean(r["Tỷ trọng rủi ro"]),
                    "gap": _clean(r["Chênh lệch"]),
                }
                for _, r in rc.iterrows()
            ],
            "method": method,
            "tickers": ctx["tickers"],
            "period": {
                "start": ctx["returns"].index[0].strftime("%Y-%m-%d"),
                "end": ctx["returns"].index[-1].strftime("%Y-%m-%d"),
                "sessions": int(len(rets)),
            },
            "market": market,
        }

    return _cached(_key("overview", tickers, start, method, market, robust, max_weight), build)


def compare_allocations(tickers, start, market="us", robust=True, max_weight=0.4) -> dict:
    """So sánh sáu thuật toán phân bổ trên cùng bộ tham số."""

    def build():
        ctx = load_market(tickers, start, market)
        rets = ctx["returns"]
        mu, cov = alloc_lab.estimate(rets, robust=robust)
        table, results = alloc_lab.compare_methods(mu, cov, max_weight=max_weight)

        return {
            "table": frame_to_table(table),
            "weights": {
                name: [
                    {"ticker": str(t), "weight": _clean(w)}
                    for t, w in res.weights.items()
                ]
                for name, res in results.items()
            },
            "scatter": [
                {
                    "method": name,
                    "vol": _clean(res.volatility),
                    "ret": _clean(res.expected_return),
                    "sharpe": _clean(res.sharpe),
                    "effective_n": _clean(alloc_lab.effective_n(res.weights)),
                }
                for name, res in results.items()
            ],
        }

    return _cached(_key("alloc", tickers, start, market, robust, max_weight), build)


# --------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------


def walkforward(tickers, start, methods, market="us", cost_bps=10.0, lookback=252, step=21) -> dict:
    """Chạy walk-forward cho nhiều phương pháp, kèm phân tích chi phí."""

    def build():
        ctx = load_market(tickers, start, market)
        rets, bench = ctx["returns"], ctx["benchmark"]

        cfg = wf.WalkForwardConfig(
            lookback=lookback,
            step=step,
            costs=wf.CostModel(
                commission_bps=cost_bps * 0.5,
                spread_bps=cost_bps * 0.2,
                slippage_bps=cost_bps * 0.3,
            ),
        )
        if len(rets) <= cfg.lookback + cfg.embargo + cfg.step:
            raise DataError(
                "Lịch sử chưa đủ dài cho cấu hình này. Hãy giảm cửa sổ ước lượng "
                "hoặc chọn ngày bắt đầu sớm hơn."
            )

        table, results = wf.compare_walk_forward(rets, list(methods), cfg)
        if not results:
            raise DataError("Không chạy được walk-forward với cấu hình hiện tại.")

        curves = {name: series_to_points(res.equity) for name, res in results.items()}
        if bench is not None:
            first = next(iter(results.values()))
            b = pd.Series(bench).reindex(first.net.index).fillna(0.0)
            curves["Thị trường"] = series_to_points((1 + b).cumprod())

        best = max(results.items(), key=lambda kv: api.sharpe(kv[1].net))
        isoos = wf.in_sample_vs_out_of_sample(rets, best[0], cfg)
        sens = wf.cost_sensitivity(best[1])

        return {
            "table": frame_to_table(table),
            "curves": curves,
            "drawdowns": {
                name: series_to_points(
                    (1 + res.net).cumprod() / (1 + res.net).cumprod().cummax() - 1
                )
                for name, res in results.items()
            },
            "best_method": best[0],
            "in_sample_vs_oos": frame_to_table(isoos),
            "cost_sensitivity": frame_to_table(sens),
            "breakeven_bps": _clean(wf.breakeven_cost_bps(best[1], bench)),
            "turnover": {
                name: _clean(res.summary()["Vòng quay/năm"]) for name, res in results.items()
            },
        }

    return _cached(_key("wf", tickers, start, methods, market, cost_bps, lookback, step), build)


# --------------------------------------------------------------------------
# Kiểm định thống kê
# --------------------------------------------------------------------------


def validate(tickers, start, method="HRP", market="us", n_trials=20) -> dict:
    """Kiểm định: PSR, Deflated Sharpe, bootstrap, phân vị ngẫu nhiên."""

    def build():
        ctx = load_market(tickers, start, market)
        rets = ctx["returns"]
        try:
            res = wf.run_walk_forward(rets, method)
        except ValueError as exc:
            raise DataError(str(exc))

        rep = validation.validation_report(res.net, n_trials, rets)
        ci = validation.confidence_intervals(res.net, n_boot=600)
        boot = validation.block_bootstrap(res.net, n_boot=600)
        pct = validation.random_portfolio_percentile(rets, res.net, n_random=1200)

        hist, edges = (
            np.histogram(boot["sharpe"], bins=40) if not boot.empty else (np.array([]), np.array([]))
        )
        dsr_row = rep[rep["Chỉ tiêu"].str.startswith("DSR")]

        return {
            "report": frame_to_table(rep),
            "confidence": frame_to_table(ci),
            "bootstrap": {
                "counts": [int(c) for c in hist],
                "edges": [_clean(e) for e in edges],
                "observed": _clean(api.sharpe(res.net)),
            },
            "percentile": dict_clean(pct),
            "dsr": _clean(dsr_row["Giá trị"].iloc[0]) if len(dsr_row) else None,
            "passed": bool(len(dsr_row) and float(dsr_row["Giá trị"].iloc[0]) > 0.95),
            "method": method,
            "n_trials": n_trials,
        }

    return _cached(_key("val", tickers, start, method, market, n_trials), build)


# --------------------------------------------------------------------------
# Chế độ thị trường
# --------------------------------------------------------------------------


def _fit_regime_cached(bench: pd.Series, tickers, start, market):
    """Khớp HMM một lần rồi dùng chung.

    Màn hình chế độ thị trường và màn hình cảnh báo đều cần mô hình này.
    Khớp HMM mất khoảng 4 giây, nên dùng lại kết quả thay vì chạy hai lần.
    """
    return _cached(_key("hmm", tickers, start, market), lambda: hmm_regime.fit_regimes(bench))


def regimes(tickers, start, market="us") -> dict:
    """Mô hình Markov ẩn nhận diện chế độ thị trường."""

    def build():
        ctx = load_market(tickers, start, market)
        bench, rets = ctx["benchmark"], ctx["returns"]
        if bench is None or len(bench) < 300:
            raise DataError("Cần dữ liệu chỉ số tham chiếu đủ dài để nhận diện chế độ.")

        model = _fit_regime_cached(bench, tickers, start, market)
        k = rets.shape[1]
        eq = pd.Series(rets.to_numpy() @ np.repeat(1 / k, k), index=rets.index)
        strat = hmm_regime.regime_switching_backtest(rets, model.states, np.repeat(1 / k, k))

        probs = model.probabilities
        if len(probs) > 900:
            probs = probs.iloc[:: max(1, len(probs) // 900)]

        return {
            "current": model.current,
            "confidence": _clean(model.confidence),
            "converged": bool(model.converged),
            "probabilities": {
                "dates": [d.strftime("%Y-%m-%d") for d in probs.index],
                "series": {
                    str(c): [_clean(v) for v in probs[c].to_numpy()] for c in probs.columns
                },
            },
            "transition": frame_to_table(model.transition, index_name="Từ / Đến"),
            "duration": [
                {"regime": str(k_), "sessions": _clean(v)}
                for k_, v in model.expected_duration().items()
            ],
            "stationary": [
                {"regime": str(k_), "share": _clean(v)}
                for k_, v in model.stationary().items()
            ],
            "statistics": frame_to_table(hmm_regime.regime_statistics(eq, model.states)),
            "strategy": {
                "switching": series_to_points((1 + strat).cumprod()),
                "buy_hold": series_to_points(
                    (1 + eq.reindex(strat.index).fillna(0)).cumprod()
                ),
                "sharpe_switching": _clean(api.sharpe(strat)),
                "sharpe_buy_hold": _clean(api.sharpe(eq)),
            },
        }

    return _cached(_key("regime", tickers, start, market), build)


# --------------------------------------------------------------------------
# Nhân tố
# --------------------------------------------------------------------------


def factors(tickers, start, method="HRP", market="us") -> dict:
    """Mô hình nhân tố với sai số chuẩn Newey–West."""

    def build():
        ctx = load_market(tickers, start, market)
        prices, rets = ctx["prices"], ctx["returns"]
        avail = [p for p in PROXIES if p in prices.columns]
        if BENCHMARK not in avail:
            raise DataError("Chưa hỗ trợ phân tích nhân tố cho thị trường này.")

        F = factor_lab.assemble_factors(prices[avail], prices[ctx["tickers"]])
        mu, cov = alloc_lab.estimate(rets)
        w = alloc_lab.allocate(method, mu, cov, max_weight=0.4).weights
        pr = pd.Series(rets.to_numpy() @ w.to_numpy(), index=rets.index)

        model, tbl, attr = factor_lab.factor_report(pr, F)
        try:
            rb = factor_lab.rolling_exposures(pr, F, stride=5)
            rolling = {
                "dates": [d.strftime("%Y-%m-%d") for d in rb.index[:: max(1, len(rb) // 600)]],
                "series": {
                    str(c): [
                        _clean(v) for v in rb[c].to_numpy()[:: max(1, len(rb) // 600)]
                    ]
                    for c in rb.columns
                },
            }
            drift = frame_to_table(factor_lab.style_drift_score(rb), index_name="Nhân tố")
        except Exception:
            rolling, drift = None, {"columns": [], "rows": []}

        # Active Return (Lợi nhuận vượt trội so với benchmark)
        active_returns = None
        bench = ctx["benchmark"]
        if bench is not None:
            active = pr - bench.reindex(pr.index).fillna(0)
            active_cum = active.cumsum()
            active_returns = series_to_points(active_cum)

        return {
            "summary": dict_clean(model.summary()),
            "alpha_significant": bool(model.alpha_is_significant),
            "hac_lags": int(model.hac_lags),
            "betas": frame_to_table(tbl, index_name="Nhân tố"),
            "attribution": frame_to_table(attr),
            "ols_vs_hac": frame_to_table(factor_lab.compare_ols_vs_hac(pr, F)),
            "vif": frame_to_table(factor_lab.naive_vs_orthogonal(prices[avail])),
            "rolling": rolling,
            "drift": drift,
            "correlation": _corr_matrix(F),
            "active_return": active_returns,
        }

    return _cached(_key("factors", tickers, start, method, market), build)


def _corr_matrix(df: pd.DataFrame) -> dict:
    c = df.corr()
    return {
        "labels": [str(x) for x in c.columns],
        "matrix": [[_clean(v) for v in row] for row in c.to_numpy()],
    }


# --------------------------------------------------------------------------
# Kịch bản
# --------------------------------------------------------------------------


def scenarios(tickers, start, shocks: dict, market="us", n_paths=3000, fat_tails=True) -> dict:
    """Phòng kịch bản: cú sốc nhân tố, khủng hoảng lịch sử, Monte Carlo."""

    def build():
        ctx = load_market(tickers, start, market)
        prices, rets, bench = ctx["prices"], ctx["returns"], ctx["benchmark"]
        k = rets.shape[1]
        pr = pd.Series(rets.to_numpy() @ np.repeat(1 / k, k), index=rets.index)

        avail = [p for p in PROXIES if p in prices.columns]
        shock_result, grid = None, []
        if avail:
            fr = api.to_returns(prices[avail])
            use = {kk: v for kk, v in shocks.items() if kk in fr.columns}
            if use:
                out = stress.scenario_shock(pr, fr, use)
                shock_result = {
                    "total": _clean(out["Tác động tổng"]),
                    "detail": [
                        {
                            "factor": f,
                            "shock": _clean(use[f]),
                            "beta": _clean(out["Beta sử dụng"].get(f, 0.0)),
                            "impact": _clean(v),
                        }
                        for f, v in out["Theo từng nhân tố"].items()
                    ],
                }
            g = stress.scenario_grid(pr, fr, BENCHMARK)
            grid = [
                {"shock": _clean(r["Cú sốc nhân tố"]), "impact": _clean(r["Tác động danh mục (ước tính)"])}
                for _, r in g.iterrows()
            ]

        hist = stress.historical_stress(pr, bench)
        paths = stress.monte_carlo_paths(pr, n_paths=int(n_paths), fat_tails=fat_tails)
        counts, edges = np.histogram(paths["Lợi nhuận"], bins=48)

        return {
            "shock": shock_result,
            "grid": grid,
            "historical": frame_to_table(hist),
            "worst_windows": frame_to_table(stress.worst_windows(pr)),
            "monte_carlo": {
                "summary": frame_to_table(stress.monte_carlo_summary(paths)),
                "counts": [int(c) for c in counts],
                "edges": [_clean(e) for e in edges],
            },
        }

    return _cached(_key("scen", tickers, start, tuple(sorted(shocks.items())), market, n_paths, fat_tails), build)


# --------------------------------------------------------------------------
# Cảnh báo
# --------------------------------------------------------------------------


def risk_alerts(tickers, start, method="HRP", market="us", thresholds: dict | None = None) -> dict:
    """Chạy động cơ cảnh báo trên trạng thái mới nhất của danh mục."""
    ctx = load_market(tickers, start, market)
    rets, bench = ctx["returns"], ctx["benchmark"]
    mu, cov = alloc_lab.estimate(rets)
    w = alloc_lab.allocate(method, mu, cov, max_weight=0.4).weights
    pr = pd.Series(rets.to_numpy() @ w.to_numpy(), index=rets.index)

    regime = regime_prev = None
    conf = 0.0
    if bench is not None and len(bench) > 300:
        try:
            m = _fit_regime_cached(bench, tickers, start, market)
            regime, conf = m.current, m.confidence
            regime_prev = str(m.states.iloc[-22]) if len(m.states) > 22 else regime
        except Exception:
            pass

    context = alerts.build_context(pr, w, rets, bench, regime, regime_prev, conf)
    t = thresholds or {}
    rules = alerts.build_default_rules(
        dd_limit=t.get("drawdown", -0.15),
        vol_limit=t.get("volatility", 0.30),
        beta_limit=t.get("beta", 1.30),
        concentration_limit=t.get("concentration", 0.40),
        var_limit=t.get("var", -0.03),
    )
    fired = alerts.run_checks(context, rules)

    return {
        "context": dict_clean(context),
        "alerts": [
            {
                "code": a.code,
                "title": a.title,
                "message": a.message,
                "severity": a.severity.value,
                "value": _clean(a.value),
                "threshold": _clean(a.threshold),
            }
            for a in fired
        ],
        "counts": {
            "total": len(fired),
            "critical": sum(1 for a in fired if a.severity == alerts.Severity.CRITICAL),
        },
        "regime": regime,
        "regime_confidence": _clean(conf),
        "preview": alerts.dry_run_webhook(fired) if fired else "",
    }


def quantamental_analysis(tickers: Sequence[str], start: str, market: str = "us", use_sample: bool = False) -> dict[str, Any]:
    def build():
        from core.data import load_market_data
        from core.portfolio import build_portfolio, ma_crossover_signals
        from quantamental.tab_ui import generate_quantamental_report, simulate_ab_testing

        try:
            md = load_market_data(list(tickers), start, use_sample=use_sample, market=market)
        except Exception:
            md = load_market_data(list(tickers), start, use_sample=True, market=market)

        weights = [1.0 / len(tickers)] * len(tickers)
        pf = build_portfolio(md, weights, 10, "Không")
        sig_report = ma_crossover_signals(md, 20, 50)
        report = generate_quantamental_report(md, sig_report, market=market)
        ab = simulate_ab_testing(md, pf, report)

        signals_list = []
        for ticker, sig in report.per_ticker.items():
            signals_list.append({
                "ticker": ticker,
                "T": sig.T,
                "S": sig.S,
                "F": sig.F,
                "pattern_code": sig.pattern_code,
                "pattern_name": sig.pattern_name,
                "badge_color": sig.badge_color,
                "recommendation": sig.recommendation,
                "weight_action": sig.weight_action,
                "description": sig.description,
                "sentiment": {
                    "score": float(sig.sentiment.score),
                    "label": sig.sentiment.label,
                    "reasoning": sig.sentiment.reasoning,
                    "key_themes": sig.sentiment.key_themes,
                },
                "fundamental": {
                    "score": float(sig.fundamental.score),
                    "label": sig.fundamental.label,
                    "reasoning": sig.fundamental.reasoning,
                    "criteria_passed": sig.fundamental.criteria_passed,
                    "criteria_failed": sig.fundamental.criteria_failed,
                }
            })

        radar_series = []
        for ticker, sig in report.per_ticker.items():
            t_val = 1.0 if sig.T == 1 else 0.2
            s_val = round(float(sig.sentiment.score), 2)
            f_val = round(float(sig.fundamental.score), 2)
            radar_series.append({
                "ticker": ticker,
                "code": sig.pattern_code,
                "color": sig.badge_color,
                "values": [t_val, s_val, f_val],
            })

        allocations_table = []
        for i, t in enumerate(ab["tickers"]):
            w_t = float(ab["w_trad"][i])
            w_q = float(ab["w_quant"][i])
            action = (
                "🚨 Blacklist (Ép về 0%)" if t in ab["blacklisted"]
                else ("⭐ Ưu tiên Tăng vốn" if t in ab["boosted"] else "Giữ nguyên")
            )
            allocations_table.append({
                "ticker": t,
                "trad_weight": round(w_t * 100, 1),
                "quant_weight": round(w_q * 100, 1),
                "action": action,
            })

        return {
            "tickers": list(tickers),
            "patterns_count": report.patterns_count,
            "signals": signals_list,
            "radar": {
                "categories": ["Kỹ thuật (T)", "Cảm xúc Báo chí (S)", "Nội tại BCTC (F)"],
                "series": radar_series,
            },
            "ab_testing": {
                "m_trad": {k: float(v) for k, v in ab["m_trad"].items()},
                "m_quant": {k: float(v) for k, v in ab["m_quant"].items()},
                "blacklisted": ab["blacklisted"],
                "boosted": ab["boosted"],
                "curves": {
                    "quant": series_to_points(ab["cum_quant"]),
                    "trad": series_to_points(ab["cum_trad"]),
                },
                "allocations": allocations_table,
            }
        }

    return _cached(_key("quantamental", list(tickers), start, market, use_sample), build)
