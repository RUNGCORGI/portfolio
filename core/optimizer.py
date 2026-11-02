"""
core/optimizer.py  —  P4 (SỞ HỮU)
Khớp bài gốc: PORTFOLIO OPTIMIZATION (MAX SHARPE) + MONTE CARLO (myapp9/13).

PHÁT TRIỂN THÊM #1 của P4: thêm 3 phương pháp phân bổ thay thế để trả lời
đúng lời phê bình của chính tác giả — "Sharpe optimization is unstable and
tends to spit out allocations that don't make much sense":
    min_variance | risk_parity | hrp (Hierarchical Risk Parity)
rồi đặt cạnh nhau trong một bảng so sánh Return / Vol / Sharpe.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from core.contracts import TRADING_DAYS, OptimizeResult

METHODS = ["max_sharpe", "min_variance", "risk_parity", "hrp", "equal_weight"]


def annualize(returns: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    return returns.mean() * TRADING_DAYS, returns.cov() * TRADING_DAYS


def perf(w: np.ndarray, mu: pd.Series, cov: pd.DataFrame) -> Tuple[float, float, float]:
    r = float(np.dot(w, mu))
    v = float(np.sqrt(np.dot(w, np.dot(cov, w))))
    return r, v, (0.0 if v == 0 else r / v)


def _solve(obj, n: int, max_w: float):
    cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},)
    bounds = tuple((0.0, max_w) for _ in range(n))
    x0 = np.ones(n) / n
    return minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)


def optimize(returns: pd.DataFrame, method: str = "max_sharpe",
             max_weight: float = 0.5) -> OptimizeResult:
    mu, cov = annualize(returns)
    tickers = list(returns.columns)
    n = len(tickers)
    max_w = max(max_weight, 1.0 / n)  # tránh bài toán vô nghiệm
    ok = True

    if method == "equal_weight":
        w = np.ones(n) / n
    elif method == "max_sharpe":
        res = _solve(lambda x: -perf(x, mu, cov)[2], n, max_w)
        w, ok = res.x, bool(res.success)
    elif method == "min_variance":
        res = _solve(lambda x: perf(x, mu, cov)[1], n, max_w)
        w, ok = res.x, bool(res.success)
    elif method == "risk_parity":
        C = cov.values

        def rp(x):
            pv = np.sqrt(x @ C @ x)
            if pv == 0:
                return 1e6
            rc = x * (C @ x) / pv
            return float(((rc - rc.mean()) ** 2).sum())

        res = _solve(rp, n, 1.0)
        w, ok = res.x, bool(res.success)
    elif method == "hrp":
        w = _hrp(returns)
    else:
        raise ValueError(f"method không hợp lệ: {method}")

    w = np.clip(w, 0, None)
    w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
    r, v, s = perf(w, mu, cov)
    return OptimizeResult(method, w, tickers, r, v, s, ok)


# ---------------------------------------------------------------- HRP
def _hrp(returns: pd.DataFrame) -> np.ndarray:
    """López de Prado (2016): phân cụm -> sắp xếp gần -> chia đôi đệ quy."""
    corr = returns.corr().values
    cov = returns.cov().values
    n = len(corr)
    if n < 2:
        return np.ones(n)

    dist = np.sqrt(np.clip((1 - corr) / 2.0, 0, 1))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")
    order = to_tree(link, rd=False).pre_order()

    w = np.ones(n)
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            h = len(c) // 2
            a, b = c[:h], c[h:]
            va, vb = _cluster_var(cov, a), _cluster_var(cov, b)
            tot = va + vb
            alpha = 0.5 if tot == 0 else 1 - va / tot
            w[a] *= alpha
            w[b] *= 1 - alpha
            nxt += [a, b]
        clusters = nxt
    return w / w.sum()


def _cluster_var(cov: np.ndarray, idx: List[int]) -> float:
    sub = cov[np.ix_(idx, idx)]
    iv = 1.0 / np.diag(sub)
    w = iv / iv.sum()
    return float(w @ sub @ w)


# ---------------------------------------------------------------- Monte Carlo
def monte_carlo(returns: pd.DataFrame, n_portfolios: int = 5000,
                seed: int = 7) -> pd.DataFrame:
    mu, cov = annualize(returns)
    rng = np.random.default_rng(seed)
    n = returns.shape[1]
    W = rng.random((n_portfolios, n))
    W /= W.sum(axis=1, keepdims=True)
    rets = W @ mu.values
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, cov.values, W))
    with np.errstate(divide="ignore", invalid="ignore"):
        sh = np.where(vols > 0, rets / vols, 0.0)
    return pd.DataFrame({"Return": rets, "Volatility": vols, "Sharpe": sh})


def compare_methods(returns: pd.DataFrame, max_weight: float = 0.5) -> pd.DataFrame:
    rows = []
    for m in METHODS:
        o = optimize(returns, m, max_weight)
        row = {"Phương pháp": m, "Return %": o.exp_return * 100,
               "Vol %": o.volatility * 100, "Sharpe": o.sharpe,
               "Max w %": float(o.weights.max() * 100)}
        row.update({t: round(float(x) * 100, 1) for t, x in zip(o.tickers, o.weights)})
        rows.append(row)
    return pd.DataFrame(rows)
