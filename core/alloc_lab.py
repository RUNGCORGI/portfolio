"""Phòng thí nghiệm phân bổ vốn — nâng cấp cho ``core/optimizer.py``.

Vấn đề của Max Sharpe cổ điển (cách bài gốc làm): nó dùng trung bình lịch
sử làm kỳ vọng lợi suất. Đó là ước lượng cực kỳ nhiễu, nên kết quả thường
dồn gần hết vốn vào mã may mắn nhất của mẫu, và đổi ngày bắt đầu vài tuần
là trọng số đổi hoàn toàn.

Module này bổ sung ba tầng khắc phục:

1. **Ước lượng tốt hơn** – co ngót Ledoit–Wolf cho hiệp phương sai,
   co ngót James–Stein cho kỳ vọng, và Black–Litterman để đưa quan điểm
   của người dùng vào một cách có kỷ luật.
2. **Nhiều thuật toán để so sánh** – trong đó có Hierarchical Risk Parity,
   phương pháp không cần nghịch đảo ma trận hiệp phương sai nên rất bền.
3. **Kiểm định độ ổn định** – đo bằng số chứ không nói suông.

Không file nào trong dự án của bạn bị sửa; đây là module đứng độc lập.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

TRADING_DAYS = 252


@dataclass
class AllocResult:
    """Kết quả của một thuật toán phân bổ."""

    method: str
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float
    converged: bool = True

    def as_row(self) -> dict:
        w = self.weights.to_numpy()
        return {
            "Phương pháp": self.method,
            "LN kỳ vọng": self.expected_return,
            "Biến động": self.volatility,
            "Sharpe": self.sharpe,
            "Số vị thế hiệu dụng": effective_n(w),
            "Trọng số lớn nhất": float(np.max(w)),
        }


def effective_n(weights) -> float:
    """1 / tổng bình phương trọng số. Phát hiện danh mục dồn hết vào một mã."""
    w = np.asarray(weights, dtype=float)
    d = float(np.sum(w**2))
    return float(1.0 / d) if d > 0 else 0.0


# --------------------------------------------------------------------------
# 1. Ước lượng tham số
# --------------------------------------------------------------------------


def sample_moments(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Kỳ vọng và hiệp phương sai mẫu, năm hoá — chính là cách bài gốc làm."""
    return returns.mean() * TRADING_DAYS, returns.cov() * TRADING_DAYS


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Hiệp phương sai co ngót Ledoit–Wolf (năm hoá).

    Kéo ma trận mẫu về phía một mục tiêu có cấu trúc (giữ nguyên phương sai
    riêng, ép tương quan về mức trung bình chung). Cường độ co ngót được
    ước lượng tự động từ dữ liệu, không phải tham số tự chọn.
    """
    x = returns.dropna().to_numpy(dtype=float)
    n, p = x.shape
    if n < 3 or p < 2:
        return returns.cov() * TRADING_DAYS

    xc = x - x.mean(axis=0)
    sample = xc.T @ xc / n

    var = np.diag(sample).copy()
    sd = np.sqrt(np.outer(var, var))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(sd > 0, sample / sd, 0.0)
    off = corr[~np.eye(p, dtype=bool)]
    rbar = float(off.mean()) if off.size else 0.0
    target = rbar * sd
    np.fill_diagonal(target, var)

    y = xc**2
    phi = float(np.sum(y.T @ y / n - sample**2))
    gamma = float(np.sum((sample - target) ** 2))
    shrink = 0.0 if gamma <= 1e-18 else float(np.clip(phi / n / gamma, 0.0, 1.0))

    shrunk = shrink * target + (1.0 - shrink) * sample
    return pd.DataFrame(
        shrunk * TRADING_DAYS, index=returns.columns, columns=returns.columns
    )


def shrunk_mean(returns: pd.DataFrame, intensity: float | None = None) -> pd.Series:
    """Co ngót James–Stein: kéo kỳ vọng từng mã về kỳ vọng bình quân.

    Đây là liều thuốc trực tiếp cho bệnh "Max Sharpe dồn hết vào NVDA".
    """
    mu = returns.mean() * TRADING_DAYS
    grand = float(mu.mean())
    if intensity is None:
        n, p = len(returns), returns.shape[1]
        disp = float(((mu - grand) ** 2).sum())
        sigma2 = float((returns.var(ddof=1) * TRADING_DAYS).mean())
        intensity = 0.5 if disp <= 1e-18 else float(
            np.clip(max(p - 2, 1) * sigma2 / max(n, 1) / disp, 0.0, 1.0)
        )
    return (1.0 - intensity) * mu + intensity * grand


def black_litterman(
    cov: pd.DataFrame,
    market_weights,
    views: dict[str, float] | None = None,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    confidence: float = 0.6,
) -> pd.Series:
    """Kỳ vọng hậu nghiệm Black–Litterman.

    Bắt đầu từ lợi suất cân bằng ngụ ý bởi danh mục thị trường (Π = λΣw),
    rồi pha trộn với quan điểm chủ quan, ví dụ ``{"NVDA": 0.25}`` nghĩa là
    "tôi cho rằng NVDA đạt 25%/năm". Càng tự tin thì quan điểm càng nặng ký.

    Điểm mạnh: trọng số đầu ra luôn neo quanh danh mục thị trường thay vì
    bay nhảy theo nhiễu của dữ liệu lịch sử.
    """
    assets = list(cov.index)
    w_mkt = np.asarray(market_weights, dtype=float).ravel()
    sigma = cov.to_numpy()
    pi = risk_aversion * sigma @ w_mkt

    rows, q = [], []
    for asset, target in (views or {}).items():
        if asset in assets:
            row = np.zeros(len(assets))
            row[assets.index(asset)] = 1.0
            rows.append(row)
            q.append(float(target))
    if not rows:
        return pd.Series(pi, index=assets, name="bl_mu")

    P, Q = np.vstack(rows), np.asarray(q)
    tau_sigma = tau * sigma
    c = float(np.clip(confidence, 1e-3, 0.999))
    omega = np.diag(np.diag(P @ tau_sigma @ P.T)) * (1 - c) / c

    inv_ts, inv_om = np.linalg.pinv(tau_sigma), np.linalg.pinv(omega)
    post = np.linalg.pinv(inv_ts + P.T @ inv_om @ P) @ (inv_ts @ pi + P.T @ inv_om @ Q)
    return pd.Series(post, index=assets, name="bl_mu")


def estimate(
    returns: pd.DataFrame,
    robust: bool = True,
    views: dict[str, float] | None = None,
    confidence: float = 0.6,
) -> tuple[pd.Series, pd.DataFrame]:
    """Một cửa duy nhất để lấy (mu, cov) theo chế độ mong muốn."""
    if not robust:
        return sample_moments(returns)
    cov = ledoit_wolf_cov(returns)
    if views:
        n = returns.shape[1]
        mu = black_litterman(cov, np.repeat(1.0 / n, n), views, confidence=confidence)
    else:
        mu = shrunk_mean(returns)
    return mu, cov


# --------------------------------------------------------------------------
# 2. Các thuật toán phân bổ
# --------------------------------------------------------------------------


def performance(w, mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0):
    w = np.asarray(w, dtype=float)
    ret = float(w @ mu.to_numpy())
    vol = float(np.sqrt(max(w @ cov.to_numpy() @ w, 1e-18)))
    return ret, vol, (ret - rf) / vol if vol > 0 else 0.0


def _solve(obj: Callable, n: int, hi: float, lo: float = 0.0):
    res = minimize(
        obj,
        np.repeat(1.0 / n, n),
        method="SLSQP",
        bounds=tuple((lo, hi) for _ in range(n)),
        constraints=({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},),
        options={"maxiter": 500, "ftol": 1e-10},
    )
    w = np.clip(res.x, lo, hi)
    s = w.sum()
    return (w / s if s > 0 else np.repeat(1.0 / n, n)), bool(res.success)


def equal_weight(mu, cov, **_):
    """Chia đều. Đối thủ khó đánh bại một cách đáng ngạc nhiên — luôn phải có."""
    n = len(mu)
    return _pack("Chia đều", np.repeat(1.0 / n, n), mu, cov)


def min_variance(mu, cov, max_weight: float = 0.4, **_):
    """Phương sai tối thiểu. Hoàn toàn không dùng kỳ vọng lợi suất nên rất bền."""
    s = cov.to_numpy()
    w, ok = _solve(lambda x: float(x @ s @ x), len(mu), max_weight)
    return _pack("Phương sai tối thiểu", w, mu, cov, ok)


def max_sharpe(mu, cov, max_weight: float = 0.4, rf: float = 0.0, **_):
    """Max Sharpe cổ điển, có chặn trần trọng số."""
    w, ok = _solve(lambda x: -performance(x, mu, cov, rf)[2], len(mu), max_weight)
    return _pack("Max Sharpe", w, mu, cov, ok)


def risk_parity(mu, cov, max_weight: float = 0.6, **_):
    """Cân bằng rủi ro: mỗi mã đóng góp đúng 1/N lượng rủi ro danh mục."""
    s = cov.to_numpy()
    target = 1.0 / len(mu)

    def obj(w):
        pv = float(w @ s @ w)
        if pv <= 0:
            return 1e6
        return float(np.sum((w * (s @ w) / pv - target) ** 2))

    w, ok = _solve(obj, len(mu), max_weight, lo=1e-4)
    return _pack("Cân bằng rủi ro", w, mu, cov, ok)


def hrp(mu, cov, **_):
    """Hierarchical Risk Parity (Lopez de Prado, 2016).

    Ba bước: phân cụm theo tương quan → sắp xếp ma trận gần đường chéo →
    chia vốn đệ quy theo nghịch đảo phương sai. Không nghịch đảo ma trận
    hiệp phương sai nên không bị khuếch đại sai số ước lượng.
    """
    assets = list(cov.index)
    n = len(assets)
    if n == 1:
        return _pack("HRP", np.array([1.0]), mu, cov)

    s = cov.to_numpy()
    sd = np.sqrt(np.clip(np.diag(s), 1e-18, None))
    corr = np.clip(np.nan_to_num(s / np.outer(sd, sd), nan=0.0), -1.0, 1.0)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)

    order = _quasi_diag(linkage(squareform(dist, checks=False), method="single"), n)
    w = pd.Series(1.0, index=[assets[i] for i in order])

    clusters = [list(w.index)]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            half = len(cl) // 2
            left, right = cl[:half], cl[half:]
            vl, vr = _cluster_var(cov, left), _cluster_var(cov, right)
            a = 1.0 - vl / (vl + vr) if (vl + vr) > 0 else 0.5
            w[left] *= a
            w[right] *= 1.0 - a
            nxt += [left, right]
        clusters = nxt

    out = w.reindex(assets).to_numpy()
    return _pack("HRP", out / out.sum(), mu, cov)


def max_diversification(mu, cov, max_weight: float = 0.4, **_):
    """Tối đa hoá tỷ lệ đa dạng hoá = bình quân vol / vol danh mục."""
    s = cov.to_numpy()
    vols = np.sqrt(np.clip(np.diag(s), 1e-18, None))

    def obj(w):
        return -float((w @ vols) / np.sqrt(max(float(w @ s @ w), 1e-18)))

    w, ok = _solve(obj, len(mu), max_weight)
    return _pack("Đa dạng hoá tối đa", w, mu, cov, ok)


METHODS: dict[str, Callable[..., AllocResult]] = {
    "Chia đều": equal_weight,
    "Phương sai tối thiểu": min_variance,
    "Max Sharpe": max_sharpe,
    "Cân bằng rủi ro": risk_parity,
    "HRP": hrp,
    "Đa dạng hoá tối đa": max_diversification,
}


def allocate(method: str, mu: pd.Series, cov: pd.DataFrame, **kw) -> AllocResult:
    if method not in METHODS:
        raise KeyError(f"Chưa có phương pháp '{method}'. Hiện có: {list(METHODS)}")
    return METHODS[method](mu, cov, **kw)


def compare_methods(mu, cov, methods: Iterable[str] | None = None, **kw):
    """Chạy nhiều thuật toán trên cùng bộ tham số rồi xếp cạnh nhau."""
    names = list(methods or METHODS.keys())
    res = {m: allocate(m, mu, cov, **kw) for m in names}
    return pd.DataFrame([r.as_row() for r in res.values()]), res


# --------------------------------------------------------------------------
# 3. Kiểm định độ ổn định của trọng số
# --------------------------------------------------------------------------


def weight_stability(
    returns: pd.DataFrame,
    method: str = "Max Sharpe",
    n_trials: int = 40,
    sample_frac: float = 0.7,
    robust: bool = False,
    seed: int = 7,
    **kw,
) -> pd.DataFrame:
    """Tối ưu lặp lại trên nhiều đoạn lịch sử con, trả về bảng trọng số.

    Độ lệch chuẩn của mỗi cột chính là **bằng chứng định lượng** cho nhận
    định "tối ưu Max Sharpe bất ổn" — và cho thấy co ngót chữa được nó.
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    size = max(60, int(n * sample_frac))
    rows = []
    for _ in range(n_trials):
        s = int(rng.integers(0, max(1, n - size)))
        win = returns.iloc[s : s + size]
        mu, cov = estimate(win, robust=robust)
        rows.append(allocate(method, mu, cov, **kw).weights)
    return pd.DataFrame(rows).reset_index(drop=True)


def stability_report(returns: pd.DataFrame, methods: Iterable[str] | None = None, **kw):
    """So sánh độ bất ổn giữa: ước lượng thô và ước lượng có co ngót."""
    out = []
    for m in list(methods or ["Max Sharpe", "Phương sai tối thiểu", "HRP"]):
        for robust, label in ((False, "Ước lượng thô"), (True, "Có co ngót")):
            w = weight_stability(returns, m, robust=robust, **kw)
            out.append(
                {
                    "Phương pháp": m,
                    "Ước lượng": label,
                    "Độ lệch chuẩn TB của trọng số": float(w.std(ddof=1).mean()),
                    "Biên độ trọng số TB": float((w.max() - w.min()).mean()),
                }
            )
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Nội bộ
# --------------------------------------------------------------------------


def _pack(name, w, mu, cov, ok: bool = True) -> AllocResult:
    ret, vol, sr = performance(w, mu, cov)
    return AllocResult(
        name, pd.Series(np.asarray(w, dtype=float), index=mu.index, name="weight"),
        ret, vol, sr, ok,
    )


def _quasi_diag(link: np.ndarray, n: int) -> list[int]:
    tree = to_tree(link, rd=False)
    order: list[int] = []

    def walk(node):
        if node.is_leaf():
            order.append(int(node.id))
            return
        walk(node.get_left())
        walk(node.get_right())

    walk(tree)
    return order[:n]


def _cluster_var(cov: pd.DataFrame, assets: list[str]) -> float:
    sub = cov.loc[assets, assets].to_numpy()
    ivp = 1.0 / np.clip(np.diag(sub), 1e-18, None)
    ivp = ivp / ivp.sum()
    return float(ivp @ sub @ ivp)
