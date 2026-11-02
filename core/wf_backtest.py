"""Walk-forward backtest có chi phí giao dịch — nâng cấp cho ``tab_backtest``.

Vì sao cần thiết
----------------
Backtest trong bài gốc tối ưu trọng số trên **toàn bộ** lịch sử rồi áp dụng
chính trọng số đó lên cũng lịch sử ấy. Đó là rò rỉ dữ liệu tương lai
(look-ahead bias): kết quả đẹp nhưng vô giá trị.

Walk-forward làm đúng quy trình của một quỹ thật:

    [--- cửa sổ ước lượng ---][cấm][--- kỳ áp dụng ---]
                                   [--- cửa sổ ước lượng ---][cấm][--- kỳ áp dụng ---]

Trọng số của mỗi kỳ chỉ dùng thông tin có trước kỳ đó. Khoảng "cấm"
(embargo) chặn nốt phần rò rỉ do tự tương quan ở ranh giới.

Mỗi lần tái cân bằng đều bị **tính phí**, và trọng số được cho **trôi theo
giá** giữa hai lần tái cân bằng — nếu bỏ bước này sẽ tính thừa vòng quay.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core import alloc_lab
from core.upgrade_api import TRADING_DAYS, cagr, max_drawdown, sharpe


@dataclass
class CostModel:
    """Chi phí giao dịch, tính bằng điểm cơ bản (bps) trên giá trị khớp lệnh."""

    commission_bps: float = 5.0
    spread_bps: float = 2.0
    slippage_bps: float = 3.0

    @property
    def rate(self) -> float:
        return (self.commission_bps + self.spread_bps + self.slippage_bps) / 10_000.0


@dataclass
class WalkForwardConfig:
    lookback: int = 252       # số phiên dùng để ước lượng
    step: int = 21            # tái cân bằng mỗi ~1 tháng
    embargo: int = 5          # số phiên cấm giữa ước lượng và áp dụng
    robust: bool = True       # dùng co ngót Ledoit-Wolf + James-Stein
    max_weight: float = 0.4
    costs: CostModel = field(default_factory=CostModel)


@dataclass
class WalkForwardResult:
    method: str
    gross: pd.Series
    net: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.net).cumprod()

    @property
    def gross_equity(self) -> pd.Series:
        return (1.0 + self.gross).cumprod()

    def summary(self) -> dict:
        n = len(self.net)
        return {
            "Phương pháp": self.method,
            "CAGR (sau phí)": cagr(self.net),
            "Sharpe (sau phí)": sharpe(self.net),
            "Sharpe (trước phí)": sharpe(self.gross),
            "Phí bào mòn/năm": float((self.gross - self.net).mean() * TRADING_DAYS),
            "Sụt giảm tối đa": max_drawdown(self.net),
            "Vòng quay/năm": float(self.turnover.sum() * TRADING_DAYS / max(n, 1)),
            "Số lần tái cân bằng": int((self.turnover > 1e-9).sum()),
        }


def _drift(w: np.ndarray, period_returns: np.ndarray) -> np.ndarray:
    """Trọng số tự trôi theo biến động giá giữa hai lần tái cân bằng."""
    grown = w * (1.0 + period_returns)
    tot = grown.sum()
    return grown / tot if tot > 0 else w


def run_walk_forward(
    returns: pd.DataFrame, method: str = "HRP", config: WalkForwardConfig | None = None
) -> WalkForwardResult:
    """Chạy walk-forward cho một phương pháp phân bổ."""
    cfg = config or WalkForwardConfig()
    rets = returns.dropna(how="all").fillna(0.0)
    n, k = rets.shape
    start = cfg.lookback + cfg.embargo
    if n <= start + cfg.step:
        raise ValueError(
            f"Không đủ dữ liệu: cần hơn {start + cfg.step} phiên, hiện có {n}."
        )

    gross = pd.Series(0.0, index=rets.index[start:], name="gross")
    turn = pd.Series(0.0, index=rets.index[start:], name="turnover")
    wlog: dict[pd.Timestamp, np.ndarray] = {}

    current = np.repeat(1.0 / k, k)
    arr = rets.to_numpy()

    for t0 in range(start, n, cfg.step):
        t1 = min(t0 + cfg.step, n)
        train = rets.iloc[t0 - cfg.embargo - cfg.lookback : t0 - cfg.embargo]
        try:
            mu, cov = alloc_lab.estimate(train, robust=cfg.robust)
            target = alloc_lab.allocate(
                method, mu, cov, max_weight=cfg.max_weight
            ).weights.to_numpy()
        except Exception:
            target = np.repeat(1.0 / k, k)   # có sự cố thì lui về chia đều

        ts = rets.index[t0]
        turn.loc[ts] = float(np.abs(target - current).sum() / 2.0)
        wlog[ts] = target.copy()
        current = target

        for t in range(t0, t1):
            gross.iloc[t - start] = float(arr[t] @ current)
            current = _drift(current, arr[t])

    cost = turn * cfg.costs.rate * 2.0
    net = (gross - cost.reindex(gross.index).fillna(0.0)).rename("net")
    wdf = pd.DataFrame(wlog).T
    wdf.columns = rets.columns
    return WalkForwardResult(method, gross, net, wdf, turn)


def compare_walk_forward(
    returns: pd.DataFrame, methods: list[str] | None = None, config=None
) -> tuple[pd.DataFrame, dict[str, WalkForwardResult]]:
    """Chạy walk-forward cho nhiều phương pháp và lập bảng so sánh."""
    methods = methods or ["Chia đều", "Phương sai tối thiểu", "Max Sharpe", "HRP"]
    res: dict[str, WalkForwardResult] = {}
    for m in methods:
        try:
            res[m] = run_walk_forward(returns, m, config)
        except Exception:
            continue
    table = pd.DataFrame([r.summary() for r in res.values()])
    return table, res


def in_sample_vs_out_of_sample(
    returns: pd.DataFrame, method: str = "Max Sharpe", config=None
) -> pd.DataFrame:
    """Đo mức "bốc hơi" của Sharpe khi chuyển từ in-sample sang out-of-sample.

    Đây là bảng số thuyết phục nhất khi bảo vệ: nó cho thấy con số Sharpe
    đẹp đẽ trong bài gốc mất bao nhiêu phần trăm khi làm đúng quy trình.
    """
    cfg = config or WalkForwardConfig()
    mu, cov = alloc_lab.estimate(returns, robust=cfg.robust)
    w_is = alloc_lab.allocate(method, mu, cov, max_weight=cfg.max_weight).weights
    is_ret = pd.Series(returns.to_numpy() @ w_is.to_numpy(), index=returns.index)

    oos = run_walk_forward(returns, method, cfg)
    s_is, s_oos = sharpe(is_ret), sharpe(oos.net)
    return pd.DataFrame(
        [
            {
                "Phương pháp": method,
                "Sharpe in-sample": s_is,
                "Sharpe out-of-sample (sau phí)": s_oos,
                "Mức suy giảm": s_is - s_oos,
                "Tỷ lệ còn lại": (s_oos / s_is) if abs(s_is) > 1e-9 else 0.0,
            }
        ]
    )


def cost_sensitivity(
    result: WalkForwardResult, bps_grid=(0, 5, 10, 20, 35, 50, 75, 100)
) -> pd.DataFrame:
    """Sharpe ở nhiều mức phí khác nhau — tìm ngưỡng chiến lược "chết"."""
    turn = result.turnover.reindex(result.gross.index).fillna(0.0)
    rows = []
    for bps in bps_grid:
        net = result.gross - turn * (bps / 10_000.0) * 2.0
        rows.append(
            {
                "Phí (bps/chiều)": bps,
                "Tổng lợi nhuận": float((1 + net).prod() - 1),
                "Sharpe": sharpe(net),
            }
        )
    return pd.DataFrame(rows)


def breakeven_cost_bps(result: WalkForwardResult, benchmark: pd.Series | None = None) -> float:
    """Mức phí làm lợi nhuận vượt trội về 0. Trả lời: "chịu được phí tới đâu?" """
    total = float(result.turnover.sum()) * 2.0
    if total <= 0:
        return float("inf")
    excess = float(result.gross.sum())
    if benchmark is not None:
        g, b = result.gross.align(benchmark, join="inner")
        excess = float((g - b).sum())
    return float(excess / total * 10_000.0) if excess > 0 else 0.0
