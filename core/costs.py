"""
core/costs.py  —  P2 (PHÁT TRIỂN THÊM #1)

Lỗ hổng lớn nhất của bài gốc, chính tác giả thừa nhận:
"Transaction costs, slippage, and turnover? Completely ignored."

Module này mô phỏng:
    - drift tỷ trọng giữa các kỳ (buy-and-hold làm weight lệch dần)
    - tái cân bằng theo tần suất (M / Q / A / never)
    - phí giao dịch tính trên turnover, đơn vị basis point
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

FREQ_MAP = {"Tháng": "ME", "Quý": "QE", "Năm": "YE", "Không": None}


def rebalance_dates(index: pd.DatetimeIndex, freq: Optional[str]) -> set:
    if freq is None:
        return set()
    marks = pd.Series(1, index=index).resample(freq).last().index
    return {index[index.get_indexer([d], method="ffill")[0]]
            for d in marks if d >= index[0]}


def apply_costs(
    asset_returns: pd.DataFrame,
    target_weights: np.ndarray,
    cost_bps: float = 10.0,
    freq_label: str = "Quý",
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Trả về (net_returns, gross_returns, turnover).

    cost_bps = 10 nghĩa là 0.10% giá trị giao dịch mỗi chiều — mức hợp lý cho
    nhà đầu tư cá nhân ở thị trường Mỹ; thị trường VN nên để 15-25.
    """
    freq = FREQ_MAP.get(freq_label, "QE")
    rb = rebalance_dates(asset_returns.index, freq)

    w = target_weights.astype(float).copy()
    gross, net, turn = [], [], []

    for dt, row in asset_returns.iterrows():
        r = np.nan_to_num(row.values)
        g = float(np.dot(w, r))

        # drift: tỷ trọng tự thay đổi theo giá
        grown = w * (1 + r)
        total = grown.sum()
        w = grown / total if total > 0 else target_weights.copy()

        t = 0.0
        if dt in rb:
            t = float(np.abs(w - target_weights).sum())
            w = target_weights.astype(float).copy()

        gross.append(g)
        net.append(g - t * cost_bps / 10_000.0)
        turn.append(t)

    idx = asset_returns.index
    return (pd.Series(net, index=idx),
            pd.Series(gross, index=idx),
            pd.Series(turn, index=idx))


def cost_drag(gross: pd.Series, net: pd.Series) -> float:
    """Chênh lệch tổng lợi nhuận trước/sau phí, tính bằng điểm phần trăm."""
    g = float((1 + gross).prod() - 1) * 100
    n = float((1 + net).prod() - 1) * 100
    return g - n
