"""
core/sizing.py  —  P2 (PHÁT TRIỂN THÊM #2)

Bài gốc dừng ở TỶ TRỌNG % — một con số trừu tượng. Nhà đầu tư thật không
mua được "33.3% NVDA", họ mua số cổ phiếu nguyên, theo lô, với số vốn cụ thể,
và phần dư nằm lại thành tiền mặt.

Module này biến tỷ trọng tối ưu thành PHIẾU LỆNH thực thi được:
    vốn ban đầu + tỷ trọng mục tiêu -> số cổ phiếu nguyên lô -> CSV lệnh mua
và đo sai lệch giữa tỷ trọng lý thuyết và tỷ trọng thực tế sau khi làm tròn.

Lô giao dịch:
    Mỹ   : 1 cổ phiếu
    HOSE : 100 cổ phiếu  -> sai lệch làm tròn lớn hơn nhiều, rất đáng chỉ ra
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

LOT_SIZE = {"US": 1, "HOSE": 100, "HNX": 100}


def build_orders(
    prices: pd.Series,
    weights: np.ndarray,
    capital: float = 100_000.0,
    market: str = "US",
    fee_bps: float = 10.0,
) -> pd.DataFrame:
    """
    prices  : giá đóng cửa mới nhất của từng mã (index = ticker)
    weights : tỷ trọng mục tiêu, tổng = 1
    capital : vốn khả dụng

    Trả về bảng lệnh: giá, tỷ trọng mục tiêu, số cổ phiếu, giá trị,
    tỷ trọng thực tế, sai lệch.
    """
    lot = LOT_SIZE.get(market, 1)
    px = prices.astype(float).values
    target_value = capital * np.asarray(weights, dtype=float)

    # làm tròn XUỐNG bội số của lô -> không bao giờ vượt vốn
    shares = np.floor(target_value / (px * lot)) * lot
    value = shares * px
    fee = value * fee_bps / 10_000.0

    invested = value.sum()
    actual_w = value / invested if invested > 0 else np.zeros_like(value)

    return pd.DataFrame({
        "Mã": list(prices.index),
        "Giá": px,
        "Tỷ trọng mục tiêu %": np.asarray(weights) * 100,
        "Số CP": shares.astype(int),
        "Giá trị": value,
        "Phí": fee,
        "Tỷ trọng thực tế %": actual_w * 100,
        "Sai lệch %": actual_w * 100 - np.asarray(weights) * 100,
    })


def summarize(orders: pd.DataFrame, capital: float) -> Dict[str, float]:
    invested = float(orders["Giá trị"].sum())
    fee = float(orders["Phí"].sum())
    return {
        "capital": capital,
        "invested": invested,
        "fee": fee,
        "cash_left": capital - invested - fee,
        "cash_pct": (capital - invested - fee) / capital * 100,
        "max_drift_pct": float(orders["Sai lệch %"].abs().max()),
        "tracking_error_pct": float(np.sqrt((orders["Sai lệch %"] ** 2).mean())),
    }


def to_csv(orders: pd.DataFrame) -> str:
    """Xuất phiếu lệnh để nộp cho môi giới / lưu hồ sơ."""
    cols = ["Mã", "Giá", "Số CP", "Giá trị", "Tỷ trọng thực tế %"]
    return orders[orders["Số CP"] > 0][cols].to_csv(index=False)
