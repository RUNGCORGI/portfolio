"""Lược đồ dữ liệu cho API.

Dùng pydantic để **kiểm tra đầu vào ngay tại biên hệ thống**. Nhờ đó các
module trong ``core/`` không bao giờ nhận tham số rác, và frontend nhận được
thông báo lỗi rõ ràng thay vì một traceback khó hiểu.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

METHODS = [
    "Chia đều",
    "Phương sai tối thiểu",
    "Max Sharpe",
    "Cân bằng rủi ro",
    "HRP",
    "Đa dạng hoá tối đa",
]


class BaseQuery(BaseModel):
    """Tham số chung cho mọi truy vấn phân tích."""

    tickers: list[str] = Field(
        default=["AAPL", "MSFT", "NVDA", "JNJ", "XOM", "KO"],
        min_length=2,
        max_length=60,
        description="Danh sách mã cổ phiếu",
    )
    start: str = Field(default="2018-01-01", description="Ngày bắt đầu, định dạng YYYY-MM-DD")
    market: Literal["us", "vn"] = "us"

    @field_validator("tickers")
    @classmethod
    def _upper(cls, v: list[str]) -> list[str]:
        out = [t.strip().upper() for t in v if t and t.strip()]
        if len(out) < 2:
            raise ValueError("Cần ít nhất 2 mã hợp lệ.")
        return list(dict.fromkeys(out))

    @field_validator("start")
    @classmethod
    def _date(cls, v: str) -> str:
        import re

        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v.strip()):
            raise ValueError("Ngày phải theo định dạng YYYY-MM-DD.")
        return v.strip()


class OverviewQuery(BaseQuery):
    method: str = "HRP"
    robust: bool = True
    max_weight: float = Field(default=0.4, ge=0.05, le=1.0)

    @field_validator("method")
    @classmethod
    def _method(cls, v: str) -> str:
        if v not in METHODS:
            raise ValueError(f"Phương pháp không hợp lệ. Hợp lệ: {', '.join(METHODS)}")
        return v


class AllocQuery(BaseQuery):
    robust: bool = True
    max_weight: float = Field(default=0.4, ge=0.05, le=1.0)


class WalkForwardQuery(BaseQuery):
    methods: list[str] = Field(default=["Chia đều", "Max Sharpe", "HRP"], min_length=1)
    cost_bps: float = Field(default=10.0, ge=0.0, le=200.0)
    lookback: int = Field(default=252, ge=60, le=756)
    step: int = Field(default=21, ge=5, le=126)

    @field_validator("methods")
    @classmethod
    def _methods(cls, v: list[str]) -> list[str]:
        bad = [m for m in v if m not in METHODS]
        if bad:
            raise ValueError(f"Phương pháp không hợp lệ: {', '.join(bad)}")
        return v


class ValidationQuery(BaseQuery):
    method: str = "HRP"
    n_trials: int = Field(default=20, ge=1, le=200)


class FactorQuery(BaseQuery):
    method: str = "HRP"


class ScenarioQuery(BaseQuery):
    shocks: dict[str, float] = Field(default={"SPY": -0.20, "^VIX": 0.80, "QQQ": -0.25})
    n_paths: int = Field(default=3000, ge=500, le=20000)
    fat_tails: bool = True

    @field_validator("shocks")
    @classmethod
    def _shocks(cls, v: dict[str, float]) -> dict[str, float]:
        for k, s in v.items():
            if not -0.95 <= s <= 5.0:
                raise ValueError(f"Cú sốc cho {k} nằm ngoài khoảng hợp lý.")
        return v


class AlertQuery(BaseQuery):
    method: str = "HRP"
    thresholds: dict[str, float] = Field(default_factory=dict)


class TradeInput(BaseModel):
    trade_date: str
    ticker: str
    side: Literal["MUA", "BAN"]
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)
    note: str = ""
    signal: str = ""
    followed: bool = True


class QuantamentalQuery(BaseQuery):
    tickers: list[str] = Field(
        default=["AAPL", "MSFT", "NVDA"],
        min_length=1,
        max_length=60,
        description="Danh sách mã cổ phiếu",
    )
    start: str = Field(default="2023-01-01", description="Ngày bắt đầu, định dạng YYYY-MM-DD")
    market: Literal["us", "vn"] = "us"
    use_sample: bool = False


class ApiError(BaseModel):
    detail: str
    hint: str = ""
