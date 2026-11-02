"""Hỗ trợ thị trường chứng khoán Việt Nam.

Vì sao đáng làm: toàn bộ phân tích đang neo vào SPY và cổ phiếu Mỹ. Thêm
được chế độ thị trường Việt Nam vừa chứng minh **lớp trừu tượng dữ liệu
hoạt động đúng** (đổi nguồn mà không phải sửa một dòng nào trong các module
phân tích), vừa khiến phần demo gần gũi hơn nhiều với hội đồng trong nước.

Ba điểm khác biệt của thị trường Việt Nam đã được xử lý:

* **Benchmark** là VNINDEX thay cho SPY.
* **Biên độ dao động** có giới hạn theo sàn (HOSE ±7%, HNX ±10%, UPCoM ±15%),
  nên phân phối lợi suất bị cắt đuôi — mô phỏng phản ánh đúng điều này.
* **Phí giao dịch và thuế** khác hẳn Mỹ: có thuế thu nhập 0,1% trên giá trị
  bán, và chu kỳ thanh toán T+2.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VN_BENCHMARK = "VNINDEX"

# Biên độ dao động tối đa một phiên theo sàn niêm yết
PRICE_LIMITS = {"HOSE": 0.07, "HNX": 0.10, "UPCOM": 0.15}

# Rổ cổ phiếu phổ biến, kèm sàn và nhóm ngành
VN_UNIVERSE: dict[str, dict] = {
    "VNINDEX": {"exchange": "HOSE", "sector": "Chỉ số", "beta": 1.00, "vol": 0.18},
    "VCB": {"exchange": "HOSE", "sector": "Ngân hàng", "beta": 1.05, "vol": 0.26},
    "TCB": {"exchange": "HOSE", "sector": "Ngân hàng", "beta": 1.20, "vol": 0.32},
    "BID": {"exchange": "HOSE", "sector": "Ngân hàng", "beta": 1.10, "vol": 0.30},
    "VHM": {"exchange": "HOSE", "sector": "Bất động sản", "beta": 1.25, "vol": 0.36},
    "VIC": {"exchange": "HOSE", "sector": "Bất động sản", "beta": 1.30, "vol": 0.38},
    "HPG": {"exchange": "HOSE", "sector": "Thép", "beta": 1.35, "vol": 0.40},
    "FPT": {"exchange": "HOSE", "sector": "Công nghệ", "beta": 0.95, "vol": 0.28},
    "MWG": {"exchange": "HOSE", "sector": "Bán lẻ", "beta": 1.15, "vol": 0.34},
    "VNM": {"exchange": "HOSE", "sector": "Tiêu dùng", "beta": 0.70, "vol": 0.22},
    "SAB": {"exchange": "HOSE", "sector": "Tiêu dùng", "beta": 0.75, "vol": 0.24},
    "GAS": {"exchange": "HOSE", "sector": "Năng lượng", "beta": 0.90, "vol": 0.30},
    "PLX": {"exchange": "HOSE", "sector": "Năng lượng", "beta": 0.95, "vol": 0.29},
    "SSI": {"exchange": "HOSE", "sector": "Chứng khoán", "beta": 1.45, "vol": 0.44},
    "VND": {"exchange": "HNX", "sector": "Chứng khoán", "beta": 1.50, "vol": 0.46},
    "SHS": {"exchange": "HNX", "sector": "Chứng khoán", "beta": 1.55, "vol": 0.48},
}


@dataclass(frozen=True)
class VNCostModel:
    """Chi phí giao dịch theo quy định hiện hành tại Việt Nam.

    Điểm khác biệt quan trọng so với mô hình Mỹ: thuế thu nhập cá nhân 0,1%
    chỉ đánh **một chiều khi bán**, nên chi phí mua và bán không đối xứng.
    """

    brokerage_bps: float = 15.0      # phí môi giới hai chiều, khoảng 0,15%
    sell_tax_bps: float = 10.0       # thuế TNCN 0,1% trên giá trị bán
    spread_bps: float = 10.0         # chênh lệch mua bán, thanh khoản thấp hơn Mỹ

    @property
    def buy_rate(self) -> float:
        return (self.brokerage_bps + self.spread_bps) / 10_000.0

    @property
    def sell_rate(self) -> float:
        return (self.brokerage_bps + self.sell_tax_bps + self.spread_bps) / 10_000.0

    @property
    def round_trip_rate(self) -> float:
        """Tổng chi phí một vòng mua rồi bán."""
        return self.buy_rate + self.sell_rate

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"Khoản mục": "Phí môi giới", "Mức": self.brokerage_bps / 10_000,
                 "Áp dụng": "Cả hai chiều"},
                {"Khoản mục": "Thuế TNCN", "Mức": self.sell_tax_bps / 10_000,
                 "Áp dụng": "Chỉ khi bán"},
                {"Khoản mục": "Chênh lệch mua bán", "Mức": self.spread_bps / 10_000,
                 "Áp dụng": "Cả hai chiều"},
                {"Khoản mục": "Tổng một vòng", "Mức": self.round_trip_rate,
                 "Áp dụng": "Mua rồi bán"},
            ]
        )


class VNStockProvider:
    """Nguồn dữ liệu chứng khoán Việt Nam.

    Thử lần lượt: thư viện ``vnstock`` nếu đã cài → file CSV trong ``data/``
    → dữ liệu mô phỏng có đặc tính thị trường Việt Nam. Nhờ chuỗi dự phòng
    này, phần demo luôn chạy được kể cả khi không có mạng hay chưa cài gì.
    """

    name = "vnstock"

    def __init__(self, seed: int = 20260101, csv_path: Path | str | None = None):
        self.seed = seed
        self.csv_path = Path(csv_path) if csv_path else DATA_DIR / "vn_prices.csv"

    # -- phát hiện nguồn khả dụng ---------------------------------------
    @staticmethod
    def has_vnstock() -> bool:
        try:
            import vnstock  # noqa: F401
        except Exception:
            return False
        return True

    def has_csv(self) -> bool:
        return self.csv_path.exists()

    def source_in_use(self) -> str:
        if self.has_vnstock():
            return "Thư viện vnstock (dữ liệu thật)"
        if self.has_csv():
            return "File CSV cục bộ"
        return "Dữ liệu mô phỏng (chế độ ngoại tuyến)"

    # -- giao diện chuẩn -------------------------------------------------
    def get_prices(
        self, tickers: Sequence[str], start: date | str, end: date | str | None = None
    ) -> pd.DataFrame:
        wanted = [str(t).upper().strip() for t in tickers]

        if self.has_vnstock():
            try:
                df = self._from_vnstock(wanted, start, end)
                if df is not None and not df.empty:
                    return df
            except Exception:
                pass

        if self.has_csv():
            try:
                df = pd.read_csv(self.csv_path, index_col=0, parse_dates=True)
                cols = [c for c in wanted if c in df.columns]
                if cols:
                    out = df.loc[df.index >= pd.Timestamp(start), cols]
                    if end is not None:
                        out = out.loc[out.index <= pd.Timestamp(end)]
                    return out.ffill().dropna(how="any")
            except Exception:
                pass

        # Thử lấy dữ liệu thật trực tiếp từ Yahoo Finance v8 (.VN)
        try:
            from core.data import _raw_download
            yahoo_tickers = [
                f"{t}.VN" if not t.endswith(".VN") and not t.startswith("^") else t
                for t in wanted
                if t != "VNINDEX"
            ]
            if yahoo_tickers:
                df_raw = _raw_download(yahoo_tickers, str(start))
                if df_raw is not None and not df_raw.empty and len(df_raw.columns) >= 1:
                    rename_map = {f"{t}.VN": t for t in wanted}
                    df_out = df_raw.rename(columns=rename_map)
                    if "VNINDEX" in wanted and "VNINDEX" not in df_out.columns:
                        df_out["VNINDEX"] = df_out.mean(axis=1)
                    if end is not None:
                        df_out = df_out.loc[df_out.index <= pd.Timestamp(end)]
                    return df_out.sort_index().ffill().dropna(how="any")
        except Exception:
            pass

        return simulate_vn_prices(wanted, start, end, seed=self.seed)

    def _from_vnstock(self, tickers, start, end) -> pd.DataFrame | None:  # pragma: no cover
        """Lấy dữ liệu thật qua ``vnstock`` (phiên bản mới)."""
        try:
            from vnstock import Quote
        except ImportError:
            return None
            
        q = Quote(source='VCI')
        frames = {}
        for t in tickers:
            try:
                # Gọi API lấy dữ liệu lịch sử
                d = q.history(symbol=t, start=str(start), end=str(end) if end else None, interval='1D')
                if d is None or len(d) == 0:
                    continue
                
                # Chuẩn hóa tên cột
                d = d.rename(columns=str.lower)
                # Xử lý cột thời gian (bỏ múi giờ để pandas dễ xử lý)
                idx = pd.to_datetime(d["time"]).dt.tz_localize(None)
                
                # Lấy giá đóng cửa
                col = "close" if "close" in d.columns else d.columns[-1]
                # vnstock trả về đơn vị nghìn đồng (VD: 85.5), nhân 1000 để ra VNĐ thật
                prices = d[col].to_numpy(dtype=float) * 1000.0 
                
                frames[t] = pd.Series(prices, index=idx)
            except Exception:
                continue
                
        if frames:
            return pd.DataFrame(frames).sort_index().ffill().dropna(how="any")
        return None


def simulate_vn_prices(
    tickers: Sequence[str],
    start: date | str = "2018-01-01",
    end: date | str | None = None,
    seed: int = 20260101,
) -> pd.DataFrame:
    """Mô phỏng giá cổ phiếu Việt Nam có áp biên độ dao động.

    Khác biệt then chốt so với mô phỏng thị trường Mỹ: lợi suất mỗi phiên bị
    **cắt theo biên độ sàn**. Điều này làm phân phối cụt đuôi, ảnh hưởng thật
    tới các chỉ số rủi ro đuôi như VaR và CVaR — một đặc điểm cấu trúc của
    thị trường Việt Nam mà mô hình chuẩn Mỹ không nắm được.
    """
    wanted = list(dict.fromkeys(str(t).upper().strip() for t in tickers))
    idx = pd.bdate_range("2016-01-04", str(end) if end else "2026-06-30")
    rng = np.random.default_rng(seed)
    n = len(idx)

    # nhân tố thị trường chung: biến động cao hơn Mỹ, có chu kỳ rõ rệt hơn
    regimes = {0: (0.00060, 0.0090), 1: (0.00010, 0.0130), 2: (-0.00110, 0.0250)}
    trans = np.array([[0.980, 0.017, 0.003], [0.035, 0.935, 0.030], [0.012, 0.070, 0.918]])

    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(3, p=trans[states[t - 1]])
    market = rng.normal([regimes[s][0] for s in states], [regimes[s][1] for s in states])

    data = {}
    for tk in wanted:
        meta = VN_UNIVERSE.get(tk)
        if meta is None:
            h = sum(ord(c) for c in tk) % 1000
            meta = {"exchange": "HOSE", "sector": "Khác",
                    "beta": 0.8 + h % 80 / 100, "vol": 0.22 + h % 25 / 100}

        limit = PRICE_LIMITS.get(meta["exchange"], 0.07)
        idio = rng.normal(0.0, meta["vol"] / np.sqrt(252), n)
        raw = meta["beta"] * market + idio

        # Biên độ của sàn áp lên lợi suất *giản đơn*, không phải log-return.
        # Vì vậy phải đổi sang lợi suất giản đơn rồi mới cắt, nếu không
        # exp(0,07) - 1 = 7,25% sẽ vượt biên độ 7% của HOSE.
        simple = np.expm1(raw)
        capped = np.clip(simple, -limit, limit)

        start_px = 1000.0 if tk == VN_BENCHMARK else 50.0
        data[tk] = start_px * np.cumprod(1.0 + capped)

    df = pd.DataFrame(data, index=idx)
    df = df.loc[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df.loc[df.index <= pd.Timestamp(end)]
    return df


def universe_table() -> pd.DataFrame:
    """Bảng rổ cổ phiếu hỗ trợ sẵn, kèm sàn và nhóm ngành."""
    rows = [
        {"Mã": k, "Sàn": v["exchange"], "Ngành": v["sector"],
         "Biên độ/phiên": PRICE_LIMITS.get(v["exchange"], 0.07)}
        for k, v in VN_UNIVERSE.items()
        if k != VN_BENCHMARK
    ]
    return pd.DataFrame(rows).sort_values(["Ngành", "Mã"]).reset_index(drop=True)


def sector_exposure(weights: pd.Series) -> pd.DataFrame:
    """Phân bổ danh mục theo nhóm ngành.

    Ở thị trường Việt Nam, nhóm ngân hàng và bất động sản chiếm tỷ trọng rất
    lớn trong chỉ số, nên rủi ro tập trung ngành là vấn đề thực tế hơn nhiều
    so với thị trường Mỹ.
    """
    rows = []
    for ticker, w in weights.items():
        meta = VN_UNIVERSE.get(str(ticker).upper(), {"sector": "Khác"})
        rows.append({"Ngành": meta["sector"], "Trọng số": float(w)})
    df = pd.DataFrame(rows).groupby("Ngành", as_index=False)["Trọng số"].sum()
    return df.sort_values("Trọng số", ascending=False).reset_index(drop=True)


def price_limit_hits(prices: pd.DataFrame) -> pd.DataFrame:
    """Đếm số phiên chạm trần hoặc sàn của từng mã.

    Chạm biên độ nhiều là dấu hiệu thanh khoản kém hoặc đầu cơ mạnh, và cũng
    cảnh báo rằng kết quả backtest có thể lạc quan: trên thực tế, khi cổ phiếu
    chạm sàn thì thường không khớp được lệnh bán.
    """
    rets = prices.pct_change().dropna()
    rows = []
    for col in rets.columns:
        meta = VN_UNIVERSE.get(str(col).upper(), {"exchange": "HOSE"})
        limit = PRICE_LIMITS.get(meta["exchange"], 0.07) - 0.002
        up = int((rets[col] >= limit).sum())
        down = int((rets[col] <= -limit).sum())
        rows.append(
            {
                "Mã": col,
                "Sàn": meta["exchange"],
                "Số phiên trần": up,
                "Số phiên sàn": down,
                "Tỷ lệ chạm biên": (up + down) / len(rets),
            }
        )
    return pd.DataFrame(rows).sort_values("Tỷ lệ chạm biên", ascending=False).reset_index(drop=True)