"""Phòng thí nghiệm kịch bản & kiểm tra sức chịu đựng.

Đây là tính năng "ăn tiền" nhất khi demo, vì nó trả lời câu hỏi mà nhà đầu
tư nào cũng quan tâm: **"Nếu thị trường sập 20% thì danh mục tôi mất bao
nhiêu?"** Các biểu đồ lịch sử không trả lời được câu đó.

Hai cách tiếp cận bổ sung cho nhau:

1. **Kịch bản theo nhân tố** – ước lượng beta của danh mục với các nhân tố
   rồi áp cú sốc giả định lên nhân tố.
2. **Phát lại lịch sử** – lấy đúng các giai đoạn khủng hoảng đã xảy ra và
   xem danh mục hiện tại sẽ ra sao trong giai đoạn đó.

Bổ sung thêm Monte Carlo có đuôi dày (phân phối t) vì cú sốc thực tế xảy ra
thường xuyên hơn nhiều so với giả định phân phối chuẩn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252

# Các giai đoạn căng thẳng kinh điển, dùng khi dữ liệu phủ được khoảng thời gian
HISTORICAL_EPISODES: dict[str, tuple[str, str]] = {
    "Khủng hoảng 2008": ("2008-09-01", "2009-03-31"),
    "Taper tantrum 2013": ("2013-05-01", "2013-09-30"),
    "Sụt giảm Q4/2018": ("2018-10-01", "2018-12-24"),
    "COVID sụp đổ 2020": ("2020-02-19", "2020-03-23"),
    "Sốc lãi suất 2022": ("2022-01-03", "2022-10-14"),
}


def factor_betas(portfolio_returns: pd.Series, factor_returns: pd.DataFrame) -> pd.Series:
    """Beta của danh mục với từng nhân tố, ước lượng bằng hồi quy bội OLS.

    Hồi quy *bội* chứ không phải từng cái một: các chỉ số thị trường tương
    quan rất mạnh, chạy riêng lẻ sẽ đếm trùng cùng một rủi ro nhiều lần.
    """
    y, X = pd.Series(portfolio_returns).align(factor_returns, join="inner", axis=0)
    A = np.column_stack([np.ones(len(y)), X.to_numpy(dtype=float)])
    coef, *_ = np.linalg.lstsq(A, y.to_numpy(dtype=float), rcond=None)
    return pd.Series(coef[1:], index=X.columns, name="beta")


def scenario_shock(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    shocks: dict[str, float],
) -> dict:
    """Ước tính tác động lên danh mục khi áp cú sốc lên các nhân tố.

    ``shocks`` nhận cú sốc dạng tổng mức, ví dụ ``{"SPY": -0.20, "^VIX": 0.80}``
    nghĩa là thị trường giảm 20% và VIX tăng 80%.
    """
    betas = factor_betas(portfolio_returns, factor_returns)
    detail = {f: float(betas.get(f, 0.0) * s) for f, s in shocks.items()}
    total = float(sum(detail.values()))
    return {
        "Tác động tổng": total,
        "Theo từng nhân tố": detail,
        "Beta sử dụng": betas.to_dict(),
    }


def scenario_grid(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    factor: str = "SPY",
    shock_range: tuple[float, ...] = (-0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Quét một dải cú sốc trên một nhân tố — ra ngay đường cong tổn thất."""
    betas = factor_betas(portfolio_returns, factor_returns)
    beta = float(betas.get(factor, 0.0))
    return pd.DataFrame(
        [
            {
                "Cú sốc nhân tố": s,
                "Tác động danh mục (ước tính)": beta * s,
                "Nhân tố": factor,
                "Beta": beta,
            }
            for s in shock_range
        ]
    )


def historical_stress(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series | None = None
) -> pd.DataFrame:
    """Phát lại các giai đoạn khủng hoảng có trong dữ liệu."""
    r = pd.Series(portfolio_returns).dropna()
    rows = []
    for name, (start, end) in HISTORICAL_EPISODES.items():
        window = r.loc[(r.index >= pd.Timestamp(start)) & (r.index <= pd.Timestamp(end))]
        if len(window) < 5:
            continue
        eq = (1 + window).cumprod()
        row = {
            "Giai đoạn": name,
            "Từ": start,
            "Đến": end,
            "Lợi nhuận danh mục": float(eq.iloc[-1] - 1),
            "Sụt giảm sâu nhất": float((eq / eq.cummax() - 1).min()),
            "Số phiên": int(len(window)),
        }
        if benchmark_returns is not None:
            b = pd.Series(benchmark_returns).reindex(window.index).dropna()
            if len(b) >= 5:
                row["Lợi nhuận benchmark"] = float((1 + b).prod() - 1)
                row["Chênh lệch"] = row["Lợi nhuận danh mục"] - row["Lợi nhuận benchmark"]
        rows.append(row)
    return pd.DataFrame(rows)


def worst_windows(portfolio_returns: pd.Series, window: int = 21, top: int = 5) -> pd.DataFrame:
    """Tìm các cửa sổ tệ nhất *do chính dữ liệu* chỉ ra, không cần ngày định sẵn.

    Hữu ích khi dữ liệu không phủ các giai đoạn khủng hoảng kinh điển.
    """
    r = pd.Series(portfolio_returns).dropna()
    roll = (1 + r).rolling(window).apply(np.prod, raw=True) - 1
    worst = roll.nsmallest(top)
    return pd.DataFrame(
        [
            {
                "Kết thúc ngày": ts.date(),
                f"Lợi nhuận {window} phiên": float(v),
                "Năm hoá tương đương": float((1 + v) ** (TRADING_DAYS / window) - 1),
            }
            for ts, v in worst.items()
        ]
    )


def monte_carlo_paths(
    portfolio_returns: pd.Series,
    horizon: int = TRADING_DAYS,
    n_paths: int = 5000,
    fat_tails: bool = True,
    df: int = 4,
    seed: int = 123,
) -> pd.DataFrame:
    """Mô phỏng phân phối giá trị danh mục trong tương lai.

    ``fat_tails=True`` dùng phân phối Student-t thay vì chuẩn. Đây không phải
    chi tiết vụn vặt: giả định chuẩn đánh giá thấp nghiêm trọng xác suất của
    những phiên sụp đổ, và đó chính là lúc danh mục thật sự gặp nguy.
    """
    r = pd.Series(portfolio_returns).dropna()
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    rng = np.random.default_rng(seed)

    if fat_tails:
        raw = stats.t.rvs(df, size=(n_paths, horizon), random_state=seed)
        raw = raw / np.sqrt(df / (df - 2))     # chuẩn hoá về phương sai 1
        sims = mu + sd * raw
    else:
        sims = rng.normal(mu, sd, (n_paths, horizon))

    terminal = np.prod(1 + sims, axis=1)
    return pd.DataFrame(
        {
            "Giá trị cuối kỳ": terminal,
            "Lợi nhuận": terminal - 1,
            "Sụt giảm tối đa": [
                float((e / np.maximum.accumulate(e) - 1).min())
                for e in np.cumprod(1 + sims, axis=1)
            ],
        }
    )


def monte_carlo_summary(paths: pd.DataFrame) -> pd.DataFrame:
    """Tóm tắt kết quả mô phỏng thành các phân vị dễ trình bày."""
    ret = paths["Lợi nhuận"]
    dd = paths["Sụt giảm tối đa"]
    rows = [
        {"Chỉ tiêu": "Lợi nhuận trung vị", "Giá trị": float(ret.median())},
        {"Chỉ tiêu": "Phân vị 5% (xấu)", "Giá trị": float(ret.quantile(0.05))},
        {"Chỉ tiêu": "Phân vị 95% (tốt)", "Giá trị": float(ret.quantile(0.95))},
        {"Chỉ tiêu": "Xác suất thua lỗ", "Giá trị": float((ret < 0).mean())},
        {"Chỉ tiêu": "Xác suất mất > 20%", "Giá trị": float((ret < -0.20).mean())},
        {"Chỉ tiêu": "Sụt giảm trung vị", "Giá trị": float(dd.median())},
        {"Chỉ tiêu": "Sụt giảm phân vị 95%", "Giá trị": float(dd.quantile(0.05))},
    ]
    return pd.DataFrame(rows)
