"""Kiểm định thống kê cho kết quả backtest — module gần như không nhóm nào làm.

Câu hỏi mà mọi hội đồng và mọi nhà đầu tư chuyên nghiệp đều hỏi:

    "Sharpe 1.87 này là **kỹ năng** hay chỉ là **may mắn** vì bạn đã thử
    rất nhiều chiến lược rồi chọn cái đẹp nhất?"

Module này trả lời bằng số:

* **PSR** – xác suất Sharpe thật sự lớn hơn một ngưỡng, có hiệu chỉnh cho
  độ lệch và độ nhọn của phân phối lợi suất.
* **Deflated Sharpe Ratio** – PSR sau khi phạt theo *số lần thử*.
  Thử 50 chiến lược thì Sharpe cao nhất gần như chắc chắn bị thổi phồng.
* **Block bootstrap** – khoảng tin cậy giữ được tự tương quan của chuỗi.
* **Xếp hạng phân vị** so với hàng nghìn danh mục ngẫu nhiên.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from core.upgrade_api import TRADING_DAYS, cagr, max_drawdown, sharpe


# --------------------------------------------------------------------------
# Sharpe đã hiệu chỉnh
# --------------------------------------------------------------------------


def probabilistic_sharpe(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado).

    Trả về xác suất Sharpe thật vượt ``benchmark_sr``. Có hiệu chỉnh cho
    skewness và kurtosis — quan trọng vì lợi suất tài chính đuôi dày, khiến
    Sharpe thông thường bị đánh giá quá lạc quan.
    """
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 10:
        return float("nan")

    sr = sharpe(r) / np.sqrt(TRADING_DAYS)        # Sharpe theo phiên
    bench = benchmark_sr / np.sqrt(TRADING_DAYS)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))

    denom = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2, 1e-12))
    z = (sr - bench) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe(returns: pd.Series, n_trials: int = 1, trial_sr_std: float | None = None) -> float:
    """Deflated Sharpe Ratio — PSR sau khi phạt vì đã thử nhiều chiến lược.

    Trực giác: nếu bạn thử 50 cấu hình, Sharpe cao nhất trong 50 cái đó tự
    nhiên đã cao hơn mức thật. DSR dựng ngưỡng kỳ vọng của "Sharpe cao nhất
    do thuần may mắn" rồi hỏi: kết quả của bạn có vượt nổi ngưỡng đó không?

    DSR < 0.95 nghĩa là chưa đủ bằng chứng để tin chiến lược có alpha thật.
    """
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 10 or n_trials < 1:
        return float("nan")

    sr_std = trial_sr_std if trial_sr_std is not None else sharpe(r) / np.sqrt(TRADING_DAYS) * 0.5
    sr_std = max(abs(float(sr_std)), 1e-6)

    if n_trials == 1:
        return probabilistic_sharpe(r, 0.0)

    gamma = 0.5772156649  # hằng số Euler–Mascheroni
    e1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    e2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    sr0_daily = sr_std * ((1 - gamma) * e1 + gamma * e2)
    return probabilistic_sharpe(r, sr0_daily * np.sqrt(TRADING_DAYS))


def min_track_record_length(returns: pd.Series, target_prob: float = 0.95) -> float:
    """Số phiên tối thiểu cần có để khẳng định Sharpe > 0 với độ tin cậy cho trước."""
    r = pd.Series(returns).dropna()
    sr = sharpe(r) / np.sqrt(TRADING_DAYS)
    if sr <= 0:
        return float("inf")
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    z = stats.norm.ppf(target_prob)
    return float(1 + (1 - skew * sr + (kurt - 1) / 4 * sr**2) * (z / sr) ** 2)


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------


def block_bootstrap(
    returns: pd.Series,
    n_boot: int = 1000,
    block: int = 21,
    seed: int = 42,
    metrics: tuple[str, ...] = ("sharpe", "cagr", "max_drawdown"),
) -> pd.DataFrame:
    """Bootstrap theo khối: lấy mẫu từng *đoạn* liên tiếp thay vì từng phiên.

    Lấy mẫu từng phiên riêng lẻ sẽ phá vỡ tự tương quan và cụm biến động,
    khiến khoảng tin cậy hẹp giả tạo. Khối ~21 phiên giữ được cấu trúc đó.
    """
    r = pd.Series(returns).dropna().to_numpy()
    n = len(r)
    if n < block * 2:
        return pd.DataFrame(columns=list(metrics))

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    fns = {"sharpe": sharpe, "cagr": cagr, "max_drawdown": max_drawdown}

    rows = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, n_blocks)
        sample = np.concatenate([r[s : s + block] for s in starts])[:n]
        s = pd.Series(sample)
        rows.append({m: fns[m](s) for m in metrics})
    return pd.DataFrame(rows)


def confidence_intervals(
    returns: pd.Series, n_boot: int = 1000, block: int = 21, level: float = 0.95, seed: int = 42
) -> pd.DataFrame:
    """Khoảng tin cậy bootstrap cho các chỉ số chính.

    Kết quả cho phép báo cáo trung thực kiểu: "Sharpe 1.87, KTC 95%
    [0.82; 2.61]" — thay vì chỉ nêu một con số trần trụi.
    """
    boot = block_bootstrap(returns, n_boot, block, seed)
    if boot.empty:
        return pd.DataFrame()
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    actual = {
        "sharpe": sharpe(returns),
        "cagr": cagr(returns),
        "max_drawdown": max_drawdown(returns),
    }
    names = {"sharpe": "Sharpe", "cagr": "CAGR", "max_drawdown": "Sụt giảm tối đa"}
    rows = []
    for col in boot.columns:
        rows.append(
            {
                "Chỉ số": names.get(col, col),
                "Giá trị thực tế": actual.get(col, np.nan),
                "Cận dưới": float(np.percentile(boot[col], lo)),
                "Cận trên": float(np.percentile(boot[col], hi)),
                "Độ lệch chuẩn": float(boot[col].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# So sánh với danh mục ngẫu nhiên
# --------------------------------------------------------------------------


def random_portfolio_percentile(
    asset_returns: pd.DataFrame, strategy_returns: pd.Series, n_random: int = 2000, seed: int = 99
) -> dict:
    """Chiến lược của bạn đứng ở phân vị nào so với danh mục ngẫu nhiên?

    Nếu phân vị chỉ khoảng 55%, nghĩa là ném phi tiêu chọn trọng số cũng cho
    kết quả gần như vậy — một bài kiểm tra tính khiêm tốn rất hữu ích.
    """
    rng = np.random.default_rng(seed)
    arr = asset_returns.dropna().to_numpy()
    k = arr.shape[1]
    sr = np.empty(n_random)
    for i in range(n_random):
        w = rng.random(k)
        w /= w.sum()
        sr[i] = sharpe(pd.Series(arr @ w))
    actual = sharpe(strategy_returns)
    return {
        "Sharpe chiến lược": actual,
        "Phân vị": float((sr < actual).mean() * 100),
        "Sharpe trung vị ngẫu nhiên": float(np.median(sr)),
        "Sharpe tốt nhất ngẫu nhiên": float(np.max(sr)),
        "Số danh mục ngẫu nhiên": n_random,
    }


def validation_report(
    strategy_returns: pd.Series, n_trials: int = 1, asset_returns: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Bảng kiểm định tổng hợp — nên đưa thẳng vào slide bảo vệ."""
    psr = probabilistic_sharpe(strategy_returns)
    dsr = deflated_sharpe(strategy_returns, n_trials)
    mtrl = min_track_record_length(strategy_returns)
    rows = [
        {"Chỉ tiêu": "Sharpe", "Giá trị": sharpe(strategy_returns),
         "Diễn giải": "Lợi nhuận trên mỗi đơn vị rủi ro"},
        {"Chỉ tiêu": "PSR", "Giá trị": psr,
         "Diễn giải": "Xác suất Sharpe thật > 0 (đã hiệu chỉnh đuôi dày)"},
        {"Chỉ tiêu": f"DSR ({n_trials} lần thử)", "Giá trị": dsr,
         "Diễn giải": "PSR sau khi phạt vì thử nhiều; cần > 0,95"},
        {"Chỉ tiêu": "Số phiên tối thiểu", "Giá trị": mtrl,
         "Diễn giải": "Độ dài lịch sử cần có để kết luận đáng tin"},
    ]
    if asset_returns is not None:
        pct = random_portfolio_percentile(asset_returns, strategy_returns)
        rows.append(
            {"Chỉ tiêu": "Phân vị so với ngẫu nhiên", "Giá trị": pct["Phân vị"],
             "Diễn giải": "Vượt bao nhiêu % danh mục chọn ngẫu nhiên"}
        )
    return pd.DataFrame(rows)
