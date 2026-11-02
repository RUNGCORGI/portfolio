"""Portfolio X-Ray — soi và so sánh nhiều danh mục cạnh nhau.

Một biểu đồ đường vốn đẹp không nói lên được danh mục A có thực sự tốt hơn
danh mục B hay không. Module này dựng **tearsheet đầy đủ** theo chuẩn mà
các quỹ dùng, cộng thêm ba thứ mà dashboard thông thường bỏ sót:

* **Phân rã rủi ro** – mã chiếm 20% vốn có thể đang gánh 45% rủi ro.
  Biểu đồ tròn phân bổ vốn không bao giờ cho thấy điều này.
* **Phân tích chồng lấn** – hai danh mục trông khác nhau trên giấy nhưng có
  thể cùng đặt cược vào một rủi ro duy nhất.
* **Hiệu quả theo từng năm** – trung bình toàn kỳ che giấu việc chiến lược
  chỉ thắng nhờ đúng một năm may mắn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.upgrade_api import TRADING_DAYS, ann_vol, cagr, max_drawdown, sharpe


# --------------------------------------------------------------------------
# Bộ chỉ số đầy đủ
# --------------------------------------------------------------------------


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    """Chỉ phạt biến động phía giảm — hợp lý hơn Sharpe khi lợi suất lệch."""
    r = pd.Series(returns).dropna() - rf / TRADING_DAYS
    down = r[r < 0]
    sd = down.std(ddof=1)
    if not np.isfinite(sd) or sd <= 1e-15:
        return 0.0
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS))


def calmar(returns: pd.Series) -> float:
    mdd = abs(max_drawdown(returns))
    return float(cagr(returns) / mdd) if mdd > 1e-12 else 0.0


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    r = pd.Series(returns).dropna()
    return float(np.percentile(r, (1 - level) * 100)) if len(r) else 0.0


def conditional_var(returns: pd.Series, level: float = 0.95) -> float:
    """CVaR: mức lỗ bình quân *trong những ngày tệ nhất*, không phải ngưỡng."""
    r = pd.Series(returns).dropna()
    var = value_at_risk(r, level)
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) else var


def ulcer_index(returns: pd.Series) -> float:
    """Chỉ số Ulcer: đo *độ sâu và độ dài* của sụt giảm, không chỉ đáy sâu nhất.

    Hai danh mục cùng sụt 30% nhưng một cái hồi phục sau 2 tháng, cái kia mất
    2 năm — nhà đầu tư cảm nhận hoàn toàn khác nhau. Max drawdown không phân
    biệt được, Ulcer thì có.
    """
    eq = (1 + pd.Series(returns).dropna()).cumprod()
    dd = (eq / eq.cummax() - 1) * 100
    return float(np.sqrt((dd**2).mean()))


def tail_ratio(returns: pd.Series) -> float:
    """Tỷ lệ giữa đuôi lãi và đuôi lỗ. Trên 1 là tốt."""
    r = pd.Series(returns).dropna()
    if len(r) < 20:
        return 0.0
    down = abs(np.percentile(r, 5))
    return float(np.percentile(r, 95) / down) if down > 1e-12 else 0.0


def tearsheet(returns: pd.Series, benchmark: pd.Series | None = None) -> dict:
    """Bộ chỉ số đầy đủ cho một chuỗi lợi suất."""
    r = pd.Series(returns).dropna()
    out = {
        "Tổng lợi nhuận": float((1 + r).prod() - 1),
        "CAGR": cagr(r),
        "Biến động năm": ann_vol(r),
        "Sharpe": sharpe(r),
        "Sortino": sortino(r),
        "Calmar": calmar(r),
        "Sụt giảm tối đa": max_drawdown(r),
        "Chỉ số Ulcer": ulcer_index(r),
        "VaR 95%": value_at_risk(r),
        "CVaR 95%": conditional_var(r),
        "Tỷ lệ đuôi": tail_ratio(r),
        "Tỷ lệ phiên lãi": float((r > 0).mean()),
        "Phiên tệ nhất": float(r.min()),
        "Số phiên": float(len(r)),
    }
    if benchmark is not None:
        b = pd.Series(benchmark).reindex(r.index).dropna()
        if len(b) > 20:
            rr, bb = r.align(b, join="inner")
            diff = rr - bb
            te = diff.std(ddof=1)
            var_b = float(bb.var())
            out["Beta"] = float(rr.cov(bb) / var_b) if var_b > 0 else 0.0
            out["Sai số theo dõi"] = float(te * np.sqrt(TRADING_DAYS))
            out["Information Ratio"] = (
                float(diff.mean() / te * np.sqrt(TRADING_DAYS)) if te > 1e-15 else 0.0
            )
            out["Lợi nhuận vượt trội"] = float((1 + rr).prod() - (1 + bb).prod())
    return out


def compare(
    portfolios: dict[str, pd.Series], benchmark: pd.Series | None = None
) -> pd.DataFrame:
    """Đặt nhiều danh mục cạnh nhau trên cùng bộ chỉ số."""
    return pd.DataFrame({name: tearsheet(s, benchmark) for name, s in portfolios.items()})


# --------------------------------------------------------------------------
# Phân rã rủi ro
# --------------------------------------------------------------------------


def risk_contribution(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """Mỗi mã đóng góp bao nhiêu phần trăm **rủi ro** danh mục.

    Cột "Chênh lệch" là thứ đáng xem nhất: giá trị dương lớn nghĩa là mã đó
    đang gánh rủi ro vượt xa phần vốn được cấp.
    """
    w = weights.reindex(cov.index).fillna(0.0).to_numpy(float)
    sigma = cov.to_numpy()
    port_var = float(w @ sigma @ w)
    if port_var <= 0:
        return pd.DataFrame()

    marginal = sigma @ w
    contrib = w * marginal / port_var
    port_vol = np.sqrt(port_var)

    df = pd.DataFrame(
        {
            "Mã": cov.index,
            "Trọng số vốn": w,
            "Tỷ trọng rủi ro": contrib,
            "Đóng góp biến động": w * marginal / port_vol,
        }
    )
    df["Chênh lệch"] = df["Tỷ trọng rủi ro"] - df["Trọng số vốn"]
    return df.sort_values("Tỷ trọng rủi ro", ascending=False).reset_index(drop=True)


def concentration_metrics(weights: pd.Series) -> dict:
    """Bộ chỉ số đo mức tập trung của danh mục."""
    w = np.asarray(weights, dtype=float)
    w = w[w > 1e-9]
    hhi = float(np.sum(w**2))
    ent = float(-np.sum(w * np.log(w))) if len(w) else 0.0
    return {
        "Số mã thực tế": int(len(w)),
        "Số vị thế hiệu dụng": 1.0 / hhi if hhi > 0 else 0.0,
        "Chỉ số HHI": hhi,
        "Entropy": ent,
        "Trọng số lớn nhất": float(w.max()) if len(w) else 0.0,
        "Ba mã lớn nhất": float(np.sort(w)[-3:].sum()) if len(w) >= 3 else float(w.sum()),
    }


def overlap_analysis(weights_a: pd.Series, weights_b: pd.Series) -> dict:
    """Hai danh mục thực chất giống nhau tới mức nào?

    Trả về độ chồng lấn theo trọng số và khoảng cách active share — chỉ số
    quỹ tương hỗ dùng để chứng minh họ không phải "quỹ chỉ số trá hình".
    """
    idx = weights_a.index.union(weights_b.index)
    a = weights_a.reindex(idx).fillna(0.0)
    b = weights_b.reindex(idx).fillna(0.0)
    overlap = float(np.minimum(a, b).sum())
    return {
        "Độ chồng lấn": overlap,
        "Active share": 1.0 - overlap,
        "Số mã chung": int(((a > 1e-9) & (b > 1e-9)).sum()),
        "Chỉ có ở danh mục A": int(((a > 1e-9) & (b <= 1e-9)).sum()),
        "Chỉ có ở danh mục B": int(((b > 1e-9) & (a <= 1e-9)).sum()),
    }


# --------------------------------------------------------------------------
# Phân tích theo thời gian
# --------------------------------------------------------------------------


def yearly_returns(portfolios: dict[str, pd.Series]) -> pd.DataFrame:
    """Lợi nhuận theo từng năm dương lịch.

    Trung bình toàn kỳ rất dễ che giấu việc chiến lược chỉ thắng nhờ đúng
    một năm bùng nổ rồi lẹt đẹt suốt phần còn lại.
    """
    rows = {}
    for name, s in portfolios.items():
        r = pd.Series(s).dropna()
        rows[name] = (1 + r).groupby(r.index.year).prod() - 1
    return pd.DataFrame(rows)


def monthly_heatmap(returns: pd.Series) -> pd.DataFrame:
    """Bảng lợi nhuận theo tháng, dòng là năm — dạng heatmap kinh điển."""
    r = pd.Series(returns).dropna()
    monthly = (1 + r).resample("ME").prod() - 1
    df = pd.DataFrame(
        {"Năm": monthly.index.year, "Tháng": monthly.index.month, "LN": monthly.values}
    )
    return df.pivot(index="Năm", columns="Tháng", values="LN")


def rolling_comparison(
    portfolios: dict[str, pd.Series], window: int = 126
) -> dict[str, pd.DataFrame]:
    """Sharpe và biến động trượt cho nhiều danh mục — xem ai dẫn trước khi nào."""
    sh, vol = {}, {}
    for name, s in portfolios.items():
        r = pd.Series(s).dropna()
        sh[name] = (
            r.rolling(window).mean() / r.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
        )
        vol[name] = r.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    return {"sharpe": pd.DataFrame(sh).dropna(), "volatility": pd.DataFrame(vol).dropna()}


def drawdown_episodes(returns: pd.Series, top: int = 5) -> pd.DataFrame:
    """Liệt kê các đợt sụt giảm sâu nhất kèm thời gian hồi phục."""
    r = pd.Series(returns).dropna()
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    under = dd < -1e-9

    eps, start = [], None
    for ts, flag in under.items():
        if flag and start is None:
            start = ts
        elif not flag and start is not None:
            eps.append(_episode(dd, start, ts, True))
            start = None
    if start is not None:
        eps.append(_episode(dd, start, dd.index[-1], False))

    if not eps:
        return pd.DataFrame(
            columns=["Bắt đầu", "Đáy", "Kết thúc", "Mức giảm", "Số phiên", "Đã hồi phục"]
        )
    return pd.DataFrame(eps).sort_values("Mức giảm").head(top).reset_index(drop=True)


def _episode(dd: pd.Series, start, end, recovered: bool) -> dict:
    w = dd.loc[start:end]
    return {
        "Bắt đầu": start.date(),
        "Đáy": w.idxmin().date(),
        "Kết thúc": end.date(),
        "Mức giảm": float(w.min()),
        "Số phiên": int(len(w)),
        "Đã hồi phục": recovered,
    }


def benchmark_suite(
    asset_returns: pd.DataFrame, bench_returns: pd.Series | None = None
) -> dict[str, pd.Series]:
    """Bộ mốc so sánh tiêu chuẩn mà mọi chiến lược đều phải vượt qua.

    Bao gồm chia đều, mua và giữ thị trường, và danh mục 60/40 kinh điển.
    Nếu chiến lược tinh vi không thắng nổi chia đều thì cần thành thật thừa nhận.
    """
    out: dict[str, pd.Series] = {}
    k = asset_returns.shape[1]
    out["Chia đều"] = pd.Series(
        asset_returns.to_numpy() @ np.repeat(1 / k, k), index=asset_returns.index
    )
    if bench_returns is not None:
        b = pd.Series(bench_returns).reindex(asset_returns.index).dropna()
        if len(b) > 20:
            out["Thị trường (SPY)"] = b
            # xấp xỉ 60/40 bằng 60% thị trường + 40% tiền mặt lãi suất 0
            out["60/40 xấp xỉ"] = b * 0.6
    return out
