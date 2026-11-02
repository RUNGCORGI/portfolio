"""
core/factors.py  —  P3 (SỞ HỮU)
Khớp bài gốc: FACTOR ENGINE + FACTOR BETAS + CAPM ALPHA (myapp8 -> myapp13).

PHÁT TRIỂN THÊM #2 của P3: hồi quy đa nhân tố BẰNG OLS CÓ THỐNG KÊ
(hệ số, t-stat, R²) thay cho cách bài gốc chỉ lấy tương quan trượt với một
"momentum giả lập từ SPY". Dùng ETF proxy nên chạy offline được:
    MKT = SPY, SIZE = IWM - SPY, TECH/GROWTH = QQQ - SPY, VOL = Δ^VIX
Khi có mạng, hàm load_fama_french() lấy dữ liệu 3-factor thật của Ken French.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from core.contracts import TRADING_DAYS, FactorReport, MarketData, PortfolioResult


def simple_betas(pf: PortfolioResult, md: MarketData) -> Dict[str, float]:
    out = {}
    for col in md.factors.columns:
        fr = md.factor_returns[col]
        j = pd.concat([pf.returns, fr], axis=1, join="inner").dropna()
        j.columns = ["p", "f"]
        v = float(j["f"].var())
        out[col] = 0.0 if v == 0 else float(j["p"].cov(j["f"]) / v)
    return out


def capm_alpha(pf: PortfolioResult, spy_beta: float) -> float:
    mkt = float(pf.benchmark_returns.mean() * TRADING_DAYS)
    return float(pf.returns.mean() * TRADING_DAYS - spy_beta * mkt)


def build_factor_matrix(md: MarketData) -> pd.DataFrame:
    """Nhân tố long-short tổng hợp từ ETF — chạy được hoàn toàn offline."""
    fr = md.factor_returns
    f = pd.DataFrame(index=fr.index)
    f["MKT"] = fr["SPY"]
    if "IWM" in fr:
        f["SIZE"] = fr["IWM"] - fr["SPY"]
    if "QQQ" in fr:
        f["GROWTH"] = fr["QQQ"] - fr["SPY"]
    if "^VIX" in fr:
        f["VOL"] = fr["^VIX"]
    return f.dropna()


def ols(y: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
    """OLS thuần numpy: hệ số, sai số chuẩn, t-stat. Không cần statsmodels."""
    Xc = np.column_stack([np.ones(len(X)), X.values])
    yv = y.values
    beta, *_ = np.linalg.lstsq(Xc, yv, rcond=None)
    resid = yv - Xc @ beta
    dof = max(len(yv) - Xc.shape[1], 1)
    s2 = resid @ resid / dof
    xtx_inv = np.linalg.pinv(Xc.T @ Xc)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * s2, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, 0.0)
    return pd.DataFrame(
        {"coef": beta, "std_err": se, "t_stat": t},
        index=["alpha"] + list(X.columns),
    )


def r_squared(y: pd.Series, X: pd.DataFrame) -> float:
    Xc = np.column_stack([np.ones(len(X)), X.values])
    beta, *_ = np.linalg.lstsq(Xc, y.values, rcond=None)
    resid = y.values - Xc @ beta
    tss = ((y.values - y.values.mean()) ** 2).sum()
    return float(1 - resid @ resid / tss) if tss > 0 else 0.0


def multifactor_regression(pf: PortfolioResult, md: MarketData):
    F = build_factor_matrix(md)
    j = pd.concat([pf.returns.rename("p"), F], axis=1, join="inner").dropna()
    if len(j) < 30:
        return None, None
    y, X = j["p"], j.drop(columns=["p"])
    params = ols(y, X)
    # quy đổi alpha ngày -> năm cho dễ đọc
    params.loc["alpha", "coef"] *= TRADING_DAYS
    return params, r_squared(y, X)


def build_factor_report(pf: PortfolioResult, md: MarketData,
                        window: int = 20) -> FactorReport:
    betas = simple_betas(pf, md)
    mom = md.benchmark_returns.rolling(12).sum()   # momentum 12 phiên, proxy 12-1
    j = pd.concat([pf.returns, mom], axis=1, join="inner").dropna()
    j.columns = ["p", "m"]
    rolling_mom = j["p"].rolling(window).corr(j["m"]).dropna()

    params, r2 = multifactor_regression(pf, md)

    # Khớp bài gốc: Treynor Ratio & Rolling Beta Matrix 60 phiên (myapp13)
    market_beta = betas.get("SPY", 1.0)
    treynor = float((pf.returns.mean() * TRADING_DAYS) / market_beta) if market_beta != 0 else 0.0

    beta_matrix = pd.DataFrame()
    spy_ret = md.factor_returns["SPY"] if "SPY" in md.factor_returns else md.benchmark_returns
    for ticker in md.tickers:
        al = pd.concat([md.returns[ticker], spy_ret], axis=1).dropna()
        var60 = al.iloc[:, 1].rolling(60).var()
        cov60 = al.iloc[:, 0].rolling(60).cov(al.iloc[:, 1])
        b = cov60 / var60
        beta_matrix[ticker] = b

    return FactorReport(
        betas=betas,
        capm_alpha=capm_alpha(pf, betas.get("SPY", 0.0)),
        rolling_momentum_corr=rolling_mom,
        ff_params=params,
        ff_r2=r2,
        treynor=treynor,
        beta_matrix=beta_matrix,
    )


def load_fama_french(start: str = "2023-01-01") -> Optional[pd.DataFrame]:
    """Dữ liệu 3-factor thật (cần mạng). Trả None nếu lỗi -> app vẫn chạy."""
    url = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_Factors_daily_CSV.zip")
    try:
        df = pd.read_csv(url, skiprows=3, compression="zip")
        df = df.rename(columns={df.columns[0]: "Date"})
        df = df[df["Date"].astype(str).str.len() == 8]
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
        df = df.set_index("Date").astype(float) / 100.0
        return df.loc[start:]
    except Exception:
        return None
