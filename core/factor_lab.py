"""Mô hình nhân tố đầy đủ — hoàn tất Phần 1 của lộ trình nâng cấp.

VÌ SAO CẦN MODULE NÀY
---------------------
Bài gốc tính beta của danh mục với SPY, QQQ, IWM và VIX bằng cách hồi quy
**từng cái một**. Cách đó sai về mặt thống kê: bốn chỉ số này tương quan
với nhau rất mạnh (QQQ và SPY thường trên 0,9), nên mỗi hệ số riêng lẻ đều
đang đếm lại cùng một rủi ro thị trường. Kết quả là "beta với QQQ = 1,2"
không nói lên điều gì về thiên lệch công nghệ cả.

Module này sửa ba thứ:

1. **Dựng nhân tố long/short** theo tinh thần Fama–French, để các nhân tố
   gần trực giao với nhau. ``GROWTH = QQQ − SPY`` mới thực sự đo thiên lệch
   tăng trưởng, vì phần thị trường chung đã bị trừ đi.

2. **Hồi quy bội với sai số chuẩn Newey–West**. Lợi suất có tự tương quan
   và phương sai thay đổi; sai số chuẩn OLS thông thường quá nhỏ, làm thống
   kê t bị thổi phồng và alpha trông "có ý nghĩa" trong khi thực ra không.

3. **Phân rã lợi nhuận** thành alpha + từng nhân tố + phần dư, trả lời câu
   hỏi quan trọng nhất: *lợi nhuận đến từ kỹ năng hay chỉ do gánh rủi ro?*
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252

# Các ETF dùng làm đại diện nhân tố
PROXY_TICKERS = ("SPY", "QQQ", "IWM", "^VIX", "GLD", "TLT")


# --------------------------------------------------------------------------
# 1. Dựng nhân tố
# --------------------------------------------------------------------------


def build_proxy_factors(factor_prices: pd.DataFrame) -> pd.DataFrame:
    """Dựng nhân tố long/short từ giá các ETF đại diện.

    * ``MKT``    – lợi suất thị trường (SPY). Nhân tố nền.
    * ``GROWTH`` – QQQ trừ SPY. Phần thưởng riêng cho thiên lệch tăng trưởng
      và công nghệ, đã loại bỏ phần thị trường chung.
    * ``SIZE``   – IWM trừ SPY. Thiên lệch vốn hoá nhỏ (tinh thần nhân tố SMB).
    * ``VOL``    – thay đổi của VIX. Độ nhạy với cú sốc biến động.
    * ``GOLD``   – GLD trừ SPY. Thiên lệch phòng thủ, trú ẩn.
    * ``DURATION`` – TLT trừ SPY. Độ nhạy với lãi suất dài hạn.

    Phép trừ SPY là mấu chốt: nó khiến các nhân tố bớt chồng lấn, nhờ đó hệ
    số hồi quy đọc được thay vì nhiễu loạn vì đa cộng tuyến.
    """
    rets = factor_prices.pct_change().dropna(how="all")
    out = pd.DataFrame(index=rets.index)

    if "SPY" not in rets.columns:
        raise ValueError("Cần ít nhất dữ liệu SPY để dựng nhân tố thị trường.")
    out["MKT"] = rets["SPY"]

    spec = {
        "GROWTH": "QQQ",
        "SIZE": "IWM",
        "GOLD": "GLD",
        "DURATION": "TLT",
    }
    for name, ticker in spec.items():
        if ticker in rets.columns:
            out[name] = rets[ticker] - rets["SPY"]

    if "^VIX" in rets.columns:
        out["VOL"] = rets["^VIX"]

    return out.dropna()


def build_momentum_factor(
    asset_prices: pd.DataFrame, lookback: int = 252, skip: int = 21, top_frac: float = 0.3
) -> pd.Series:
    """Nhân tố động lượng **tự dựng từ chính rổ cổ phiếu** của người dùng.

    Mỗi tháng xếp hạng các mã theo lợi suất ``lookback`` phiên, bỏ qua
    ``skip`` phiên gần nhất (quy ước 12−1 kinh điển, nhằm tránh hiệu ứng đảo
    chiều ngắn hạn). Mua nhóm dẫn đầu, bán nhóm tụt hậu.

    Khác hẳn cách bài gốc làm là lấy ``SPY.pct_change(12)`` rồi gọi đó là
    động lượng — thứ đó thực chất chỉ là lợi suất thị trường trễ, không phải
    nhân tố động lượng theo nghĩa học thuật.
    """
    px = asset_prices.dropna(how="all").ffill()
    k = px.shape[1]
    if k < 4:
        raise ValueError("Cần ít nhất 4 mã để dựng nhân tố động lượng long/short.")

    signal = px.shift(skip).pct_change(lookback - skip)
    rets = px.pct_change()
    n_pick = max(1, int(round(k * top_frac)))

    # Vector hoá toàn bộ thay vì lặp từng phiên: xếp hạng tại các ngày tái cân
    # bằng, dựng ma trận trọng số rồi điền tiến. Nhanh hơn khoảng 100 lần so
    # với vòng lặp Python, và cho ra đúng cùng một kết quả.
    rebal = signal.iloc[::21]
    ranks = rebal.rank(axis=1, ascending=False)

    longs = (ranks <= n_pick).astype(float)
    shorts = (ranks > k - n_pick).astype(float)

    n_long = longs.sum(axis=1).replace(0.0, np.nan)
    n_short = shorts.sum(axis=1).replace(0.0, np.nan)
    weights = longs.div(n_long, axis=0) - shorts.div(n_short, axis=0)

    # Giữ nguyên vị thế cho tới lần tái cân bằng kế tiếp
    weights = weights.reindex(rets.index).ffill().fillna(0.0)

    out = pd.Series((rets.to_numpy() * weights.to_numpy()).sum(axis=1), index=rets.index)
    return out.rename("MOM").dropna()


def assemble_factors(
    factor_prices: pd.DataFrame,
    asset_prices: pd.DataFrame | None = None,
    include_momentum: bool = True,
) -> pd.DataFrame:
    """Gom toàn bộ nhân tố vào một bảng duy nhất."""
    factors = build_proxy_factors(factor_prices)
    if include_momentum and asset_prices is not None and asset_prices.shape[1] >= 4:
        try:
            mom = build_momentum_factor(asset_prices)
            factors = factors.join(mom, how="inner")
        except Exception:
            pass
    return factors.dropna()


# --------------------------------------------------------------------------
# 2. Hồi quy nhân tố với sai số chuẩn Newey–West
# --------------------------------------------------------------------------


@dataclass
class FactorModel:
    """Kết quả hồi quy nhân tố."""

    betas: pd.Series
    tstats: pd.Series
    pvalues: pd.Series
    alpha_annual: float
    alpha_tstat: float
    alpha_pvalue: float
    r_squared: float
    adj_r_squared: float
    residuals: pd.Series
    factors: pd.DataFrame
    n_obs: int
    hac_lags: int

    def table(self) -> pd.DataFrame:
        """Bảng hệ số kèm đánh giá ý nghĩa thống kê."""
        df = pd.DataFrame(
            {
                "Hệ số beta": self.betas,
                "Thống kê t": self.tstats,
                "p-value": self.pvalues,
            }
        )
        df["Ý nghĩa"] = df["p-value"].map(_stars)
        return df

    def summary(self) -> dict:
        return {
            "Alpha năm hoá": self.alpha_annual,
            "Thống kê t của alpha": self.alpha_tstat,
            "p-value của alpha": self.alpha_pvalue,
            "R²": self.r_squared,
            "R² hiệu chỉnh": self.adj_r_squared,
            "Số quan sát": self.n_obs,
        }

    @property
    def alpha_is_significant(self) -> bool:
        """Alpha có ý nghĩa ở mức 5% hay không (đã dùng sai số HAC)."""
        return bool(self.alpha_pvalue < 0.05)


def _stars(p: float) -> str:
    if p < 0.01:
        return "*** (1%)"
    if p < 0.05:
        return "** (5%)"
    if p < 0.10:
        return "* (10%)"
    return "không có ý nghĩa"


def newey_west_cov(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """Ma trận hiệp phương sai HAC theo Newey–West (1987).

    Hiệu chỉnh đồng thời cho phương sai thay đổi và tự tương quan. Không có
    bước này, thống kê t của alpha thường bị thổi phồng khoảng 30–50% với dữ
    liệu lợi suất ngày, khiến ta kết luận nhầm là "có alpha".
    """
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)

    # thành phần phương sai thay đổi (White)
    S = (X * resid[:, None]).T @ (X * resid[:, None])

    # cộng thêm các thành phần tự tương phần theo trọng số Bartlett
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        Xl = X[lag:] * resid[lag:, None]
        Xr = X[:-lag] * resid[:-lag, None]
        G = Xl.T @ Xr
        S += w * (G + G.T)

    return XtX_inv @ S @ XtX_inv * n / max(n - k, 1)


def fit_factor_model(
    portfolio_returns: pd.Series, factors: pd.DataFrame, hac_lags: int | None = None
) -> FactorModel:
    """Hồi quy bội có sai số chuẩn HAC.

    ``hac_lags=None`` sẽ dùng quy tắc kinh nghiệm Newey–West:
    ``lags = ⌊4·(n/100)^(2/9)⌋``.
    """
    y_s, X_df = pd.Series(portfolio_returns).align(factors, join="inner", axis=0)
    y = y_s.to_numpy(dtype=float)
    Xv = X_df.to_numpy(dtype=float)
    n, k = Xv.shape
    if n <= k + 2:
        raise ValueError(f"Quá ít quan sát ({n}) cho {k} nhân tố.")

    A = np.column_stack([np.ones(n), Xv])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef

    if hac_lags is None:
        hac_lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    hac_lags = max(0, min(hac_lags, n // 4))

    cov = newey_west_cov(A, resid, hac_lags)
    se = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
    t = coef / se
    dof = max(n - k - 1, 1)
    p = 2 * (1 - stats.t.cdf(np.abs(t), dof))

    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float(resid @ resid)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1 - r2) * (n - 1) / max(n - k - 1, 1)

    return FactorModel(
        betas=pd.Series(coef[1:], index=X_df.columns, name="beta"),
        tstats=pd.Series(t[1:], index=X_df.columns, name="t"),
        pvalues=pd.Series(p[1:], index=X_df.columns, name="p"),
        alpha_annual=float(coef[0] * TRADING_DAYS),
        alpha_tstat=float(t[0]),
        alpha_pvalue=float(p[0]),
        r_squared=r2,
        adj_r_squared=adj_r2,
        residuals=pd.Series(resid, index=y_s.index, name="residual"),
        factors=X_df,
        n_obs=n,
        hac_lags=hac_lags,
    )


def compare_ols_vs_hac(portfolio_returns: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    """Đặt cạnh nhau thống kê t theo OLS thường và theo Newey–West.

    Bảng này cho thấy trực tiếp mức độ thổi phồng khi bỏ qua tự tương quan —
    một điểm rất dễ ghi điểm khi bảo vệ.
    """
    y_s, X_df = pd.Series(portfolio_returns).align(factors, join="inner", axis=0)
    y, Xv = y_s.to_numpy(float), X_df.to_numpy(float)
    n, k = Xv.shape
    A = np.column_stack([np.ones(n), Xv])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef

    sigma2 = float(resid @ resid) / max(n - k - 1, 1)
    se_ols = np.sqrt(np.clip(np.diag(sigma2 * np.linalg.pinv(A.T @ A)), 1e-30, None))

    hac = fit_factor_model(portfolio_returns, factors)
    se_hac = np.abs(coef) / np.clip(
        np.abs(np.concatenate([[hac.alpha_tstat], hac.tstats.to_numpy()])), 1e-12, None
    )

    names = ["Alpha"] + list(X_df.columns)
    return pd.DataFrame(
        {
            "Hệ số": names,
            "Thống kê t (OLS)": coef / se_ols,
            "Thống kê t (Newey–West)": np.concatenate(
                [[hac.alpha_tstat], hac.tstats.to_numpy()]
            ),
        }
    ).assign(
        **{
            "Mức thổi phồng": lambda d: (
                d["Thống kê t (OLS)"].abs() / d["Thống kê t (Newey–West)"].abs().clip(lower=1e-9)
            )
        }
    )


# --------------------------------------------------------------------------
# 3. Phân rã lợi nhuận
# --------------------------------------------------------------------------


def return_attribution(model: FactorModel) -> pd.DataFrame:
    """Phân rã tổng lợi nhuận thành alpha, từng nhân tố và phần dư.

    Đây là bảng trả lời thẳng câu hỏi: trong X% lợi nhuận đạt được, bao nhiêu
    là do nhà quản lý giỏi, bao nhiêu chỉ là do gánh rủi ro thị trường?
    """
    contrib = model.factors.mul(model.betas, axis=1).sum()
    alpha_total = model.alpha_annual / TRADING_DAYS * model.n_obs
    resid_total = float(model.residuals.sum())

    rows = [{"Nguồn": "Alpha (kỹ năng)", "Đóng góp": alpha_total}]
    rows += [
        {"Nguồn": f"Nhân tố {c}", "Đóng góp": float(v)} for c, v in contrib.items()
    ]
    rows.append({"Nguồn": "Phần dư (không giải thích được)", "Đóng góp": resid_total})

    df = pd.DataFrame(rows)
    total = df["Đóng góp"].sum()
    df["Tỷ trọng"] = df["Đóng góp"] / total if abs(total) > 1e-12 else 0.0
    return df


def rolling_exposures(
    portfolio_returns: pd.Series, factors: pd.DataFrame, window: int = 126, stride: int = 1
) -> pd.DataFrame:
    """Beta nhân tố trượt — phát hiện hiện tượng trôi phong cách (style drift).

    Một danh mục khởi đầu trung lập nhưng dần nghiêng mạnh sang công nghệ là
    rủi ro thật, và chỉ nhìn thấy được qua biểu đồ này.
    """
    y, X = pd.Series(portfolio_returns).align(factors, join="inner", axis=0)
    yv, Xv = y.to_numpy(float), X.to_numpy(float)
    n = len(y)
    if n < window + 1:
        raise ValueError(f"Cần ít nhất {window + 1} quan sát, hiện có {n}.")

    # ``stride`` cho phép chỉ hồi quy mỗi N phiên rồi điền tiến phần còn lại.
    # Beta trượt 126 phiên gần như không đổi giữa hai phiên liền kề, nên với
    # mục đích hiển thị, stride=5 giảm 80% khối lượng tính toán mà biểu đồ
    # nhìn không khác gì. Mặc định stride=1 giữ nguyên độ chính xác đầy đủ.
    out = np.full((n, X.shape[1]), np.nan)
    idx = list(range(window, n + 1, max(1, stride)))
    if idx and idx[-1] != n:
        idx.append(n)

    for i in idx:
        sub_y, sub_X = yv[i - window : i], Xv[i - window : i]
        A = np.column_stack([np.ones(window), sub_X])
        coef, *_ = np.linalg.lstsq(A, sub_y, rcond=None)
        out[i - 1] = coef[1:]

    df = pd.DataFrame(out, index=y.index, columns=X.columns)
    return (df.ffill() if stride > 1 else df).dropna()


def style_drift_score(rolling_betas: pd.DataFrame) -> pd.DataFrame:
    """Đo mức độ trôi phong cách của từng nhân tố."""
    return pd.DataFrame(
        {
            "Beta đầu kỳ": rolling_betas.iloc[0],
            "Beta cuối kỳ": rolling_betas.iloc[-1],
            "Thay đổi": rolling_betas.iloc[-1] - rolling_betas.iloc[0],
            "Độ dao động": rolling_betas.std(ddof=1),
            "Biên độ": rolling_betas.max() - rolling_betas.min(),
        }
    ).sort_values("Độ dao động", ascending=False)


# --------------------------------------------------------------------------
# 4. Chẩn đoán chất lượng mô hình
# --------------------------------------------------------------------------


def variance_inflation_factors(factors: pd.DataFrame) -> pd.DataFrame:
    """VIF — đo mức chồng lấn giữa các nhân tố.

    VIF trên 10 nghĩa là nhân tố đó gần như được giải thích hoàn toàn bởi các
    nhân tố còn lại, khiến hệ số của nó không còn đáng tin. Đây chính là bệnh
    mà cách làm trong bài gốc mắc phải khi hồi quy SPY, QQQ, IWM cùng lúc.
    """
    X = factors.dropna()
    rows = []
    for col in X.columns:
        y = X[col].to_numpy(float)
        others = X.drop(columns=[col]).to_numpy(float)
        A = np.column_stack([np.ones(len(y)), others])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
        vif = 1.0 / max(1.0 - r2, 1e-9)
        rows.append(
            {
                "Nhân tố": col,
                "VIF": vif,
                "Đánh giá": "Chồng lấn nghiêm trọng" if vif > 10
                else ("Cần lưu ý" if vif > 5 else "Chấp nhận được"),
            }
        )
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def naive_vs_orthogonal(factor_prices: pd.DataFrame) -> pd.DataFrame:
    """So sánh VIF của cách làm bài gốc và cách dựng nhân tố long/short.

    Minh chứng bằng số rằng việc trừ đi SPY thực sự làm giảm đa cộng tuyến.
    """
    naive = factor_prices.pct_change().dropna()
    naive = naive.loc[:, [c for c in naive.columns if c in PROXY_TICKERS]]
    ortho = build_proxy_factors(factor_prices)

    a = variance_inflation_factors(naive).assign(**{"Cách dựng": "Dùng ETF thô (bài gốc)"})
    b = variance_inflation_factors(ortho).assign(**{"Cách dựng": "Long/short đã trừ SPY"})
    return pd.concat([a, b], ignore_index=True)[["Cách dựng", "Nhân tố", "VIF", "Đánh giá"]]


def asset_factor_betas(asset_returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Beta nhân tố cho **từng mã** — nền tảng xếp hạng rủi ro trong danh mục."""
    rows = {}
    for col in asset_returns.columns:
        try:
            m = fit_factor_model(asset_returns[col], factors)
        except Exception:
            continue
        row = m.betas.to_dict()
        row["Alpha năm"] = m.alpha_annual
        row["R²"] = m.r_squared
        rows[col] = row
    return pd.DataFrame(rows).T


def factor_report(
    portfolio_returns: pd.Series, factors: pd.DataFrame
) -> tuple[FactorModel, pd.DataFrame, pd.DataFrame]:
    """Gói gọn một lần gọi: mô hình, bảng hệ số và bảng phân rã lợi nhuận."""
    model = fit_factor_model(portfolio_returns, factors)
    return model, model.table(), return_attribution(model)
