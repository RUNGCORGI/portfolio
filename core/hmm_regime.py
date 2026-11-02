"""Nhận diện chế độ thị trường bằng Markov ẩn — nâng cấp cho ``core/regime.py``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class RegimeModel:
    """Kết quả HMM đã khớp."""

    probabilities: pd.DataFrame   # xác suất hậu nghiệm theo ngày
    states: pd.Series             # nhãn chế độ (đường Viterbi)
    transition: pd.DataFrame      # ma trận chuyển trạng thái
    means: pd.DataFrame           # trung bình đặc trưng từng chế độ
    log_likelihood: float
    converged: bool

    @property
    def current(self) -> str:
        return str(self.states.iloc[-1])

    @property
    def confidence(self) -> float:
        """Độ chắc chắn của chế độ hiện tại — thứ mô hình quy tắc không có."""
        return float(self.probabilities.iloc[-1].max())

    def expected_duration(self) -> pd.Series:
        """Số phiên kỳ vọng ở lại mỗi chế độ = 1 / (1 − xác suất tự lặp)."""
        diag = np.diag(self.transition.to_numpy())
        return pd.Series(
            1.0 / np.clip(1.0 - diag, 1e-6, None),
            index=self.transition.index,
            name="Số phiên kỳ vọng",
        )

    def stationary(self) -> pd.Series:
        """Phân phối dài hạn: % thời gian thị trường ở mỗi chế độ."""
        vals, vecs = np.linalg.eig(self.transition.to_numpy().T)
        v = np.abs(np.real(vecs[:, int(np.argmin(np.abs(vals - 1.0)))]))
        return pd.Series(v / v.sum(), index=self.transition.index, name="Tỷ trọng thời gian")


class GaussianHMM:
    """HMM Gauss nhiều biến (hiệp phương sai chéo), EM có chuẩn hoá tỷ lệ."""

    def __init__(self, n_states: int = 3, n_iter: int = 200, tol: float = 1e-6, seed: int = 11):
        self.n_states, self.n_iter, self.tol, self.seed = n_states, n_iter, tol, seed
        self.converged_, self.loglik_ = False, -np.inf

    def _emission(self, X: np.ndarray) -> np.ndarray:
        B = np.empty((X.shape[0], self.n_states))
        for k in range(self.n_states):
            diff = X - self.means_[k]
            var = np.clip(self.vars_[k], 1e-12, None)
            logp = -0.5 * (np.sum(diff**2 / var, axis=1) + np.sum(np.log(2 * np.pi * var)))
            B[:, k] = np.exp(np.clip(logp, -700, 700))
        return np.clip(B, 1e-300, None)

    def _init(self, X: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        # khởi tạo theo phân vị của đặc trưng thứ nhất → trạng thái đã có thứ tự
        chunks = np.array_split(np.argsort(X[:, 0]), self.n_states)
        self.means_ = np.vstack([X[c].mean(axis=0) for c in chunks])
        self.vars_ = np.vstack([X[c].var(axis=0) + 1e-8 for c in chunks])
        self.means_ += rng.normal(0, 1e-9, self.means_.shape)
        self.start_ = np.repeat(1.0 / self.n_states, self.n_states)
        self.trans_ = np.full((self.n_states, self.n_states), 0.1 / (self.n_states - 1))
        np.fill_diagonal(self.trans_, 0.9)

    def _forward_backward(self, B: np.ndarray):
        n, k = B.shape
        alpha = np.zeros((n, k))
        scale = np.ones(n)

        alpha[0] = self.start_ * B[0]
        scale[0] = max(alpha[0].sum(), 1e-300)
        alpha[0] /= scale[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ self.trans_) * B[t]
            scale[t] = max(alpha[t].sum(), 1e-300)
            alpha[t] /= scale[t]

        beta = np.zeros((n, k))
        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = (self.trans_ @ (B[t + 1] * beta[t + 1])) / scale[t + 1]

        gamma = alpha * beta
        gamma /= np.clip(gamma.sum(axis=1, keepdims=True), 1e-300, None)

        xi = np.zeros((k, k))
        for t in range(n - 1):
            num = np.outer(alpha[t], B[t + 1] * beta[t + 1]) * self.trans_
            xi += num / max(num.sum(), 1e-300)

        return gamma, xi, float(np.sum(np.log(scale)))

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        # ==========================================
        # SỬA LỖI: BẢO VỆ DỮ LIỆU ĐẦU VÀO
        # ==========================================
        if X is None or len(X) == 0:
            raise ValueError("Dữ liệu đầu vào rỗng. Không thể chạy HMM.")
            
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if np.isnan(X).any():
            raise ValueError("Dữ liệu đầu vào chứa giá trị NaN. Vui lòng làm sạch dữ liệu (dropna) trước khi huấn luyện.")
        # ==========================================

        self._init(X)
        prev = -np.inf
        gamma = None
        for _ in range(self.n_iter):
            gamma, xi, ll = self._forward_backward(self._emission(X))

            self.start_ = np.clip(gamma[0], 1e-12, None)
            self.start_ /= self.start_.sum()
            self.trans_ = xi / np.clip(xi.sum(axis=1, keepdims=True), 1e-300, None)

            w = np.clip(gamma.sum(axis=0), 1e-12, None)
            self.means_ = (gamma.T @ X) / w[:, None]
            for k in range(self.n_states):
                d = X - self.means_[k]
                self.vars_[k] = (gamma[:, k] @ (d**2)) / w[k] + 1e-10

            if abs(ll - prev) < self.tol:
                self.converged_, prev = True, ll
                break
            prev = ll

        self.loglik_, self.gamma_ = prev, gamma
        return self

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        """Đường trạng thái hợp lý nhất (quy hoạch động trên log-xác suất)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        B = np.log(self._emission(X))
        logA = np.log(np.clip(self.trans_, 1e-300, None))
        n, k = B.shape
        delta, psi = np.zeros((n, k)), np.zeros((n, k), dtype=int)
        delta[0] = np.log(np.clip(self.start_, 1e-300, None)) + B[0]
        for t in range(1, n):
            sc = delta[t - 1][:, None] + logA
            psi[t] = np.argmax(sc, axis=0)
            delta[t] = sc[psi[t], np.arange(k)] + B[t]
        path = np.zeros(n, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(n - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path


def build_features(market_returns: pd.Series, vol_window: int = 20) -> pd.DataFrame:
    """Đặc trưng đưa vào HMM: lợi suất và biến động trượt, đều đã chuẩn hoá."""
    r = pd.Series(market_returns).dropna()
    if len(r) < vol_window + 1:
        # Nếu dữ liệu quá ngắn, trả về DataFrame rỗng để báo lỗi
        return pd.DataFrame()
        
    vol = r.rolling(vol_window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    df = pd.DataFrame({"ret": r, "vol": vol}).dropna()
    
    if df.empty:
        return pd.DataFrame()
        
    for c in df.columns:
        df[c] = (df[c] - df[c].mean()) / (df[c].std(ddof=1) + 1e-12)
    return df


def fit_regimes(
    market_returns: pd.Series, n_states: int = 3, vol_window: int = 20, seed: int = 11
) -> RegimeModel:
    """Khớp HMM lên lợi suất thị trường, gán nhãn BEAR / NEUTRAL / BULL."""
    feats = build_features(market_returns, vol_window)
    if feats.empty:
         raise ValueError("Dữ liệu quá ngắn để tính toán các đặc trưng (features) cho HMM.")
         
    X = feats.to_numpy()
    model = GaussianHMM(n_states=n_states, seed=seed).fit(X)

    # gán nhãn theo lợi suất trung bình: thấp nhất là BEAR, cao nhất là BULL
    order = np.argsort(model.means_[:, 0])
    if n_states == 3:
        labels = {int(order[0]): "BEAR", int(order[1]): "NEUTRAL", int(order[2]): "BULL"}
    else:
        labels = {int(s): f"S{i}" for i, s in enumerate(order)}

    names = [labels[k] for k in range(n_states)]
    probs = pd.DataFrame(model.gamma_, index=feats.index, columns=names)
    states = pd.Series(
        [labels[int(s)] for s in model.viterbi(X)], index=feats.index, name="regime"
    )
    trans = pd.DataFrame(model.trans_, index=names, columns=names)
    means = pd.DataFrame(model.means_, index=names, columns=feats.columns)

    if n_states == 3:
        want = ["BEAR", "NEUTRAL", "BULL"]
        probs, trans, means = probs[want], trans.loc[want, want], means.loc[want]

    return RegimeModel(probs, states, trans, means, model.loglik_, model.converged_)


def rule_based_regime(market_prices: pd.Series, vol_window: int = 20) -> pd.Series:
    """Quy tắc cộng điểm của bài gốc — giữ lại để đặt cạnh HMM mà so sánh."""
    px = pd.Series(market_prices).dropna()
    r = px.pct_change()
    vol = r.rolling(vol_window).std(ddof=1)
    score = (
        (px > px.rolling(50).mean()).astype(int)
        + (px.rolling(50).mean() > px.rolling(200).mean()).astype(int)
        + (px.pct_change(60) > 0).astype(int)
        - (vol > vol.rolling(50).mean()).astype(int)
    ).dropna()
    return score.map(
        lambda s: "BULL" if s >= 2 else ("BEAR" if s <= 0 else "NEUTRAL")
    ).rename("regime_rule")


def regime_statistics(returns: pd.Series, states: pd.Series) -> pd.DataFrame:
    """Hiệu quả của danh mục trong từng chế độ — kiểm tra chế độ có ý nghĩa không."""
    r, s = pd.Series(returns).align(states, join="inner")
    rows = []
    for label, grp in r.groupby(s):
        rows.append(
            {
                "Chế độ": label,
                "Số phiên": int(len(grp)),
                "Tỷ trọng thời gian": len(grp) / len(r),
                "LN năm hoá": float(grp.mean() * TRADING_DAYS),
                "Biến động năm hoá": float(grp.std(ddof=1) * np.sqrt(TRADING_DAYS)),
                "Tỷ lệ phiên lãi": float((grp > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("LN năm hoá").reset_index(drop=True)


def regime_switching_backtest(
    asset_returns: pd.DataFrame,
    states: pd.Series,
    bull_weights: np.ndarray,
    bear_weights: np.ndarray | None = None,
    neutral_weights: np.ndarray | None = None,
    lag: int = 1,
) -> pd.Series:
    """Backtest chiến lược chuyển chế độ, **có độ trễ tín hiệu**.

    ``lag=1`` là chi tiết sống còn: chế độ của hôm nay chỉ được dùng để giao
    dịch từ hôm sau. Thiếu dòng này là backtest nhìn trộm tương lai.
    """
    k = asset_returns.shape[1]
    bear = bear_weights if bear_weights is not None else np.zeros(k)
    neutral = neutral_weights if neutral_weights is not None else np.repeat(1.0 / k, k)
    table = {"BULL": np.asarray(bull_weights, float), "BEAR": bear, "NEUTRAL": neutral}

    sig = states.shift(lag).dropna()
    r, s = asset_returns.align(sig, join="inner", axis=0)
    W = np.vstack([table.get(str(x), neutral) for x in s])
    return pd.Series((r.to_numpy() * W).sum(axis=1), index=r.index, name="regime_strategy")