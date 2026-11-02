"""Động cơ cảnh báo rủi ro — biến dashboard thụ động thành công cụ giám sát.

Một dashboard chỉ hữu ích khi có người mở ra xem. Rủi ro thì không chờ ai
cả. Module này định nghĩa các luật cảnh báo, kiểm tra chúng trên dữ liệu mới
nhất, ghi nhật ký và gửi thông báo qua webhook (Teams, Slack, Telegram...).

Thiết kế theo hướng **khai báo**: mỗi luật là một đối tượng ``AlertRule``
độc lập, nên thêm luật mới chỉ là thêm một dòng, không phải sửa logic.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from core.upgrade_api import TRADING_DAYS, max_drawdown

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "alerts.db"


class Severity(str, Enum):
    INFO = "Thông tin"
    WARNING = "Cảnh báo"
    CRITICAL = "Nghiêm trọng"


@dataclass
class Alert:
    """Một cảnh báo đã kích hoạt."""

    code: str
    title: str
    message: str
    severity: Severity
    value: float
    threshold: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return {
            "Thời điểm": self.timestamp,
            "Mã": self.code,
            "Mức độ": self.severity.value,
            "Tiêu đề": self.title,
            "Nội dung": self.message,
            "Giá trị": self.value,
            "Ngưỡng": self.threshold,
        }


@dataclass
class AlertRule:
    """Một luật cảnh báo khai báo được."""

    code: str
    title: str
    check: Callable[[dict], tuple[bool, float]]
    threshold: float
    severity: Severity
    message_template: str

    def evaluate(self, context: dict) -> Alert | None:
        try:
            triggered, value = self.check(context)
        except Exception:
            return None
        if not triggered:
            return None
        return Alert(
            code=self.code,
            title=self.title,
            message=self.message_template.format(value=value, threshold=self.threshold),
            severity=self.severity,
            value=float(value),
            threshold=float(self.threshold),
        )


# --------------------------------------------------------------------------
# Bộ luật mặc định
# --------------------------------------------------------------------------


def build_default_rules(
    dd_limit: float = -0.15,
    vol_limit: float = 0.30,
    beta_limit: float = 1.30,
    concentration_limit: float = 0.40,
    var_limit: float = -0.03,
) -> list[AlertRule]:
    """Bộ luật giám sát tiêu chuẩn cho một danh mục dài hạn.

    Các ngưỡng mặc định lấy theo thông lệ quản trị rủi ro phổ biến, nhưng đều
    chỉnh được qua tham số để phù hợp khẩu vị rủi ro từng người.
    """
    return [
        AlertRule(
            code="DRAWDOWN",
            title="Sụt giảm vượt ngưỡng",
            check=lambda c: (c["drawdown"] < dd_limit, c["drawdown"]),
            threshold=dd_limit,
            severity=Severity.CRITICAL,
            message_template=(
                "Danh mục đang sụt {value:.1%} so với đỉnh, vượt ngưỡng chịu đựng "
                "{threshold:.1%}. Cân nhắc rà soát lại mức độ giải ngân."
            ),
        ),
        AlertRule(
            code="VOLATILITY",
            title="Biến động tăng cao",
            check=lambda c: (c["vol_20d"] > vol_limit, c["vol_20d"]),
            threshold=vol_limit,
            severity=Severity.WARNING,
            message_template=(
                "Biến động 20 phiên đạt {value:.1%}/năm, vượt ngưỡng {threshold:.1%}. "
                "Thị trường đang bất ổn hơn bình thường."
            ),
        ),
        AlertRule(
            code="BETA",
            title="Độ nhạy thị trường quá cao",
            check=lambda c: (c["beta"] > beta_limit, c["beta"]),
            threshold=beta_limit,
            severity=Severity.WARNING,
            message_template=(
                "Beta hiện tại {value:.2f}, vượt ngưỡng {threshold:.2f}. Danh mục "
                "đang nhạy hơn thị trường, rủi ro khuếch đại khi thị trường giảm."
            ),
        ),
        AlertRule(
            code="CONCENTRATION",
            title="Danh mục quá tập trung",
            check=lambda c: (c["max_weight"] > concentration_limit, c["max_weight"]),
            threshold=concentration_limit,
            severity=Severity.WARNING,
            message_template=(
                "Một mã chiếm {value:.1%} danh mục, vượt ngưỡng {threshold:.1%}. "
                "Rủi ro riêng lẻ đang lấn át lợi ích đa dạng hoá."
            ),
        ),
        AlertRule(
            code="VAR",
            title="Rủi ro đuôi gia tăng",
            check=lambda c: (c["var_95"] < var_limit, c["var_95"]),
            threshold=var_limit,
            severity=Severity.WARNING,
            message_template=(
                "VaR 95% một phiên ở mức {value:.2%}, xấu hơn ngưỡng {threshold:.2%}."
            ),
        ),
        AlertRule(
            code="REGIME",
            title="Thị trường chuyển sang chế độ xấu",
            check=lambda c: (
                c.get("regime") == "BEAR" and c.get("regime_prev") != "BEAR",
                float(c.get("regime_confidence", 0.0)),
            ),
            threshold=0.0,
            severity=Severity.CRITICAL,
            message_template=(
                "Mô hình vừa chuyển nhận định sang chế độ BEAR với độ chắc chắn "
                "{value:.0%}. Đây là tín hiệu giảm mức độ giải ngân."
            ),
        ),
        AlertRule(
            code="CORRELATION",
            title="Đa dạng hoá đang mất tác dụng",
            check=lambda c: (c["avg_corr"] > 0.80, c["avg_corr"]),
            threshold=0.80,
            severity=Severity.WARNING,
            message_template=(
                "Tương quan trung bình giữa các mã đạt {value:.2f}. Khi thị trường "
                "căng thẳng, mọi tài sản có xu hướng giảm cùng nhau."
            ),
        ),
    ]


# --------------------------------------------------------------------------
# Thu thập bối cảnh và chạy kiểm tra
# --------------------------------------------------------------------------


def build_context(
    portfolio_returns: pd.Series,
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    regime: str | None = None,
    regime_prev: str | None = None,
    regime_confidence: float = 0.0,
) -> dict:
    """Tính toàn bộ số liệu cần cho các luật cảnh báo."""
    r = pd.Series(portfolio_returns).dropna()
    recent = r.tail(20)

    beta = 0.0
    if benchmark_returns is not None:
        b = pd.Series(benchmark_returns).reindex(r.index).dropna()
        if len(b) > 60:
            rr, bb = r.align(b, join="inner")
            rr, bb = rr.tail(252), bb.tail(252)
            var_b = float(bb.var())
            beta = float(rr.cov(bb) / var_b) if var_b > 0 else 0.0

    corr = asset_returns.tail(252).corr().to_numpy()
    off = corr[~np.eye(corr.shape[0], dtype=bool)]

    return {
        "drawdown": float(max_drawdown(r.tail(252))),
        "vol_20d": float(recent.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(recent) > 5 else 0.0,
        "beta": beta,
        "max_weight": float(np.max(np.abs(weights.to_numpy()))) if len(weights) else 0.0,
        "var_95": float(np.percentile(r.tail(252), 5)) if len(r) > 60 else 0.0,
        "avg_corr": float(np.mean(off)) if off.size else 0.0,
        "regime": regime,
        "regime_prev": regime_prev,
        "regime_confidence": regime_confidence,
    }


def run_checks(context: dict, rules: list[AlertRule] | None = None) -> list[Alert]:
    """Chạy toàn bộ luật, trả về danh sách cảnh báo đã kích hoạt."""
    rules = rules or build_default_rules()
    fired = [a for a in (rule.evaluate(context) for rule in rules) if a is not None]
    order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
    return sorted(fired, key=lambda a: order[a.severity])


def alerts_to_frame(alerts: list[Alert]) -> pd.DataFrame:
    if not alerts:
        return pd.DataFrame(
            columns=["Thời điểm", "Mã", "Mức độ", "Tiêu đề", "Nội dung", "Giá trị", "Ngưỡng"]
        )
    return pd.DataFrame([a.to_dict() for a in alerts])


# --------------------------------------------------------------------------
# Nhật ký cảnh báo
# --------------------------------------------------------------------------


def _connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS alerts(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            code TEXT NOT NULL,
            severity TEXT NOT NULL,
            title TEXT,
            message TEXT,
            value REAL,
            threshold REAL,
            portfolio TEXT
        )"""
    )
    conn.commit()
    return conn


def log_alerts(alerts: list[Alert], portfolio: str = "default", db_path=DB_PATH) -> int:
    """Ghi cảnh báo vào nhật ký SQLite. Trả về số bản ghi đã lưu."""
    if not alerts:
        return 0
    conn = _connect(db_path)
    try:
        conn.executemany(
            """INSERT INTO alerts(timestamp, code, severity, title, message, value,
               threshold, portfolio) VALUES (?,?,?,?,?,?,?,?)""",
            [
                (a.timestamp, a.code, a.severity.value, a.title, a.message,
                 a.value, a.threshold, portfolio)
                for a in alerts
            ],
        )
        conn.commit()
        return len(alerts)
    finally:
        conn.close()


def alert_history(limit: int = 100, portfolio: str | None = None, db_path=DB_PATH) -> pd.DataFrame:
    """Đọc lại lịch sử cảnh báo."""
    conn = _connect(db_path)
    try:
        q = "SELECT timestamp, code, severity, title, message, value, threshold, portfolio FROM alerts"
        params: list = []
        if portfolio:
            q += " WHERE portfolio = ?"
            params.append(portfolio)
        q += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        df = pd.read_sql_query(q, conn, params=params)
    finally:
        conn.close()
    return df.rename(
        columns={
            "timestamp": "Thời điểm", "code": "Mã", "severity": "Mức độ",
            "title": "Tiêu đề", "message": "Nội dung", "value": "Giá trị",
            "threshold": "Ngưỡng", "portfolio": "Danh mục",
        }
    )


def clear_history(db_path=DB_PATH) -> int:
    conn = _connect(db_path)
    try:
        n = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        conn.execute("DELETE FROM alerts")
        conn.commit()
        return int(n)
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Gửi thông báo qua webhook
# --------------------------------------------------------------------------


def format_webhook_payload(alerts: list[Alert], portfolio: str = "Danh mục") -> dict:
    """Dựng gói tin dạng thẻ thích ứng, hiển thị tốt trên Teams và Slack."""
    crit = sum(1 for a in alerts if a.severity == Severity.CRITICAL)
    lines = [f"**{a.severity.value} · {a.title}**\n{a.message}" for a in alerts]
    return {
        "text": f"Cảnh báo rủi ro — {portfolio}",
        "summary": f"{len(alerts)} cảnh báo ({crit} nghiêm trọng)",
        "sections": [{"text": t} for t in lines],
        "alerts": [a.to_dict() for a in alerts],
    }


def send_webhook(alerts: list[Alert], url: str, portfolio: str = "Danh mục", timeout: int = 8) -> dict:
    """Gửi cảnh báo tới một webhook URL.

    Mọi lỗi mạng đều được bắt và trả về dưới dạng kết quả, **không ném ngoại
    lệ** — một dashboard không bao giờ được sập chỉ vì webhook chết.
    """
    if not alerts:
        return {"sent": False, "reason": "Không có cảnh báo nào để gửi."}
    if not url or not url.startswith(("http://", "https://")):
        return {"sent": False, "reason": "URL webhook không hợp lệ."}

    payload = format_webhook_payload(alerts, portfolio)
    try:
        import requests

        resp = requests.post(
            url, data=json.dumps(payload),
            headers={"Content-Type": "application/json"}, timeout=timeout,
        )
        return {
            "sent": 200 <= resp.status_code < 300,
            "status_code": resp.status_code,
            "count": len(alerts),
        }
    except Exception as exc:
        return {"sent": False, "reason": f"Lỗi khi gửi: {exc}"}


def dry_run_webhook(alerts: list[Alert], portfolio: str = "Danh mục") -> str:
    """Xem trước nội dung thông báo mà không thực sự gửi đi.

    Rất hữu ích khi demo: cho người xem thấy nội dung cảnh báo mà không cần
    cấu hình webhook thật.
    """
    payload = format_webhook_payload(alerts, portfolio)
    lines = [f"### {payload['text']}", f"_{payload['summary']}_", ""]
    lines += [s["text"] for s in payload["sections"]]
    return "\n\n".join(lines)
