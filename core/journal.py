"""Sổ nhật ký giao dịch mô phỏng (paper trading) trên SQLite.

Khoảng cách lớn nhất giữa một bài tập và một công cụ thật là: backtest giả
định bạn luôn thực hiện đúng tín hiệu. Thực tế con người do dự, vào lệnh trễ,
hoặc bỏ qua tín hiệu vì sợ.

Module này ghi lại **quyết định thực tế** của người dùng, tính lãi lỗ theo
thời gian và so sánh với kết quả mà mô hình lẽ ra đã tạo ra. Phần chênh lệch
giữa hai đường chính là *khoảng cách thực thi* — con số mà không backtest nào
đo được.

Dữ liệu lưu trong SQLite nên tồn tại qua các lần khởi động lại ứng dụng.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "journal.db"


@dataclass
class Trade:
    """Một giao dịch mô phỏng."""

    trade_date: str
    ticker: str
    side: str          # MUA hoặc BAN
    quantity: float
    price: float
    note: str = ""
    signal: str = ""   # tín hiệu mô hình tại thời điểm đó
    followed: bool = True   # người dùng có làm theo mô hình không

    @property
    def value(self) -> float:
        return self.quantity * self.price

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.side.upper() == "MUA" else -self.quantity


# --------------------------------------------------------------------------
# Khởi tạo cơ sở dữ liệu
# --------------------------------------------------------------------------


def _connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS trades(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            note TEXT,
            signal TEXT,
            followed INTEGER DEFAULT 1,
            created_at TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS snapshots(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snap_date TEXT NOT NULL UNIQUE,
            nav REAL NOT NULL,
            model_nav REAL,
            note TEXT
        )"""
    )
    conn.commit()
    return conn


def record_trade(trade: Trade, db_path=DB_PATH) -> int:
    """Ghi một giao dịch, trả về id bản ghi."""
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            """INSERT INTO trades(trade_date, ticker, side, quantity, price, note,
               signal, followed, created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                trade.trade_date, trade.ticker.upper(), trade.side.upper(),
                float(trade.quantity), float(trade.price), trade.note,
                trade.signal, int(trade.followed),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def delete_trade(trade_id: int, db_path=DB_PATH) -> bool:
    conn = _connect(db_path)
    try:
        cur = conn.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def all_trades(db_path=DB_PATH) -> pd.DataFrame:
    """Đọc toàn bộ nhật ký giao dịch."""
    conn = _connect(db_path)
    try:
        df = pd.read_sql_query(
            """SELECT id, trade_date, ticker, side, quantity, price, note, signal,
               followed FROM trades ORDER BY trade_date, id""",
            conn,
        )
    finally:
        conn.close()
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["Giá trị"] = df["quantity"] * df["price"]
    return df.rename(
        columns={
            "id": "ID", "trade_date": "Ngày", "ticker": "Mã", "side": "Lệnh",
            "quantity": "Khối lượng", "price": "Giá", "note": "Ghi chú",
            "signal": "Tín hiệu mô hình", "followed": "Làm theo mô hình",
        }
    )


# --------------------------------------------------------------------------
# Tính vị thế và lãi lỗ
# --------------------------------------------------------------------------


def current_positions(db_path=DB_PATH) -> pd.DataFrame:
    """Vị thế đang nắm giữ, kèm giá vốn bình quân.

    Giá vốn tính theo phương pháp bình quân gia quyền: khi bán, giá vốn giữ
    nguyên và phần chênh lệch được ghi nhận thành lãi lỗ đã thực hiện.
    """
    df = all_trades(db_path)
    if df.empty:
        return pd.DataFrame(columns=["Mã", "Khối lượng", "Giá vốn BQ", "Giá trị vốn"])

    rows = []
    for ticker, grp in df.groupby("Mã"):
        qty, cost = 0.0, 0.0
        realized = 0.0
        for _, t in grp.iterrows():
            q, p = float(t["Khối lượng"]), float(t["Giá"])
            if t["Lệnh"] == "MUA":
                cost += q * p
                qty += q
            else:
                avg = cost / qty if qty > 1e-9 else p
                sell_q = min(q, qty)
                realized += sell_q * (p - avg)
                cost -= sell_q * avg
                qty -= sell_q
        if abs(qty) > 1e-9 or abs(realized) > 1e-9:
            rows.append(
                {
                    "Mã": ticker,
                    "Khối lượng": qty,
                    "Giá vốn BQ": cost / qty if qty > 1e-9 else 0.0,
                    "Giá trị vốn": cost,
                    "Lãi lỗ đã thực hiện": realized,
                }
            )
    return pd.DataFrame(rows)


def mark_to_market(prices: pd.DataFrame, db_path=DB_PATH) -> pd.DataFrame:
    """Định giá vị thế theo giá thị trường mới nhất."""
    pos = current_positions(db_path)
    if pos.empty:
        return pos

    last = prices.iloc[-1]
    pos["Giá thị trường"] = pos["Mã"].map(lambda t: float(last.get(t, np.nan)))
    pos["Giá trị thị trường"] = pos["Khối lượng"] * pos["Giá thị trường"]
    pos["Lãi lỗ chưa thực hiện"] = pos["Giá trị thị trường"] - pos["Giá trị vốn"]
    pos["Tỷ suất"] = np.where(
        pos["Giá trị vốn"].abs() > 1e-9,
        pos["Lãi lỗ chưa thực hiện"] / pos["Giá trị vốn"].abs(),
        0.0,
    )
    return pos.sort_values("Giá trị thị trường", ascending=False).reset_index(drop=True)


def portfolio_summary(prices: pd.DataFrame, db_path=DB_PATH) -> dict:
    """Tổng hợp tình trạng danh mục mô phỏng."""
    pos = mark_to_market(prices, db_path)
    if pos.empty:
        return {"Số vị thế": 0, "Giá trị thị trường": 0.0, "Tổng lãi lỗ": 0.0}

    mv = float(pos["Giá trị thị trường"].sum())
    cost = float(pos["Giá trị vốn"].sum())
    unreal = float(pos["Lãi lỗ chưa thực hiện"].sum())
    real = float(pos.get("Lãi lỗ đã thực hiện", pd.Series([0.0])).sum())
    return {
        "Số vị thế": int((pos["Khối lượng"].abs() > 1e-9).sum()),
        "Giá trị vốn": cost,
        "Giá trị thị trường": mv,
        "Lãi lỗ chưa thực hiện": unreal,
        "Lãi lỗ đã thực hiện": real,
        "Tổng lãi lỗ": unreal + real,
        "Tỷ suất tổng": (unreal + real) / cost if abs(cost) > 1e-9 else 0.0,
    }


# --------------------------------------------------------------------------
# Khoảng cách thực thi
# --------------------------------------------------------------------------


def discipline_report(db_path=DB_PATH) -> pd.DataFrame:
    """Người dùng tuân thủ tín hiệu mô hình tới mức nào?

    Đây là loại phân tích mà chỉ một nhật ký giao dịch thật mới cho ra được,
    và thường hé lộ điều bất ngờ: nhiều người bỏ qua tín hiệu đúng lúc nó có
    giá trị nhất, tức là khi thị trường đang hoảng loạn.
    """
    df = all_trades(db_path)
    if df.empty:
        return pd.DataFrame(columns=["Chỉ tiêu", "Giá trị"])

    followed = df["Làm theo mô hình"].astype(bool)
    with_signal = df[df["Tín hiệu mô hình"].astype(str).str.len() > 0]

    rows = [
        {"Chỉ tiêu": "Tổng số lệnh", "Giá trị": float(len(df))},
        {"Chỉ tiêu": "Số lệnh theo mô hình", "Giá trị": float(followed.sum())},
        {"Chỉ tiêu": "Tỷ lệ tuân thủ", "Giá trị": float(followed.mean())},
        {"Chỉ tiêu": "Số lệnh tự quyết", "Giá trị": float((~followed).sum())},
        {
            "Chỉ tiêu": "Giá trị lệnh tự quyết",
            "Giá trị": float(df.loc[~followed, "Giá trị"].sum()),
        },
    ]
    if not with_signal.empty:
        rows.append(
            {
                "Chỉ tiêu": "Số lệnh có ghi tín hiệu",
                "Giá trị": float(len(with_signal)),
            }
        )
    return pd.DataFrame(rows)


def save_snapshot(
    nav: float, model_nav: float | None = None, snap_date: str | None = None,
    note: str = "", db_path=DB_PATH,
) -> bool:
    """Lưu ảnh chụp NAV theo ngày, phục vụ vẽ đường vốn thực tế."""
    conn = _connect(db_path)
    try:
        conn.execute(
            """INSERT INTO snapshots(snap_date, nav, model_nav, note)
               VALUES (?,?,?,?)
               ON CONFLICT(snap_date) DO UPDATE SET nav=excluded.nav,
               model_nav=excluded.model_nav, note=excluded.note""",
            (snap_date or date.today().isoformat(), float(nav),
             float(model_nav) if model_nav is not None else None, note),
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def nav_history(db_path=DB_PATH) -> pd.DataFrame:
    """Đọc lại lịch sử NAV thực tế so với NAV mô hình."""
    conn = _connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT snap_date, nav, model_nav, note FROM snapshots ORDER BY snap_date", conn
        )
    finally:
        conn.close()
    if df.empty:
        return df
    df["snap_date"] = pd.to_datetime(df["snap_date"])
    return df.rename(
        columns={"snap_date": "Ngày", "nav": "NAV thực tế",
                 "model_nav": "NAV mô hình", "note": "Ghi chú"}
    ).set_index("Ngày")


def execution_gap(db_path=DB_PATH) -> dict:
    """Đo khoảng cách giữa kết quả thực tế và kết quả mô hình.

    Giá trị âm nghĩa là việc thực thi đang bào mòn lợi nhuận so với mô hình —
    thường do vào lệnh trễ, bỏ lỡ tín hiệu, hoặc giao dịch quá nhiều.
    """
    hist = nav_history(db_path)
    if hist.empty or hist["NAV mô hình"].isna().all() or len(hist) < 2:
        return {"Có đủ dữ liệu": False}

    h = hist.dropna(subset=["NAV mô hình"])
    if len(h) < 2:
        return {"Có đủ dữ liệu": False}

    actual = float(h["NAV thực tế"].iloc[-1] / h["NAV thực tế"].iloc[0] - 1)
    model = float(h["NAV mô hình"].iloc[-1] / h["NAV mô hình"].iloc[0] - 1)
    return {
        "Có đủ dữ liệu": True,
        "Lợi nhuận thực tế": actual,
        "Lợi nhuận mô hình": model,
        "Khoảng cách thực thi": actual - model,
        "Số kỳ theo dõi": int(len(h)),
    }


def clear_journal(db_path=DB_PATH) -> dict:
    """Xoá toàn bộ nhật ký. Trả về số bản ghi đã xoá từng bảng."""
    conn = _connect(db_path)
    try:
        n_t = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        n_s = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM snapshots")
        conn.commit()
        return {"Giao dịch": int(n_t), "Ảnh chụp NAV": int(n_s)}
    finally:
        conn.close()
