"""Các điểm cuối giám sát: cảnh báo, nhật ký giao dịch, thông tin hệ thống."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from api import services as svc
from api.schemas import AlertQuery, BaseQuery, TradeInput
from core import journal, vn_market

router = APIRouter(prefix="/api", tags=["Giám sát"])


@router.post("/alerts", summary="Cảnh báo rủi ro")
def alerts(q: AlertQuery):
    """Chạy bộ luật cảnh báo trên trạng thái mới nhất của danh mục."""
    try:
        return svc.risk_alerts(q.tickers, q.start, q.method, q.market, q.thresholds)
    except svc.DataError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Lỗi khi kiểm tra cảnh báo: {exc}")


@router.get("/journal/trades", summary="Danh sách giao dịch")
def list_trades():
    df = journal.all_trades()
    return svc.frame_to_table(df)


@router.post("/journal/trades", summary="Ghi giao dịch")
def add_trade(t: TradeInput):
    tid = journal.record_trade(
        journal.Trade(
            t.trade_date, t.ticker, t.side, t.quantity, t.price, t.note, t.signal, t.followed
        )
    )
    return {"id": tid, "message": f"Đã ghi lệnh {t.side} {t.quantity:,.0f} {t.ticker}."}


@router.delete("/journal/trades/{trade_id}", summary="Xoá giao dịch")
def remove_trade(trade_id: int):
    ok = journal.delete_trade(trade_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Không tìm thấy giao dịch.")
    return {"message": "Đã xoá giao dịch."}


@router.post("/journal/positions", summary="Vị thế theo giá thị trường")
def positions(q: BaseQuery):
    try:
        ctx = svc.load_market(q.tickers, q.start, q.market)
    except svc.DataError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    prices = ctx["prices"]
    return {
        "positions": svc.frame_to_table(journal.mark_to_market(prices)),
        "summary": svc.dict_clean(journal.portfolio_summary(prices)),
        "discipline": svc.frame_to_table(journal.discipline_report()),
    }


@router.get("/market/vn/universe", summary="Rổ cổ phiếu Việt Nam")
def vn_universe():
    return {
        "universe": svc.frame_to_table(vn_market.universe_table()),
        "costs": svc.frame_to_table(vn_market.VNCostModel().describe()),
        "source": vn_market.VNStockProvider().source_in_use(),
    }


@router.get("/system/info", summary="Thông tin hệ thống")
def system_info():
    """Cho frontend biết môi trường đang chạy ở chế độ nào."""
    from core import upgrade_api as api_mod

    has_yf = False
    try:
        import yfinance  # noqa: F401

        has_yf = True
    except Exception:
        pass

    return {
        "data_source": "yfinance (dữ liệu thật)" if has_yf else "Mô phỏng (chế độ ngoại tuyến)",
        "offline": not has_yf,
        "vn_source": vn_market.VNStockProvider().source_in_use(),
        "methods": [
            "Chia đều",
            "Phương sai tối thiểu",
            "Max Sharpe",
            "Cân bằng rủi ro",
            "HRP",
            "Đa dạng hoá tối đa",
        ],
    }


@router.post("/system/cache/clear", summary="Xoá bộ nhớ đệm")
def clear_cache():
    return {"cleared": svc.clear_cache(), "message": "Đã xoá bộ nhớ đệm."}
