"""Các điểm cuối phân tích danh mục."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api import services as svc
from api.schemas import (
    AllocQuery,
    FactorQuery,
    OverviewQuery,
    ScenarioQuery,
    ValidationQuery,
    WalkForwardQuery,
    BaseQuery,
    QuantamentalQuery,
)

router = APIRouter(prefix="/api", tags=["Phân tích"])


def _guard(fn, *args, **kwargs):
    """Biến lỗi nội bộ thành phản hồi HTTP có thông điệp tiếng Việt."""
    try:
        return fn(*args, **kwargs)
    except svc.DataError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xử lý yêu cầu: {type(exc).__name__}: {exc}",
        )


@router.post("/overview", summary="Tổng quan danh mục")
def overview(q: OverviewQuery):
    """Chỉ số chính, đường vốn, phân bổ vốn và phân rã rủi ro."""
    return _guard(
        svc.overview, q.tickers, q.start, q.method, q.market, q.robust, q.max_weight
    )


@router.post("/allocations", summary="So sánh thuật toán phân bổ")
def allocations(q: AllocQuery):
    """Chạy sáu thuật toán phân bổ trên cùng bộ tham số."""
    return _guard(svc.compare_allocations, q.tickers, q.start, q.market, q.robust, q.max_weight)


@router.post("/walkforward", summary="Backtest walk-forward")
def walkforward(q: WalkForwardQuery):
    """Backtest theo quy trình thật, có chi phí giao dịch và khoảng cấm."""
    return _guard(
        svc.walkforward, q.tickers, q.start, q.methods, q.market, q.cost_bps, q.lookback, q.step
    )


@router.post("/validation", summary="Kiểm định thống kê")
def validation(q: ValidationQuery):
    """PSR, Deflated Sharpe, bootstrap theo khối và phân vị ngẫu nhiên."""
    return _guard(svc.validate, q.tickers, q.start, q.method, q.market, q.n_trials)


@router.post("/regimes", summary="Chế độ thị trường")
def regimes(q: BaseQuery):
    """Mô hình Markov ẩn ba trạng thái kèm ma trận chuyển trạng thái."""
    return _guard(svc.regimes, q.tickers, q.start, q.market)


@router.post("/factors", summary="Phân tích nhân tố")
def factors(q: FactorQuery):
    """Hồi quy nhân tố với sai số chuẩn Newey–West và phân rã lợi nhuận."""
    return _guard(svc.factors, q.tickers, q.start, q.method, q.market)


@router.post("/scenarios", summary="Kịch bản và sức chịu đựng")
def scenarios(q: ScenarioQuery):
    """Cú sốc nhân tố, phát lại khủng hoảng lịch sử và mô phỏng Monte Carlo."""
    return _guard(
        svc.scenarios, q.tickers, q.start, q.shocks, q.market, q.n_paths, q.fat_tails
    )


@router.post("/quantamental", summary="Phân tích Quantamental AI & Dual-RAG")
def quantamental(q: QuantamentalQuery):
    """Phân tích đa trụ cột: Kỹ thuật (T) + Cảm xúc (S) + Nội tại (F)."""
    return _guard(svc.quantamental_analysis, q.tickers, q.start, q.market, q.use_sample)

