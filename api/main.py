"""Ứng dụng FastAPI — backend cho giao diện web độc lập.

Kiến trúc::

    web/ (HTML, CSS, JS thuần)  ←HTTP JSON→  api/  →  core/ (toàn bộ tính toán)

Tách lớp như vậy giải quyết đúng hạn chế mà bài báo gốc nêu ra: khi ứng dụng
vượt khỏi giai đoạn nguyên mẫu, việc trộn lẫn logic phân tích với mã giao
diện khiến hệ thống khó mở rộng và khó bảo trì.

Chạy::

    python run_api.py
    # hoặc: uvicorn api.main:app --reload --port 8000

Tài liệu API tự sinh tại ``/docs``.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles

from api.routers import analytics, monitor

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(
    title="Quant Lab API",
    description=(
        "Backend cho nền tảng phân tích danh mục đầu tư. Toàn bộ logic định "
        "lượng nằm trong gói core/, tầng này chỉ phục vụ HTTP."
    ),
    version="2.0.0",
    default_response_class=ORJSONResponse,
)

# Nén phản hồi: các chuỗi thời gian dài tiết kiệm được rất nhiều băng thông
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.include_router(analytics.router)
app.include_router(monitor.router)


@app.middleware("http")
async def add_timing(request: Request, call_next):
    """Gắn thời gian xử lý vào header để frontend hiển thị được và vô hiệu hóa cache trình duyệt."""
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"

    path = request.url.path
    if path == "/" or path.startswith("/js") or path.startswith("/css"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):  # pragma: no cover
    """Không bao giờ để lộ traceback ra ngoài."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Đã xảy ra lỗi không mong muốn khi xử lý yêu cầu.",
            "hint": f"{type(exc).__name__}",
        },
    )


@app.get("/health", tags=["Hệ thống"], summary="Kiểm tra tình trạng")
def health():
    return {"status": "ok", "service": "quant-lab-api"}


# --------------------------------------------------------------------------
# Phục vụ giao diện web tĩnh
# --------------------------------------------------------------------------

if WEB_DIR.exists():
    app.mount("/css", StaticFiles(directory=WEB_DIR / "css"), name="css")
    app.mount("/js", StaticFiles(directory=WEB_DIR / "js"), name="js")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")
