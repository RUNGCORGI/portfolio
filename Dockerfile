# =============================================================================
# P1 sở hữu file này — PHÁT TRIỂN THÊM: multi-stage build + non-root + health
# Bài gốc hướng dẫn cài Conda 5 bước (~10 phút). Ở đây: 1 lệnh.
# =============================================================================

# ---------- Stage 1: builder (có compiler, KHÔNG đi vào image cuối) ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ---------- Stage 2: runtime (gọn, không compiler) ----------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Portfolio Terminal - Nhom 4"
LABEL org.opencontainers.image.description="Refactor myapp13.py thanh kien truc module"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY --from=builder /install /usr/local

# Chạy bằng user thường, không phải root
RUN useradd --create-home --uid 1000 quant
WORKDIR /app

COPY --chown=quant:quant core/ ./core/
COPY --chown=quant:quant tabs/ ./tabs/
COPY --chown=quant:quant tests/ ./tests/
COPY --chown=quant:quant app.py ./
COPY --chown=quant:quant data/ ./data/

USER quant
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py"]
