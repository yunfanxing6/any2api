# ── Builder ───────────────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    curl \
    libcurl4-openssl-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    cargo \
    rustc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m venv /opt/venv

COPY pyproject.toml config.defaults.toml ./
COPY app ./app
COPY embedded_qwen ./embedded_qwen
COPY scripts ./scripts

RUN pip install --upgrade pip setuptools wheel \
    && pip install . \
    && find /opt/venv -type d \
         \( -name "__pycache__" -o -name "tests" -o -name "test" -o -name "testing" \) \
         -prune -exec rm -rf {} + \
    && find /opt/venv -type f -name "*.pyc" -delete \
    && rm -rf /root/.cache /tmp/*

# ── Runtime ───────────────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    VIRTUAL_ENV=/opt/venv \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8000 \
    SERVER_WORKERS=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    ca-certificates \
    curl \
    wget \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdbus-glib-1-2 \
    libdrm2 \
    libgbm1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpangocairo-1.0-0 \
    libpulse0 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    libxshmfence1 \
    fonts-liberation \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY pyproject.toml config.defaults.toml ./
COPY app ./app
COPY embedded_qwen ./embedded_qwen
COPY scripts ./scripts

RUN mkdir -p /app/data /app/logs \
    && chmod +x /app/scripts/entrypoint.sh /app/scripts/init_storage.sh \
    && python -m camoufox fetch

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["sh", "-c", "wget -qO /dev/null http://127.0.0.1:${SERVER_PORT}/health || exit 1"]

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["sh", "-c", "exec granian --interface asgi --host ${SERVER_HOST} --port ${SERVER_PORT} --workers ${SERVER_WORKERS} app.main:app"]
