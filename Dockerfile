FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY gateway ./gateway
COPY providers.*.yaml ./
RUN useradd --uid 10001 --create-home app && mkdir /data && chown app /data
USER app
ENV STATE_DB=/data/state.db PORT=8000
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["/app/.venv/bin/uvicorn", "gateway.app:build", "--factory", "--host", "0.0.0.0", "--port", "8000"]
