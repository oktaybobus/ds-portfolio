# Slim runtime image for the portfolio: CLI, tabular projects and Streamlit apps.
#
# Two extras are deliberately left out. TensorFlow (`dl`) would add ~2 GB for
# projects whose datasets are downloaded from Kaggle at training time anyway,
# and `xgboost` pulls nvidia-nccl-cu12 (~326 MB) of CUDA runtime that a CPU-only
# image never touches. CatBoost alone covers every model the shipped configs
# name; the registry degrades gracefully for the rest.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    DSJOURNEY_ROOT=/app

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv

# Dependencies first: this layer is cached until pyproject or the lock changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra catboost --extra app --extra api

COPY src/ ./src/
COPY projects/ ./projects/
COPY service/ ./service/
COPY assets.yaml Makefile ./
COPY scripts/ ./scripts/
COPY data/raw/laptop_price/ ./data/raw/laptop_price/

RUN uv sync --frozen --extra catboost --extra app --extra api

ENV PATH="/app/.venv/bin:${PATH}"

# A non-root user, so a mounted artifacts volume does not end up root-owned.
RUN useradd --create-home --uid 1000 portfolio \
    && mkdir -p /app/artifacts /app/data/raw \
    && chown -R portfolio:portfolio /app
USER portfolio

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD dsj list || exit 1

ENTRYPOINT ["dsj"]
CMD ["list"]
