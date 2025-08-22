FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Allow callers to choose which optional dependencies to install.
ARG MATRIX_EXTRAS="vllm_083"
RUN python -m pip install --upgrade pip setuptools wheel \
  && python -m pip install .[${MATRIX_EXTRAS}]

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose the Matrix CLI as the container entrypoint so users can execute
# commands directly, e.g. `docker run matrix --help`
ENTRYPOINT ["matrix"]
CMD ["--help"]
