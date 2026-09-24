FROM python:3.11.16-slim-trixie@sha256:be1575ed968de893bd54f4c56315ff7c4736ce522c1bca08fd521731aafc0d76

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# cmake / swig: python-libnuml, python-libcombine, python-libsedml は sdist からのビルドになる
RUN apt-get update && apt-get install -y \
    git curl make build-essential cmake swig zlib1g-dev libxml2-dev libbz2-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.6@sha256:88bc6eb1ccd4b82efd0e1b530caffabddf50dc2bf612e66c14ea25b8ee8a4d3d /uv /usr/local/bin/uv

WORKDIR /workspace
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-cache --group eval
ENV UV_NO_SYNC=1

COPY . .
RUN mkdir -p .research/results
CMD ["bash"]
