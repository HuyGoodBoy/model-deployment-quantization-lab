FROM python:3.11-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89
COPY --from=ghcr.io/astral-sh/uv:0.12.2 /uv /uvx /bin/
ENV OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=-1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libstdc++6 && rm -rf /var/lib/apt/lists/*
COPY requirements-lock.txt /tmp/requirements-lock.txt
# Route only PyTorch packages to the official CPU index; retain the measured pins.
RUN uv pip sync --system --torch-backend cpu /tmp/requirements-lock.txt && uv pip check --system
WORKDIR /workspace
