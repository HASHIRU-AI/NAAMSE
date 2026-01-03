# NAAMSE Green Agent Docker Image
# ================================
# Multi-stage build using uv for fast dependency installation.
# to build and run:
# docker build -t naamse-green-agent:latest .
#  docker run --rm --add-host host.docker.internal:host-gateway -p 8000:8000 -e GOOGLE_API_KEY="key" naamse-green-agent:latest
# Test with green agent client:
# python test_green_agent.py --target http://host.docker.internal:5000 --green-agent http://localhost:8000 --iterations 1 --mutations 1

# Stage 1: Build the application
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Disable Python downloads - use the system interpreter across both images
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install dependencies first (cached if pyproject.toml/uv.lock unchanged)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Copy only what's needed for the application
COPY pyproject.toml uv.lock /app/
COPY src/ /app/src/

# Install the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


# Stage 2: Runtime image without uv
FROM python:3.11-slim-bookworm

# Install runtime dependencies (libgomp1 needed for torch/numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the application from the builder
COPY --from=builder /app /app

# Place executables in the environment at the front of the path
# Set PYTHONPATH so Python can find the src module
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app"

# Run the NAAMSE green agent server
CMD ["python", "-m", "src.agentbeats.server", "--host", "0.0.0.0", "--port", "8000"]
