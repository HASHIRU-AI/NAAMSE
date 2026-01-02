# NAAMSE Green Agent Docker Image
# ================================
# This Dockerfile packages the NAAMSE fuzzer green agent for end-to-end deployment.

# Stage 1: Build stage
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy dependency files and source code
COPY pyproject.toml ./
COPY src/ ./src/

# Install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install .

# Stage 2: Runtime stage
FROM python:3.11-slim AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Default port for the agent
    PORT=8000

# Install runtime dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY src/ ./src/
COPY pyproject.toml ./

# Download spaCy model (required for presidio-analyzer)
RUN python -m spacy download en_core_web_lg

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the default port
EXPOSE 8000

# Health check to verify the agent is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/.well-known/agent.json')" || exit 1

# Entry point - run the NAAMSE green agent server
# Use 0.0.0.0 to accept connections from outside the container
ENTRYPOINT ["python", "-m", "src.agentbeats.server"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
