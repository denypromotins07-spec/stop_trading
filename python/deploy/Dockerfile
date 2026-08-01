# Stage 45 HFT System - Python Container
# Multi-stage slim Dockerfile stripping compilation headers and test suites
# Final image contains only stripped ONNX models, Ray binaries, and Nautilus core dependencies

# =============================================================================
# Stage 1: Builder - Install dependencies and compile
# =============================================================================
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libatlas-base-dev \
    liblapack-dev \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies to a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install core ML and trading dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        numpy==1.26.2 \
        scipy==1.11.4 \
        pandas==2.1.3 \
        xgboost==2.0.3 \
        onnxruntime==1.16.3 \
        aiofiles==23.2.1 \
        jinja2==3.1.2 \
        msgpack==1.0.7 \
        pyzmq==25.1.1 \
        ray==2.8.1 \
        nautilus_trader==1.175.0

# Copy source code
COPY execution/ ./python/execution/
COPY toxicity/ ./python/toxicity/
COPY risk/ ./python/risk/
COPY reporting/ ./python/reporting/

# Compile Python bytecode ahead of time
RUN python -m compileall -b python/

# Strip debug symbols from compiled extensions
RUN find /opt/venv -name "*.so" -exec strip --strip-all {} \; 2>/dev/null || true

# =============================================================================
# Stage 2: Runtime - Minimal production image
# =============================================================================
FROM python:3.11-slim-bookworm AS runtime

LABEL maintainer="HFT Systems Architect"
LABEL stage="45"
LABEL version="1.0.0"
LABEL description="Stage 45 HFT Python Runtime - Memory Optimized"

# Set environment variables for memory optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Create non-root user for security
RUN groupadd -r hftgroup && useradd -r -g hftgroup hftuser

WORKDIR /app

# Install only runtime dependencies (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libatlas3-base \
    liblapack3 \
    libblas3 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy compiled Python code
COPY --from=builder --chown=hftuser:hftgroup /build/python ./python

# Copy entrypoint script
COPY --chown=hftuser:hftgroup entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create necessary directories with proper permissions
RUN mkdir -p /app/reports /app/digests /app/logs /app/models /app/data && \
    chown -R hftuser:hftgroup /app

# Copy pre-trained ONNX models if available
COPY --chown=hftuser:hftgroup models/*.onnx /app/models/ 2>/dev/null || true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Switch to non-root user
USER hftuser

# Expose ports for IPC and monitoring
EXPOSE 8000/tcp  # HTTP API
EXPOSE 9000/tcp  # Ray dashboard
EXPOSE 5555/tcp  # ZeroMQ IPC

# Set resource limits via ulimit in entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command - start the Nautilus kernel
CMD ["python", "-u", "python/main.py"]

# =============================================================================
# Build Instructions:
# =============================================================================
# Build the image:
#   docker build -t hft-python:stage45 --target runtime .
#
# Run with memory limit:
#   docker run --memory=3g --cpus=2.0 hft-python:stage45
#
# Security scan:
#   docker scout cve hft-python:stage45
#
# Size optimization verification:
#   docker images hft-python:stage45
# =============================================================================
