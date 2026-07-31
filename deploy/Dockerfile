# Multi-stage Dockerfile for HFT Crypto Bot
# Optimized for minimal footprint and cold-start latency

# =============================================================================
# Stage 1: Build Environment
# =============================================================================
FROM rust:1.75-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy manifests first for better caching
COPY Cargo.toml ./
COPY build.rs ./

# Create dummy src for dependency caching
RUN mkdir -p src && echo "fn main() {}" > src/main.rs

# Download and compile dependencies (cached layer)
RUN cargo fetch
RUN cargo build --release --bin hft_crypto_bot 2>&1 | head -1 || true

# Remove dummy binary
RUN rm -f target/release/hft_crypto_bot

# Copy actual source code
COPY . .

# Build optimized release binary
ARG RUSTFLAGS="-C target-cpu=native -C lto=fat -C codegen-units=1"
RUN cargo build --release --bin hft_crypto_bot

# Strip binary to reduce size
RUN strip target/release/hft_crypto_bot

# =============================================================================
# Stage 2: Runtime Image (Scratch-based for minimal footprint)
# =============================================================================
FROM scratch AS runtime

# Copy CA certificates for HTTPS connections
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Copy the stripped binary
COPY --from=builder /build/target/release/hft_crypto_bot /hft_crypto_bot

# Copy SOUL.md documentation
COPY --from=builder /build/SOUL.md /SOUL.md

# Copy deployment scripts
COPY --from=builder /build/deploy/systemd.service /etc/systemd/system/hft_crypto_bot.service

# Set non-root user for security (UID 1000)
# Note: In scratch image, we need to create user manually or use numeric UID
USER 1000:1000

# Set working directory
WORKDIR /app

# Expose metrics port (if applicable)
EXPOSE 9090

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/hft_crypto_bot", "--health-check"]

# Entry point
ENTRYPOINT ["/hft_crypto_bot"]

# Default arguments
CMD ["--config", "/app/config.toml"]

# =============================================================================
# Stage 3: Debug Image (Optional - for troubleshooting)
# =============================================================================
FROM debian:bookworm-slim AS debug

# Install debugging tools
RUN apt-get update && apt-get install -y \
    gdb \
    strace \
    ltrace \
    procps \
    net-tools \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy binary with symbols (from builder before stripping)
COPY --from=builder /build/target/release/hft_crypto_bot /hft_crypto_bot

# Copy SOUL.md
COPY SOUL.md /SOUL.md

WORKDIR /app

ENTRYPOINT ["/bin/bash"]
