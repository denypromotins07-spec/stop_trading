#!/bin/bash
# Binary Optimization Script for HFT Crypto Bot
# Applies release build, stripping, and UPX compression

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BINARY_NAME="hft_crypto_bot"
RELEASE_DIR="$PROJECT_ROOT/target/release/$BINARY_NAME"
OPTIMIZED_DIR="$PROJECT_ROOT/target/optimized"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v cargo &> /dev/null; then
        log_error "cargo not found. Please install Rust."
        exit 1
    fi
    
    if ! command -v strip &> /dev/null; then
        log_error "strip not found. Please install binutils."
        exit 1
    fi
    
    # UPX is optional
    if ! command -v upx &> /dev/null; then
        log_warn "upx not found. Skipping compression step."
        SKIP_UPX=true
    else
        SKIP_UPX=false
        log_info "UPX version: $(upx --version | head -1)"
    fi
}

# Build release binary with maximum optimization
build_release() {
    log_info "Building release binary with LTO and codegen-units=1..."
    
    cd "$PROJECT_ROOT"
    
    # Set RUSTFLAGS for maximum optimization
    export RUSTFLAGS="-C target-cpu=native -C lto=fat -C codegen-units=1"
    
    # Build with release profile
    cargo build --release --locked
    
    if [[ ! -f "$RELEASE_DIR" ]]; then
        log_error "Release binary not found at $RELEASE_DIR"
        exit 1
    fi
    
    local original_size=$(stat -c%s "$RELEASE_DIR" 2>/dev/null || stat -f%z "$RELEASE_DIR" 2>/dev/null)
    log_info "Original binary size: $(numfmt --to=iec-i --suffix=B "$original_size" 2>/dev/null || echo "$original_size bytes")"
}

# Strip debug symbols
strip_binary() {
    log_info "Stripping debug symbols..."
    
    mkdir -p "$OPTIMIZED_DIR"
    local stripped_path="$OPTIMIZED_DIR/${BINARY_NAME}_stripped"
    
    cp "$RELEASE_DIR" "$stripped_path"
    strip --strip-all "$stripped_path"
    
    local stripped_size=$(stat -c%s "$stripped_path" 2>/dev/null || stat -f%z "$stripped_path" 2>/dev/null)
    log_info "Stripped binary size: $(numfmt --to=iec-i --suffix=B "$stripped_size" 2>/dev/null || echo "$stripped_size bytes")"
}

# Apply UPX compression (optional)
compress_binary() {
    if [[ "$SKIP_UPX" == "true" ]]; then
        log_warn "Skipping UPX compression (upx not installed)"
        return
    fi
    
    log_info "Applying UPX compression..."
    
    local compressed_path="$OPTIMIZED_DIR/${BINARY_NAME}_compressed"
    
    # UPX best compression level
    upx --best --lzma -o "$compressed_path" "$OPTIMIZED_DIR/${BINARY_NAME}_stripped"
    
    local compressed_size=$(stat -c%s "$compressed_path" 2>/dev/null || stat -f%z "$compressed_path" 2>/dev/null)
    log_info "Compressed binary size: $(numfmt --to=iec-i --suffix=B "$compressed_size" 2>/dev/null || echo "$compressed_size bytes")"
    
    # Verify compressed binary
    if upx -t "$compressed_path" > /dev/null 2>&1; then
        log_info "UPX verification passed"
    else
        log_error "UPX verification failed!"
        exit 1
    fi
}

# Generate build metadata
generate_metadata() {
    log_info "Generating build metadata..."
    
    local metadata_file="$OPTIMIZED_DIR/build_info.json"
    local git_hash=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    local git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    local build_timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local rust_version=$(rustc --version)
    
    cat > "$metadata_file" << EOF
{
    "binary": "$BINARY_NAME",
    "git_hash": "$git_hash",
    "git_branch": "$git_branch",
    "build_timestamp": "$build_timestamp",
    "rust_version": "$rust_version",
    "optimization_flags": "-C target-cpu=native -C lto=fat -C codegen-units=1",
    "stripped": true,
    "compressed": $([ "$SKIP_UPX" == "true" ] && echo "false" || echo "true")
}
EOF
    
    log_info "Build metadata written to $metadata_file"
}

# Main execution
main() {
    log_info "=========================================="
    log_info "HFT Crypto Bot Binary Optimization"
    log_info "=========================================="
    
    check_prerequisites
    build_release
    strip_binary
    compress_binary
    generate_metadata
    
    log_info "=========================================="
    log_info "Optimization complete!"
    log_info "Output directory: $OPTIMIZED_DIR"
    log_info "=========================================="
    
    # List generated files
    log_info "Generated files:"
    ls -lh "$OPTIMIZED_DIR"
}

main "$@"
