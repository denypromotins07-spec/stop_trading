#!/bin/bash
# Binary Analysis Script for HFT Crypto Bot
# Runs cargo-bloat and cargo-llvm-lines to identify binary fat

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

log_section() {
    echo -e "\n${BLUE}=========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=========================================${NC}\n"
}

# Check and install tools if needed
check_tools() {
    log_info "Checking analysis tools..."
    
    # Check cargo-bloat
    if ! command -v cargo-bloat &> /dev/null; then
        log_warn "cargo-bloat not found. Installing..."
        cargo install cargo-bloat --locked
    fi
    
    # Check cargo-llvm-lines
    if ! command -v cargo-llvm-lines &> /dev/null; then
        log_warn "cargo-llvm-lines not found. Installing..."
        cargo install cargo-llvm-lines --locked
    fi
    
    # Check nm (part of binutils)
    if ! command -v nm &> /dev/null; then
        log_error "nm not found. Please install binutils."
        exit 1
    fi
    
    log_info "All tools available."
}

# Analyze binary size with cargo-bloat
analyze_bloat() {
    log_section "Binary Size Analysis (cargo-bloat)"
    
    cd "$PROJECT_ROOT"
    
    log_info "Analyzing release binary..."
    
    # Run cargo-bloat on release build
    cargo bloat --release \
        --crates \
        -n 50 \
        --message-format=json \
        > /tmp/bloat_crates.json 2>/dev/null || true
    
    # Human-readable output
    cargo bloat --release \
        --crates \
        -n 30
    
    echo ""
    log_info "Top functions by size:"
    cargo bloat --release \
        -n 20 \
        --sort=size
    
    # Save detailed report
    local report_file="$PROJECT_ROOT/target/analysis/bloat_report.txt"
    mkdir -p "$(dirname "$report_file")"
    
    cargo bloat --release --crates -n 100 > "$report_file" 2>&1 || true
    log_info "Full bloat report saved to: $report_file"
}

# Analyze LLVM IR lines with cargo-llvm-lines
analyze_llvm_lines() {
    log_section "LLVM IR Lines Analysis (cargo-llvm-lines)"
    
    cd "$PROJECT_ROOT"
    
    log_info "Analyzing LLVM IR line counts..."
    
    # Run cargo-llvm-lines
    cargo llvm-lines \
        --release \
        --sort=lines \
        > /tmp/llvm_lines_output.txt 2>&1 || true
    
    # Display top results
    echo ""
    log_info "Top crates by LLVM IR lines:"
    head -50 /tmp/llvm_lines_output.txt
    
    # Save full report
    local report_file="$PROJECT_ROOT/target/analysis/llvm_lines_report.txt"
    cp /tmp/llvm_lines_output.txt "$report_file"
    log_info "Full LLVM lines report saved to: $report_file"
}

# Analyze symbol table
analyze_symbols() {
    log_section "Symbol Table Analysis"
    
    local binary="$PROJECT_ROOT/target/release/hft_crypto_bot"
    
    if [[ ! -f "$binary" ]]; then
        log_warn "Release binary not found. Building first..."
        cargo build --release
    fi
    
    if [[ ! -f "$binary" ]]; then
        log_error "Binary still not found after build."
        return
    fi
    
    log_info "Analyzing symbol table..."
    
    # Count symbols
    local total_symbols=$(nm "$binary" 2>/dev/null | wc -l || echo "0")
    log_info "Total symbols: $total_symbols"
    
    # Text section symbols (code)
    local text_symbols=$(nm "$binary" 2>/dev/null | grep -E '^[0-9a-f]+ T ' | wc -l || echo "0")
    log_info "Text section symbols (code): $text_symbols"
    
    # Data section symbols
    local data_symbols=$(nm "$binary" 2>/dev/null | grep -E '^[0-9a-f]+ [Dd] ' | wc -l || echo "0")
    log_info "Data section symbols: $data_symbols"
    
    # Largest symbols
    echo ""
    log_info "Largest symbols (by address range estimation):"
    nm --print-size --size-sort "$binary" 2>/dev/null | tail -20 || true
    
    # Demangled Rust symbols
    echo ""
    log_info "Top mangled Rust symbols:"
    nm "$binary" 2>/dev/null | grep -E '_ZN' | head -20 || true
}

# Generate optimization recommendations
generate_recommendations() {
    log_section "Optimization Recommendations"
    
    cat << 'EOF'
Based on the analysis, consider the following optimizations:

1. CODEGEN UNITS
   - Ensure codegen-units=1 in release profile for better inlining
   - Already configured in Cargo.toml

2. LTO (Link Time Optimization)
   - Use lto=fat for maximum optimization
   - Increases build time but reduces binary size

3. STRIP DEBUG SYMBOLS
   - Run: strip --strip-all target/release/hft_crypto_bot
   - Reduces binary size by 60-80%

4. REMOVE UNUSED DEPENDENCIES
   - Run: cargo udeps (requires cargo-udeps)
   - Remove unused crates from Cargo.toml

5. FEATURE FLAGS
   - Disable unnecessary features in dependencies
   - Use --no-default-features where possible

6. PANIC ABORT
   - Consider panic = "abort" in release profile
   - Removes unwinding code (~10% size reduction)

7. DYNAMIC VS STATIC LINKING
   - Static linking increases size but improves portability
   - Consider dynamic linking for system libraries

8. PROFILE GUIDED OPTIMIZATION (PGO)
   - Advanced: Use PGO for hot path optimization
   - Requires profiling run first
EOF
}

# Main execution
main() {
    log_info "=========================================="
    log_info "HFT Crypto Bot Binary Analysis"
    log_info "=========================================="
    
    check_tools
    analyze_bloat
    analyze_llvm_lines
    analyze_symbols
    generate_recommendations
    
    log_info ""
    log_info "=========================================="
    log_info "Analysis complete!"
    log_info "Reports saved to: $PROJECT_ROOT/target/analysis/"
    log_info "=========================================="
}

main "$@"
