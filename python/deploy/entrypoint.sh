#!/bin/bash
# Stage 45 HFT System - Python Container Entrypoint
# Strict memory enforcement with ulimit -v capping virtual memory at exactly 3072MB
# Initializes shared memory segments and waits for Rust core's "READY" IPC flag

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
MEMORY_LIMIT_KB=3145728  # 3072 MB in KB (exactly)
SHM_SIZE="256m"          # Shared memory size for zero-copy IPC
RUST_READY_FLAG="/tmp/rust_core_ready"
RUST_READY_TIMEOUT=120   # Seconds to wait for Rust core
PYTHON_MALLOC="malloc"   # Use system malloc for better control

# =============================================================================
# Logging Functions
# =============================================================================
log_info() {
    echo "[INFO] $(date -u +"%Y-%m-%dT%H:%M:%SZ") $*"
}

log_error() {
    echo "[ERROR] $(date -u +"%Y-%m-%dT%H:%M:%SZ") $*" >&2
}

log_warn() {
    echo "[WARN] $(date -u +"%Y-%m-%dT%H:%M:%SZ") $*" >&2
}

# =============================================================================
# Memory Limit Enforcement
# =============================================================================
enforce_memory_limit() {
    log_info "Enforcing virtual memory limit: ${MEMORY_LIMIT_KB}KB (3072MB)"
    
    # Set ulimit for virtual memory (in KB)
    ulimit -v "${MEMORY_LIMIT_KB}"
    
    # Also set RSS limit as secondary protection
    ulimit -m "${MEMORY_LIMIT_KB}"
    
    # Verify the limit was applied
    local current_vmem
    current_vmem=$(ulimit -v)
    if [[ "${current_vmem}" != "${MEMORY_LIMIT_KB}" ]]; then
        log_warn "Virtual memory limit may not be enforced correctly: ${current_vmem}KB"
    fi
    
    log_info "Memory limits applied successfully"
}

# =============================================================================
# Shared Memory Initialization
# =============================================================================
init_shared_memory() {
    log_info "Initializing shared memory for zero-copy IPC with Rust"
    
    # Check if /dev/shm exists and is mounted
    if ! mountpoint -q /dev/shm 2>/dev/null; then
        log_warn "/dev/shm is not a mountpoint, checking availability..."
        if [[ ! -d /dev/shm ]]; then
            log_error "/dev/shm does not exist - shared memory IPC will fail"
            return 1
        fi
    fi
    
    # Check available shared memory
    local shm_available
    shm_available=$(df -k /dev/shm 2>/dev/null | tail -1 | awk '{print $4}')
    
    if [[ -n "${shm_available}" ]] && [[ "${shm_available}" -lt 134217728 ]]; then
        # Less than 128MB available
        log_warn "Limited shared memory available: ${shm_available}KB"
    else
        log_info "Shared memory available: ${shm_available:-unknown}KB"
    fi
    
    # Create shared memory directory for our IPC segments
    mkdir -p /dev/shm/hft_stage45
    chmod 755 /dev/shm/hft_stage45
    
    log_info "Shared memory initialized at /dev/shm/hft_stage45"
}

# =============================================================================
# Wait for Rust Core Ready Flag
# =============================================================================
wait_for_rust_core() {
    log_info "Waiting for Rust core READY signal at ${RUST_READY_FLAG}"
    
    local elapsed=0
    local interval=1
    
    while [[ ${elapsed} -lt ${RUST_READY_TIMEOUT} ]]; do
        if [[ -f "${RUST_READY_FLAG}" ]]; then
            # Verify the flag contains "READY"
            local content
            content=$(cat "${RUST_READY_FLAG}" 2>/dev/null || echo "")
            if [[ "${content}" == "READY" ]]; then
                log_info "Rust core READY signal received after ${elapsed}s"
                return 0
            fi
        fi
        
        sleep "${interval}"
        elapsed=$((elapsed + interval))
        
        # Log progress every 10 seconds
        if (( elapsed % 10 == 0 )); then
            log_info "Still waiting for Rust core... (${elapsed}s/${RUST_READY_TIMEOUT}s)"
        fi
    done
    
    log_error "Timeout waiting for Rust core READY signal after ${RUST_READY_TIMEOUT}s"
    return 1
}

# =============================================================================
# Environment Setup
# =============================================================================
setup_environment() {
    log_info "Setting up runtime environment"
    
    # Set Python memory allocator
    export PYTHONMALLOC="${PYTHON_MALLOC}"
    
    # Disable Python import caching to save memory
    export PYTHONDONTWRITEBYTECODE=1
    
    # Set thread limits for deterministic performance
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    
    # Set memory-efficient GC settings
    export PYTHON_GC_DEBUG=0
    
    # Configure Ray for low-memory operation
    export RAY_DISABLE_IMPORT_WARNING=1
    
    log_info "Environment configured for memory-constrained operation"
}

# =============================================================================
# Health Check Function
# =============================================================================
health_check() {
    local check_type="${1:-all}"
    
    case "${check_type}" in
        memory)
            local mem_usage
            mem_usage=$(ps -o vsz= -p $$ 2>/dev/null || echo "0")
            if [[ "${mem_usage}" -gt "${MEMORY_LIMIT_KB}" ]]; then
                log_error "Memory usage exceeds limit: ${mem_usage}KB > ${MEMORY_LIMIT_KB}KB"
                return 1
            fi
            ;;
        rust_ipc)
            if [[ ! -f "${RUST_READY_FLAG}" ]]; then
                log_error "Rust core IPC not ready"
                return 1
            fi
            ;;
        *)
            health_check memory || return 1
            health_check rust_ipc || return 1
            ;;
    esac
    
    return 0
}

# =============================================================================
# Cleanup Handler
# =============================================================================
cleanup() {
    log_info "Cleanup initiated..."
    
    # Remove shared memory segments
    rm -rf /dev/shm/hft_stage45/* 2>/dev/null || true
    
    # Remove ready flag if we created it
    rm -f "${RUST_READY_FLAG}" 2>/dev/null || true
    
    log_info "Cleanup complete"
}

# =============================================================================
# Main Entry Point
# =============================================================================
main() {
    log_info "=========================================="
    log_info "Stage 45 HFT Python Container Starting"
    log_info "=========================================="
    
    # Set trap for cleanup on exit
    trap cleanup EXIT INT TERM
    
    # Step 1: Setup environment
    setup_environment
    
    # Step 2: Enforce memory limits
    enforce_memory_limit
    
    # Step 3: Initialize shared memory
    init_shared_memory || log_warn "Shared memory initialization had issues"
    
    # Step 4: Wait for Rust core (optional based on env var)
    if [[ "${SKIP_RUST_WAIT:-false}" != "true" ]]; then
        wait_for_rust_core || {
            log_error "Failed to receive Rust core READY signal"
            exit 1
        }
    else
        log_info "Skipping Rust core wait (SKIP_RUST_WAIT=true)"
    fi
    
    # Step 5: Final health check before starting
    health_check || {
        log_error "Pre-start health check failed"
        exit 1
    }
    
    log_info "All checks passed, starting Python application..."
    log_info "Command: $*"
    
    # Execute the main command
    exec "$@"
}

# Run main function with all arguments
main "$@"
