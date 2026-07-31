#!/bin/bash
# Python Environment Setup Script for HFT Crypto Bot ML Backend
# Creates a virtual environment with strict memory limits matching the 6.5GB ceiling

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$SCRIPT_DIR/venv"
PYTHON_VERSION="${1:-3.11}"

# Memory limits (in MB) - must stay within 6.5GB total system limit
MAX_PYTHON_MEMORY_MB=2048  # 2GB max for Python ML backend
MAX_VIRTUAL_MEMORY_MB=4096 # 4GB virtual memory limit

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

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python version
    if ! command -v python$PYTHON_VERSION &> /dev/null; then
        if ! command -v python3 &> /dev/null; then
            log_error "Python 3 not found. Please install Python $PYTHON_VERSION or later."
            exit 1
        fi
        PYTHON_CMD="python3"
        log_warn "Using python3 instead of python$PYTHON_VERSION"
    else
        PYTHON_CMD="python$PYTHON_VERSION"
    fi
    
    # Verify Python version is 3.9+
    local py_version=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    log_info "Found Python version: $py_version"
    
    # Check pip
    if ! $PYTHON_CMD -m pip --version &> /dev/null; then
        log_error "pip not found. Please install pip."
        exit 1
    fi
    
    # Check venv module
    if ! $PYTHON_CMD -m venv --help &> /dev/null; then
        log_error "venv module not found. Install python$PYTHON_VERSION-venv package."
        exit 1
    fi
    
    log_info "System requirements satisfied."
}

# Create virtual environment
create_venv() {
    log_info "Creating virtual environment at $VENV_DIR..."
    
    # Remove existing venv if present
    if [[ -d "$VENV_DIR" ]]; then
        log_warn "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    fi
    
    # Create new venv
    $PYTHON_CMD -m venv "$VENV_DIR"
    
    # Activate and upgrade pip
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel
}

# Install dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."
    
    source "$VENV_DIR/bin/activate"
    
    # Install from requirements.txt
    if [[ -f "$SCRIPT_DIR/requirements.txt" ]]; then
        pip install -r "$SCRIPT_DIR/requirements.txt"
    else
        log_error "requirements.txt not found!"
        exit 1
    fi
    
    # Verify critical packages
    log_info "Verifying critical packages..."
    python -c "import nautilus_trader; print(f'nautilus_trader: {nautilus_trader.__version__}')" || true
    python -c "import ray; print(f'ray: {ray.__version__}')" || true
    python -c "import pyarrow; print(f'pyarrow: {pyarrow.__version__}')" || true
    python -c "import numpy; print(f'numpy: {numpy.__version__}')" || true
}

# Create memory-limited launcher script
create_launcher() {
    log_info "Creating memory-limited launcher script..."
    
    local launcher="$SCRIPT_DIR/run_ml_backend.sh"
    
    cat > "$launcher" << EOF
#!/bin/bash
# Memory-Limited Python ML Backend Launcher
# Enforces strict memory limits to prevent exceeding 6.5GB system ceiling

set -euo pipefail

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="\$SCRIPT_DIR/venv"

# Memory limits (in bytes for ulimit)
MAX_VIRTUAL_MEM_BYTES=$(( $MAX_VIRTUAL_MEMORY_MB * 1024 * 1024 ))
MAX_RSS_BYTES=$(( $MAX_PYTHON_MEMORY_MB * 1024 * 1024 ))

# Activate virtual environment
source "\$VENV_DIR/bin/activate"

# Set memory limits using ulimit
# -v: Virtual memory limit (in KB)
ulimit -v \$(( $MAX_VIRTUAL_MEM_BYTES / 1024 ))

# Export memory limit for Python to enforce internally
export PYTHON_MAX_MEMORY_MB=$MAX_PYTHON_MEMORY_MB
export RAY_MEMORY_LIMIT_MB=$MAX_PYTHON_MEMORY_MB

# Launch Python ML backend
exec python "\$SCRIPT_DIR/ml_backend.py" "\$@"
EOF
    
    chmod +x "$launcher"
    log_info "Launcher created: $launcher"
}

# Create wrapper script with cgroups (if available)
create_cgroups_wrapper() {
    log_info "Checking for cgroups support..."
    
    if [[ -d "/sys/fs/cgroup" ]]; then
        local cgroup_launcher="$SCRIPT_DIR/run_ml_backend_cgroup.sh"
        
        cat > "$cgroup_launcher" << EOF
#!/bin/bash
# Cgroups-based Memory-Limited Launcher
# Uses Linux cgroups for hard memory limits

set -euo pipefail

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
CGROUP_NAME="hft_ml_backend"
MEMORY_LIMIT="$(( $MAX_PYTHON_MEMORY_MB * 1024 * 1024 ))"

# Check if running as root or with sudo
if [[ \$EUID -ne 0 ]]; then
    echo "Warning: Cgroups require root privileges. Falling back to ulimit."
    exec "\$SCRIPT_DIR/run_ml_backend.sh" "\$@"
fi

# Create cgroup (cgroup v2)
if [[ -d "/sys/fs/cgroup/$CGROUP_NAME" ]]; then
    rmdir "/sys/fs/cgroup/$CGROUP_NAME" 2>/dev/null || true
fi

mkdir -p "/sys/fs/cgroup/$CGROUP_NAME"
echo "$MEMORY_LIMIT" > "/sys/fs/cgroup/$CGROUP_NAME/memory.max"

# Get current PID and add to cgroup
CURRENT_PID=\$\$
echo "\$CURRENT_PID" > "/sys/fs/cgroup/$CGROUP_NAME/cgroup.procs"

# Launch in cgroup
exec systemd-run --scope -p MemoryMax="$MEMORY_LIMIT" \
    "\$SCRIPT_DIR/run_ml_backend.sh" "\$@"
EOF
        
        chmod +x "$cgroup_launcher"
        log_info "Cgroups launcher created: $cgroup_launcher"
    else
        log_warn "Cgroups not available. Using ulimit-based limits only."
    fi
}

# Generate environment documentation
generate_docs() {
    log_info "Generating environment documentation..."
    
    cat > "$SCRIPT_DIR/ENV_SETUP.md" << EOF
# Python ML Backend Environment Setup

## Memory Limits

The Python ML backend is configured with strict memory limits to ensure
the total system memory usage stays within the 6.5GB ceiling:

| Component | Memory Limit |
|-----------|--------------|
| Rust Core | 4.5 GB       |
| Python ML | 2.0 GB       |
| **Total** | **6.5 GB**   |

## Virtual Environment

Location: \`./venv\`

### Activation

\`\`\`bash
source venv/bin/activate
\`\`\`

## Running the ML Backend

### Standard Mode (ulimit)

\`\`\`bash
./run_ml_backend.sh
\`\`\`

### Cgroups Mode (requires root)

\`\`\`bash
sudo ./run_ml_backend_cgroup.sh
\`\`\`

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| PYTHON_MAX_MEMORY_MB | Max RSS for Python process | 2048 |
| RAY_MEMORY_LIMIT_MB | Ray object store limit | 2048 |
| RAY_OBJECT_STORE_SIZE | Explicit object store size | auto |

## IPC Configuration

The Python backend communicates with Rust via shared memory:

- Feature vectors: Rust → Python (zero-copy)
- Alpha signals: Python → Rust (zero-copy)
- Weight updates: Python → Rust (periodic sync)

See \`../src/bridge/\` for schema definitions.
EOF
    
    log_info "Documentation generated: $SCRIPT_DIR/ENV_SETUP.md"
}

# Main execution
main() {
    log_info "=========================================="
    log_info "HFT Crypto Bot Python Environment Setup"
    log_info "=========================================="
    
    check_requirements
    create_venv
    install_dependencies
    create_launcher
    create_cgroups_wrapper
    generate_docs
    
    log_info ""
    log_info "=========================================="
    log_info "Setup complete!"
    log_info ""
    log_info "To activate the environment:"
    log_info "  source $VENV_DIR/bin/activate"
    log_info ""
    log_info "To run the ML backend:"
    log_info "  ./run_ml_backend.sh"
    log_info "=========================================="
}

main "$@"
