#!/bin/bash
# Packaging Script for HFT Crypto Bot
# Bundles stripped binary, config files, and documentation into secure tarball

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PACKAGE_NAME="hft_crypto_bot"
VERSION="${1:-$(git describe --tags --always --dirty 2>/dev/null || echo "dev")}"
BUILD_DIR="$PROJECT_ROOT/target/package"
OUTPUT_FILE="$PROJECT_ROOT/releases/${PACKAGE_NAME}-${VERSION}.tar.gz"

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

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    local binary="$PROJECT_ROOT/target/release/$PACKAGE_NAME"
    
    if [[ ! -f "$binary" ]]; then
        log_error "Release binary not found at $binary"
        log_info "Run: cargo build --release"
        exit 1
    fi
    
    if [[ ! -f "$PROJECT_ROOT/SOUL.md" ]]; then
        log_error "SOUL.md not found in project root"
        exit 1
    fi
    
    if [[ ! -f "$PROJECT_ROOT/config.toml" ]] && [[ ! -f "$PROJECT_ROOT/.env.example" ]]; then
        log_warn "No config.toml or .env.example found. Creating default config."
    fi
    
    log_info "Prerequisites validated."
}

# Create package directory structure
create_package_structure() {
    log_info "Creating package directory structure..."
    
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR/$PACKAGE_NAME"
    mkdir -p "$BUILD_DIR/$PACKAGE_NAME/bin"
    mkdir -p "$BUILD_DIR/$PACKAGE_NAME/config"
    mkdir -p "$BUILD_DIR/$PACKAGE_NAME/logs"
    mkdir -p "$BUILD_DIR/$PACKAGE_NAME/data"
    mkdir -p "$PROJECT_ROOT/releases"
    
    # Copy and strip binary
    log_info "Copying and stripping binary..."
    cp "$PROJECT_ROOT/target/release/$PACKAGE_NAME" "$BUILD_DIR/$PACKAGE_NAME/bin/"
    strip --strip-all "$BUILD_DIR/$PACKAGE_NAME/bin/$PACKAGE_NAME"
    
    # Set executable permissions
    chmod 755 "$BUILD_DIR/$PACKAGE_NAME/bin/$PACKAGE_NAME"
}

# Copy configuration files
copy_config_files() {
    log_info "Copying configuration files..."
    
    # Copy SOUL.md (documentation)
    cp "$PROJECT_ROOT/SOUL.md" "$BUILD_DIR/$PACKAGE_NAME/"
    
    # Copy or create config.toml
    if [[ -f "$PROJECT_ROOT/config.toml" ]]; then
        cp "$PROJECT_ROOT/config.toml" "$BUILD_DIR/$PACKAGE_NAME/config/"
    else
        cat > "$BUILD_DIR/$PACKAGE_NAME/config/config.toml" << 'EOF'
# HFT Crypto Bot Configuration
# Copy this file to config.toml and modify as needed

[general]
log_level = "info"
data_dir = "./data"
log_dir = "./logs"

[network]
bind_address = "0.0.0.0"
api_port = 8080

[risk]
max_position_size = 1.0
max_daily_loss = 0.05
max_order_size = 0.1

[exchanges]
# Exchange configurations go here
EOF
    fi
    
    # Copy or create .env example
    if [[ -f "$PROJECT_ROOT/.env.example" ]]; then
        cp "$PROJECT_ROOT/.env.example" "$BUILD_DIR/$PACKAGE_NAME/config/"
    elif [[ -f "$PROJECT_ROOT/.env" ]]; then
        # Create sanitized version without secrets
        grep -v "^API_KEY\|^SECRET_KEY\|^PASSWORD" "$PROJECT_ROOT/.env" \
            > "$BUILD_DIR/$PACKAGE_NAME/config/.env.example" 2>/dev/null || true
    else
        cat > "$BUILD_DIR/$PACKAGE_NAME/config/.env.example" << 'EOF'
# Environment Variables Example
# Copy to .env and fill in your values

# Exchange API Credentials
# EXCHANGE_API_KEY=your_api_key_here
# EXCHANGE_SECRET_KEY=your_secret_key_here

# Network Settings
# LISTEN_ADDRESS=0.0.0.0
# PORT=8080

# Risk Parameters
# MAX_POSITION=1.0
# DAILY_LOSS_LIMIT=0.05
EOF
    fi
    
    # Copy systemd service file
    if [[ -f "$PROJECT_ROOT/deploy/systemd.service" ]]; then
        cp "$PROJECT_ROOT/deploy/systemd.service" "$BUILD_DIR/$PACKAGE_NAME/config/hft_crypto_bot.service"
    fi
    
    # Copy install script
    if [[ -f "$PROJECT_ROOT/deploy/install.sh" ]]; then
        cp "$PROJECT_ROOT/deploy/install.sh" "$BUILD_DIR/$PACKAGE_NAME/install.sh"
        chmod +x "$BUILD_DIR/$PACKAGE_NAME/install.sh"
    fi
}

# Create README for package
create_readme() {
    log_info "Creating package README..."
    
    cat > "$BUILD_DIR/$PACKAGE_NAME/README.md" << EOF
# HFT Crypto Bot - Release Package

**Version:** $VERSION  
**Build Date:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")

## Quick Start

### Installation

1. Extract the package:
   \`\`\`bash
   tar -xzf ${PACKAGE_NAME}-${VERSION}.tar.gz
   cd ${PACKAGE_NAME}
   \`\`\`

2. Run the installation script (requires sudo):
   \`\`\`bash
   sudo ./install.sh
   \`\`\`

3. Configure your environment:
   \`\`\`bash
   cp config/.env.example config/.env
   # Edit config/.env with your credentials
   \`\`\`

4. Start the bot:
   \`\`\`bash
   ./bin/hft_crypto_bot
   \`\`\`

### Systemd Service

To run as a system service:

\`\`\`bash
sudo cp config/hft_crypto_bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable hft_crypto_bot
sudo systemctl start hft_crypto_bot
\`\`\`

## Directory Structure

\`\`\`
${PACKAGE_NAME}/
├── bin/
│   └── hft_crypto_bot    # Main binary (stripped)
├── config/
│   ├── config.toml       # Application config
│   ├── .env.example      # Environment template
│   └── hft_crypto_bot.service  # Systemd unit
├── logs/                  # Log files (created at runtime)
├── data/                  # Data files (created at runtime)
├── SOUL.md               # Documentation
├── README.md             # This file
└── install.sh            # Installation script
\`\`\`

## Security Notes

- The binary is stripped of debug symbols
- Default memory limit: 6.5GB (configured in systemd)
- CPU affinity set to isolated cores (2-7)
- Seccomp and AppArmor profiles available in security/

## Support

See SOUL.md for detailed documentation.
EOF
}

# Generate checksums
generate_checksums() {
    log_info "Generating checksums..."
    
    cd "$BUILD_DIR"
    
    # Generate SHA256 checksums
    find "$PACKAGE_NAME" -type f -exec sha256sum {} \; > "$PACKAGE_NAME/SHA256SUMS"
    
    # Display checksums
    log_info "Package checksums:"
    cat "$PACKAGE_NAME/SHA256SUMS"
}

# Create tarball
create_tarball() {
    log_info "Creating release tarball..."
    
    cd "$BUILD_DIR"
    
    # Create compressed tarball
    tar --sort=name \
        --owner=0 --group=0 --numeric-owner \
        -czf "$OUTPUT_FILE" "$PACKAGE_NAME"
    
    # Generate tarball checksum
    sha256sum "$OUTPUT_FILE" > "${OUTPUT_FILE}.sha256"
    
    local tarball_size=$(stat -c%s "$OUTPUT_FILE" 2>/dev/null || stat -f%z "$OUTPUT_FILE" 2>/dev/null)
    log_info "Tarball created: $OUTPUT_FILE"
    log_info "Tarball size: $(numfmt --to=iec-i --suffix=B "$tarball_size" 2>/dev/null || echo "$tarball_size bytes")"
    log_info "Checksum: $(cat "${OUTPUT_FILE}.sha256")"
}

# Cleanup
cleanup() {
    log_info "Cleaning up build directory..."
    rm -rf "$BUILD_DIR"
}

# Main execution
main() {
    log_info "=========================================="
    log_info "HFT Crypto Bot Packaging"
    log_info "Version: $VERSION"
    log_info "=========================================="
    
    validate_prerequisites
    create_package_structure
    copy_config_files
    create_readme
    generate_checksums
    create_tarball
    cleanup
    
    log_info ""
    log_info "=========================================="
    log_info "Packaging complete!"
    log_info "Release: $OUTPUT_FILE"
    log_info "=========================================="
    
    # List releases directory
    log_info "Available releases:"
    ls -lh "$PROJECT_ROOT/releases/"
}

main "$@"
