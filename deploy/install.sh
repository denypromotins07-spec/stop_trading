#!/bin/bash
# =============================================================================
# HFT Crypto Bot - Ubuntu Host Installation and Hardening Script
# =============================================================================
# This script configures an Ubuntu host for ultra-low-latency trading:
# - Disables SWAP to prevent memory paging
# - Enables HugePages for TLB optimization
# - Configures CPU isolation (isolcpus) for dedicated trading cores
# - Applies network and kernel optimizations
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TRADING_CORES="2-7"      # Cores dedicated to trading (adjust for your CPU)
SYSTEM_CORES="0-1"       # Cores for OS and background tasks
HUGEPAGES_COUNT=2048     # Number of 2MB hugepages (4GB total)
MAX_FILE_DESCRIPTORS=65536

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (sudo)"
        exit 1
    fi
}

check_ubuntu() {
    if ! grep -q "Ubuntu" /etc/os-release; then
        log_warning "This script is designed for Ubuntu. Proceeding with caution..."
    fi
}

# =============================================================================
# Disable SWAP
# =============================================================================
disable_swap() {
    log_info "Disabling SWAP..."
    
    # Turn off swap immediately
    swapoff -a 2>/dev/null || true
    
    # Remove swap from fstab (comment out swap entries)
    if grep -q "swap" /etc/fstab; then
        cp /etc/fstab /etc/fstab.backup
        sed -i '/swap/s/^/#/' /etc/fstab
        log_success "SWAP disabled in /etc/fstab (backup created at /etc/fstab.backup)"
    else
        log_info "No SWAP entries found in /etc/fstab"
    fi
    
    # Verify swap is off
    if [[ $(swapon --show | wc -l) -eq 0 ]]; then
        log_success "SWAP is now disabled"
    else
        log_warning "SWAP may still be active. Reboot recommended."
    fi
}

# =============================================================================
# Enable HugePages
# =============================================================================
enable_hugepages() {
    log_info "Enabling HugePages (${HUGEPAGES_COUNT} pages = $((HUGEPAGES_COUNT * 2)) MB)..."
    
    # Set hugepages count
    echo "${HUGEPAGES_COUNT}" > /proc/sys/vm/nr_hugepages
    
    # Make persistent across reboots
    if ! grep -q "vm.nr_hugepages" /etc/sysctl.conf; then
        echo "vm.nr_hugepages = ${HUGEPAGES_COUNT}" >> /etc/sysctl.conf
        log_success "HugePages configured in /etc/sysctl.conf"
    else
        log_info "HugePages already configured in /etc/sysctl.conf"
    fi
    
    # Verify hugepages are allocated
    local allocated=$(grep "^HugePages_Total:" /proc/meminfo | awk '{print $2}')
    if [[ "${allocated}" -eq "${HUGEPAGES_COUNT}" ]]; then
        log_success "HugePages enabled: ${allocated} pages allocated"
    else
        log_warning "HugePages allocation may require reboot. Current: ${allocated}/${HUGEPAGES_COUNT}"
    fi
    
    # Set transparent hugepages to madvise (better for databases)
    if [[ -f /sys/kernel/mm/transparent_hugepage/enabled ]]; then
        echo "madvise" > /sys/kernel/mm/transparent_hugepage/enabled
        echo "never" > /sys/kernel/mm/transparent_hugepage/defrag
        log_success "Transparent HugePages set to madvise"
    fi
}

# =============================================================================
# Configure CPU Isolation (isolcpus)
# =============================================================================
configure_cpu_isolation() {
    log_info "Configuring CPU isolation (isolcpus=${TRADING_CORES})..."
    
    local grub_config="/etc/default/grub"
    local backup="${grub_config}.backup"
    
    # Backup original grub config
    if [[ ! -f "${backup}" ]]; then
        cp "${grub_config}" "${backup}"
        log_info "Backed up GRUB config to ${backup}"
    fi
    
    # Check if isolcpus is already configured
    if grep -q "isolcpus" "${grub_config}"; then
        log_warning "isolcpus already configured in GRUB. Skipping..."
        log_warning "Current setting: $(grep 'isolcpus' "${grub_config}")"
        return
    fi
    
    # Add isolcpus, nohz_full, and rcu_nocbs to GRUB_CMDLINE_LINUX
    local cpu_params="isolcpus=${TRADING_CORES},nohz_full=${TRADING_CORES},rcu_nocbs=${TRADING_CORES}"
    
    # Also add interrupt affinity hint disabling
    cpu_params="${cpu_params},irqaffinity=${SYSTEM_CORES}"
    
    sed -i "s/GRUB_CMDLINE_LINUX=\"\(.*\)\"/GRUB_CMDLINE_LINUX=\"\1 ${cpu_params}\"/" "${grub_config}"
    
    # Update GRUB
    update-grub
    
    log_success "CPU isolation configured. Reboot required for changes to take effect."
    log_info "Trading cores ${TRADING_CORES} will be isolated from OS scheduler"
    log_info "System cores ${SYSTEM_CORES} will handle interrupts and OS tasks"
}

# =============================================================================
# Apply sysctl Network and Kernel Optimizations
# =============================================================================
apply_sysctl_tweaks() {
    log_info "Applying sysctl optimizations..."
    
    local sysctl_file="/etc/sysctl.d/99-hft-crypto-bot.conf"
    
    cat > "${sysctl_file}" << 'EOF'
# =============================================================================
# HFT Crypto Bot - Kernel and Network Optimizations
# =============================================================================

# -----------------------------------------------------------------------------
# Network Stack Optimizations
# -----------------------------------------------------------------------------

# Increase TCP buffer sizes for high-throughput
net.core.rmem_default = 16777216
net.core.rmem_max = 134217728
net.core.wmem_default = 16777216
net.core.wmem_max = 134217728

# TCP congestion control (use BBR for low latency)
net.core.default_qdisc = fq
net.ipv4.tcp_congestion_control = bbr

# Reduce TCP latency
net.ipv4.tcp_low_latency = 1
net.ipv4.tcp_no_metrics_save = 1
net.ipv4.tcp_moderate_rcvbuf = 1

# TCP Fast Open (reduce connection establishment latency)
net.ipv4.tcp_fastopen = 3

# Increase connection queue sizes
net.core.netdev_max_backlog = 65536
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65536

# TCP keepalive settings
net.ipv4.tcp_keepalive_time = 600
net.ipv4.tcp_keepalive_probes = 3
net.ipv4.tcp_keepalive_intvl = 30

# Disable IPv6 (if not needed, reduces complexity)
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
net.ipv6.conf.lo.disable_ipv6 = 1

# Optimize route cache
net.ipv4.route.flush = 1

# Increase ephemeral port range
net.ipv4.ip_local_port_range = 1024 65535

# Disable TCP timestamps (minor latency improvement)
net.ipv4.tcp_timestamps = 0

# Enable TCP window scaling
net.ipv4.tcp_window_scaling = 1

# Disable slow start on idle connections
net.ipv4.tcp_slow_start_after_idle = 0

# -----------------------------------------------------------------------------
# Memory Management
# -----------------------------------------------------------------------------

# Reduce swappiness (avoid swapping at all costs)
vm.swappiness = 1

# Overcommit memory (allow allocating more than physical RAM)
vm.overcommit_memory = 1

# Dirty page writeback tuning
vm.dirty_ratio = 10
vm.dirty_background_ratio = 5
vm.dirty_expire_centisecs = 3000
vm.dirty_writeback_centisecs = 500

# VMA (Virtual Memory Area) limit for large allocations
vm.max_map_count = 262144

# NUMA balancing (disable for manual pinning)
kernel.numa_balancing = 0

# -----------------------------------------------------------------------------
# File System
# -----------------------------------------------------------------------------

# Increase file descriptor limits
fs.file-max = 2097152
fs.inotify.max_user_watches = 524288
fs.inotify.max_user_instances = 512

# AIO (Asynchronous I/O) limits
fs.aio-max-nr = 1048576

# -----------------------------------------------------------------------------
# Kernel Scheduling
# -----------------------------------------------------------------------------

# Real-time priority for trading threads
kernel.sched_rt_runtime_us = 950000
kernel.sched_rt_period_us = 1000000

# Energy performance (prefer performance over power saving)
kernel.energy_perf_bias = 0

# Watchdog timeout (increase for stability)
kernel.watchdog_thresh = 10
EOF

    # Apply sysctl settings
    sysctl --system
    
    log_success "Sysctl optimizations applied from ${sysctl_file}"
}

# =============================================================================
# Configure File Descriptor Limits
# =============================================================================
configure_limits() {
    log_info "Configuring file descriptor limits..."
    
    local limits_file="/etc/security/limits.d/99-hft-crypto-bot.conf"
    
    cat > "${limits_file}" << EOF
# =============================================================================
# HFT Crypto Bot - Resource Limits
# =============================================================================

# File descriptors
* soft nofile ${MAX_FILE_DESCRIPTORS}
* hard nofile ${MAX_FILE_DESCRIPTORS}

# Process/thread limits
* soft nproc 65536
* hard nproc 65536

# Memory lock (for mlock/mlockall)
* soft memlock unlimited
* hard memlock unlimited

# Real-time priority
* soft rtprio 99
* hard rtprio 99

# Nice priority
* soft nice -20
* hard nice -20
EOF

    log_success "Resource limits configured in ${limits_file}"
}

# =============================================================================
# Create Dedicated User
# =============================================================================
create_user() {
    log_info "Creating dedicated hftbot user..."
    
    if id "hftbot" &>/dev/null; then
        log_info "User hftbot already exists"
    else
        useradd -r -s /bin/false -d /opt/hft_crypto_bot hftbot
        log_success "Created system user hftbot"
    fi
    
    # Create directories
    mkdir -p /opt/hft_crypto_bot/data
    mkdir -p /var/log/hft_crypto_bot
    chown -R hftbot:hftbot /opt/hft_crypto_bot
    chown -R hftbot:hftbot /var/log/hft_crypto_bot
    
    log_success "Created data directories"
}

# =============================================================================
# Install Required Packages
# =============================================================================
install_packages() {
    log_info "Installing required packages..."
    
    apt-get update -qq
    
    # Essential packages for HFT
    local packages=(
        "linux-tools-generic"      # perf for profiling
        "htop"                     # Process monitoring
        "iotop"                    # I/O monitoring
        "tcpdump"                  # Network capture
        "ethtool"                  # Network interface tuning
        "irqbalance"               # IRQ management (will be disabled)
        "numactl"                  # NUMA awareness
        "hugepages"                # Hugepage management
    )
    
    DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"
    
    log_success "Required packages installed"
}

# =============================================================================
# Disable Unnecessary Services
# =============================================================================
disable_services() {
    log_info "Disabling unnecessary services..."
    
    # Disable irqbalance (we'll manage IRQ affinity manually)
    systemctl disable irqbalance 2>/dev/null || true
    systemctl stop irqbalance 2>/dev/null || true
    
    # Disable bluetooth (not needed for trading)
    systemctl disable bluetooth 2>/dev/null || true
    systemctl stop bluetooth 2>/dev/null || true
    
    # Disable cups (printing not needed)
    systemctl disable cups 2>/dev/null || true
    systemctl stop cups 2>/dev/null || true
    
    log_success "Unnecessary services disabled"
}

# =============================================================================
# Main Execution
# =============================================================================
main() {
    echo ""
    echo "======================================================================"
    echo "  HFT Crypto Bot - Ubuntu Host Installation & Hardening"
    echo "======================================================================"
    echo ""
    
    check_root
    check_ubuntu
    
    log_warning "This script will make significant system changes."
    log_warning "A reboot will be required after installation."
    echo ""
    
    read -p "Do you want to proceed? (y/N): " confirm
    if [[ ! "${confirm}" =~ ^[Yy]$ ]]; then
        log_info "Installation cancelled"
        exit 0
    fi
    
    echo ""
    
    # Execute installation steps
    install_packages
    disable_swap
    enable_hugepages
    configure_cpu_isolation
    apply_sysctl_tweaks
    configure_limits
    create_user
    disable_services
    
    echo ""
    echo "======================================================================"
    echo "  Installation Complete!"
    echo "======================================================================"
    echo ""
    log_success "Host hardening completed successfully"
    echo ""
    log_warning "IMPORTANT: A reboot is required for CPU isolation to take effect"
    echo ""
    log_info "After reboot, verify configuration:"
    echo "  - Check SWAP:        swapon --show (should be empty)"
    echo "  - Check HugePages:   grep HugePages /proc/meminfo"
    echo "  - Check isolcpus:    cat /proc/cmdline | grep isolcpus"
    echo "  - Check IRQ affinity: cat /proc/irq/*/smp_affinity_list"
    echo ""
    log_info "Next steps:"
    echo "  1. Reboot the system: sudo reboot"
    echo "  2. Install systemd service: sudo cp deploy/systemd.service /etc/systemd/system/"
    echo "  3. Enable service: sudo systemctl enable hft_crypto_bot"
    echo "  4. Start service: sudo systemctl start hft_crypto_bot"
    echo ""
}

# Run main function
main "$@"
