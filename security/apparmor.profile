# =============================================================================
# AppArmor Profile for HFT Crypto Bot
# =============================================================================
# This Mandatory Access Control (MAC) profile restricts the trading bot's
# filesystem access to only essential paths, preventing unauthorized access
# even if the binary is compromised.
# =============================================================================

#include <tunables/global>

# Profile declaration
profile hft_crypto_bot flags=(attach_disconnected,mediate_deleted) {
  
  # ===========================================================================
  # Network Rules
  # ===========================================================================
  
  # Allow network access for exchange connections
  network inet stream,
  network inet6 stream,
  network inet dgram,
  network inet6 dgram,
  
  # Allow raw sockets for network diagnostics (optional)
  # network inet raw,
  
  # ===========================================================================
  # File System Access - Trading Directory
  # ===========================================================================
  
  # Main application directory (read/write for data files)
  /opt/hft_crypto_bot/ r,
  /opt/hft_crypto_bot/** rw,
  
  # Configuration files (read-only after startup)
  /opt/hft_crypto_bot/config.toml r,
  /opt/hft_crypto_bot/*.toml r,
  
  # Data directory for tick database and logs
  /opt/hft_crypto_bot/data/ r,
  /opt/hft_crypto_bot/data/** rw,
  
  # Log directory
  /var/log/hft_crypto_bot/ r,
  /var/log/hft_crypto_bot/** rw,
  
  # ===========================================================================
  # Essential System Files
  # ===========================================================================
  
  # CA certificates for HTTPS/TLS connections
  /etc/ssl/certs/ r,
  /etc/ssl/certs/** r,
  /etc/ssl/** r,
  
  # Timezone information
  /etc/timezone r,
  /usr/share/zoneinfo/** r,
  
  # Hostname and network configuration
  /etc/hostname r,
  /etc/hosts r,
  /etc/resolv.conf r,
  /etc/nsswitch.conf r,
  
  # Dynamic linker
  /lib64/ld-linux-x86-64.so.2 mr,
  
  # Shared libraries (essential system libs)
  /lib/x86_64-linux-gnu/** mr,
  /usr/lib/x86_64-linux-gnu/** mr,
  
  # ===========================================================================
  # Proc and Sysfs Access
  # ===========================================================================
  
  # Process information (for monitoring and pinning)
  /proc/ r,
  /proc/[0-9]*/ r,
  /proc/[0-9]*/cmdline r,
  /proc/[0-9]*/stat r,
  /proc/[0-9]*/status r,
  /proc/[0-9]*/maps r,
  /proc/[0-9]*/fd/ r,
  /proc/[0-9]*/fd/* r,
  /proc/self/** r,
  
  # CPU and memory information
  /proc/cpuinfo r,
  /proc/meminfo r,
  /proc/stat r,
  /proc/uptime r,
  /proc/version r,
  /proc/loadavg r,
  
  # NUMA information
  /proc/sys/kernel/numa_balancing r,
  /sys/devices/system/node/** r,
  
  # HugePages information
  /proc/sys/vm/nr_hugepages r,
  /proc/meminfo r,
  
  # ===========================================================================
  # Restricted Paths (Explicitly Denied)
  # ===========================================================================
  
  # Deny access to home directories
  deny /home/** rw,
  deny /root/** rw,
  
  # Deny access to other user data
  deny /tmp/** w,
  deny /var/tmp/** w,
  
  # Deny access to sensitive system files
  deny /etc/shadow r,
  deny /etc/passwd r,
  deny /etc/sudoers r,
  deny /etc/ssh/** r,
  deny /etc/gshadow r,
  
  # Deny access to kernel modules
  deny /lib/modules/** r,
  deny /sys/module/** r,
  
  # Deny access to device files (except essential)
  deny /dev/** w,
  
  # Deny ptrace access (prevent debugging)
  deny ptrace (trace,traced,read) peer=unconfined,
  
  # ===========================================================================
  # Capabilities
  # ===========================================================================
  
  # Required capabilities for low-latency operations
  capability ipc_lock,        # Lock memory (mlock)
  capability net_bind_service, # Bind to privileged ports if needed
  capability sys_nice,        # Set real-time priority
  capability sys_resource,    # Override resource limits
  
  # ===========================================================================
  # Unix Sockets
  # ===========================================================================
  
  # Allow communication via Unix domain sockets
  unix stream connect type=stream,
  unix dgram connect type=dgram,
  
  # ===========================================================================
  # Signal Handling
  # ===========================================================================
  
  # Allow receiving signals for graceful shutdown
  signal (receive) peer=unconfined,
  
  # ===========================================================================
  # Audit Rules
  # ===========================================================================
  
  # Log denied access attempts
  audit deny /** w,
  
}

# =============================================================================
# Local Overrides (Optional)
# =============================================================================
# Place any site-specific overrides in:
# /etc/apparmor.d/local/hft_crypto_bot
# =============================================================================
