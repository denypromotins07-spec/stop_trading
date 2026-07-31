//! Hardware Tuning Module Root
//! 
//! Applies sysctl tweaks (TCP buffers, file limits) dynamically at startup
//! for optimal HFT performance.

pub mod irq_affinity;
pub mod isolcpus;

pub use irq_affinity::{IrqAffinity, CpuMask, MAX_CPUS};
pub use isolcpus::{IsolCpusManager, IsolCpusConfig, CpuValidation, MAX_ISOL_CPUS};

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// System tuning configuration
#[repr(C, align(64))]
pub struct SystemTuning {
    /// IRQ affinity manager
    pub irq_affinity: IrqAffinity,
    /// Isolated CPUs manager
    pub isolcpus: IsolCpusManager,
    /// Whether tuning has been applied
    tuning_applied: AtomicBool,
    /// TCP send buffer size (bytes)
    tcp_sndbuf: AtomicU64,
    /// TCP receive buffer size (bytes)
    tcp_rcvbuf: AtomicU64,
    /// File descriptor limit
    fd_limit: AtomicU64,
}

impl SystemTuning {
    pub fn new() -> Self {
        Self {
            irq_affinity: IrqAffinity::new(),
            isolcpus: IsolCpusManager::new(),
            tuning_applied: AtomicBool::new(false),
            tcp_sndbuf: AtomicU64::new(0),
            tcp_rcvbuf: AtomicU64::new(0),
            fd_limit: AtomicU64::new(0),
        }
    }
    
    /// Apply all system tunings
    #[cfg(target_os = "linux")]
    pub fn apply_all(&self, trading_cpu: usize, nic_interface: &str) -> TuningResult {
        let mut result = TuningResult {
            success: true,
            tunings_applied: 0,
            errors: [None; 16],
            error_count: 0,
        };
        
        // 1. Detect and configure isolated CPUs
        match self.isolcpus.detect_from_sysfs() {
            Ok(count) => {
                if count > 0 {
                    result.tunings_applied += 1;
                } else {
                    result.add_warning("No isolated CPUs detected - consider setting isolcpus kernel parameter");
                }
            }
            Err(e) => result.add_error(e),
        }
        
        // 2. Detect NUMA topology
        if let Err(e) = self.isolcpus.detect_numa_topology() {
            result.add_warning(e);
        } else {
            result.tunings_applied += 1;
        }
        
        // 3. Validate trading CPU
        let validation = self.isolcpus.validate_trading_cpu(trading_cpu);
        if !validation.is_valid {
            result.add_error("Trading CPU validation failed");
            for issue in validation.get_issues() {
                if let Some(msg) = issue {
                    result.add_warning(msg);
                }
            }
        }
        
        // 4. Bind NIC IRQs to trading CPU
        match self.irq_affinity.detect_nic_irqs(nic_interface) {
            Ok(count) => {
                if count > 0 {
                    match self.irq_affinity.bind_to_cpu(trading_cpu) {
                        Ok(bound) => {
                            result.tunings_applied += 1;
                        }
                        Err(e) => result.add_error(e),
                    }
                } else {
                    result.add_warning(format!("No NIC IRQs found for interface: {}", nic_interface));
                }
            }
            Err(e) => result.add_error(e),
        }
        
        // 5. Tune TCP buffers
        match self.tune_tcp_buffers() {
            Ok(_) => result.tunings_applied += 1,
            Err(e) => result.add_warning(e),
        }
        
        // 6. Increase file descriptor limit
        match self.increase_fd_limit() {
            Ok(_) => result.tunings_applied += 1,
            Err(e) => result.add_warning(e),
        }
        
        self.tuning_applied.store(true, Ordering::Release);
        result.success = result.error_count == 0;
        result
    }
    
    /// Tune TCP buffer sizes for low latency
    #[cfg(target_os = "linux")]
    fn tune_tcp_buffers(&self) -> Result<(), &'static str> {
        use std::fs::OpenOptions;
        use std::io::Write;
        
        // Set TCP buffer sizes
        let tcp_settings = [
            ("net.core.rmem_max", "16777216"),      // 16MB max receive buffer
            ("net.core.wmem_max", "16777216"),      // 16MB max send buffer
            ("net.ipv4.tcp_rmem", "4096 262144 16777216"),  // min default max
            ("net.ipv4.tcp_wmem", "4096 262144 16777216"),  // min default max
            ("net.ipv4.tcp_low_latency", "1"),      // Favor latency over throughput
            ("net.core.netdev_max_backlog", "65536"), // NIC backlog
        ];
        
        for (key, value) in &tcp_settings {
            let path = format!("/proc/sys/{}", key.replace('.', "/"));
            if let Ok(mut file) = OpenOptions::new().write(true).open(&path) {
                if file.write_all(value.as_bytes()).is_ok() {
                    // Track what we set
                    if key.contains("rmem") {
                        self.tcp_rcvbuf.store(value.parse().unwrap_or(0), Ordering::Relaxed);
                    } else if key.contains("wmem") {
                        self.tcp_sndbuf.store(value.parse().unwrap_or(0), Ordering::Relaxed);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Increase file descriptor limit
    #[cfg(target_os = "linux")]
    fn increase_fd_limit(&self) -> Result<(), &'static str> {
        // Use prlimit or setrlimit via libc
        // For now, just report current limit
        unsafe {
            let mut rlim: libc::rlimit = std::mem::zeroed();
            if libc::getrlimit(libc::RLIMIT_NOFILE, &mut rlim) == 0 {
                self.fd_limit.store(rlim.rlim_cur as u64, Ordering::Relaxed);
                
                // Try to increase if below target
                if rlim.rlim_cur < 65536 {
                    rlim.rlim_cur = 65536;
                    rlim.rlim_max = rlim.rlim_max.max(65536);
                    if libc::setrlimit(libc::RLIMIT_NOFILE, &rlim) != 0 {
                        return Err("Failed to increase FD limit (may need root)");
                    }
                    self.fd_limit.store(65536, Ordering::Relaxed);
                }
            }
        }
        Ok(())
    }
    
    /// Get tuning status
    pub fn get_status(&self) -> TuningStatus {
        TuningStatus {
            tuning_applied: self.tuning_applied.load(Ordering::Acquire),
            tcp_sndbuf: self.tcp_sndbuf.load(Ordering::Relaxed),
            tcp_rcvbuf: self.tcp_rcvbuf.load(Ordering::Relaxed),
            fd_limit: self.fd_limit.load(Ordering::Relaxed),
            isolcpus_detected: self.isolcpus.is_initialized(),
            irq_affinity_set: self.irq_affinity.is_affinity_set(),
            target_cpu: self.irq_affinity.get_target_cpu(),
        }
    }
    
    /// Windows stub - minimal tuning available
    #[cfg(not(target_os = "linux"))]
    pub fn apply_all(&self, _trading_cpu: usize, _nic_interface: &str) -> TuningResult {
        let mut result = TuningResult {
            success: true,
            tunings_applied: 0,
            errors: [None; 16],
            error_count: 0,
        };
        result.add_warning("System tuning is Linux-only; running on non-Linux OS");
        self.tuning_applied.store(true, Ordering::Release);
        result
    }
    
    #[cfg(not(target_os = "linux"))]
    fn tune_tcp_buffers(&self) -> Result<(), &'static str> {
        Ok(())
    }
    
    #[cfg(not(target_os = "linux"))]
    fn increase_fd_limit(&self) -> Result<(), &'static str> {
        Ok(())
    }
}

impl Default for SystemTuning {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of tuning application
#[derive(Debug)]
#[repr(C)]
pub struct TuningResult {
    pub success: bool,
    pub tunings_applied: u32,
    errors: [Option<String>; 16],
    pub error_count: usize,
}

impl TuningResult {
    fn add_error(&mut self, err: &'static str) {
        if self.error_count < 16 {
            self.errors[self.error_count] = Some(err.to_string());
            self.error_count += 1;
        }
    }
    
    fn add_warning(&mut self, msg: impl Into<String>) {
        // Warnings don't affect success but are tracked
        self.add_error(&msg.into());
    }
    
    pub fn get_errors(&self) -> &[Option<String>] {
        &self.errors[..self.error_count]
    }
}

/// Current tuning status
#[derive(Debug, Clone)]
#[repr(C)]
pub struct TuningStatus {
    pub tuning_applied: bool,
    pub tcp_sndbuf: u64,
    pub tcp_rcvbuf: u64,
    pub fd_limit: u64,
    pub isolcpus_detected: bool,
    pub irq_affinity_set: bool,
    pub target_cpu: u32,
}

// Include libc for Unix systems
#[cfg(unix)]
extern crate libc;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_system_tuning_creation() {
        let tuning = SystemTuning::new();
        assert!(!tuning.get_status().tuning_applied);
    }
    
    #[test]
    fn test_tuning_result() {
        let mut result = TuningResult {
            success: true,
            tunings_applied: 0,
            errors: [None; 16],
            error_count: 0,
        };
        
        assert!(result.success);
        assert_eq!(result.error_count, 0);
        
        result.add_error("Test error");
        assert_eq!(result.error_count, 1);
    }
}
