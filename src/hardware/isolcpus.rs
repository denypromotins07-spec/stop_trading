//! Isolated CPU Detection and Management
//! 
//! Detects and utilizes `isolcpus` kernel parameters to ensure trading threads
//! never suffer OS scheduler preemption. Critical for deterministic latency.
//! 
//! Linux-only with graceful Windows fallbacks.

#[cfg(target_os = "linux")]
use std::fs;

/// Maximum CPUs supported
pub const MAX_ISOL_CPUS: usize = 256;

/// Isolated CPU configuration
#[repr(C, align(64))]
pub struct IsolCpusConfig {
    /// Bitmask of isolated CPUs
    isolated_mask: [u64; MAX_ISOL_CPUS / 64],
    /// Number of isolated CPUs detected
    isolated_count: u32,
    /// Whether isolcpus was detected in kernel cmdline
    isolcpus_detected: bool,
    /// Housekeeping CPUs (non-isolated)
    housekeeping_mask: [u64; MAX_ISOL_CPUS / 64],
    /// NUMA node for each CPU
    cpu_numa_node: [i8; MAX_ISOL_CPUS],
}

impl IsolCpusConfig {
    pub const fn new() -> Self {
        Self {
            isolated_mask: [0; MAX_ISOL_CPUS / 64],
            isolated_count: 0,
            isolcpus_detected: false,
            housekeeping_mask: [0; MAX_ISOL_CPUS / 64],
            cpu_numa_node: [-1; MAX_ISOL_CPUS],
        }
    }
    
    /// Check if a CPU is isolated
    #[inline]
    pub fn is_isolated(&self, cpu: usize) -> bool {
        if cpu >= MAX_ISOL_CPUS {
            return false;
        }
        let idx = cpu / 64;
        let bit = cpu % 64;
        (self.isolated_mask[idx] & (1u64 << bit)) != 0
    }
    
    /// Check if a CPU is housekeeping (scheduler-managed)
    #[inline]
    pub fn is_housekeeping(&self, cpu: usize) -> bool {
        if cpu >= MAX_ISOL_CPUS {
            return false;
        }
        let idx = cpu / 64;
        let bit = cpu % 64;
        (self.housekeeping_mask[idx] & (1u64 << bit)) != 0
    }
    
    /// Get list of isolated CPUs
    pub fn get_isolated_list(&self) -> Vec<usize> {
        let mut list = Vec::with_capacity(self.isolated_count as usize);
        for cpu in 0..MAX_ISOL_CPUS {
            if self.is_isolated(cpu) {
                list.push(cpu);
            }
        }
        list
    }
    
    /// Get NUMA node for a CPU
    pub fn get_cpu_numa(&self, cpu: usize) -> i8 {
        if cpu < MAX_ISOL_CPUS {
            self.cpu_numa_node[cpu]
        } else {
            -1
        }
    }
}

impl Default for IsolCpusConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Isolated CPU manager
#[repr(C, align(64))]
pub struct IsolCpusManager {
    /// Current configuration
    config: IsolCpusConfig,
    /// Whether manager is initialized
    initialized: bool,
}

impl IsolCpusManager {
    pub const fn new() -> Self {
        Self {
            config: IsolCpusConfig::new(),
            initialized: false,
        }
    }
    
    /// Parse isolcpus from kernel command line
    #[cfg(target_os = "linux")]
    pub fn detect_from_cmdline(&mut self) -> Result<bool, &'static str> {
        let cmdline = fs::read_to_string("/proc/cmdline")
            .map_err(|_| "Failed to read /proc/cmdline")?;
        
        self.config.isolcpus_detected = false;
        self.config.isolated_count = 0;
        
        // Reset masks
        self.config.isolated_mask = [0; MAX_ISOL_CPUS / 64];
        self.config.housekeeping_mask = [0; MAX_ISOL_CPUS / 64];
        
        // Look for isolcpus= parameter
        for param in cmdline.split_whitespace() {
            if let Some(value) = param.strip_prefix("isolcpus=") {
                self.config.isolcpus_detected = true;
                
                // Parse CPU list/ranges (e.g., "4-7,12-15" or "nohz_full=4-7")
                for part in value.split(',') {
                    let part = part.trim();
                    if part.contains('-') {
                        // Range like "4-7"
                        let parts: Vec<&str> = part.split('-').collect();
                        if parts.len() == 2 {
                            if let (Ok(start), Ok(end)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                                for cpu in start..=end.min(MAX_ISOL_CPUS - 1) {
                                    self.set_isolated(cpu);
                                }
                            }
                        }
                    } else if let Ok(cpu) = part.parse::<usize>() {
                        self.set_isolated(cpu);
                    }
                }
            }
        }
        
        // Set housekeeping mask (complement of isolated)
        for cpu in 0..MAX_ISOL_CPUS {
            if !self.config.is_isolated(cpu) {
                self.set_housekeeping(cpu);
            }
        }
        
        self.initialized = true;
        Ok(self.config.isolcpus_detected)
    }
    
    /// Detect isolated CPUs from /sys/devices/system/cpu/isolated
    #[cfg(target_os = "linux")]
    pub fn detect_from_sysfs(&mut self) -> Result<usize, &'static str> {
        let isolated = fs::read_to_string("/sys/devices/system/cpu/isolated")
            .map_err(|_| "Failed to read isolated CPUs from sysfs")?;
        
        self.config.isolated_count = 0;
        self.config.isolated_mask = [0; MAX_ISOL_CPUS / 64];
        
        let isolated = isolated.trim();
        if isolated.is_empty() {
            return Ok(0);
        }
        
        for part in isolated.split(',') {
            let part = part.trim();
            if part.contains('-') {
                let parts: Vec<&str> = part.split('-').collect();
                if parts.len() == 2 {
                    if let (Ok(start), Ok(end)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                        for cpu in start..=end.min(MAX_ISOL_CPUS - 1) {
                            self.set_isolated(cpu);
                        }
                    }
                }
            } else if let Ok(cpu) = part.parse::<usize>() {
                self.set_isolated(cpu);
            }
        }
        
        // Update count
        self.config.isolated_count = self.config.get_isolated_list().len() as u32;
        self.initialized = true;
        
        Ok(self.config.isolated_count as usize)
    }
    
    /// Detect NUMA topology
    #[cfg(target_os = "linux")]
    pub fn detect_numa_topology(&mut self) -> Result<(), &'static str> {
        for cpu in 0..MAX_ISOL_CPUS {
            let path = format!("/sys/devices/system/cpu/cpu{}/topology/physical_package_id", cpu);
            if let Ok(content) = fs::read_to_string(&path) {
                if let Ok(node) = content.trim().parse::<i8>() {
                    self.config.cpu_numa_node[cpu] = node;
                }
            }
        }
        Ok(())
    }
    
    /// Mark a CPU as isolated
    fn set_isolated(&mut self, cpu: usize) {
        if cpu < MAX_ISOL_CPUS {
            let idx = cpu / 64;
            let bit = cpu % 64;
            self.config.isolated_mask[idx] |= 1u64 << bit;
            self.config.isolated_count += 1;
        }
    }
    
    /// Mark a CPU as housekeeping
    fn set_housekeeping(&mut self, cpu: usize) {
        if cpu < MAX_ISOL_CPUS {
            let idx = cpu / 64;
            let bit = cpu % 64;
            self.config.housekeeping_mask[idx] |= 1u64 << bit;
        }
    }
    
    /// Validate that a CPU is suitable for real-time trading
    pub fn validate_trading_cpu(&self, cpu: usize) -> CpuValidation {
        let mut validation = CpuValidation {
            cpu,
            is_valid: true,
            is_isolated: false,
            numa_node: -1,
            issues: [None; 8],
            issue_count: 0,
        };
        
        if cpu >= MAX_ISOL_CPUS {
            validation.is_valid = false;
            validation.add_issue("CPU index out of range");
            return validation;
        }
        
        validation.is_isolated = self.config.is_isolated(cpu);
        validation.numa_node = self.config.get_cpu_numa(cpu);
        
        if !validation.is_isolated {
            validation.add_issue("CPU is not isolated - may experience scheduler preemption");
        }
        
        // Additional checks could be added here:
        // - Check if CPU is online
        // - Check governor setting
        // - Check IRQ affinity
        
        validation.is_valid = validation.issue_count == 0;
        validation
    }
    
    /// Get current configuration
    pub fn get_config(&self) -> &IsolCpusConfig {
        &self.config
    }
    
    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
    
    /// Windows stub
    #[cfg(not(target_os = "linux"))]
    pub fn detect_from_cmdline(&mut self) -> Result<bool, &'static str> {
        // Windows doesn't have isolcpus
        self.initialized = true;
        Ok(false)
    }
    
    #[cfg(not(target_os = "linux"))]
    pub fn detect_from_sysfs(&mut self) -> Result<usize, &'static str> {
        Ok(0)
    }
    
    #[cfg(not(target_os = "linux"))]
    pub fn detect_numa_topology(&mut self) -> Result<(), &'static str> {
        Ok(())
    }
}

impl Default for IsolCpusManager {
    fn default() -> Self {
        Self::new()
    }
}

/// CPU validation result
#[derive(Debug, Clone)]
#[repr(C)]
pub struct CpuValidation {
    pub cpu: usize,
    pub is_valid: bool,
    pub is_isolated: bool,
    pub numa_node: i8,
    issues: [Option<&'static str>; 8],
    issue_count: usize,
}

impl CpuValidation {
    fn add_issue(&mut self, issue: &'static str) {
        if self.issue_count < 8 {
            self.issues[self.issue_count] = Some(issue);
            self.issue_count += 1;
        }
    }
    
    pub fn get_issues(&self) -> &[Option<&'static str>] {
        &self.issues[..self.issue_count]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_isol_config_basic() {
        let mut config = IsolCpusConfig::new();
        
        // Manually set some isolated CPUs
        config.isolated_mask[0] = 0b11110000; // CPUs 4-7
        
        assert!(config.is_isolated(4));
        assert!(config.is_isolated(5));
        assert!(config.is_isolated(6));
        assert!(config.is_isolated(7));
        assert!(!config.is_isolated(0));
        assert!(!config.is_isolated(8));
        
        let list = config.get_isolated_list();
        assert_eq!(list, vec![4, 5, 6, 7]);
    }
    
    #[test]
    fn test_manager_creation() {
        let manager = IsolCpusManager::new();
        assert!(!manager.is_initialized());
    }
}
