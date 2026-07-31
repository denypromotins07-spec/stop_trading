//! IRQ Affinity Management
//! 
//! Programmatically binds network card (NIC) hardware interrupts to specific
//! isolated CPU cores on AMD Ryzen. Ensures network packets are processed on
//! the exact same NUMA node and core as the trading thread.
//! 
//! Linux-only with graceful Windows fallbacks.

#[cfg(target_os = "linux")]
use std::fs::{File, OpenOptions};
#[cfg(target_os = "linux")]
use std::io::{Read, Write};
#[cfg(target_os = "linux")]
use std::path::Path;

/// Maximum number of CPUs supported
pub const MAX_CPUS: usize = 256;

/// CPU affinity mask (bitmask for CPU selection)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CpuMask {
    bits: [u64; MAX_CPUS / 64],
}

impl CpuMask {
    pub const fn empty() -> Self {
        Self {
            bits: [0; MAX_CPUS / 64],
        }
    }
    
    /// Set a specific CPU in the mask
    #[inline]
    pub fn set_cpu(&mut self, cpu: usize) {
        if cpu < MAX_CPUS {
            let idx = cpu / 64;
            let bit = cpu % 64;
            self.bits[idx] |= 1u64 << bit;
        }
    }
    
    /// Clear a specific CPU from the mask
    #[inline]
    pub fn clear_cpu(&mut self, cpu: usize) {
        if cpu < MAX_CPUS {
            let idx = cpu / 64;
            let bit = cpu % 64;
            self.bits[idx] &= !(1u64 << bit);
        }
    }
    
    /// Check if CPU is in mask
    #[inline]
    pub fn has_cpu(&self, cpu: usize) -> bool {
        if cpu >= MAX_CPUS {
            return false;
        }
        let idx = cpu / 64;
        let bit = cpu % 64;
        (self.bits[idx] & (1u64 << bit)) != 0
    }
    
    /// Convert to hex string for /proc/irq write
    #[cfg(target_os = "linux")]
    pub fn to_hex(&self) -> String {
        let mut s = String::new();
        for i in (0..self.bits.len()).rev() {
            if i < self.bits.len() - 1 || self.bits[i] != 0 {
                s.push_str(&format!("{:016x}", self.bits[i]));
            }
        }
        if s.is_empty() {
            s.push('0');
        }
        s
    }
    
    /// Create mask for single CPU
    pub fn single(cpu: usize) -> Self {
        let mut mask = Self::empty();
        mask.set_cpu(cpu);
        mask
    }
    
    /// Create mask for CPU range
    pub fn range(start: usize, end: usize) -> Self {
        let mut mask = Self::empty();
        for cpu in start..=end.min(MAX_CPUS - 1) {
            mask.set_cpu(cpu);
        }
        mask
    }
}

impl Default for CpuMask {
    fn default() -> Self {
        Self::empty()
    }
}

/// IRQ affinity manager for Linux systems
#[repr(C, align(64))]
pub struct IrqAffinity {
    /// Detected NIC IRQs
    nic_irqs: [i32; 32],
    /// Number of detected NIC IRQs
    nic_irq_count: usize,
    /// Target CPU for NIC interrupts
    target_cpu: u32,
    /// Whether affinity was successfully set
    affinity_set: bool,
    /// Original affinity masks (for restoration)
    original_masks: [CpuMask; 32],
}

impl IrqAffinity {
    pub const fn new() -> Self {
        Self {
            nic_irqs: [-1; 32],
            nic_irq_count: 0,
            target_cpu: 0,
            affinity_set: false,
            original_masks: [CpuMask::empty(); 32],
        }
    }
    
    /// Detect network interface IRQs
    #[cfg(target_os = "linux")]
    pub fn detect_nic_irqs(&mut self, interface: &str) -> Result<usize, &'static str> {
        self.nic_irq_count = 0;
        
        // Read /proc/interrupts to find NIC IRQs
        let content = std::fs::read_to_string("/proc/interrupts")
            .map_err(|_| "Failed to read /proc/interrupts")?;
        
        for line in content.lines() {
            let parts: Vec<&str> = line.split(':').collect();
            if parts.is_empty() {
                continue;
            }
            
            let irq_num: i32 = match parts[0].trim().parse() {
                Ok(n) => n,
                Err(_) => continue,
            };
            
            // Check if this is a NIC interrupt
            if line.contains(interface) || line.contains("eth") || line.contains("enp") {
                if self.nic_irq_count < 32 {
                    self.nic_irqs[self.nic_irq_count] = irq_num;
                    self.nic_irq_count += 1;
                }
            }
        }
        
        Ok(self.nic_irq_count)
    }
    
    /// Get current IRQ affinity mask
    #[cfg(target_os = "linux")]
    pub fn get_irq_affinity(&self, irq: i32) -> Result<CpuMask, &'static str> {
        let path = format!("/proc/irq/{}/smp_affinity", irq);
        let mut file = File::open(&path)
            .map_err(|_| "Failed to open affinity file")?;
        
        let mut content = String::new();
        file.read_to_string(&mut content)
            .map_err(|_| "Failed to read affinity")?;
        
        // Parse hex mask
        let hex = content.trim().trim_start_matches("0x");
        let mut mask = CpuMask::empty();
        
        // Parse from right to left (little endian)
        let chars: Vec<char> = hex.chars().rev().collect();
        for (i, chunk) in chars.chunks(16).enumerate() {
            let chunk_str: String = chunk.iter().rev().collect();
            if let Ok(val) = u64::from_str_radix(&chunk_str, 16) {
                mask.bits[i] = val;
            }
        }
        
        Ok(mask)
    }
    
    /// Set IRQ affinity mask
    #[cfg(target_os = "linux")]
    pub fn set_irq_affinity(&mut self, irq: i32, mask: &CpuMask) -> Result<(), &'static str> {
        let path = format!("/proc/irq/{}/smp_affinity", irq);
        
        // Save original mask
        if let Ok(orig) = self.get_irq_affinity(irq) {
            for i in 0..32 {
                if self.nic_irqs[i] == irq {
                    self.original_masks[i] = orig;
                    break;
                }
            }
        }
        
        // Write new mask
        let mut file = OpenOptions::new()
            .write(true)
            .open(&path)
            .map_err(|_| "Failed to open affinity file for writing")?;
        
        file.write_all(mask.to_hex().as_bytes())
            .map_err(|_| "Failed to write affinity mask")?;
        
        self.affinity_set = true;
        Ok(())
    }
    
    /// Bind all NIC IRQs to specific CPU
    #[cfg(target_os = "linux")]
    pub fn bind_to_cpu(&mut self, cpu: usize) -> Result<usize, &'static str> {
        if cpu >= MAX_CPUS {
            return Err("CPU index out of range");
        }
        
        self.target_cpu = cpu as u32;
        let mask = CpuMask::single(cpu);
        
        let mut bound = 0usize;
        for i in 0..self.nic_irq_count {
            let irq = self.nic_irqs[i];
            if irq >= 0 {
                if self.set_irq_affinity(irq, &mask).is_ok() {
                    bound += 1;
                }
            }
        }
        
        Ok(bound)
    }
    
    /// Restore original IRQ affinities
    #[cfg(target_os = "linux")]
    pub fn restore_original(&self) -> Result<(), &'static str> {
        for i in 0..self.nic_irq_count {
            let irq = self.nic_irqs[i];
            if irq >= 0 {
                let path = format!("/proc/irq/{}/smp_affinity", irq);
                let mut file = OpenOptions::new()
                    .write(true)
                    .open(&path)
                    .map_err(|_| "Failed to open affinity file")?;
                
                file.write_all(self.original_masks[i].to_hex().as_bytes())
                    .map_err(|_| "Failed to write affinity mask")?;
            }
        }
        Ok(())
    }
    
    /// Get detected NIC IRQs
    pub fn get_nic_irqs(&self) -> &[i32] {
        &self.nic_irqs[..self.nic_irq_count]
    }
    
    /// Check if affinity was set
    pub fn is_affinity_set(&self) -> bool {
        self.affinity_set
    }
    
    /// Get target CPU
    pub fn get_target_cpu(&self) -> u32 {
        self.target_cpu
    }
    
    /// Windows stub - returns success but does nothing
    #[cfg(not(target_os = "linux"))]
    pub fn detect_nic_irqs(&mut self, _interface: &str) -> Result<usize, &'static str> {
        // Windows would require different API (SetupAPI, etc.)
        // For now, return empty result gracefully
        Ok(0)
    }
    
    #[cfg(not(target_os = "linux"))]
    pub fn bind_to_cpu(&mut self, _cpu: usize) -> Result<usize, &'static str> {
        // Windows would use SetThreadAffinityMask
        Ok(0)
    }
    
    #[cfg(not(target_os = "linux"))]
    pub fn restore_original(&self) -> Result<(), &'static str> {
        Ok(())
    }
}

impl Default for IrqAffinity {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_mask_single() {
        let mask = CpuMask::single(4);
        assert!(mask.has_cpu(4));
        assert!(!mask.has_cpu(3));
        assert!(!mask.has_cpu(5));
    }
    
    #[test]
    fn test_cpu_mask_range() {
        let mask = CpuMask::range(2, 5);
        assert!(mask.has_cpu(2));
        assert!(mask.has_cpu(3));
        assert!(mask.has_cpu(4));
        assert!(mask.has_cpu(5));
        assert!(!mask.has_cpu(1));
        assert!(!mask.has_cpu(6));
    }
    
    #[test]
    fn test_cpu_mask_operations() {
        let mut mask = CpuMask::empty();
        mask.set_cpu(0);
        mask.set_cpu(64);
        mask.set_cpu(128);
        
        assert!(mask.has_cpu(0));
        assert!(mask.has_cpu(64));
        assert!(mask.has_cpu(128));
        
        mask.clear_cpu(64);
        assert!(!mask.has_cpu(64));
    }
    
    #[test]
    fn test_irq_affinity_creation() {
        let affinity = IrqAffinity::new();
        assert_eq!(affinity.nic_irq_count, 0);
        assert!(!affinity.is_affinity_set());
    }
}
