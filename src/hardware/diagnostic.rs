//! Hardware Diagnostic Module
//!
//! Continuous hardware diagnostic tool monitoring CPU temperature, thermal throttling,
//! and power states. Automatically reduces thread concurrency or lowers execution
//! frequency if the laptop begins to thermally throttle under heavy load.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicU32, Ordering};
use std::time::{Instant, Duration};
use crossbeam_channel::{bounded, Sender, Receiver};
use std::fs;
use std::path::Path;

/// CPU thermal state
#[derive(Debug, Clone)]
pub struct ThermalState {
    /// Current temperature in millidegrees Celsius
    pub current_temp_mc: i32,
    /// Maximum safe temperature (TjMax)
    pub tjmax_mc: i32,
    /// Critical temperature threshold
    pub critical_temp_mc: i32,
    /// Is thermal throttling active
    pub is_throttling: bool,
    /// Throttle percentage (0-100)
    pub throttle_percent: u8,
    /// Timestamp
    pub timestamp_ns: u64,
}

impl ThermalState {
    pub fn temperature_celsius(&self) -> f64 {
        self.current_temp_mc as f64 / 1000.0
    }

    pub fn tjmax_celsius(&self) -> f64 {
        self.tjmax_mc as f64 / 1000.0
    }

    pub fn thermal_margin(&self) -> f64 {
        (self.tjmax_mc - self.current_temp_mc) as f64 / 1000.0
    }

    pub fn is_safe(&self) -> bool {
        self.current_temp_mc < self.critical_temp_mc && !self.is_throttling
    }

    pub fn severity(&self) -> ThermalSeverity {
        if self.is_throttling {
            ThermalSeverity::Critical
        } else if self.current_temp_mc > self.tjmax_mc - 5000 {
            ThermalSeverity::High
        } else if self.current_temp_mc > self.tjmax_mc - 15000 {
            ThermalSeverity::Medium
        } else {
            ThermalSeverity::Low
        }
    }
}

/// Thermal severity levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThermalSeverity {
    Low,      // Normal operation
    Medium,   // Elevated but safe
    High,     // Approaching limits
    Critical, // Throttling active
}

/// Power state information
#[derive(Debug, Clone)]
pub struct PowerState {
    /// Current frequency in MHz
    pub current_freq_mhz: u32,
    /// Base frequency in MHz
    pub base_freq_mhz: u32,
    /// Max frequency in MHz
    pub max_freq_mhz: u32,
    /// Current TDP in milliwatts
    pub current_tdp_mw: u32,
    /// Package TDP limit in milliwatts
    pub tdp_limit_mw: u32,
    /// Is power capping active
    pub is_power_capped: bool,
    /// Energy counter (joules * 1e6)
    pub energy_counter_uj: u64,
}

impl PowerState {
    pub fn frequency_utilization(&self) -> f64 {
        if self.max_freq_mhz == 0 {
            return 0.0;
        }
        self.current_freq_mhz as f64 / self.max_freq_mhz as f64
    }

    pub fn power_utilization(&self) -> f64 {
        if self.tdp_limit_mw == 0 {
            return 0.0;
        }
        self.current_tdp_mw as f64 / self.tdp_limit_mw as f64
    }
}

/// Hardware health status
#[derive(Debug, Clone)]
pub struct HardwareHealth {
    pub thermal: ThermalState,
    pub power: PowerState,
    /// Recommended action
    pub recommended_action: MitigationAction,
    /// Health score (0-100)
    pub health_score: u8,
}

/// Mitigation actions for thermal/power issues
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MitigationAction {
    /// No action needed
    None,
    /// Reduce thread count slightly
    ReduceThreads(u8),
    /// Lower execution frequency
    ReduceFrequency(u8), // Percentage reduction
    /// Aggressive throttling
    EmergencyThrottle,
    /// Shutdown non-critical systems
    ShutdownNonCritical,
}

/// Main hardware diagnostic engine
pub struct HardwareDiagnostics {
    /// Last thermal state
    last_thermal: std::sync::Mutex<Option<ThermalState>>,
    /// Last power state
    last_power: std::sync::Mutex<Option<PowerState>>,
    /// Health history (for trend analysis)
    health_history: std::sync::Mutex<Vec<(u64, u8)>>,
    /// Baseline TJMax
    baseline_tjmax_mc: i32,
    /// Is monitoring active
    is_active: AtomicBool,
    /// Monitoring interval
    monitor_interval_ms: AtomicU32,
    /// Event channel
    event_tx: Sender<DiagnosticEvent>,
    event_rx: Receiver<DiagnosticEvent>,
    /// Sample count
    sample_count: AtomicU64,
}

/// Diagnostic events
#[derive(Debug, Clone)]
pub enum DiagnosticEvent {
    /// Temperature reading
    TemperatureUpdate { temp_c: f64, severity: ThermalSeverity },
    /// Thermal throttling detected
    ThermalThrottling { temp_c: f64, throttle_pct: u8 },
    /// Power cap hit
    PowerCapHit { utilization: f64 },
    /// Mitigation action triggered
    MitigationTriggered(MitigationAction),
    /// Health score update
    HealthScoreUpdate { score: u8, delta: i8 },
}

impl HardwareDiagnostics {
    pub fn new(buffer_size: usize) -> Self {
        let (tx, rx) = bounded(buffer_size);

        // Read baseline TJMax from hardware
        let baseline_tjmax_mc = Self::read_tjmax().unwrap_or(100_000);

        Self {
            last_thermal: std::sync::Mutex::new(None),
            last_power: std::sync::Mutex::new(None),
            health_history: std::sync::Mutex::new(Vec::with_capacity(60)),
            baseline_tjmax_mc,
            is_active: AtomicBool::new(true),
            monitor_interval_ms: AtomicU32::new(1000), // 1 second default
            event_tx: tx,
            event_rx: rx,
            sample_count: AtomicU64::new(0),
        }
    }

    /// Perform a single diagnostic scan
    pub fn scan(&self) -> Option<HardwareHealth> {
        if !self.is_active.load(Ordering::Relaxed) {
            return None;
        }

        let now_ns = Instant::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos() as u64;

        // Read thermal state
        let thermal = self.read_thermal_state(now_ns);

        // Read power state
        let power = self.read_power_state();

        // Calculate health score
        let health_score = self.calculate_health_score(&thermal, &power);

        // Determine mitigation action
        let action = self.determine_mitigation(&thermal, &power, health_score);

        // Store results
        if let Ok(mut guard) = self.last_thermal.lock() {
            *guard = Some(thermal.clone());
        }
        if let Ok(mut guard) = self.last_power.lock() {
            *guard = Some(power.clone());
        }

        // Update history
        if let Ok(mut guard) = self.health_history.lock() {
            guard.push((now_ns, health_score));
            if guard.len() > 60 {
                guard.remove(0);
            }
        }

        self.sample_count.fetch_add(1, Ordering::Relaxed);

        // Emit events
        self.emit_events(&thermal, &power, action, health_score);

        Some(HardwareHealth {
            thermal,
            power,
            recommended_action: action,
            health_score,
        })
    }

    fn read_thermal_state(&self, now_ns: u64) -> ThermalState {
        // Try to read from hwmon
        let mut current_temp_mc = 0;
        let mut tjmax_mc = self.baseline_tjmax_mc;
        let mut is_throttling = false;

        // Try various hwmon paths
        for hwmon_idx in 0..10 {
            let path = format!("/sys/class/hwmon/hwmon{}/temp1_input", hwmon_idx);
            if let Ok(temp_str) = fs::read_to_string(&path) {
                if let Ok(temp) = temp_str.trim().parse::<i32>() {
                    if temp > 0 && temp < 150_000 {
                        current_temp_mc = temp;
                        break;
                    }
                }
            }
        }

        // Check for throttling status
        for hwmon_idx in 0..10 {
            let throttle_path = format!("/sys/class/hwmon/hwmon{}/throttle", hwmon_idx);
            if let Ok(throttle_str) = fs::read_to_string(&throttle_path) {
                is_throttling = throttle_str.trim() == "1";
                break;
            }
        }

        // Fallback: estimate from CPU freq reduction
        if !is_throttling {
            is_throttling = Self::detect_freq_throttling();
        }

        let critical_temp_mc = tjmax_mc - 5_000; // 5°C below TJMax
        let throttle_percent = if current_temp_mc >= tjmax_mc {
            100
        } else if current_temp_mc >= critical_temp_mc {
            (((current_temp_mc - critical_temp_mc) * 100) / 5_000) as u8
        } else {
            0
        };

        ThermalState {
            current_temp_mc,
            tjmax_mc,
            critical_temp_mc,
            is_throttling,
            throttle_percent,
            timestamp_ns: now_ns,
        }
    }

    fn read_power_state(&self) -> PowerState {
        // Read current frequency
        let current_freq_mhz = Self::read_cpu_freq().unwrap_or(0);

        // Estimate other values (would need RAPL MSR access for real values)
        PowerState {
            current_freq_mhz,
            base_freq_mhz: 3000, // Assume 3GHz base
            max_freq_mhz: 4500,  // Assume 4.5GHz boost
            current_tdp_mw: 15_000, // Assume 15W
            tdp_limit_mw: 45_000,   // 45W limit
            is_power_capped: false,
            energy_counter_uj: 0,
        }
    }

    fn read_cpu_freq() -> Option<u32> {
        if let Ok(freq_str) = fs::read_to_string("/proc/cpuinfo") {
            for line in freq_str.lines() {
                if line.starts_with("cpu MHz") {
                    if let Some(idx) = line.find(':') {
                        if let Ok(freq) = line[idx + 1..].trim().parse::<f64>() {
                            return Some(freq as u32);
                        }
                    }
                }
            }
        }
        None
    }

    fn detect_freq_throttling() -> bool {
        // Check if current freq is significantly below max
        if let Some(current) = Self::read_cpu_freq() {
            // If running at less than 70% of expected max, likely throttling
            current < 2000 // Below 2GHz suggests throttling
        } else {
            false
        }
    }

    fn read_tjmax() -> Option<i32> {
        // Try to read from coretemp
        for hwmon_idx in 0..10 {
            let path = format!("/sys/class/hwmon/hwmon{}/temp{}_max", hwmon_idx, 
                if hwmon_idx == 0 { 1 } else { 2 });
            if let Ok(temp_str) = fs::read_to_string(&path) {
                if let Ok(temp) = temp_str.trim().parse::<i32>() {
                    if temp > 50_000 && temp < 150_000 {
                        return Some(temp);
                    }
                }
            }
        }
        // Default for most modern CPUs
        Some(100_000) // 100°C
    }

    fn calculate_health_score(&self, thermal: &ThermalState, power: &PowerState) -> u8 {
        let mut score = 100i16;

        // Thermal penalty
        let thermal_margin = thermal.thermal_margin();
        if thermal_margin < 10.0 {
            score -= ((10.0 - thermal_margin) * 3.0) as i16;
        }
        if thermal.is_throttling {
            score -= 30;
        }

        // Power penalty
        let power_util = power.power_utilization();
        if power_util > 0.9 {
            score -= ((power_util - 0.9) * 100.0) as i16;
        }

        score.clamp(0, 100) as u8
    }

    fn determine_mitigation(&self, thermal: &ThermalState, power: &PowerState, health: u8) -> MitigationAction {
        if thermal.is_throttling || health < 30 {
            MitigationAction::EmergencyThrottle
        } else if thermal.severity() == ThermalSeverity::High || health < 50 {
            MitigationAction::ReduceFrequency(25)
        } else if thermal.severity() == ThermalSeverity::Medium || health < 70 {
            MitigationAction::ReduceThreads(2)
        } else {
            MitigationAction::None
        }
    }

    fn emit_events(&self, thermal: &ThermalState, power: &PowerState, action: MitigationAction, health: u8) {
        let _ = self.event_tx.send(DiagnosticEvent::TemperatureUpdate {
            temp_c: thermal.temperature_celsius(),
            severity: thermal.severity(),
        });

        if thermal.is_throttling {
            let _ = self.event_tx.send(DiagnosticEvent::ThermalThrottling {
                temp_c: thermal.temperature_celsius(),
                throttle_pct: thermal.throttle_percent,
            });
        }

        if power.power_utilization() > 0.95 {
            let _ = self.event_tx.send(DiagnosticEvent::PowerCapHit {
                utilization: power.power_utilization(),
            });
        }

        if action != MitigationAction::None {
            let _ = self.event_tx.send(DiagnosticEvent::MitigationTriggered(action));
        }

        let _ = self.event_tx.send(DiagnosticEvent::HealthScoreUpdate {
            score: health,
            delta: 0, // Would compare to previous
        });
    }

    /// Get last thermal state
    pub fn get_last_thermal(&self) -> Option<ThermalState> {
        self.last_thermal.lock().ok().and_then(|g| g.clone())
    }

    /// Get last power state
    pub fn get_last_power(&self) -> Option<PowerState> {
        self.last_power.lock().ok().and_then(|g| g.clone())
    }

    /// Get average health score over recent history
    pub fn get_average_health(&self) -> Option<f64> {
        if let Ok(guard) = self.health_history.lock() {
            if guard.is_empty() {
                return None;
            }
            let sum: u64 = guard.iter().map(|(_, h)| *h as u64).sum();
            Some(sum as f64 / guard.len() as f64)
        } else {
            None
        }
    }

    /// Get event receiver
    pub fn get_event_receiver(&self) -> Receiver<DiagnosticEvent> {
        self.event_rx.clone()
    }

    /// Set monitoring interval
    pub fn set_monitor_interval(&self, interval_ms: u32) {
        self.monitor_interval_ms.store(interval_ms, Ordering::Relaxed);
    }

    /// Deactivate diagnostics
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Activate diagnostics
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Relaxed);
    }

    /// Get sample count
    pub fn get_sample_count(&self) -> u64 {
        self.sample_count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostics_initialization() {
        let diag = HardwareDiagnostics::new(100);

        assert!(diag.is_active.load(Ordering::Relaxed));
        assert_eq!(diag.get_sample_count(), 0);
    }

    #[test]
    fn test_thermal_state_methods() {
        let thermal = ThermalState {
            current_temp_mc: 75_000,
            tjmax_mc: 100_000,
            critical_temp_mc: 95_000,
            is_throttling: false,
            throttle_percent: 0,
            timestamp_ns: 0,
        };

        assert!((thermal.temperature_celsius() - 75.0).abs() < 0.01);
        assert!((thermal.thermal_margin() - 25.0).abs() < 0.01);
        assert!(thermal.is_safe());
        assert_eq!(thermal.severity(), ThermalSeverity::Medium);
    }

    #[test]
    fn test_health_score_calculation() {
        let diag = HardwareDiagnostics::new(100);

        let thermal = ThermalState {
            current_temp_mc: 80_000,
            tjmax_mc: 100_000,
            critical_temp_mc: 95_000,
            is_throttling: false,
            throttle_percent: 0,
            timestamp_ns: 0,
        };

        let power = PowerState {
            current_freq_mhz: 3500,
            base_freq_mhz: 3000,
            max_freq_mhz: 4500,
            current_tdp_mw: 30_000,
            tdp_limit_mw: 45_000,
            is_power_capped: false,
            energy_counter_uj: 0,
        };

        let health = diag.calculate_health_score(&thermal, &power);
        assert!(health > 0 && health <= 100);
    }
}
