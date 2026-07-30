//! Ensemble Weighting Engine
//! 
//! Dynamically blends Rust quantitative signals with Python deep learning predictions.
//! Uses confidence scores from SOUL.md to scale position sizing via Kelly Criterion.

use std::{
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, RwLock,
    },
};

/// Cache-line padding constant
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of ensemble members
pub const MAX_ENSEMBLE_MEMBERS: usize = 16;

/// Ensemble member configuration
#[derive(Debug, Clone)]
pub struct EnsembleMember {
    /// Member ID
    pub id: String,
    /// Member type (Rust or Python)
    pub member_type: MemberType,
    /// Current weight (0.0 to 1.0)
    pub weight: f32,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Historical win rate
    pub win_rate: f32,
    /// Average return per trade
    pub avg_return: f32,
    /// Sharpe ratio
    pub sharpe_ratio: f32,
}

/// Member type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemberType {
    /// Rust SMC engine
    RustSMC,
    /// Rust mean reversion
    RustMeanReversion,
    /// Rust momentum
    RustMomentum,
    /// Python LSTM ensemble
    PythonLSTM,
    /// Python Transformer
    PythonTransformer,
    /// Python gradient boosting
    PythonXGBoost,
}

impl MemberType {
    pub fn is_rust(&self) -> bool {
        matches!(self, Self::RustSMC | Self::RustMeanReversion | Self::RustMomentum)
    }

    pub fn is_python(&self) -> bool {
        !self.is_rust()
    }
}

/// Blended signal output from ensemble
#[derive(Debug, Clone)]
pub struct BlendedSignal {
    /// Symbol
    pub symbol: String,
    /// Blended direction (-1.0 to 1.0)
    pub direction: f32,
    /// Combined confidence (0.0 to 1.0)
    pub confidence: f32,
    /// Kelly-optimal position size (0.0 to 1.0)
    pub kelly_position: f32,
    /// Fractional Kelly (for risk management)
    pub fractional_kelly: f32,
    /// Timestamp
    pub timestamp_ns: u64,
    /// Contributing members
    pub member_contributions: Vec<(String, f32)>,
}

impl BlendedSignal {
    /// Get the recommended action
    pub fn get_action(&self) -> Action {
        if self.direction > 0.15 {
            Action::Long
        } else if self.direction < -0.15 {
            Action::Short
        } else {
            Action::Flat
        }
    }
}

/// Trading action recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Long,
    Short,
    Flat,
}

/// Ensemble weighting engine
pub struct EnsembleEngine {
    /// Ensemble members
    members: Arc<RwLock<Vec<EnsembleMember>>>,
    /// Global confidence multiplier
    confidence_multiplier: Arc<RwLock<f32>>,
    /// Kelly fraction (typically 0.25 for quarter-Kelly)
    kelly_fraction: Arc<RwLock<f32>>,
    /// Total blending operations
    blend_count: Arc<AtomicU64>,
    /// Last update timestamp
    last_update_ns: Arc<AtomicU64>,
}

unsafe impl Send for EnsembleEngine {}
unsafe impl Sync for EnsembleEngine {}

impl EnsembleEngine {
    /// Create a new ensemble engine
    pub fn new() -> Self {
        Self {
            members: Arc::new(RwLock::new(Vec::new())),
            confidence_multiplier: Arc::new(RwLock::new(1.0)),
            kelly_fraction: Arc::new(RwLock::new(0.25)), // Quarter Kelly by default
            blend_count: Arc::new(AtomicU64::new(0)),
            last_update_ns: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Add an ensemble member
    pub fn add_member(&self, member: EnsembleMember) {
        if let Ok(mut members) = self.members.write() {
            if members.len() < MAX_ENSEMBLE_MEMBERS {
                members.push(member);
            }
        }
    }

    /// Update member weights dynamically
    pub fn update_weights(&self, updates: &[(String, f32)]) {
        if let Ok(mut members) = self.members.write() {
            for (id, new_weight) in updates {
                for member in members.iter_mut() {
                    if member.id == *id {
                        member.weight = new_weight.clamp(0.0, 1.0);
                    }
                }
            }

            // Normalize weights to sum to 1.0
            let total: f32 = members.iter().map(|m| m.weight).sum();
            if total > 0.0 {
                for member in members.iter_mut() {
                    member.weight /= total;
                }
            }
        }
    }

    /// Update member confidence from SOUL feedback
    pub fn update_confidence(&self, member_id: &str, confidence: f32, win_rate: f32) {
        if let Ok(mut members) = self.members.write() {
            for member in members.iter_mut() {
                if member.id == member_id {
                    member.confidence = confidence.clamp(0.0, 1.0);
                    member.win_rate = win_rate.clamp(0.0, 1.0);
                }
            }
        }
    }

    /// Blend signals from all ensemble members
    pub fn blend_signals(
        &self,
        symbol: &str,
        individual_signals: &[(String, f32)], // (member_id, signal)
    ) -> Option<BlendedSignal> {
        let members = self.members.read().ok()?;
        let confidence_mult = *self.confidence_multiplier.read().ok()?;
        let kelly_frac = *self.kelly_fraction.read().ok()?;

        if members.is_empty() || individual_signals.is_empty() {
            return None;
        }

        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        let mut weighted_confidence = 0.0;
        let mut contributions = Vec::new();

        // Calculate weighted average of signals
        for (member_id, signal) in individual_signals {
            if let Some(member) = members.iter().find(|m| m.id == *member_id) {
                let effective_weight = member.weight * member.confidence * confidence_mult;
                weighted_sum += signal * effective_weight;
                total_weight += effective_weight;
                weighted_confidence += member.confidence * effective_weight;
                contributions.push((member_id.clone(), signal * effective_weight));
            }
        }

        if total_weight == 0.0 {
            return None;
        }

        let direction = weighted_sum / total_weight;
        let confidence = (weighted_confidence / total_weight).clamp(0.0, 1.0);

        // Calculate Kelly Criterion position size
        // Kelly % = W - [(1-W) / R] where W = win rate, R = win/loss ratio
        let avg_win_rate = members.iter().map(|m| m.win_rate).sum::<f32>() / members.len() as f32;
        let avg_return = members.iter().map(|m| m.avg_return.abs()).sum::<f32>() / members.len() as f32;
        
        let kelly_pct = if avg_return > 0.0 {
            avg_win_rate - ((1.0 - avg_win_rate) / avg_return)
        } else {
            0.0
        };

        let kelly_position = kelly_pct.max(0.0).min(1.0);
        let fractional_kelly = kelly_position * kelly_frac;

        // Apply confidence scaling to position size
        let final_position = fractional_kelly * confidence;

        self.blend_count.fetch_add(1, Ordering::Relaxed);
        self.last_update_ns.store(get_timestamp_ns(), Ordering::Release);

        Some(BlendedSignal {
            symbol: symbol.to_string(),
            direction,
            confidence,
            kelly_position,
            fractional_kelly,
            timestamp_ns: get_timestamp_ns(),
            member_contributions: contributions,
        })
    }

    /// Set global confidence multiplier (from external factors)
    pub fn set_confidence_multiplier(&self, multiplier: f32) {
        if let Ok(mut cm) = self.confidence_multiplier.write() {
            *cm = multiplier.clamp(0.0, 2.0);
        }
    }

    /// Set Kelly fraction for risk management
    pub fn set_kelly_fraction(&self, fraction: f32) {
        if let Ok(mut kf) = self.kelly_fraction.write() {
            *kf = fraction.clamp(0.0, 1.0);
        }
    }

    /// Get ensemble statistics
    pub fn get_stats(&self) -> EnsembleStats {
        let members = self.members.read().unwrap_or_else(|e| e.into_inner());
        
        let total_members = members.len();
        let rust_count = members.iter().filter(|m| m.member_type.is_rust()).count();
        let python_count = members.iter().filter(|m| m.member_type.is_python()).count();
        
        let avg_weight = if total_members > 0 {
            members.iter().map(|m| m.weight).sum::<f32>() / total_members as f32
        } else {
            0.0
        };
        
        let avg_confidence = if total_members > 0 {
            members.iter().map(|m| m.confidence).sum::<f32>() / total_members as f32
        } else {
            0.0
        };

        EnsembleStats {
            total_members,
            rust_count,
            python_count,
            avg_weight,
            avg_confidence,
            blend_count: self.blend_count.load(Ordering::Relaxed),
        }
    }

    /// Get member by ID
    pub fn get_member(&self, member_id: &str) -> Option<EnsembleMember> {
        let members = self.members.read().ok()?;
        members.iter().find(|m| m.id == member_id).cloned()
    }

    /// Remove a member
    pub fn remove_member(&self, member_id: &str) -> bool {
        if let Ok(mut members) = self.members.write() {
            let initial_len = members.len();
            members.retain(|m| m.id != member_id);
            return members.len() < initial_len;
        }
        false
    }
}

impl Default for EnsembleEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Ensemble statistics
#[derive(Debug, Clone)]
pub struct EnsembleStats {
    pub total_members: usize,
    pub rust_count: usize,
    pub python_count: usize,
    pub avg_weight: f32,
    pub avg_confidence: f32,
    pub blend_count: u64,
}

/// Get current timestamp in nanoseconds
fn get_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Kelly Criterion calculator
pub struct KellyCalculator {
    /// Win rate
    win_rate: f32,
    /// Average win amount
    avg_win: f32,
    /// Average loss amount
    avg_loss: f32,
}

impl KellyCalculator {
    pub fn new(win_rate: f32, avg_win: f32, avg_loss: f32) -> Self {
        Self {
            win_rate: win_rate.clamp(0.0, 1.0),
            avg_win: avg_win.abs(),
            avg_loss: avg_loss.abs(),
        }
    }

    /// Calculate full Kelly percentage
    pub fn kelly_percentage(&self) -> f32 {
        if self.avg_loss == 0.0 {
            return 1.0;
        }

        let win_loss_ratio = self.avg_win / self.avg_loss;
        let kelly = self.win_rate - ((1.0 - self.win_rate) / win_loss_ratio);
        kelly.clamp(0.0, 1.0)
    }

    /// Calculate fractional Kelly (safer)
    pub fn fractional_kelly(&self, fraction: f32) -> f32 {
        self.kelly_percentage() * fraction.clamp(0.0, 1.0)
    }

    /// Calculate optimal position size given account value
    pub fn position_size(&self, account_value: f64, fraction: f32) -> f64 {
        account_value * self.fractional_kelly(fraction) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_engine_creation() {
        let engine = EnsembleEngine::new();
        let stats = engine.get_stats();
        
        assert_eq!(stats.total_members, 0);
        assert_eq!(stats.rust_count, 0);
        assert_eq!(stats.python_count, 0);
    }

    #[test]
    fn test_add_members() {
        let engine = EnsembleEngine::new();
        
        engine.add_member(EnsembleMember {
            id: "rust_smc".to_string(),
            member_type: MemberType::RustSMC,
            weight: 0.4,
            confidence: 0.85,
            win_rate: 0.65,
            avg_return: 0.02,
            sharpe_ratio: 1.5,
        });
        
        engine.add_member(EnsembleMember {
            id: "python_lstm".to_string(),
            member_type: MemberType::PythonLSTM,
            weight: 0.6,
            confidence: 0.78,
            win_rate: 0.58,
            avg_return: 0.015,
            sharpe_ratio: 1.2,
        });
        
        let stats = engine.get_stats();
        assert_eq!(stats.total_members, 2);
        assert_eq!(stats.rust_count, 1);
        assert_eq!(stats.python_count, 1);
    }

    #[test]
    fn test_signal_blending() {
        let engine = EnsembleEngine::new();
        
        engine.add_member(EnsembleMember {
            id: "smc".to_string(),
            member_type: MemberType::RustSMC,
            weight: 0.5,
            confidence: 0.9,
            win_rate: 0.7,
            avg_return: 0.03,
            sharpe_ratio: 2.0,
        });
        
        engine.add_member(EnsembleMember {
            id: "ml".to_string(),
            member_type: MemberType::PythonLSTM,
            weight: 0.5,
            confidence: 0.8,
            win_rate: 0.6,
            avg_return: 0.02,
            sharpe_ratio: 1.5,
        });
        
        let signals = vec![
            ("smc".to_string(), 0.8),  // Strong long
            ("ml".to_string(), 0.4),   // Moderate long
        ];
        
        let blended = engine.blend_signals("BTCUSDT", &signals).unwrap();
        
        assert!(blended.direction > 0.0);
        assert!(blended.confidence > 0.8);
        assert_eq!(blended.get_action(), Action::Long);
    }

    #[test]
    fn test_kelly_calculator() {
        let calc = KellyCalculator::new(0.6, 100.0, 50.0);
        
        let kelly = calc.kelly_percentage();
        // Kelly = 0.6 - (0.4 / 2) = 0.6 - 0.2 = 0.4
        assert!((kelly - 0.4).abs() < 0.01);
        
        let fractional = calc.fractional_kelly(0.5);
        assert!((fractional - 0.2).abs() < 0.01);
        
        let position = calc.position_size(100000.0, 0.5);
        assert!((position - 20000.0).abs() < 1.0);
    }

    #[test]
    fn test_weight_normalization() {
        let engine = EnsembleEngine::new();
        
        engine.add_member(EnsembleMember {
            id: "m1".to_string(),
            member_type: MemberType::RustSMC,
            weight: 0.8,
            confidence: 0.9,
            win_rate: 0.6,
            avg_return: 0.02,
            sharpe_ratio: 1.0,
        });
        
        engine.add_member(EnsembleMember {
            id: "m2".to_string(),
            member_type: MemberType::PythonLSTM,
            weight: 0.4,
            confidence: 0.8,
            win_rate: 0.55,
            avg_return: 0.015,
            sharpe_ratio: 0.9,
        });
        
        // Normalize weights
        engine.update_weights(&[]);
        
        let stats = engine.get_stats();
        // After normalization, weights should sum to ~1.0
        // Average should be ~0.5
        assert!(stats.avg_weight > 0.4 && stats.avg_weight < 0.6);
    }
}
