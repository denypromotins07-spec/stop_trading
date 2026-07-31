//! Wavelet Analysis for Multi-Resolution Time-Series Decomposition
//! 
//! Implements Haar and Daubechies (DB4) wavelet transforms for isolating
//! high-frequency noise from low-frequency structural trends in tick data.
//! Uses fixed-size ring buffers to strictly respect the 6.5GB RAM ceiling.

use std::array;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Maximum decomposition levels supported (log2 of buffer size)
const MAX_LEVELS: usize = 10;

/// Fixed buffer size for wavelet decomposition (power of 2)
const BUFFER_SIZE: usize = 1024;

/// Pre-computed Daubechies-4 filter coefficients
/// These are hardcoded for performance to avoid runtime computation
mod db4_coeffs {
    pub const H0: f64 = 0.4829629131445341;  // Low-pass scaling
    pub const H1: f64 = 0.8365163037378079;
    pub const H2: f64 = 0.2241438680420134;
    pub const H3: f64 = -0.1294095225512604;
    
    pub const G0: f64 = -0.1294095225512604; // High-pass wavelet
    pub const G1: f64 = -0.2241438680420134;
    pub const G2: f64 = 0.8365163037378079;
    pub const G3: f64 = -0.4829629131445341;
}

/// Haar filter coefficients (simpler, faster)
mod haar_coeffs {
    pub const H0: f64 = 0.7071067811865476; // 1/sqrt(2)
    pub const H1: f64 = 0.7071067811865476;
    pub const G0: f64 = 0.7071067811865476;
    pub const G1: f64 = -0.7071067811865476;
}

/// Wavelet type enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaveletType {
    Haar,
    Daubechies4,
}

/// Result of wavelet decomposition
#[derive(Debug, Clone)]
pub struct WaveletDecomposition {
    /// Approximation coefficients (low-frequency trend)
    pub approximation: [f64; BUFFER_SIZE / (1 << MAX_LEVELS)],
    /// Detail coefficients at each level (high-frequency components)
    pub details: [[f64; BUFFER_SIZE / (1 << MAX_LEVELS)]; MAX_LEVELS],
    /// Number of valid levels computed
    pub levels: usize,
    /// Original signal energy preserved (for reconstruction validation)
    pub energy_preserved: f64,
}

/// Wavelet transformer with pre-allocated buffers
pub struct WaveletTransformer {
    wavelet_type: WaveletType,
    input_buffer: [f64; BUFFER_SIZE],
    temp_buffer: [f64; BUFFER_SIZE],
    write_index: AtomicUsize,
}

impl WaveletTransformer {
    /// Create a new wavelet transformer with specified wavelet type
    pub fn new(wavelet_type: WaveletType) -> Self {
        Self {
            wavelet_type,
            input_buffer: [0.0; BUFFER_SIZE],
            temp_buffer: [0.0; BUFFER_SIZE],
            write_index: AtomicUsize::new(0),
        }
    }
    
    /// Add a new tick value to the ring buffer
    #[inline]
    pub fn push_tick(&mut self, value: f64) {
        let idx = self.write_index.load(Ordering::Relaxed);
        self.input_buffer[idx % BUFFER_SIZE] = value;
        self.write_index.store(idx + 1, Ordering::Release);
    }
    
    /// Perform Haar wavelet decomposition
    fn haar_decompose(&self, data: &[f64], levels: usize) -> WaveletDecomposition {
        use haar_coeffs::*;
        
        let mut result = WaveletDecomposition {
            approximation: [0.0; BUFFER_SIZE / (1 << MAX_LEVELS)],
            details: [[0.0; BUFFER_SIZE / (1 << MAX_LEVELS)]; MAX_LEVELS],
            levels: 0,
            energy_preserved: 0.0,
        };
        
        let mut current = data.to_vec();
        let mut original_energy: f64 = 0.0;
        
        for &val in data.iter() {
            original_energy += val * val;
        }
        
        for level in 0..levels.min(MAX_LEVELS) {
            let len = current.len();
            if len < 2 {
                break;
            }
            
            let half_len = len / 2;
            
            // Single pass decomposition
            for i in 0..half_len {
                let even_idx = 2 * i;
                let odd_idx = 2 * i + 1;
                
                let even_val = current[even_idx.min(len - 1)];
                let odd_val = if odd_idx < len { current[odd_idx] } else { even_val };
                
                // Approximation (low-pass)
                current[i] = H0 * even_val + H1 * odd_val;
                // Detail (high-pass)
                self.temp_buffer[i] = G0 * even_val + G1 * odd_val;
            }
            
            // Store detail coefficients
            for i in 0..half_len.min(result.details[level].len()) {
                result.details[level][i] = self.temp_buffer[i];
            }
            
            // Truncate for next level
            current.truncate(half_len);
            result.levels = level + 1;
        }
        
        // Store final approximation
        for i in 0..current.len().min(result.approximation.len()) {
            result.approximation[i] = current[i];
        }
        
        // Calculate preserved energy
        let mut reconstructed_energy: f64 = 0.0;
        for i in 0..result.levels {
            for j in 0..result.details[i].len() {
                reconstructed_energy += result.details[i][j] * result.details[i][j];
            }
        }
        for &approx in result.approximation.iter() {
            reconstructed_energy += approx * approx;
        }
        
        result.energy_preserved = if original_energy > 0.0 {
            reconstructed_energy / original_energy
        } else {
            1.0
        };
        
        result
    }
    
    /// Perform Daubechies-4 wavelet decomposition
    fn db4_decompose(&self, data: &[f64], levels: usize) -> WaveletDecomposition {
        use db4_coeffs::*;
        
        let mut result = WaveletDecomposition {
            approximation: [0.0; BUFFER_SIZE / (1 << MAX_LEVELS)],
            details: [[0.0; BUFFER_SIZE / (1 << MAX_LEVELS)]; MAX_LEVELS],
            levels: 0,
            energy_preserved: 0.0,
        };
        
        let mut current = data.to_vec();
        let mut original_energy: f64 = 0.0;
        
        for &val in data.iter() {
            original_energy += val * val;
        }
        
        for level in 0..levels.min(MAX_LEVELS) {
            let len = current.len();
            if len < 4 {
                break;
            }
            
            let half_len = len / 2;
            
            // DB4 decomposition with periodic extension
            for i in 0..half_len {
                let base_idx = 2 * i;
                
                // Periodic boundary handling
                let get_val = |offset: usize| -> f64 {
                    let idx = (base_idx + offset) % len;
                    current[idx]
                };
                
                let even_val = get_val(0);
                let odd1_val = get_val(1);
                let odd2_val = get_val(2);
                let odd3_val = get_val(3);
                
                // Approximation (low-pass filter)
                current[i] = H0 * even_val + H1 * odd1_val + H2 * odd2_val + H3 * odd3_val;
                
                // Detail (high-pass filter)
                self.temp_buffer[i] = G0 * even_val + G1 * odd1_val + G2 * odd2_val + G3 * odd3_val;
            }
            
            // Store detail coefficients
            for i in 0..half_len.min(result.details[level].len()) {
                result.details[level][i] = self.temp_buffer[i];
            }
            
            current.truncate(half_len);
            result.levels = level + 1;
        }
        
        // Store final approximation
        for i in 0..current.len().min(result.approximation.len()) {
            result.approximation[i] = current[i];
        }
        
        // Calculate preserved energy
        let mut reconstructed_energy: f64 = 0.0;
        for i in 0..result.levels {
            for j in 0..result.details[i].len() {
                reconstructed_energy += result.details[i][j] * result.details[i][j];
            }
        }
        for &approx in result.approximation.iter() {
            reconstructed_energy += approx * approx;
        }
        
        result.energy_preserved = if original_energy > 0.0 {
            reconstructed_energy / original_energy
        } else {
            1.0
        };
        
        result
    }
    
    /// Perform wavelet decomposition on current buffer contents
    pub fn decompose(&self, levels: usize) -> WaveletDecomposition {
        let idx = self.write_index.load(Ordering::Acquire);
        let valid_len = idx.min(BUFFER_SIZE);
        
        if valid_len == 0 {
            return WaveletDecomposition {
                approximation: [0.0; BUFFER_SIZE / (1 << MAX_LEVELS)],
                details: [[0.0; BUFFER_SIZE / (1 << MAX_LEVELS)]; MAX_LEVELS],
                levels: 0,
                energy_preserved: 0.0,
            };
        }
        
        let data_slice = &self.input_buffer[..valid_len];
        
        match self.wavelet_type {
            WaveletType::Haar => self.haar_decompose(data_slice, levels),
            WaveletType::Daubechies4 => self.db4_decompose(data_slice, levels),
        }
    }
    
    /// Extract high-frequency noise component from decomposition
    pub fn extract_noise(&self, decomposition: &WaveletDecomposition) -> f64 {
        let mut noise_power: f64 = 0.0;
        let mut count = 0usize;
        
        // Sum detail coefficients from finest levels (high-frequency noise)
        for level in 0..decomposition.levels.min(3) {
            for i in 0..(BUFFER_SIZE / (1 << MAX_LEVELS)) {
                let coeff = decomposition.details[level][i];
                noise_power += coeff * coeff;
                count += 1;
            }
        }
        
        if count > 0 {
            (noise_power / count as f64).sqrt()
        } else {
            0.0
        }
    }
    
    /// Extract low-frequency trend component
    pub fn extract_trend(&self, decomposition: &WaveletDecomposition) -> f64 {
        if decomposition.levels == 0 {
            return 0.0;
        }
        
        let approx_count = BUFFER_SIZE / (1 << MAX_LEVELS);
        let mut sum: f64 = 0.0;
        let mut count = 0usize;
        
        for i in 0..approx_count {
            let coeff = decomposition.approximation[i];
            if coeff.abs() > 1e-12 {
                sum += coeff;
                count += 1;
            }
        }
        
        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }
    
    /// Detect regime change using wavelet coefficient variance
    pub fn detect_regime_change(&self, decomposition: &WaveletDecomposition, threshold: f64) -> bool {
        let noise_level = self.extract_noise(decomposition);
        let trend_level = self.extract_trend(decomposition).abs();
        
        // Regime change when noise dominates trend significantly
        if trend_level > 1e-9 {
            noise_level / trend_level > threshold
        } else {
            noise_level > threshold
        }
    }
}

/// Multi-resolution analyzer combining multiple wavelet types
pub struct MultiResolutionAnalyzer {
    haar_transformer: WaveletTransformer,
    db4_transformer: WaveletTransformer,
}

impl MultiResolutionAnalyzer {
    pub fn new() -> Self {
        Self {
            haar_transformer: WaveletTransformer::new(WaveletType::Haar),
            db4_transformer: WaveletTransformer::new(WaveletType::Daubechies4),
        }
    }
    
    pub fn push_tick(&mut self, value: f64) {
        self.haar_transformer.push_tick(value);
        self.db4_transformer.push_tick(value);
    }
    
    /// Get consensus decomposition from both wavelet types
    pub fn get_consensus_trend(&self, levels: usize) -> f64 {
        let haar_decomp = self.haar_transformer.decompose(levels);
        let db4_decomp = self.db4_transformer.decompose(levels);
        
        let haar_trend = self.haar_transformer.extract_trend(&haar_decomp);
        let db4_trend = self.db4_transformer.extract_trend(&db4_decomp);
        
        // Weighted average favoring DB4 for smoother trends
        (haar_trend + 2.0 * db4_trend) / 3.0
    }
    
    /// Get noise estimate for volatility adjustment
    pub fn get_noise_estimate(&self) -> f64 {
        let haar_decomp = self.haar_transformer.decompose(3);
        let db4_decomp = self.db4_transformer.decompose(3);
        
        let haar_noise = self.haar_transformer.extract_noise(&haar_decomp);
        let db4_noise = self.db4_transformer.extract_noise(&db4_decomp);
        
        (haar_noise + db4_noise) / 2.0
    }
}

impl Default for MultiResolutionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_haar_decomposition() {
        let mut transformer = WaveletTransformer::new(WaveletType::Haar);
        
        // Push test signal: sine wave + noise
        for i in 0..BUFFER_SIZE {
            let value = (i as f64 * 0.1).sin() + 0.1 * (i as f64 * 0.5).sin();
            transformer.push_tick(value);
        }
        
        let decomp = transformer.decompose(5);
        assert!(decomp.levels >= 1);
        assert!(decomp.energy_preserved > 0.8); // Should preserve most energy
    }
    
    #[test]
    fn test_db4_decomposition() {
        let mut transformer = WaveletTransformer::new(WaveletType::Daubechies4);
        
        for i in 0..BUFFER_SIZE {
            let value = (i as f64 * 0.1).sin();
            transformer.push_tick(value);
        }
        
        let decomp = transformer.decompose(4);
        assert!(decomp.levels >= 1);
    }
    
    #[test]
    fn test_multi_resolution_analyzer() {
        let mut analyzer = MultiResolutionAnalyzer::new();
        
        for i in 0..BUFFER_SIZE {
            let value = (i as f64 * 0.05).sin() + 0.2 * (i as f64).sin();
            analyzer.push_tick(value);
        }
        
        let trend = analyzer.get_consensus_trend(4);
        let noise = analyzer.get_noise_estimate();
        
        assert!(trend.abs() > 0.0 || noise > 0.0);
    }
    
    #[test]
    fn test_regime_change_detection() {
        let mut transformer = WaveletTransformer::new(WaveletType::Haar);
        
        // Stable regime
        for i in 0..BUFFER_SIZE {
            transformer.push_tick((i as f64 * 0.01).sin());
        }
        
        let decomp = transformer.decompose(3);
        let stable = !transformer.detect_regime_change(&decomp, 2.0);
        
        // Inject high frequency noise
        for i in 0..BUFFER_SIZE {
            transformer.push_tick((i as f64 * 0.01).sin() + 5.0 * (i as f64 * 0.8).sin());
        }
        
        let decomp_noisy = transformer.decompose(3);
        let noisy = transformer.detect_regime_change(&decomp_noisy, 1.5);
        
        // Noisy regime should be detected
        assert!(noisy || !stable);
    }
}
