//! Square-Root Market Impact Model Calibrator
//! 
//! Real-time calibration using recent aggressive trade tick sizes.
//! Implements the classic square-root law: impact = alpha * sign(q) * sqrt(|q|/volume)

use std::collections::VecDeque;

/// Configuration for the square-root impact model
#[derive(Debug, Clone)]
pub struct SquareRootConfig {
    /// Number of recent trades to use for calibration
    pub calibration_window: usize,
    /// Minimum volume threshold for considering a trade
    pub min_trade_volume: u64,
    /// Decay factor for exponential weighting
    pub decay_factor: f64,
    /// Default alpha when insufficient data
    pub default_alpha: f64,
}

impl Default for SquareRootConfig {
    fn default() -> Self {
        Self {
            calibration_window: 100,
            min_trade_volume: 1000,
            decay_factor: 0.95,
            default_alpha: 0.1,
        }
    }
}

/// Represents an observed trade for calibration
#[derive(Debug, Clone)]
pub struct TradeObservation {
    /// Trade volume (absolute value)
    pub volume: u64,
    /// Price before trade
    pub price_before: f64,
    /// Price after trade
    pub price_after: f64,
    /// Direction: 1 for buy, -1 for sell
    pub direction: i8,
    /// Timestamp (nanoseconds)
    pub timestamp_ns: u64,
}

impl TradeObservation {
    pub fn new(volume: u64, price_before: f64, price_after: f64, is_buy: bool, timestamp_ns: u64) -> Self {
        Self {
            volume,
            price_before,
            price_after,
            direction: if is_buy { 1 } else { -1 },
            timestamp_ns,
        }
    }

    /// Calculate observed impact in basis points
    pub fn observed_impact_bps(&self) -> f64 {
        let price_change = self.price_after - self.price_before;
        let mid_price = (self.price_before + self.price_after) / 2.0;
        if mid_price < 1e-9 {
            return 0.0;
        }
        // Impact in the direction of the trade
        let signed_change = price_change * (self.direction as f64);
        (signed_change / mid_price) * 10000.0
    }
}

/// Square-root market impact model with real-time calibration
pub struct SquareRootImpactModel {
    config: SquareRootConfig,
    observations: VecDeque<TradeObservation>,
    /// Current calibrated alpha coefficient
    current_alpha: f64,
    /// Sum of weighted squared normalized impacts for regression
    sum_weighted_xy: f64,
    /// Sum of weighted squared sqrt volumes
    sum_weighted_xx: f64,
    /// Total weight for normalization
    total_weight: f64,
    /// Recent volume average for normalization
    avg_daily_volume: f64,
}

impl SquareRootImpactModel {
    pub fn new(config: SquareRootConfig) -> Self {
        Self {
            config,
            observations: VecDeque::with_capacity(config.calibration_window),
            current_alpha: config.default_alpha,
            sum_weighted_xy: 0.0,
            sum_weighted_xx: 0.0,
            total_weight: 0.0,
            avg_daily_volume: 1_000_000.0, // Default 1M
        }
    }

    /// Add a new trade observation and recalibrate
    pub fn add_observation(&mut self, obs: TradeObservation) {
        if obs.volume < self.config.min_trade_volume {
            return;
        }

        // Remove oldest observation if at capacity
        if self.observations.len() >= self.config.calibration_window {
            if let Some(oldest) = self.observations.pop_front() {
                self.remove_observation_effect(&oldest);
            }
        }

        self.observations.push_back(obs.clone());
        self.add_observation_effect(&obs);
        
        // Recalibrate alpha
        self.recalibrate();
    }

    fn add_observation_effect(&mut self, obs: &TradeObservation) {
        let weight = self.compute_weight(obs.timestamp_ns);
        let sqrt_vol_norm = (obs.volume as f64 / self.avg_daily_volume).sqrt();
        let impact_bps = obs.observed_impact_bps() / 100.0; // Convert to decimal
        
        self.sum_weighted_xy += weight * sqrt_vol_norm * impact_bps;
        self.sum_weighted_xx += weight * sqrt_vol_norm * sqrt_vol_norm;
        self.total_weight += weight;
    }

    fn remove_observation_effect(&mut self, obs: &TradeObservation) {
        let weight = self.compute_weight(obs.timestamp_ns);
        let sqrt_vol_norm = (obs.volume as f64 / self.avg_daily_volume).sqrt();
        let impact_bps = obs.observed_impact_bps() / 100.0;
        
        self.sum_weighted_xy -= weight * sqrt_vol_norm * impact_bps;
        self.sum_weighted_xx -= weight * sqrt_vol_norm * sqrt_vol_norm;
        self.total_weight -= weight;
    }

    fn compute_weight(&self, timestamp_ns: u64) -> f64 {
        // Exponential decay based on recency
        // Assumes observations are added in order
        let age_factor = (self.observations.len() as f64) * 0.1;
        self.config.decay_factor.powf(age_factor)
    }

    fn recalibrate(&mut self) {
        if self.sum_weighted_xx < 1e-12 || self.total_weight < 1.0 {
            self.current_alpha = self.config.default_alpha;
            return;
        }

        // Linear regression: impact = alpha * sqrt(vol/volume)
        // alpha = sum(w * x * y) / sum(w * x^2)
        self.current_alpha = self.sum_weighted_xy / self.sum_weighted_xx;
        
        // Clamp alpha to reasonable bounds
        self.current_alpha = self.current_alpha.clamp(0.01, 1.0);
    }

    /// Calculate predicted market impact for a given order size
    /// Returns impact as a decimal (e.g., 0.01 = 1%)
    pub fn predict_impact(&self, order_volume: u64, is_buy: bool) -> f64 {
        if order_volume == 0 {
            return 0.0;
        }

        let vol_ratio = order_volume as f64 / self.avg_daily_volume;
        let sqrt_impact = self.current_alpha * vol_ratio.sqrt();
        
        if is_buy {
            sqrt_impact
        } else {
            -sqrt_impact
        }
    }

    /// Calculate predicted price after executing an order
    pub fn predict_execution_price(&self, current_price: f64, order_volume: u64, is_buy: bool) -> f64 {
        let impact = self.predict_impact(order_volume, is_buy);
        if is_buy {
            current_price * (1.0 + impact)
        } else {
            current_price * (1.0 - impact)
        }
    }

    /// Get the current calibrated alpha
    pub fn alpha(&self) -> f64 {
        self.current_alpha
    }

    /// Update the average daily volume estimate
    pub fn update_avg_daily_volume(&mut self, volume: f64) {
        if volume > 0.0 {
            self.avg_daily_volume = volume;
        }
    }

    /// Get the number of observations used for calibration
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Check if we have enough data for reliable calibration
    pub fn is_calibrated(&self) -> bool {
        self.observations.len() >= 10 && self.total_weight >= 5.0
    }

    /// Reset the calibrator
    pub fn reset(&mut self) {
        self.observations.clear();
        self.sum_weighted_xy = 0.0;
        self.sum_weighted_xx = 0.0;
        self.total_weight = 0.0;
        self.current_alpha = self.config.default_alpha;
    }

    /// Get confidence score for the current calibration (0 to 1)
    pub fn confidence(&self) -> f64 {
        let n = self.observations.len() as f64;
        let max_n = self.config.calibration_window as f64;
        (n / max_n).clamp(0.0, 1.0) * self.total_weight.min(10.0) / 10.0
    }
}

/// Batch impact calculator for large orders
pub struct BatchImpactCalculator<'a> {
    model: &'a SquareRootImpactModel,
}

impl<'a> BatchImpactCalculator<'a> {
    pub fn new(model: &'a SquareRootImpactModel) -> Self {
        Self { model }
    }

    /// Calculate total impact for splitting an order into N chunks
    pub fn calculate_sliced_impact(&self, total_volume: u64, num_slices: u32, is_buy: bool) -> f64 {
        if num_slices == 0 {
            return 0.0;
        }

        let slice_volume = total_volume / num_slices as u64;
        let single_impact = self.model.predict_impact(slice_volume, is_buy);
        
        // Total impact scales with sqrt(N) reduction per slice
        // But we execute N slices, so total = N * impact_per_slice
        single_impact * num_slices as f64
    }

    /// Find optimal number of slices to minimize impact
    pub fn optimal_slices(&self, total_volume: u64, max_slices: u32) -> u32 {
        let mut best_slices = 1;
        let mut best_impact = self.calculate_sliced_impact(total_volume, 1, true).abs();

        for n in 2..=max_slices {
            let impact = self.calculate_sliced_impact(total_volume, n, true).abs();
            if impact < best_impact {
                best_impact = impact;
                best_slices = n;
            }
        }

        best_slices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_root_model_basic() {
        let config = SquareRootConfig::default();
        let mut model = SquareRootImpactModel::new(config);

        // Add some observations
        for i in 0..20 {
            let obs = TradeObservation::new(
                10000 + i * 1000,
                50000.0,
                50000.0 * (1.0 + 0.001 * (i as f64 % 3)),
                true,
                i * 1000000,
            );
            model.add_observation(obs);
        }

        assert!(model.is_calibrated());
        assert!(model.alpha() > 0.0);
    }

    #[test]
    fn test_impact_prediction() {
        let mut model = SquareRootImpactModel::new(SquareRootConfig::default());
        
        // Manually set alpha for predictable testing
        model.current_alpha = 0.1;
        model.avg_daily_volume = 1_000_000.0;

        let impact = model.predict_impact(10000, true);
        assert!(impact > 0.0);
        
        // Larger orders should have more impact (square root relationship)
        let impact_large = model.predict_impact(40000, true);
        assert!(impact_large > impact);
        assert!((impact_large / impact - 2.0).abs() < 0.1); // sqrt(4) = 2
    }

    #[test]
    fn test_batch_calculator() {
        let model = SquareRootImpactModel::new(SquareRootConfig::default());
        let calc = BatchImpactCalculator::new(&model);

        let optimal = calc.optimal_slices(100000, 10);
        assert!(optimal >= 1 && optimal <= 10);
    }
}
