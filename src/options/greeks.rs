//! Options Greeks Engine
//! Calculates Delta, Gamma, Theta, Vega, and Rho using finite difference methods in O(1) time.
//! Designed to feed Gamma/Delta alpha signals for HFT strategies.

use super::pricing::{bs_price, OptionType, OptionPrice};

/// Fixed-point scaling factors
const PRICE_SCALE: i64 = 1_000_000_000;
const GREEK_DELTA_SCALE: i64 = 100_000_000;   // 1e8 for delta (0-1 range)
const GREEK_GAMMA_SCALE: i64 = 1_000_000_000_000; // 1e12 for gamma
const GREEK_THETA_SCALE: i64 = 100_000_000;   // 1e8 for theta
const GREEK_VEGA_SCALE: i64 = 100_000_000;    // 1e8 for vega
const GREEK_RHO_SCALE: i64 = 100_000_000;     // 1e8 for rho

/// All Greeks packed together for SIMD-friendly processing
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Greeks {
    pub delta: i64,    // First derivative w.r.t. spot
    pub gamma: i64,    // Second derivative w.r.t. spot
    pub theta: i64,    // First derivative w.r.t. time
    pub vega: i64,     // First derivative w.r.t. volatility
    pub rho: i64,      // First derivative w.r.t. interest rate
    pub vanna: i64,    // Cross derivative: d²V/dSdσ
    pub volga: i64,    // Second derivative w.r.t. volatility
    _padding: [u8; 8], // Cache-line alignment to 64 bytes
}

impl Greeks {
    pub const fn new() -> Self {
        Self {
            delta: 0,
            gamma: 0,
            theta: 0,
            vega: 0,
            rho: 0,
            vanna: 0,
            volga: 0,
            _padding: [0; 8],
        }
    }

    /// Convert analytical Greeks from OptionPrice
    pub fn from_option_price(price: &OptionPrice) -> Self {
        Self {
            delta: price.delta,
            gamma: price.gamma,
            theta: price.theta,
            vega: price.vega,
            rho: price.rho,
            vanna: 0,
            volga: 0,
            _padding: [0; 8],
        }
    }
}

/// Finite difference calculator for Greeks
/// Uses central difference method for O(1) computation
pub struct GreeksCalculator {
    /// Spot bump size in basis points for finite difference
    spot_bump_bps: i64,
    /// Vol bump size in basis points
    vol_bump_bps: i64,
    /// Time bump in seconds
    time_bump: i64,
    /// Rate bump in basis points
    rate_bump_bps: i64,
}

impl Default for GreeksCalculator {
    fn default() -> Self {
        Self::new(10, 10, 1, 10)
    }
}

impl GreeksCalculator {
    /// Create a new Greeks calculator with specified bump sizes
    pub const fn new(
        spot_bump_bps: i64,
        vol_bump_bps: i64,
        time_bump_seconds: i64,
        rate_bump_bps: i64,
    ) -> Self {
        Self {
            spot_bump_bps,
            vol_bump_bps,
            time_bump: time_bump_seconds,
            rate_bump_bps,
        }
    }

    /// Calculate all Greeks using finite differences in O(1) time
    /// Uses central difference for better accuracy
    #[inline]
    pub fn calculate_greeks_fd(
        &self,
        spot: i64,
        strike: i64,
        time_to_expiry: i64,
        volatility: i64,
        risk_free_rate: i64,
        option_type: OptionType,
    ) -> Greeks {
        let base_price = bs_price(spot, strike, time_to_expiry, volatility, risk_free_rate, option_type);
        
        // Delta: central difference dV/dS
        let delta = self.calculate_delta_fd(spot, strike, time_to_expiry, volatility, risk_free_rate, option_type);
        
        // Gamma: second derivative d²V/dS²
        let gamma = self.calculate_gamma_fd(spot, strike, time_to_expiry, volatility, risk_free_rate, option_type);
        
        // Theta: dV/dt (negative because time decreases)
        let theta = self.calculate_theta_fd(spot, strike, time_to_expiry, volatility, risk_free_rate, option_type);
        
        // Vega: dV/dσ
        let vega = self.calculate_vega_fd(spot, strike, time_to_expiry, volatility, risk_free_rate, option_type);
        
        // Rho: dV/dr
        let rho = self.calculate_rho_fd(spot, strike, time_to_expiry, volatility, risk_free_rate, option_type);
        
        // Vanna: d²V/dSdσ
        let vanna = self.calculate_vanna_fd(spot, strike, time_to_expiry, volatility, risk_free_rate, option_type);
        
        // Volga: d²V/dσ²
        let volga = self.calculate_volga_fd(spot, strike, time_to_expiry, volatility, risk_free_rate, option_type);
        
        Greeks {
            delta,
            gamma,
            theta,
            vega,
            rho,
            vanna,
            volga,
            _padding: [0; 8],
        }
    }

    /// Calculate Delta using central finite difference
    #[inline]
    fn calculate_delta_fd(
        &self,
        spot: i64,
        strike: i64,
        time_to_expiry: i64,
        volatility: i64,
        risk_free_rate: i64,
        option_type: OptionType,
    ) -> i64 {
        let bump = (spot * self.spot_bump_bps / 10000).max(1);
        
        let price_up = bs_price(spot + bump, strike, time_to_expiry, volatility, risk_free_rate, option_type);
        let price_down = bs_price(spot - bump, strike, time_to_expiry, volatility, risk_free_rate, option_type);
        
        // Central difference: (f(x+h) - f(x-h)) / (2h)
        let delta_raw = (price_up.price - price_down.price) / (2 * bump);
        
        // Scale to fixed-point (delta is unitless, scaled by 1e8)
        delta_raw * GREEK_DELTA_SCALE / PRICE_SCALE
    }

    /// Calculate Gamma using central finite difference
    #[inline]
    fn calculate_gamma_fd(
        &self,
        spot: i64,
        strike: i64,
        time_to_expiry: i64,
        volatility: i64,
        risk_free_rate: i64,
        option_type: OptionType,
    ) -> i64 {
        let bump = (spot * self.spot_bump_bps / 10000).max(1);
        
        let price_up = bs_price(spot + bump, strike, time_to_expiry, volatility, risk_free_rate, option_type);
        let price_base = bs_price(spot, strike, time_to_expiry, volatility, risk_free_rate, option_type);
        let price_down = bs_price(spot - bump, strike, time_to_expiry, volatility, risk_free_rate, option_type);
        
        // Central second difference: (f(x+h) - 2f(x) + f(x-h)) / h²
        let gamma_raw = (price_up.price - 2 * price_base.price + price_down.price) / (bump * bump);
        
        // Scale to fixed-point
        gamma_raw * GREEK_GAMMA_SCALE / PRICE_SCALE
    }

    /// Calculate Theta using forward finite difference (time only moves forward)
    #[inline]
    fn calculate_theta_fd(
        &self,
        spot: i64,
        strike: i64,
        time_to_expiry: i64,
        volatility: i64,
        risk_free_rate: i64,
        option_type: OptionType,
    ) -> i64 {
        if time_to_expiry <= self.time_bump {
            return 0;
        }
        
        let price_now = bs_price(spot, strike, time_to_expiry, volatility, risk_free_rate, option_type);
        let price_later = bs_price(spot, strike, time_to_expiry - self.time_bump, volatility, risk_free_rate, option_type);
        
        // Theta is negative of dV/dt (option loses value as time passes)
        let theta_raw = (price_later.price - price_now.price) / self.time_bump;
        
        // Annualize (convert from per-second to per-year)
        let theta_annual = theta_raw * 31536000;
        
        // Scale to fixed-point
        theta_annual * GREEK_THETA_SCALE / PRICE_SCALE
    }

    /// Calculate Vega using central finite difference
    #[inline]
    fn calculate_vega_fd(
        &self,
        spot: i64,
        strike: i64,
        time_to_expiry: i64,
        volatility: i64,
        risk_free_rate: i64,
        option_type: OptionType,
    ) -> i64 {
        let bump = (volatility * self.vol_bump_bps / 10000).max(PRICE_SCALE / 100);
        
        let price_up = bs_price(spot, strike, time_to_expiry, volatility + bump, risk_free_rate, option_type);
        let price_down = bs_price(spot, strike, time_to_expiry, volatility - bump, risk_free_rate, option_type);
        
        // Central difference
        let vega_raw = (price_up.price - price_down.price) / (2 * bump);
        
        // Scale: vega is typically expressed per 1% vol change
        let vega_scaled = vega_raw * volatility / 100;
        
        vega_scaled * GREEK_VEGA_SCALE / PRICE_SCALE
    }

    /// Calculate Rho using central finite difference
    #[inline]
    fn calculate_rho_fd(
        &self,
        spot: i64,
        strike: i64,
        time_to_expiry: i64,
        volatility: i64,
        risk_free_rate: i64,
        option_type: OptionType,
    ) -> i64 {
        let bump = (risk_free_rate * self.rate_bump_bps / 10000).max(PRICE_SCALE / 1000);
        
        let price_up = bs_price(spot, strike, time_to_expiry, volatility, risk_free_rate + bump, option_type);
        let price_down = bs_price(spot, strike, time_to_expiry, volatility, risk_free_rate - bump, option_type);
        
        // Central difference
        let rho_raw = (price_up.price - price_down.price) / (2 * bump);
        
        // Scale: rho per 1% rate change
        let rho_scaled = rho_raw * risk_free_rate / 100;
        
        rho_scaled * GREEK_RHO_SCALE / PRICE_SCALE
    }

    /// Calculate Vanna (cross derivative d²V/dSdσ)
    #[inline]
    fn calculate_vanna_fd(
        &self,
        spot: i64,
        strike: i64,
        time_to_expiry: i64,
        volatility: i64,
        risk_free_rate: i64,
        option_type: OptionType,
    ) -> i64 {
        let spot_bump = (spot * self.spot_bump_bps / 10000).max(1);
        let vol_bump = (volatility * self.vol_bump_bps / 10000).max(PRICE_SCALE / 100);
        
        // Four corners for cross derivative
        let p_uu = bs_price(spot + spot_bump, strike, time_to_expiry, volatility + vol_bump, risk_free_rate, option_type).price;
        let p_ud = bs_price(spot + spot_bump, strike, time_to_expiry, volatility - vol_bump, risk_free_rate, option_type).price;
        let p_du = bs_price(spot - spot_bump, strike, time_to_expiry, volatility + vol_bump, risk_free_rate, option_type).price;
        let p_dd = bs_price(spot - spot_bump, strike, time_to_expiry, volatility - vol_bump, risk_free_rate, option_type).price;
        
        // Cross derivative: (f(x+h,y+k) - f(x+h,y-k) - f(x-h,y+k) + f(x-h,y-k)) / (4hk)
        let vanna_raw = (p_uu - p_ud - p_du + p_dd) / (4 * spot_bump * vol_bump);
        
        vanna_raw * GREEK_DELTA_SCALE / PRICE_SCALE
    }

    /// Calculate Volga (second derivative w.r.t. volatility d²V/dσ²)
    #[inline]
    fn calculate_volga_fd(
        &self,
        spot: i64,
        strike: i64,
        time_to_expiry: i64,
        volatility: i64,
        risk_free_rate: i64,
        option_type: OptionType,
    ) -> i64 {
        let bump = (volatility * self.vol_bump_bps / 10000).max(PRICE_SCALE / 100);
        
        let price_up = bs_price(spot, strike, time_to_expiry, volatility + bump, risk_free_rate, option_type);
        let price_base = bs_price(spot, strike, time_to_expiry, volatility, risk_free_rate, option_type);
        let price_down = bs_price(spot, strike, time_to_expiry, volatility - bump, risk_free_rate, option_type);
        
        // Central second difference
        let volga_raw = (price_up.price - 2 * price_base.price + price_down.price) / (bump * bump);
        
        volga_raw * GREEK_GAMMA_SCALE / PRICE_SCALE
    }

    /// Calculate Delta-adjusted position P&L sensitivity
    #[inline]
    pub fn delta_adjusted_pnl(&self, greeks: &Greeks, spot_move_bps: i64) -> i64 {
        // Approximate P&L: delta * spot_move + 0.5 * gamma * spot_move²
        let spot_move = spot_move_bps * spot_move_bps;
        let delta_pnl = greeks.delta * spot_move_bps / 10000;
        let gamma_pnl = greeks.gamma * spot_move / (2 * 10000 * 10000);
        
        delta_pnl + gamma_pnl
    }

    /// Calculate total portfolio Greeks from individual positions
    pub fn aggregate_greeks<'a, I>(&self, positions: I) -> Greeks
    where
        I: Iterator<Item = &'a Greeks>,
    {
        let mut total = Greeks::new();
        
        for greeks in positions {
            total.delta = total.delta.saturating_add(greeks.delta);
            total.gamma = total.gamma.saturating_add(greeks.gamma);
            total.theta = total.theta.saturating_add(greeks.theta);
            total.vega = total.vega.saturating_add(greeks.vega);
            total.rho = total.rho.saturating_add(greeks.rho);
            total.vanna = total.vanna.saturating_add(greeks.vanna);
            total.volga = total.volga.saturating_add(greeks.volga);
        }
        
        total
    }
}

/// Gamma/Delta signal generator for alpha detection
pub struct GammaDeltaSignals {
    /// Threshold for gamma squeeze detection
    gamma_threshold: i64,
    /// Threshold for delta imbalance
    delta_imbalance_threshold: i64,
}

impl GammaDeltaSignals {
    pub const fn new(gamma_threshold: i64, delta_threshold: i64) -> Self {
        Self {
            gamma_threshold,
            delta_imbalance_threshold: delta_threshold,
        }
    }

    /// Detect potential gamma squeeze
    /// High positive gamma means market makers need to buy as price rises
    #[inline]
    pub fn is_gamma_squeeze(&self, greeks: &Greeks, price_direction: i8) -> bool {
        // price_direction: 1 = up, -1 = down
        if price_direction > 0 {
            greeks.gamma > self.gamma_threshold && greeks.delta > 0
        } else {
            greeks.gamma > self.gamma_threshold && greeks.delta < 0
        }
    }

    /// Detect delta rebalancing opportunity
    /// Large delta imbalances indicate hedging flows
    #[inline]
    pub fn detect_delta_imbalance(&self, call_delta: i64, put_delta: i64) -> Option<i8> {
        let net_delta = call_delta - put_delta.abs();
        
        if net_delta > self.delta_imbalance_threshold {
            Some(1) // Call skew - bullish hedging needed
        } else if net_delta < -self.delta_imbalance_threshold {
            Some(-1) // Put skew - bearish hedging needed
        } else {
            None
        }
    }

    /// Calculate gamma exposure (GEX) level
    /// Positive GEX = stable market, Negative GEX = volatile market
    #[inline]
    pub fn calculate_gex(&self, greeks: &Greeks, open_interest: u64) -> i64 {
        // GEX = Gamma * Open Interest * Spot
        // Simplified: just gamma * OI for relative comparison
        (greeks.gamma * open_interest as i64) / GREEK_GAMMA_SCALE
    }

    /// Detect zero-gamma level (market maker hedging flip point)
    pub fn find_zero_gamma_level(
        &self,
        strikes: &[i64],
        gammas: &[i64],
        current_spot: i64,
    ) -> Option<i64> {
        if strikes.len() != gammas.len() || strikes.is_empty() {
            return None;
        }

        // Find where cumulative gamma crosses zero
        let mut cum_gamma: i64 = 0;
        
        for i in 0..strikes.len() {
            let strike = unsafe { strikes.get_unchecked(i) };
            let gamma = unsafe { gammas.get_unchecked(i) };
            
            // Weight gamma by distance from spot (closer strikes matter more)
            let distance = (*strike - current_spot).abs();
            let weight = 1000000 / (distance.max(1));
            let weighted_gamma = gamma * weight / 1000000;
            
            cum_gamma += weighted_gamma;
            
            if cum_gamma <= 0 && i > 0 {
                return Some(*strike);
            }
        }
        
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greeks_calculation() {
        let calc = GreeksCalculator::default();
        
        let spot = 100 * PRICE_SCALE;
        let strike = 100 * PRICE_SCALE;
        let time = 31536000; // 1 year
        let vol = 20 * PRICE_SCALE / 100;
        let rate = 5 * PRICE_SCALE / 100;
        
        let greeks = calc.calculate_greeks_fd(spot, strike, time, vol, rate, OptionType::Call);
        
        // Delta should be around 0.5-0.6 for ATM call
        assert!(greeks.delta > 40_000_000 && greeks.delta < 70_000_000);
        
        // Gamma should be positive
        assert!(greeks.gamma > 0);
        
        // Theta should be negative for long options
        assert!(greeks.theta < 0);
    }

    #[test]
    fn test_gamma_squeeze_detection() {
        let signals = GammaDeltaSignals::new(1_000_000, 10_000_000);
        
        let greeks = Greeks {
            delta: 50_000_000,
            gamma: 2_000_000,
            ..Greeks::new()
        };
        
        assert!(signals.is_gamma_squeeze(&greeks, 1));
        assert!(!signals.is_gamma_squeeze(&greeks, -1));
    }
}
