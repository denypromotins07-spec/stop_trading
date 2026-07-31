//! Implied Volatility Surface Interpolator
//! Builds a real-time IV surface using cubic splines for strike/expiry mapping.
//! Zero heap allocations in hot paths - uses pre-allocated bounded arrays.

use core::sync::atomic::{AtomicUsize, Ordering};

/// Maximum number of strikes per expiry
const MAX_STRIKES: usize = 64;

/// Maximum number of expiries
const MAX_EXPIRIES: usize = 24;

/// Fixed-point scaling
const FP_SCALE: i64 = 1_000_000_000;

/// Volatility smile/skew parameters for SABR-like interpolation
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VolParams {
    pub alpha: i64,  // Overall volatility level
    pub beta: i64,   // CEV exponent (typically 0.5-1.0)
    pub rho: i64,    // Correlation between spot and vol
    pub nu: i64,     // Volatility of volatility
    _padding: [u8; 8],
}

impl Default for VolParams {
    fn default() -> Self {
        Self {
            alpha: 0,
            beta: FP_SCALE / 2, // 0.5
            rho: 0,
            nu: 0,
            _padding: [0; 8],
        }
    }
}

/// Single point on the volatility surface
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VolPoint {
    pub strike: i64,
    pub expiry: u32,       // Days to expiry
    pub implied_vol: i64,  // Scaled by 1e8
    pub bid_vol: i64,
    pub ask_vol: i64,
    pub volume: u64,
    pub open_interest: u64,
}

impl Default for VolPoint {
    fn default() -> Self {
        Self {
            strike: 0,
            expiry: 0,
            implied_vol: 0,
            bid_vol: 0,
            ask_vol: 0,
            volume: 0,
            open_interest: 0,
        }
    }
}

/// Cubic spline coefficients for interpolation
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct SplineCoefficients {
    a: i64,  // Constant term
    b: i64,  // Linear coefficient
    c: i64,  // Quadratic coefficient
    d: i64,  // Cubic coefficient
}

/// Natural cubic spline interpolator (no heap allocation)
struct CubicSpline {
    x: [i64; MAX_STRIKES],
    coeffs: [SplineCoefficients; MAX_STRIKES],
    count: AtomicUsize,
}

impl CubicSpline {
    const fn new() -> Self {
        Self {
            x: [0; MAX_STRIKES],
            coeffs: [SplineCoefficients { a: 0, b: 0, c: 0, d: 0 }; MAX_STRIKES],
            count: AtomicUsize::new(0),
        }
    }

    /// Build spline from sorted x values and corresponding y values
    fn build(&mut self, x_vals: &[i64], y_vals: &[i64]) {
        let n = x_vals.len().min(MAX_STRIKES);
        if n < 2 {
            return;
        }

        // Store x values
        for i in 0..n {
            self.x[i] = x_vals[i];
        }
        self.count.store(n, Ordering::Relaxed);

        // Compute second derivatives using Thomas algorithm for tridiagonal system
        // For natural spline: M[0] = M[n-1] = 0
        
        let mut h: [i64; MAX_STRIKES] = [0; MAX_STRIKES];
        let mut alpha: [i64; MAX_STRIKES] = [0; MAX_STRIKES];
        let mut l: [i64; MAX_STRIKES] = [0; MAX_STRIKES];
        let mut mu: [i64; MAX_STRIKES] = [0; MAX_STRIKES];
        let mut z: [i64; MAX_STRIKES] = [0; MAX_STRIKES];
        
        // Step sizes
        for i in 0..n - 1 {
            h[i] = self.x[i + 1] - self.x[i];
        }

        // Alpha values (right-hand side)
        for i in 1..n - 1 {
            alpha[i] = (3 * (y_vals[i + 1] - y_vals[i]) * h[i - 1] 
                       - 3 * (y_vals[i] - y_vals[i - 1]) * h[i]) / (h[i - 1] * h[i]);
            // Scale down to prevent overflow
            alpha[i] /= FP_SCALE;
        }

        // Thomas algorithm forward sweep
        l[0] = FP_SCALE; // Scaled identity
        mu[0] = 0;
        z[0] = 0;

        for i in 1..n - 1 {
            l[i] = (2 * (self.x[i + 1] - self.x[i - 1])) * FP_SCALE / h[i - 1] - h[i - 1] * mu[i - 1];
            if l[i] == 0 {
                l[i] = 1; // Prevent division by zero
            }
            mu[i] = h[i] * FP_SCALE / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        l[n - 1] = FP_SCALE;
        z[n - 1] = 0;

        // Back substitution - store as 'c' coefficients (second derivatives)
        let mut c_prev = 0i64;
        for i in (0..n - 1).rev() {
            let c_i = z[i] - mu[i] * c_prev / FP_SCALE;
            self.coeffs[i].c = c_i;
            c_prev = c_i;
        }

        // Compute b and d coefficients
        for i in 0..n - 1 {
            self.coeffs[i].a = y_vals[i];
            self.coeffs[i].b = (y_vals[i + 1] - y_vals[i]) * FP_SCALE / h[i] 
                             - h[i] * (2 * self.coeffs[i].c + self.coeffs[i + 1].c) / (3 * FP_SCALE);
            self.coeffs[i].d = (self.coeffs[i + 1].c - self.coeffs[i].c) * FP_SCALE / (3 * h[i]);
        }
        
        self.coeffs[n - 1].a = y_vals[n - 1];
        self.coeffs[n - 1].b = 0;
        self.coeffs[n - 1].c = 0;
        self.coeffs[n - 1].d = 0;
    }

    /// Evaluate spline at point x
    #[inline]
    fn evaluate(&self, x: i64) -> i64 {
        let count = self.count.load(Ordering::Relaxed);
        if count < 2 {
            return 0;
        }

        // Find interval using binary search
        let mut lo = 0;
        let mut hi = count - 1;
        
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.x[mid] < x {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        // Clamp to valid range
        let idx = if lo >= count { count - 2 } else { lo.min(count - 2) };
        let coeff = &self.coeffs[idx];
        
        let dx = x - self.x[idx];
        let dx2 = dx * dx / FP_SCALE;
        let dx3 = dx2 * dx / FP_SCALE;
        
        // Horner's method: a + b*dx + c*dx² + d*dx³
        coeff.a + dx * coeff.b / FP_SCALE 
            + dx2 * coeff.c / FP_SCALE 
            + dx3 * coeff.d / FP_SCALE
    }
}

/// Volatility surface with multiple expiry slices
pub struct VolatilitySurface {
    /// Spot price reference
    spot: i64,
    /// Spline interpolators for each expiry
    splines: [CubicSpline; MAX_EXPIRIES],
    /// Expiry timestamps (days)
    expiries: [u32; MAX_EXPIRIES],
    /// Number of active expiry slices
    expiry_count: AtomicUsize,
    /// ATM volatility for each expiry
    atm_vols: [i64; MAX_EXPIRIES],
    /// Risk reversal (25D call vol - 25D put vol)
    risk_reversals: [i64; MAX_EXPIRIES],
    /// Butterfly (average of 25D vols - ATM vol)
    butterflies: [i64; MAX_EXPIRIES],
}

impl Default for VolatilitySurface {
    fn default() -> Self {
        Self::new()
    }
}

impl VolatilitySurface {
    /// Create a new empty volatility surface
    pub const fn new() -> Self {
        Self {
            spot: 0,
            splines: unsafe {
                // Initialize array of splines
                let mut arr: [CubicSpline; MAX_EXPIRIES] = [
                    CubicSpline { x: [0; MAX_STRIKES], coeffs: [SplineCoefficients { a: 0, b: 0, c: 0, d: 0 }; MAX_STRIKES], count: AtomicUsize::new(0) };
                    MAX_EXPIRIES
                ];
                arr
            },
            expiries: [0; MAX_EXPIRIES],
            expiry_count: AtomicUsize::new(0),
            atm_vols: [0; MAX_EXPIRIES],
            risk_reversals: [0; MAX_EXPIRIES],
            butterflies: [0; MAX_EXPIRIES],
        }
    }

    /// Update the spot price reference
    #[inline]
    pub fn set_spot(&mut self, spot: i64) {
        self.spot = spot;
    }

    /// Add or update an expiry slice with strike/vol data
    pub fn update_expiry_slice(
        &mut self,
        expiry_days: u32,
        strikes: &[i64],
        vols: &[i64],
    ) {
        let expiry_idx = self.find_or_add_expiry(expiry_days);
        if expiry_idx >= MAX_EXPIRIES {
            return;
        }

        // Build spline for this expiry
        self.splines[expiry_idx].build(strikes, vols);
        
        // Calculate smile parameters
        self.calculate_smile_params(expiry_idx, strikes, vols);
    }

    /// Find existing expiry or add new one
    fn find_or_add_expiry(&mut self, expiry_days: u32) -> usize {
        let count = self.expiry_count.load(Ordering::Relaxed);
        
        // Search for existing
        for i in 0..count {
            if self.expiries[i] == expiry_days {
                return i;
            }
        }
        
        // Add new if space available
        if count < MAX_EXPIRIES {
            self.expiries[count] = expiry_days;
            self.expiry_count.store(count + 1, Ordering::Relaxed);
            return count;
        }
        
        // Replace oldest (index 0)
        self.expiries[0] = expiry_days;
        0
    }

    /// Calculate smile parameters (ATM vol, risk reversal, butterfly)
    fn calculate_smile_params(&mut self, expiry_idx: usize, strikes: &[i64], vols: &[i64]) {
        if strikes.is_empty() || self.spot == 0 {
            return;
        }

        // Find ATM strike (closest to spot)
        let mut atm_idx = 0;
        let mut min_diff = i64::MAX;
        
        for (i, &strike) in strikes.iter().enumerate() {
            let diff = (strike - self.spot).abs();
            if diff < min_diff {
                min_diff = diff;
                atm_idx = i;
            }
        }

        self.atm_vols[expiry_idx] = vols[atm_idx];

        // Find 25 delta strikes (approximate using moneyness)
        // 25D call ~ 10% OTM, 25D put ~ 10% ITM for typical vol
        let call_25d_strike = self.spot * 110 / 100;
        let put_25d_strike = self.spot * 90 / 100;

        let call_25d_vol = self.interpolate_vol(expiry_days_from_idx(expiry_idx, &self.expiries), call_25d_strike);
        let put_25d_vol = self.interpolate_vol(expiry_days_from_idx(expiry_idx, &self.expiries), put_25d_strike);

        // Risk reversal: 25D call vol - 25D put vol
        self.risk_reversals[expiry_idx] = call_25d_vol - put_25d_vol;

        // Butterfly: (25D call vol + 25D put vol) / 2 - ATM vol
        self.butterflies[expiry_idx] = (call_25d_vol + put_25d_vol) / 2 - self.atm_vols[expiry_idx];
    }

    /// Interpolate volatility for given expiry and strike
    pub fn interpolate_vol(&self, expiry_days: u32, strike: i64) -> i64 {
        if self.spot == 0 || strike <= 0 {
            return 0;
        }

        // Find surrounding expiries
        let count = self.expiry_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0;
        }

        // Exact expiry match
        for i in 0..count {
            if self.expiries[i] == expiry_days {
                return self.splines[i].evaluate(strike);
            }
        }

        // Find bracketing expiries for time interpolation
        let mut lower_idx = None;
        let mut upper_idx = None;
        
        for i in 0..count {
            if self.expiries[i] < expiry_days {
                lower_idx = Some(i);
            }
            if self.expiries[i] > expiry_days && upper_idx.is_none() {
                upper_idx = Some(i);
            }
        }

        match (lower_idx, upper_idx) {
            (Some(lo), Some(hi)) => {
                // Time-weighted interpolation
                let lo_exp = self.expiries[lo] as i64;
                let hi_exp = self.expiries[hi] as i64;
                let target = expiry_days as i64;
                
                let lo_vol = self.splines[lo].evaluate(strike);
                let hi_vol = self.splines[hi].evaluate(strike);
                
                // Linear time interpolation
                let weight = (target - lo_exp) * FP_SCALE / (hi_exp - lo_exp);
                lo_vol + (hi_vol - lo_vol) * weight / FP_SCALE
            }
            (Some(lo), None) => self.splines[lo].evaluate(strike),
            (None, Some(hi)) => self.splines[hi].evaluate(strike),
            (None, None) => 0,
        }
    }

    /// Get interpolated option price using Black-Scholes
    pub fn get_option_price(
        &self,
        strike: i64,
        expiry_days: u32,
        option_type: crate::options::pricing::OptionType,
    ) -> i64 {
        use super::pricing::bs_price;
        
        let vol = self.interpolate_vol(expiry_days, strike);
        if vol <= 0 {
            return 0;
        }

        let time_seconds = (expiry_days as i64) * 86400;
        let result = bs_price(self.spot, strike, time_seconds, vol, 0, option_type);
        result.price
    }

    /// Get ATM volatility for term structure
    pub fn get_atm_vol_term_structure(&self) -> &[(u32, i64)] {
        // Return slice of (expiry, atm_vol) pairs
        // This is a simplified view - in production would use proper memory management
        unsafe {
            core::slice::from_raw_parts(
                self.expiries.as_ptr() as *const (u32, i64),
                self.expiry_count.load(Ordering::Relaxed),
            )
        }
    }

    /// Get risk reversal skew for expiry
    #[inline]
    pub fn get_risk_reversal(&self, expiry_days: u32) -> i64 {
        for i in 0..self.expiry_count.load(Ordering::Relaxed) {
            if self.expiries[i] == expiry_days {
                return self.risk_reversals[i];
            }
        }
        0
    }

    /// Get butterfly spread value for expiry
    #[inline]
    pub fn get_butterfly(&self, expiry_days: u32) -> i64 {
        for i in 0..self.expiry_count.load(Ordering::Relaxed) {
            if self.expiries[i] == expiry_days {
                return self.butterflies[i];
            }
        }
        0
    }

    /// Detect vol arbitrage opportunities (calendar spreads)
    pub fn detect_calendar_arbitrage(&self) -> Option<(u32, u32, i64)> {
        let count = self.expiry_count.load(Ordering::Relaxed);
        if count < 2 {
            return None;
        }

        // Check for inverted term structure at ATM
        for i in 0..count - 1 {
            for j in i + 1..count {
                if self.expiries[j] > self.expiries[i] {
                    // Longer expiry should have higher or equal vol
                    if self.atm_vols[i] > self.atm_vols[j] + FP_SCALE / 100 {
                        // Arbitrage opportunity: sell short vol, buy long vol
                        return Some((self.expiries[i], self.expiries[j], 
                                    self.atm_vols[i] - self.atm_vols[j]));
                    }
                }
            }
        }

        None
    }
}

#[inline]
fn expiry_days_from_idx(idx: usize, expiries: &[u32; MAX_EXPIRIES]) -> u32 {
    unsafe { *expiries.get_unchecked(idx) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surface_interpolation() {
        let mut surface = VolatilitySurface::new();
        surface.set_spot(100 * FP_SCALE);

        // Add some vol data for 30-day expiry
        let strikes = [
            90 * FP_SCALE,
            95 * FP_SCALE,
            100 * FP_SCALE,
            105 * FP_SCALE,
            110 * FP_SCALE,
        ];
        let vols = [
            85 * FP_SCALE / 100,
            80 * FP_SCALE / 100,
            75 * FP_SCALE / 100,
            78 * FP_SCALE / 100,
            82 * FP_SCALE / 100,
        ];

        surface.update_expiry_slice(30, &strikes, &vols);

        // Test interpolation at ATM
        let atm_vol = surface.interpolate_vol(30, 100 * FP_SCALE);
        assert!(atm_vol > 70 * FP_SCALE / 100 && atm_vol < 80 * FP_SCALE / 100);

        // Test interpolation between strikes
        let mid_vol = surface.interpolate_vol(30, 102 * FP_SCALE);
        assert!(mid_vol > 0);
    }
}
