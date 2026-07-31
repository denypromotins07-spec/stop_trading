//! Cross-Asset Implied Volatility Surface Fitter
//! 
//! Real-time IV surface fitting for crypto options using cubic spline interpolation.

use std::collections::HashMap;

/// Represents a single option contract
#[derive(Debug, Clone)]
pub struct OptionContract {
    pub strike: f64,
    pub expiry_days: u32,
    pub is_call: bool,
    pub market_price: f64,
    pub underlying_price: f64,
    pub risk_free_rate: f64,
}

/// A point on the volatility surface
#[derive(Debug, Clone, Copy)]
pub struct VolPoint {
    pub moneyness: f64,      // strike / spot
    pub time_to_expiry: f64, // in years
    pub implied_vol: f64,
}

/// Cubic spline coefficients for interpolation
#[derive(Debug, Clone)]
struct SplineCoefficients {
    a: Vec<f64>, // function values
    b: Vec<f64>, // first derivative coefficients
    c: Vec<f64>, // second derivative coefficients
    d: Vec<f64>, // third derivative coefficients
    x: Vec<f64>, // knot positions
}

impl SplineCoefficients {
    /// Build natural cubic spline from data points
    fn new(x: Vec<f64>, y: Vec<f64>) -> Option<Self> {
        let n = x.len();
        if n < 2 || x.len() != y.len() {
            return None;
        }

        let mut a = y.clone();
        let mut c = vec![0.0; n];
        let mut h = Vec::with_capacity(n - 1);
        
        for i in 0..n - 1 {
            h.push(x[i + 1] - x[i]);
        }

        // Solve tridiagonal system for natural spline
        let mut alpha = vec![0.0; n];
        let mut l = vec![1.0; n];
        let mut mu = vec![0.0; n];
        let mut z = vec![0.0; n];

        for i in 1..n - 1 {
            alpha[i] = (3.0 / h[i]) * (a[i + 1] - a[i]) 
                     - (3.0 / h[i - 1]) * (a[i] - a[i - 1]);
        }

        for i in 1..n - 1 {
            l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        c[n - 1] = 0.0;
        let mut b = vec![0.0; n];
        let mut d = vec![0.0; n];

        for j in (0..n - 1).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
        }

        Some(Self { a, b, c, d, x })
    }

    /// Evaluate spline at given point
    fn evaluate(&self, x_val: f64) -> Option<f64> {
        if self.x.is_empty() {
            return None;
        }

        // Find the right interval using binary search
        let mut lo = 0;
        let mut hi = self.x.len() - 1;

        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if self.x[mid] <= x_val {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }

        let i = lo.min(self.x.len() - 2);
        let dx = x_val - self.x[i];

        Some(self.a[i] + self.b[i] * dx + self.c[i] * dx.powi(2) + self.d[i] * dx.powi(3))
    }
}

/// Volatility surface representation with interpolation
pub struct VolatilitySurface {
    /// Map of expiry -> spline for that expiry slice
    expiry_splines: HashMap<u32, SplineCoefficients>,
    /// Raw vol points for each expiry
    vol_points: HashMap<u32, Vec<VolPoint>>,
    /// ATM vol term structure
    atm_term_structure: Vec<(f64, f64)>, // (time, vol)
    /// ATM term spline
    atm_spline: Option<SplineCoefficients>,
}

impl VolatilitySurface {
    pub fn new() -> Self {
        Self {
            expiry_splines: HashMap::new(),
            vol_points: HashMap::new(),
            atm_term_structure: Vec::new(),
            atm_spline: None,
        }
    }

    /// Add a volatility observation to the surface
    pub fn add_point(&mut self, point: VolPoint) {
        let expiry_days = (point.time_to_expiry * 365.0) as u32;
        
        self.vol_points.entry(expiry_days)
            .or_insert_with(Vec::new)
            .push(point);

        // Check if this is near ATM (moneyness close to 1)
        if (point.moneyness - 1.0).abs() < 0.02 {
            self.atm_term_structure.push((point.time_to_expiry, point.implied_vol));
        }
    }

    /// Build splines from accumulated points
    pub fn build_surface(&mut self) {
        // Build smile splines for each expiry
        for (expiry, points) in &self.vol_points {
            if points.len() >= 3 {
                let mut sorted_points = points.clone();
                sorted_points.sort_by(|a, b| a.moneyness.partial_cmp(&b.moneyness).unwrap());

                let x: Vec<f64> = sorted_points.iter().map(|p| p.moneyness).collect();
                let y: Vec<f64> = sorted_points.iter().map(|p| p.implied_vol).collect();

                if let Some(spline) = SplineCoefficients::new(x, y) {
                    self.expiry_splines.insert(*expiry, spline);
                }
            }
        }

        // Build term structure spline
        if self.atm_term_structure.len() >= 2 {
            self.atm_term_structure.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            
            let x: Vec<f64> = self.atm_term_structure.iter().map(|p| p.0).collect();
            let y: Vec<f64> = self.atm_term_structure.iter().map(|p| p.1).collect();
            
            self.atm_spline = SplineCoefficients::new(x, y);
        }
    }

    /// Get interpolated implied vol for given moneyness and expiry
    pub fn get_implied_vol(&self, moneyness: f64, expiry_days: u32) -> Option<f64> {
        // First try direct expiry lookup
        if let Some(spline) = self.expiry_splines.get(&expiry_days) {
            return spline.evaluate(moneyness);
        }

        // Find nearest expiries and interpolate in time
        let expiries: Vec<u32> = self.expiry_splines.keys().copied().collect();
        if expiries.len() < 2 {
            return None;
        }

        let mut lower_expiry = None;
        let mut upper_expiry = None;

        for &e in &expiries {
            if e <= expiry_days {
                lower_expiry = Some(e);
            }
            if e >= expiry_days && upper_expiry.is_none() {
                upper_expiry = Some(e);
            }
        }

        match (lower_expiry, upper_expiry) {
            (Some(lo), Some(hi)) if lo != hi => {
                let lo_vol = self.expiry_splines.get(&lo)?.evaluate(moneyness)?;
                let hi_vol = self.expiry_splines.get(&hi)?.evaluate(moneyness)?;
                
                // Linear interpolation in time
                let t_lo = lo as f64;
                let t_hi = hi as f64;
                let t = expiry_days as f64;
                
                if (t_hi - t_lo).abs() < 1e-9 {
                    return Some(lo_vol);
                }
                
                let weight = (t - t_lo) / (t_hi - t_lo);
                Some(lo_vol * (1.0 - weight) + hi_vol * weight)
            }
            (Some(lo), _) => self.expiry_splines.get(&lo)?.evaluate(moneyness),
            (_, Some(hi)) => self.expiry_splines.get(&hi)?.evaluate(moneyness),
            _ => None,
        }
    }

    /// Get ATM vol for given expiry
    pub fn get_atm_vol(&self, expiry_days: u32) -> Option<f64> {
        let time_years = expiry_days as f64 / 365.0;
        
        // Try direct ATM term structure
        if let Some(ref spline) = self.atm_spline {
            return spline.evaluate(time_years);
        }

        // Fallback: use smile at moneyness = 1.0
        self.get_implied_vol(1.0, expiry_days)
    }

    /// Calculate skew: difference between OTM put vol and ATM vol
    pub fn get_skew(&self, expiry_days: u32, delta: f64) -> Option<f64> {
        let atm_vol = self.get_atm_vol(expiry_days)?;
        let otm_moneyness = 1.0 - delta; // For puts
        let otm_vol = self.get_implied_vol(otm_moneyness, expiry_days)?;
        Some(otm_vol - atm_vol)
    }

    /// Get the full vol surface as a grid
    pub fn get_surface_grid(&self, moneyness_range: (f64, f64), num_strikes: usize) -> Vec<Vec<Option<f64>>> {
        let expiries: Vec<u32> = self.expiry_splines.keys().copied().collect();
        let mut grid = Vec::with_capacity(expiries.len());

        let step = (moneyness_range.1 - moneyness_range.0) / (num_strikes - 1) as f64;

        for expiry in expiries {
            let mut row = Vec::with_capacity(num_strikes);
            for i in 0..num_strikes {
                let m = moneyness_range.0 + i as f64 * step;
                row.push(self.get_implied_vol(m, expiry));
            }
            grid.push(row);
        }

        grid
    }

    /// Number of vol points loaded
    pub fn point_count(&self) -> usize {
        self.vol_points.values().map(|v| v.len()).sum()
    }

    /// Clear the surface
    pub fn clear(&mut self) {
        self.expiry_splines.clear();
        self.vol_points.clear();
        self.atm_term_structure.clear();
        self.atm_spline = None;
    }
}

impl Default for VolatilitySurface {
    fn default() -> Self {
        Self::new()
    }
}

/// Black-Scholes utilities for IV calculation
pub mod black_scholes {
    const SQRT_2PI: f64 = 2.5066282746310002;

    /// Standard normal CDF approximation
    fn norm_cdf(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let d = 0.3989423 * (-x * x / 2.0).exp();
        let prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        if x > 0.0 {
            1.0 - prob
        } else {
            prob
        }
    }

    /// Calculate call option price
    pub fn call_price(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
        if t <= 0.0 || sigma <= 0.0 {
            return (s - k).max(0.0);
        }

        let d1 = (s / k).ln() + (r + sigma * sigma / 2.0) * t;
        let d1 = d1 / (sigma * t.sqrt());
        let d2 = d1 - sigma * t.sqrt();

        s * norm_cdf(d1) - k * (-r * t).exp() * norm_cdf(d2)
    }

    /// Calculate put option price
    pub fn put_price(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
        if t <= 0.0 || sigma <= 0.0 {
            return (k - s).max(0.0);
        }

        let d1 = (s / k).ln() + (r + sigma * sigma / 2.0) * t;
        let d1 = d1 / (sigma * t.sqrt());
        let d2 = d1 - sigma * t.sqrt();

        k * (-r * t).exp() * norm_cdf(-d2) - s * norm_cdf(-d1)
    }

    /// Newton-Raphson implied vol solver
    pub fn implied_vol(price: f64, s: f64, k: f64, t: f64, r: f64, is_call: bool) -> Option<f64> {
        if t <= 0.0 || price <= 0.0 {
            return None;
        }

        let mut sigma = 0.5; // Initial guess
        const MAX_ITER: usize = 50;
        const TOLERANCE: f64 = 1e-8;

        for _ in 0..MAX_ITER {
            let bs_price = if is_call {
                call_price(s, k, t, r, sigma)
            } else {
                put_price(s, k, t, r, sigma)
            };

            let diff = bs_price - price;
            if diff.abs() < TOLERANCE {
                return Some(sigma);
            }

            // Vega: same for calls and puts
            let d1 = (s / k).ln() + (r + sigma * sigma / 2.0) * t;
            let d1 = d1 / (sigma * t.sqrt());
            let vega = s * t.sqrt() * (-d1 * d1 / 2.0).exp() / SQRT_2PI;

            if vega < 1e-12 {
                break;
            }

            sigma = sigma - diff / vega;
            sigma = sigma.max(0.001).min(5.0);
        }

        Some(sigma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volatility_surface_basic() {
        let mut surface = VolatilitySurface::new();

        // Add some vol points
        surface.add_point(VolPoint {
            moneyness: 0.9,
            time_to_expiry: 30.0 / 365.0,
            implied_vol: 0.75,
        });
        surface.add_point(VolPoint {
            moneyness: 1.0,
            time_to_expiry: 30.0 / 365.0,
            implied_vol: 0.70,
        });
        surface.add_point(VolPoint {
            moneyness: 1.1,
            time_to_expiry: 30.0 / 365.0,
            implied_vol: 0.72,
        });

        surface.build_surface();

        let vol = surface.get_implied_vol(1.0, 30);
        assert!(vol.is_some());
        assert!((vol.unwrap() - 0.70).abs() < 0.05);
    }

    #[test]
    fn test_black_scholes() {
        let iv = black_scholes::implied_vol(5000.0, 50000.0, 50000.0, 30.0 / 365.0, 0.05, true);
        assert!(iv.is_some());
    }
}
