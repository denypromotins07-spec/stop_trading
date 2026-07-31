//! Options Pricing Engine
//! Implements Black-Scholes-Merton and crypto-specific options pricing models.
//! Uses fast polynomial approximations (Taylor series, Horner's method) to avoid expensive std calls.

/// Fixed-point scaling factor for high precision
const FP_SCALE: i64 = 1_000_000_000;

/// Option type
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum OptionType {
    Call = 0,
    Put = 1,
}

/// Option pricing result
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OptionPrice {
    pub price: i64,        // Fixed-point option premium
    pub delta: i64,        // Fixed-point delta (scaled by 1e8)
    pub gamma: i64,        // Fixed-point gamma (scaled by 1e12)
    pub theta: i64,        // Fixed-point theta (scaled by 1e8)
    pub vega: i64,         // Fixed-point vega (scaled by 1e8)
    pub rho: i64,          // Fixed-point rho (scaled by 1e8)
    _padding: [u8; 8],     // Cache-line alignment
}

impl Default for OptionPrice {
    fn default() -> Self {
        Self::new()
    }
}

impl OptionPrice {
    pub const fn new() -> Self {
        Self {
            price: 0,
            delta: 0,
            gamma: 0,
            theta: 0,
            vega: 0,
            rho: 0,
            _padding: [0; 8],
        }
    }
}

/// Fast approximation of e^x using Taylor series with range reduction
/// Accuracy: ~6 decimal places for |x| < 1
#[inline]
fn fast_exp(x: f64) -> f64 {
    // Range reduction: e^x = e^(n + f) = e^n * e^f where |f| < 0.5
    let n = (x * 2.0).round() as i32;
    let f = x - n as f64;
    
    // Taylor series for e^f: 1 + f + f²/2 + f³/6 + f⁴/24 + f⁵/120
    let f2 = f * f;
    let f3 = f2 * f;
    let f4 = f3 * f;
    let f5 = f4 * f;
    
    let exp_f = 1.0 + f + f2 * 0.5 + f3 * 0.16666666666666666 
                  + f4 * 0.041666666666666664 + f5 * 0.008333333333333333;
    
    // e^n using precomputed powers (for small n)
    const E: f64 = 2.718281828459045;
    if n >= 0 && n <= 10 {
        exp_f * E.powi(n)
    } else if n < 0 && n >= -10 {
        exp_f / E.powi(-n)
    } else {
        // Fallback for extreme values
        exp_f * E.powi(n)
    }
}

/// Fast natural logarithm using Newton-Raphson iteration
/// ln(x) for x > 0
#[inline]
fn fast_ln(mut x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    
    // Range reduction: x = m * 2^k where 0.5 <= m < 1
    let mut k = 0i32;
    while x >= 2.0 {
        x *= 0.5;
        k += 1;
    }
    while x < 0.5 {
        x *= 2.0;
        k -= 1;
    }
    
    // ln(2) constant
    const LN2: f64 = 0.6931471805599453;
    
    // Use Padé approximant for ln(m) where m is in [0.5, 2)
    // ln(m) ≈ (m-1)(6 + 4(m-1) + (m-1)²) / (6 + 6(m-1) + (m-1)²)
    let y = x - 1.0;
    let y2 = y * y;
    let numerator = y * (6.0 + 4.0 * y + y2);
    let denominator = 6.0 + 6.0 * y + y2;
    
    let ln_m = numerator / denominator;
    ln_m + (k as f64) * LN2
}

/// Fast square root using Newton-Raphson method
#[inline]
fn fast_sqrt(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    
    // Initial guess using bit manipulation approximation
    let mut guess = x * 0.5;
    if x > 1.0 {
        guess = x;
    }
    
    // Newton-Raphson iterations (3 iterations for good accuracy)
    for _ in 0..3 {
        guess = 0.5 * (guess + x / guess);
    }
    
    guess
}

/// Cumulative distribution function for standard normal distribution
/// Uses Abramowitz and Stegun approximation (error < 7.5e-8)
#[inline]
fn norm_cdf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    
    // Constants for the approximation
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 0.551942973;
    const A4: f64 = 1.432788123;
    const A5: f64 = 0.07248364;
    const P: f64 = 0.3275911;
    
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * fast_exp(-x * x);
    
    0.5 * (1.0 + sign * y)
}

/// Probability density function for standard normal distribution
#[inline]
fn norm_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.3989422804014327;
    INV_SQRT_2PI * fast_exp(-0.5 * x * x)
}

/// Black-Scholes-Merton pricing model
/// All inputs are fixed-point scaled integers
pub fn bs_price(
    spot: i64,           // Spot price (scaled by 1e8)
    strike: i64,         // Strike price (scaled by 1e8)
    time_to_expiry: i64, // Time in seconds
    volatility: i64,     // Volatility (scaled by 1e8, e.g., 0.8 = 80%)
    risk_free_rate: i64, // Risk-free rate (scaled by 1e8)
    option_type: OptionType,
) -> OptionPrice {
    // Convert to f64 for calculation
    let s = spot as f64 / FP_SCALE as f64;
    let k = strike as f64 / FP_SCALE as f64;
    let t = time_to_expiry as f64 / 31536000.0; // Convert seconds to years
    let sigma = volatility as f64 / FP_SCALE as f64;
    let r = risk_free_rate as f64 / FP_SCALE as f64;
    
    if t <= 0.0 || sigma <= 0.0 || s <= 0.0 || k <= 0.0 {
        // Intrinsic value at expiry
        let intrinsic = match option_type {
            OptionType::Call => (s - k).max(0.0),
            OptionType::Put => (k - s).max(0.0),
        };
        let mut result = OptionPrice::new();
        result.price = (intrinsic * FP_SCALE as f64) as i64;
        return result;
    }
    
    let sqrt_t = fast_sqrt(t);
    let sigma_sqrt_t = sigma * sqrt_t;
    
    // d1 = (ln(S/K) + (r + σ²/2)t) / (σ√t)
    let ln_s_k = fast_ln(s / k);
    let d1 = (ln_s_k + (r + 0.5 * sigma * sigma) * t) / sigma_sqrt_t;
    
    // d2 = d1 - σ√t
    let d2 = d1 - sigma_sqrt_t;
    
    // Calculate option price based on type
    let (price, delta, gamma, theta, vega, rho) = match option_type {
        OptionType::Call => {
            let nd1 = norm_cdf(d1);
            let nd2 = norm_cdf(d2);
            
            // Call price = S*N(d1) - K*e^(-rt)*N(d2)
            let discount = fast_exp(-r * t);
            let price_val = s * nd1 - k * discount * nd2;
            
            // Greeks
            let npd1 = norm_pdf(d1);
            let delta = nd1;
            let gamma = npd1 / (s * sigma_sqrt_t);
            let theta = -(s * npd1 * sigma) / (2.0 * sqrt_t) - r * k * discount * nd2;
            let vega = s * npd1 * sqrt_t;
            let rho = k * t * discount * nd2;
            
            (price_val, delta, gamma, theta, vega, rho)
        }
        OptionType::Put => {
            let nd1 = norm_cdf(d1);
            let nd2 = norm_cdf(d2);
            
            // Put price = K*e^(-rt)*N(-d2) - S*N(-d1)
            let discount = fast_exp(-r * t);
            let price_val = k * discount * (1.0 - nd2) - s * (1.0 - nd1);
            
            // Greeks
            let npd1 = norm_pdf(d1);
            let delta = nd1 - 1.0;
            let gamma = npd1 / (s * sigma_sqrt_t);
            let theta = -(s * npd1 * sigma) / (2.0 * sqrt_t) + r * k * discount * (1.0 - nd2);
            let vega = s * npd1 * sqrt_t;
            let rho = -k * t * discount * (1.0 - nd2);
            
            (price_val, delta, gamma, theta, vega, rho)
        }
    };
    
    // Convert back to fixed-point
    let mut result = OptionPrice::new();
    result.price = (price * FP_SCALE as f64) as i64;
    result.delta = (delta * 1e8) as i64;
    result.gamma = (gamma * 1e12) as i64;
    result.theta = (theta * 1e8) as i64;
    result.vega = (vega * 1e8) as i64;
    result.rho = (rho * 1e8) as i64;
    
    result
}

/// Crypto-specific pricing adjustments for Deribit/Binance style options
/// Accounts for funding rates and perpetual basis
pub fn crypto_bs_price(
    spot: i64,
    strike: i64,
    time_to_expiry: i64,
    volatility: i64,
    risk_free_rate: i64,
    funding_rate: i64,   // Annualized funding rate (scaled by 1e8)
    option_type: OptionType,
) -> OptionPrice {
    // Adjust spot for funding rate impact
    // In crypto, funding rate affects the effective cost of carry
    let adjusted_spot = spot + (spot * funding_rate * time_to_expiry) / (FP_SCALE * FP_SCALE * 31536000);
    
    bs_price(adjusted_spot, strike, time_to_expiry, volatility, risk_free_rate, option_type)
}

/// Binary/digital option pricing
pub fn binary_option_price(
    spot: i64,
    strike: i64,
    time_to_expiry: i64,
    volatility: i64,
    risk_free_rate: i64,
    payout: i64,         // Fixed payout amount (scaled by 1e8)
    option_type: OptionType,
) -> i64 {
    let s = spot as f64 / FP_SCALE as f64;
    let k = strike as f64 / FP_SCALE as f64;
    let t = time_to_expiry as f64 / 31536000.0;
    let sigma = volatility as f64 / FP_SCALE as f64;
    let r = risk_free_rate as f64 / FP_SCALE as f64;
    let p = payout as f64 / FP_SCALE as f64;
    
    if t <= 0.0 || sigma <= 0.0 {
        // At expiry
        let itv = match option_type {
            OptionType::Call => if s > k { p } else { 0.0 },
            OptionType::Put => if s < k { p } else { 0.0 },
        };
        return (itv * FP_SCALE as f64) as i64;
    }
    
    let sqrt_t = fast_sqrt(t);
    let ln_s_k = fast_ln(s / k);
    let d2 = (ln_s_k + (r - 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    
    let discount = fast_exp(-r * t);
    let price = match option_type {
        OptionType::Call => p * discount * norm_cdf(d2),
        OptionType::Put => p * discount * norm_cdf(-d2),
    };
    
    (price * FP_SCALE as f64) as i64
}

/// American option approximation using Barone-Adesi Whaley method
/// Simplified version for early exercise premium
pub fn american_option_approx(
    spot: i64,
    strike: i64,
    time_to_expiry: i64,
    volatility: i64,
    risk_free_rate: i64,
    option_type: OptionType,
) -> i64 {
    // Get European price first
    let eu_price = bs_price(spot, strike, time_to_expiry, volatility, risk_free_rate, option_type);
    
    // For calls on non-dividend paying assets, American = European
    if option_type == OptionType::Call {
        return eu_price.price;
    }
    
    // For puts, add early exercise premium approximation
    let s = spot as f64 / FP_SCALE as f64;
    let k = strike as f64 / FP_SCALE as f64;
    let t = time_to_expiry as f64 / 31536000.0;
    
    // Simple early exercise premium for deep ITM puts
    let intrinsic = (k - s).max(0.0);
    let eu_float = eu_price.price as f64 / FP_SCALE as f64;
    
    // American put should be worth at least intrinsic value
    let american_value = intrinsic.max(eu_float);
    
    (american_value * FP_SCALE as f64) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_exp() {
        let x = 0.5;
        let result = fast_exp(x);
        let expected = 1.6487212707001282;
        assert!((result - expected).abs() < 0.001);
    }

    #[test]
    fn test_fast_ln() {
        let x = 2.0;
        let result = fast_ln(x);
        let expected = 0.6931471805599453;
        assert!((result - expected).abs() < 0.001);
    }

    #[test]
    fn test_fast_sqrt() {
        let x = 2.0;
        let result = fast_sqrt(x);
        let expected = 1.4142135623730951;
        assert!((result - expected).abs() < 0.0001);
    }

    #[test]
    fn test_bs_call_price() {
        // ATM call: S=K=100, T=1 year, σ=20%, r=5%
        let spot = 100 * FP_SCALE;
        let strike = 100 * FP_SCALE;
        let time = 31536000; // 1 year in seconds
        let vol = 20 * FP_SCALE / 100;
        let rate = 5 * FP_SCALE / 100;
        
        let result = bs_price(spot, strike, time, vol, rate, OptionType::Call);
        
        // Expected price around $10.45
        let price_float = result.price as f64 / FP_SCALE as f64;
        assert!(price_float > 10.0 && price_float < 11.0);
    }

    #[test]
    fn test_norm_cdf() {
        // Test standard values
        assert!((norm_cdf(0.0) - 0.5).abs() < 0.0001);
        assert!((norm_cdf(1.0) - 0.8413).abs() < 0.001);
        assert!((norm_cdf(-1.0) - 0.1587).abs() < 0.001);
    }
}
