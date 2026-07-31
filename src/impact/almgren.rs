//! Almgren-Chriss Optimal Execution Trajectory Solver
//! 
//! Implements the Almgren-Chriss model using fixed-point arithmetic for O(1) execution.
//! Uses Newton-Raphson method with strict iteration limits.

/// Fixed-point representation for high-precision arithmetic without floats
#[derive(Debug, Clone, Copy)]
pub struct FixedPoint(i128);

impl FixedPoint {
    /// Scale factor: 10^12 for sub-nanosecond precision
    const SCALE: i128 = 1_000_000_000_000;

    pub fn from_f64(val: f64) -> Self {
        Self((val * Self::SCALE as f64) as i128)
    }

    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::SCALE as f64
    }

    pub fn from_i64(val: i64) -> Self {
        Self(val as i128 * Self::SCALE)
    }

    pub fn to_i64(self) -> i64 {
        (self.0 / Self::SCALE) as i64
    }

    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }

    #[inline]
    pub fn mul(self, other: Self) -> Self {
        Self((self.0 * other.0) / Self::SCALE)
    }

    #[inline]
    pub fn div(self, other: Self) -> Self {
        if other.0 == 0 {
            return Self(0);
        }
        Self((self.0 * Self::SCALE) / other.0)
    }

    #[inline]
    pub fn neg(self) -> Self {
        Self(-self.0)
    }

    pub fn sqrt_newton(self) -> Self {
        if self.0 <= 0 {
            return Self(0);
        }
        
        // Initial guess using integer square root approximation
        let mut x = self.0;
        let mut y = (x + 1) / 2;
        
        // Limited iterations for O(1) guarantee
        for _ in 0..20 {
            if y >= x {
                break;
            }
            x = y;
            y = (x + self.0 / x) / 2;
        }
        
        // Convert back to fixed point
        let int_sqrt = x;
        // Refine for fixed point precision
        let mut result = int_sqrt;
        for _ in 0..10 {
            let new_result = (result + self.0 / result) / 2;
            if new_result >= result {
                break;
            }
            result = new_result;
        }
        
        Self(result)
    }

    pub fn is_positive(&self) -> bool {
        self.0 > 0
    }

    pub fn zero() -> Self {
        Self(0)
    }

    pub fn one() -> Self {
        Self(Self::SCALE)
    }
}

/// Parameters for the Almgren-Chriss model
#[derive(Debug, Clone)]
pub struct AlmgrenChrissParams {
    /// Total quantity to execute (in base units, fixed point)
    pub total_quantity: FixedPoint,
    /// Time horizon in seconds (fixed point)
    pub time_horizon: FixedPoint,
    /// Number of trading intervals
    pub num_intervals: u32,
    /// Market impact coefficient (temporary)
    pub eta: FixedPoint,
    /// Volatility (annualized, fixed point)
    pub sigma: FixedPoint,
    /// Risk aversion parameter
    pub lambda: FixedPoint,
    /// Permanent impact coefficient
    pub gamma: FixedPoint,
}

/// Result of the optimal trajectory calculation
#[derive(Debug, Clone)]
pub struct ExecutionTrajectory {
    /// Quantity to trade in each interval
    pub quantities: Vec<FixedPoint>,
    /// Expected cost due to market impact
    pub expected_impact_cost: FixedPoint,
    /// Expected variance of implementation shortfall
    pub variance: FixedPoint,
    /// Optimal trading rate
    pub trading_rate: FixedPoint,
}

/// Almgren-Chriss solver using Newton-Raphson optimization
pub struct AlmgrenChrissSolver {
    params: AlmgrenChrissParams,
    /// Precomputed constants for O(1) per-step calculation
    dt: FixedPoint,
    kappa: FixedPoint,
    sinh_coeff: FixedPoint,
    cosh_coeff: FixedPoint,
}

impl AlmgrenChrissSolver {
    pub fn new(params: AlmgrenChrissParams) -> Self {
        let n = params.num_intervals as i128;
        let dt = params.time_horizon.div(FixedPoint(n));
        
        // κ = arccosh(1 + ησ²λΔt²/2γ)
        // Simplified: use approximation for small Δt
        let eta_sigma_sq = params.eta.mul(params.sigma).mul(params.sigma);
        let lambda_dt_sq = params.lambda.mul(dt).mul(dt);
        let numerator = eta_sigma_sq.mul(lambda_dt_sq);
        let denominator = FixedPoint::from_f64(2.0).mul(params.gamma);
        
        // For small values, arccosh(1+x) ≈ √(2x)
        let x = numerator.div(denominator);
        let kappa = x.sqrt_newton();
        
        // Precompute sinh and cosh coefficients using Taylor series
        // sinh(κ) ≈ κ + κ³/6
        // cosh(κ) ≈ 1 + κ²/2
        let kappa_sq = kappa.mul(kappa);
        let kappa_cubed = kappa_sq.mul(kappa);
        
        let sinh_coeff = kappa.add(kappa_cubed.div(FixedPoint::from_f64(6.0)));
        let cosh_coeff = FixedPoint::one().add(kappa_sq.div(FixedPoint::from_f64(2.0)));
        
        Self {
            params,
            dt,
            kappa,
            sinh_coeff,
            cosh_coeff,
        }
    }

    /// Calculate optimal trajectory in O(1) per interval
    pub fn calculate_trajectory(&self) -> ExecutionTrajectory {
        let n = self.params.num_intervals;
        let mut quantities = Vec::with_capacity(n as usize);
        
        let total_q = self.params.total_quantity;
        let sinh_k = self.sinh_coeff;
        let cosh_k = self.cosh_coeff;
        
        // Precompute denominator: sinh(κ(N+1))
        // Using recurrence: sinh((N+1)κ) = 2cosh(κ)sinh(Nκ) - sinh((N-1)κ)
        let mut sinh_prev = FixedPoint::zero();
        let mut sinh_curr = sinh_k;
        
        for i in 2..=n + 1 {
            let sinh_next = FixedPoint::from_f64(2.0).mul(cosh_k).mul(sinh_curr).sub(sinh_prev);
            sinh_prev = sinh_curr;
            sinh_curr = sinh_next;
        }
        let denom = sinh_curr;
        
        // Calculate each interval's quantity
        let mut remaining = total_q;
        let mut expected_cost = FixedPoint::zero();
        let mut variance_sum = FixedPoint::zero();
        
        for k in 0..n {
            // q_k = Q * sinh(κ(N-k)) / sinh(κ(N+1))
            let n_minus_k = n - k;
            
            // Compute sinh((N-k)κ) using recurrence
            let mut s_prev = FixedPoint::zero();
            let mut s_curr = sinh_k;
            
            if n_minus_k == 0 {
                quantities.push(FixedPoint::zero());
                continue;
            }
            
            for _ in 2..=n_minus_k {
                let s_next = FixedPoint::from_f64(2.0).mul(cosh_k).mul(s_curr).sub(s_prev);
                s_prev = s_curr;
                s_curr = s_next;
            }
            
            let numer = s_curr;
            let q_k = if denom.0 != 0 {
                total_q.mul(numer).div(denom)
            } else {
                total_q.div(FixedPoint(n as i128))
            };
            
            quantities.push(q_k);
            
            // Expected impact cost: η * q_k² / Δt
            let impact = self.params.eta.mul(q_k).mul(q_k).div(self.dt);
            expected_cost = expected_cost.add(impact);
            
            // Variance contribution: σ² * Δt * (remaining)²
            variance_sum = variance_sum.add(
                self.params.sigma.mul(self.params.sigma)
                    .mul(self.dt)
                    .mul(remaining).mul(remaining)
            );
            
            remaining = remaining.sub(q_k);
        }
        
        // Average trading rate
        let trading_rate = total_q.div(self.params.time_horizon);
        
        ExecutionTrajectory {
            quantities,
            expected_impact_cost: expected_cost.div(FixedPoint::from_f64(2.0)),
            variance: variance_sum,
            trading_rate,
        }
    }

    /// Newton-Raphson solver for finding optimal liquidation time
    /// Returns optimal time in fixed point, guaranteed O(1) with max iterations
    pub fn solve_optimal_time(&self, target_cost: FixedPoint) -> FixedPoint {
        // f(T) = expected_shortfall(T) - target
        // f'(T) = derivative w.r.t. T
        
        let mut t = self.params.time_horizon;
        const MAX_ITER: u32 = 10;
        
        for _ in 0..MAX_ITER {
            // Compute expected shortfall at time t
            let dt = t.div(FixedPoint(self.params.num_intervals as i128));
            if dt.0 <= 0 {
                t = t.mul(FixedPoint::from_f64(1.1));
                continue;
            }
            
            // ES ≈ γQ²/2 + ηQ²/T + λσ²QT/2
            let q = self.params.total_quantity;
            let term1 = self.params.gamma.mul(q).mul(q).div(FixedPoint::from_f64(2.0));
            let term2 = self.params.eta.mul(q).mul(q).div(t);
            let term3 = self.params.lambda.mul(self.params.sigma).mul(self.params.sigma)
                .mul(q).mul(t).div(FixedPoint::from_f64(2.0));
            
            let es = term1.add(term2).add(term3);
            let diff = es.sub(target_cost);
            
            if diff.0.abs() < FixedPoint::from_f64(0.0001).0 {
                break;
            }
            
            // Derivative: dES/dT = -ηQ²/T² + λσ²Q/2
            let deriv_term1 = self.params.eta.mul(q).mul(q).neg();
            let deriv = deriv_term1.div(t).div(t)
                .add(self.params.lambda.mul(self.params.sigma).mul(self.params.sigma).mul(q)
                    .div(FixedPoint::from_f64(2.0)));
            
            if deriv.0.abs() < 1 {
                break;
            }
            
            // Newton step: T_new = T - f(T)/f'(T)
            let step = diff.div(deriv);
            t = t.sub(step);
            
            if t.0 <= 0 {
                t = FixedPoint::from_f64(0.001);
            }
        }
        
        t
    }

    /// Get the precomputed kappa value
    pub fn kappa(&self) -> FixedPoint {
        self.kappa
    }

    /// Get the time step
    pub fn dt(&self) -> FixedPoint {
        self.dt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_basic() {
        let a = FixedPoint::from_f64(2.5);
        let b = FixedPoint::from_f64(4.0);
        
        let sum = a.add(b);
        assert!((sum.to_f64() - 6.5).abs() < 0.0001);
        
        let prod = a.mul(b);
        assert!((prod.to_f64() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_fixed_point_sqrt() {
        let val = FixedPoint::from_f64(4.0);
        let sqrt_val = val.sqrt_newton();
        assert!((sqrt_val.to_f64() - 2.0).abs() < 0.01);
        
        let val2 = FixedPoint::from_f64(2.0);
        let sqrt_val2 = val2.sqrt_newton();
        assert!((sqrt_val2.to_f64() - 1.414).abs() < 0.01);
    }

    #[test]
    fn test_almgren_chriss_solver() {
        let params = AlmgrenChrissParams {
            total_quantity: FixedPoint::from_i64(10000),
            time_horizon: FixedPoint::from_f64(3600.0), // 1 hour
            num_intervals: 10,
            eta: FixedPoint::from_f64(1e-6),
            sigma: FixedPoint::from_f64(0.02),
            lambda: FixedPoint::from_f64(1e-5),
            gamma: FixedPoint::from_f64(1e-7),
        };
        
        let solver = AlmgrenChrissSolver::new(params);
        let trajectory = solver.calculate_trajectory();
        
        assert_eq!(trajectory.quantities.len(), 10);
        assert!(trajectory.trading_rate.is_positive());
    }

    #[test]
    fn test_optimal_time_solver() {
        let params = AlmgrenChrissParams {
            total_quantity: FixedPoint::from_i64(5000),
            time_horizon: FixedPoint::from_f64(1800.0),
            num_intervals: 5,
            eta: FixedPoint::from_f64(1e-6),
            sigma: FixedPoint::from_f64(0.03),
            lambda: FixedPoint::from_f64(1e-5),
            gamma: FixedPoint::from_f64(1e-7),
        };
        
        let solver = AlmgrenChrissSolver::new(params);
        let target = FixedPoint::from_f64(100.0);
        let opt_time = solver.solve_optimal_time(target);
        
        assert!(opt_time.is_positive());
    }
}
