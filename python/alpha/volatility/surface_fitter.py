"""
Volatility Surface Fitter using SVI and SABR parameterization.
Real-time implied volatility surface fitting for crypto options chains.
Optimized with strict iteration limits to prevent CPU blocking.
Strictly NumPy/SciPy based - no Pandas in hot path.
"""

import numpy as np
from scipy import optimize
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class SurfaceModel(Enum):
    """Volatility surface model types."""
    SVI = "svi"           # Stochastic Volatility Inspired
    SABR = "sabr"         # SABR model
    HYBRID = "hybrid"     # SVI + SABR hybrid


@dataclass
class SVIParameters:
    """SVI model parameters."""
    a: float      # Overall level
    b: float      # Slope
    rho: float    # Correlation parameter
    m: float      # ATM forward log-moneyness shift
    sigma: float  # Volatility of volatility
    
    def validate(self) -> bool:
        """Check if parameters are valid (no arbitrage)."""
        return (self.b >= 0 and 
                abs(self.rho) <= 1 and 
                self.sigma > 0)


@dataclass
class SABRParameters:
    """SABR model parameters."""
    alpha: float  # Initial volatility
    beta: float   # Elasticity (usually fixed)
    rho: float    # Correlation
    nu: float     # Volatility of volatility
    
    def validate(self) -> bool:
        """Check if parameters are valid."""
        return (self.alpha > 0 and 
                0 <= self.beta <= 1 and 
                abs(self.rho) < 1 and 
                self.nu > 0)


@dataclass
class SurfaceFitResult:
    """Result from volatility surface fitting."""
    model: SurfaceModel
    params: object  # SVIParameters or SABRParameters
    rmse: float
    max_error: float
    convergence: bool
    n_iterations: int
    timestamp_ns: int


class SVIFitter:
    """
    Fits the Stochastic Volatility Inspired (SVI) model to implied volatility data.
    
    SVI parameterization:
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    
    where w is total variance (sigma_imp^2 * T), k is log-moneyness.
    
    Reference: Gatheral & Jacquier (2014) "Arbitrage-free SVI volatility surfaces"
    """
    
    def __init__(self, 
                 max_iterations: int = 50,
                 tolerance: float = 1e-6,
                 min_tenor: float = 0.01,
                 max_tenor: float = 2.0):
        """
        Args:
            max_iterations: Maximum optimization iterations (CPU safety)
            tolerance: Convergence tolerance
            min_tenor: Minimum time to expiry in years
            max_tenor: Maximum time to expiry in years
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.min_tenor = min_tenor
        self.max_tenor = max_tenor
        
        # Parameter bounds for optimization
        self.param_bounds = {
            'a': (0.0, 2.0),
            'b': (0.0, 1.0),
            'rho': (-1.0, 1.0),
            'm': (-0.5, 0.5),
            'sigma': (0.01, 2.0)
        }
        
    def svi_variance(self, k: np.ndarray, params: SVIParameters) -> np.ndarray:
        """
        Calculate SVI total variance for given log-moneyness.
        
        Args:
            k: Log-moneyness array (ln(K/F))
            params: SVI parameters
            
        Returns:
            Total variance array
        """
        a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma
        
        term = np.sqrt((k - m) ** 2 + sigma ** 2)
        w = a + b * (rho * (k - m) + term)
        
        return w
    
    def fit(self, 
            log_moneyness: np.ndarray,
            implied_vol: np.ndarray,
            tenor: float,
            initial_guess: Optional[SVIParameters] = None) -> SurfaceFitResult:
        """
        Fit SVI model to implied volatility data.
        
        Args:
            log_moneyness: Array of log-moneyness values
            implied_vol: Array of implied volatilities
            tenor: Time to expiry in years
            initial_guess: Optional initial parameter guess
            
        Returns:
            SurfaceFitResult with fitted parameters
        """
        if len(log_moneyness) < 3:
            return SurfaceFitResult(
                model=SurfaceModel.SVI,
                params=None,
                rmse=float('inf'),
                max_error=float('inf'),
                convergence=False,
                n_iterations=0,
                timestamp_ns=0
            )
        
        # Convert to total variance
        w_market = implied_vol ** 2 * tenor
        
        # Initial guess
        if initial_guess is None:
            initial_params = self._initial_guess(log_moneyness, w_market)
        else:
            initial_params = [initial_guess.a, initial_guess.b, 
                             initial_guess.rho, initial_guess.m, initial_guess.sigma]
        
        # Objective function: sum of squared errors
        def objective(params):
            svi_params = SVIParameters(*params)
            w_model = self.svi_variance(log_moneyness, svi_params)
            
            # Penalize invalid parameters
            if not svi_params.validate():
                return 1e10
            
            # Ensure positive variance
            if np.any(w_model < 0):
                return 1e10
            
            return np.sum((w_model - w_market) ** 2)
        
        # Bounds for L-BFGS-B
        bounds = [
            self.param_bounds['a'],
            self.param_bounds['b'],
            self.param_bounds['rho'],
            self.param_bounds['m'],
            self.param_bounds['sigma']
        ]
        
        # Optimize with strict iteration limit
        result = optimize.minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        # Extract results
        fitted_params = SVIParameters(*result.x)
        
        # Calculate metrics
        w_fitted = self.svi_variance(log_moneyness, fitted_params)
        residuals = w_fitted - w_market
        
        rmse = np.sqrt(np.mean(residuals ** 2))
        max_error = np.max(np.abs(residuals))
        
        return SurfaceFitResult(
            model=SurfaceModel.SVI,
            params=fitted_params,
            rmse=rmse,
            max_error=max_error,
            convergence=result.success,
            n_iterations=result.nit,
            timestamp_ns=0
        )
    
    def _initial_guess(self, k: np.ndarray, w: np.ndarray) -> List[float]:
        """Generate reasonable initial parameter guess."""
        # Simple heuristic initialization
        a = np.median(w) * 0.5
        b = 0.3
        rho = 0.0
        m = k[np.argmin(np.abs(w - np.median(w)))]
        sigma = 0.3
        
        return [a, b, rho, m, sigma]
    
    def get_implied_vol(self, 
                        params: SVIParameters,
                        log_moneyness: np.ndarray,
                        tenor: float) -> np.ndarray:
        """Convert SVI variance back to implied volatility."""
        w = self.svi_variance(log_moneyness, params)
        iv = np.sqrt(w / tenor)
        return np.clip(iv, 0.01, 5.0)  # Sanity bounds


class SABRFitter:
    """
    Fits the SABR stochastic volatility model to implied volatility data.
    
    SABR model provides closed-form approximation for implied volatility.
    
    Reference: Hagan et al. (2002) "Managing Smile Risk"
    """
    
    def __init__(self,
                 max_iterations: int = 50,
                 tolerance: float = 1e-6,
                 beta_fixed: float = 0.5):
        """
        Args:
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
            beta_fixed: Fixed beta parameter (typically 0.5 for crypto)
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.beta_fixed = beta_fixed
        
    def sabr_implied_vol(self, 
                         F: float, K: np.ndarray, T: float,
                         params: SABRParameters) -> np.ndarray:
        """
        Calculate SABR implied volatility using Hagan's approximation.
        
        Args:
            F: Forward price
            K: Strike array
            T: Time to expiry
            params: SABR parameters
            
        Returns:
            Implied volatility array
        """
        alpha, beta, rho, nu = params.alpha, params.beta, params.rho, params.nu
        
        # Handle ATM case separately
        atm_mask = np.abs(K - F) < 1e-10 * F
        non_atm_mask = ~atm_mask
        
        iv = np.zeros_like(K)
        
        # ATM volatility
        if np.any(atm_mask):
            fk_mid = F
            sabr_atm = (alpha / (fk_mid ** (1 - beta))) * (
                1 + (
                    ((1 - beta) ** 2 / 24) * (alpha ** 2) / (fk_mid ** (2 - 2 * beta))
                    + (rho * beta * nu * alpha) / (4 * fk_mid ** (1 - beta))
                    + (2 - 3 * rho ** 2) * nu ** 2 / 24
                ) * T
            )
            iv[atm_mask] = sabr_atm
        
        # Non-ATM volatility
        if np.any(non_atm_mask):
            K_non = K[non_atm_mask]
            fk = np.sqrt(F * K_non)
            
            # Log term
            log_term = np.log(F / K_non)
            
            # z and x(z) calculations
            z = (nu / alpha) * (fk ** (1 - beta)) * log_term
            x = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
            
            # Pre-factor
            pre_factor = alpha / ((F * K_non) ** ((1 - beta) / 2) * (1 + (1 - beta) ** 2 / 24 * log_term ** 2))
            
            # SABR formula
            sabr_non = pre_factor * (z / x) * (
                1 + (
                    ((1 - beta) ** 2 / 24) * (alpha ** 2) / (fk ** (2 - 2 * beta))
                    + (rho * beta * nu * alpha) / (4 * fk ** (1 - beta))
                    + (2 - 3 * rho ** 2) * nu ** 2 / 24
                ) * T
            )
            
            iv[non_atm_mask] = sabr_non
        
        return np.clip(iv, 0.01, 5.0)
    
    def fit(self,
            strikes: np.ndarray,
            forward: float,
            implied_vol: np.ndarray,
            tenor: float,
            initial_guess: Optional[SABRParameters] = None) -> SurfaceFitResult:
        """
        Fit SABR model to implied volatility data.
        
        Args:
            strikes: Strike array
            forward: Forward/spot price
            implied_vol: Market implied volatilities
            tenor: Time to expiry
            initial_guess: Optional initial parameters
            
        Returns:
            SurfaceFitResult with fitted parameters
        """
        if len(strikes) < 3:
            return SurfaceFitResult(
                model=SurfaceModel.SABR,
                params=None,
                rmse=float('inf'),
                max_error=float('inf'),
                convergence=False,
                n_iterations=0,
                timestamp_ns=0
            )
        
        # Initial guess
        if initial_guess is None:
            # Heuristic: estimate alpha from ATM vol
            atm_idx = np.argmin(np.abs(strikes - forward))
            alpha_init = implied_vol[atm_idx] * (forward ** (1 - self.beta_fixed))
            initial_params = [alpha_init, self.beta_fixed, 0.0, 0.5]
        else:
            initial_params = [initial_guess.alpha, initial_guess.beta,
                             initial_guess.rho, initial_guess.nu]
        
        # Objective function
        def objective(params):
            sabr_params = SABRParameters(
                alpha=params[0],
                beta=self.beta_fixed,  # Keep beta fixed
                rho=params[1],
                nu=params[2]
            )
            
            if not sabr_params.validate():
                return 1e10
            
            iv_model = self.sabr_implied_vol(forward, strikes, tenor, sabr_params)
            return np.sum((iv_model - implied_vol) ** 2)
        
        # Bounds
        bounds = [
            (0.01, 5.0),   # alpha
            (-0.99, 0.99), # rho
            (0.01, 3.0)    # nu
        ]
        
        # Optimize
        result = optimize.minimize(
            objective,
            initial_params[0::2] if len(initial_params) == 4 else initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        # Extract fitted parameters
        fitted_params = SABRParameters(
            alpha=result.x[0],
            beta=self.beta_fixed,
            rho=result.x[1],
            nu=result.x[2] if len(result.x) > 2 else 0.5
        )
        
        # Calculate metrics
        iv_fitted = self.sabr_implied_vol(forward, strikes, tenor, fitted_params)
        residuals = iv_fitted - implied_vol
        
        rmse = np.sqrt(np.mean(residuals ** 2))
        max_error = np.max(np.abs(residuals))
        
        return SurfaceFitResult(
            model=SurfaceModel.SABR,
            params=fitted_params,
            rmse=rmse,
            max_error=max_error,
            convergence=result.success,
            n_iterations=result.nit,
            timestamp_ns=0
        )


class VolatilitySurfaceFitter:
    """
    Complete volatility surface fitter handling multiple expiries.
    Combines SVI for cross-section and SABR for dynamics.
    """
    
    def __init__(self, 
                 svi_max_iter: int = 50,
                 sabr_max_iter: int = 50,
                 use_hybrid: bool = True):
        """
        Args:
            svi_max_iter: Max iterations for SVI fitting
            sabr_max_iter: Max iterations for SABR fitting
            use_hybrid: Use hybrid SVI-SABR approach
        """
        self.svi_fitter = SVIFitter(max_iterations=svi_max_iter)
        self.sabr_fitter = SABRFitter(max_iterations=sabr_max_iter)
        self.use_hybrid = use_hybrid
        
        # Storage for fitted surfaces
        self.fitted_surfaces = {}  # (asset, tenor) -> SurfaceFitResult
        
    def fit_surface(self,
                    asset: str,
                    strikes: np.ndarray,
                    forward: float,
                    implied_vols: np.ndarray,
                    tenor: float,
                    model: SurfaceModel = SurfaceModel.HYBRID) -> SurfaceFitResult:
        """
        Fit volatility surface for a single expiry.
        
        Args:
            asset: Asset identifier
            strikes: Strike array
            forward: Forward price
            implied_vols: Implied volatilities
            tenor: Time to expiry
            model: Model to use
            
        Returns:
            SurfaceFitResult
        """
        import time
        timestamp_ns = time.time_ns()
        
        # Calculate log-moneyness
        log_k = np.log(strikes / forward)
        
        # Choose model
        if model == SurfaceModel.SVI or (model == SurfaceModel.HYBRID and tenor > 0.1):
            result = self.svi_fitter.fit(log_k, implied_vols, tenor)
        elif model == SurfaceModel.SABR:
            result = self.sabr_fitter.fit(strikes, forward, implied_vols, tenor)
        else:  # Hybrid
            # Use SVI for longer tenors, SABR for short
            if tenor > 0.1:
                result = self.svi_fitter.fit(log_k, implied_vols, tenor)
            else:
                result = self.sabr_fitter.fit(strikes, forward, implied_vols, tenor)
        
        result.timestamp_ns = timestamp_ns
        
        # Store result
        self.fitted_surfaces[(asset, tenor)] = result
        
        return result
    
    def get_interpolated_vol(self,
                             asset: str,
                             strike: float,
                             forward: float,
                             tenor: float) -> Optional[float]:
        """
        Get interpolated implied volatility from fitted surface.
        
        Args:
            asset: Asset identifier
            strike: Target strike
            forward: Forward price
            tenor: Time to expiry
            
        Returns:
            Interpolated implied volatility or None
        """
        # Find closest tenor
        available_tenors = [t for (a, t) in self.fitted_surfaces.keys() if a == asset]
        
        if not available_tenors:
            return None
        
        # Simple nearest-neighbor interpolation
        closest_tenor = min(available_tenors, key=lambda t: abs(t - tenor))
        result = self.fitted_surfaces.get((asset, closest_tenor))
        
        if result is None or result.params is None:
            return None
        
        log_k = np.log(strike / forward)
        
        if isinstance(result.params, SVIParameters):
            w = self.svi_fitter.svi_variance(np.array([log_k]), result.params)
            iv = np.sqrt(w / closest_tenor)[0]
        elif isinstance(result.params, SABRParameters):
            iv = self.sabr_fitter.sabr_implied_vol(
                forward, np.array([strike]), closest_tenor, result.params
            )[0]
        else:
            return None
        
        return iv
    
    def get_all_surfaces(self) -> Dict:
        """Get all fitted surfaces."""
        return {
            f"{asset}_{tenor}": {
                'model': result.model.value,
                'params': vars(result.params) if result.params else None,
                'rmse': result.rmse,
                'convergence': result.convergence
            }
            for (asset, tenor), result in self.fitted_surfaces.items()
        }
    
    def clear_old_surfaces(self, max_age_seconds: float = 60.0):
        """Clear surfaces older than specified age."""
        import time
        current_ns = time.time_ns()
        max_age_ns = int(max_age_seconds * 1e9)
        
        keys_to_remove = [
            key for key, result in self.fitted_surfaces.items()
            if current_ns - result.timestamp_ns > max_age_ns
        ]
        
        for key in keys_to_remove:
            del self.fitted_surfaces[key]


__all__ = [
    'VolatilitySurfaceFitter',
    'SVIFitter',
    'SABRFitter',
    'SVIParameters',
    'SABRParameters',
    'SurfaceModel',
    'SurfaceFitResult'
]
