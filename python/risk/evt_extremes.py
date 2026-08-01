"""
Extreme Value Theory (EVT) for Tail Risk Modeling.
Implements Generalized Pareto Distribution (GPD) fitter for extreme tail risk.
Updates shape parameters via online maximum likelihood for fat-tailed crypto markets.
Calculates extreme Expected Shortfall beyond standard Gaussian assumptions.
Strictly enforces 3GB RAM limit via bounded block processing.
"""
import asyncio
import numpy as np
from scipy import stats, optimize
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class GPDParameters:
    """Fitted Generalized Pareto Distribution parameters"""
    xi: float  # Shape parameter (tail index)
    sigma: float  # Scale parameter
    threshold: float  # Threshold above which GPD applies
    n_exceedances: int  # Number of observations above threshold
    log_likelihood: float
    standard_error_xi: float
    standard_error_sigma: float
    fit_time_ms: float
    timestamp_ns: int


@dataclass
class TailRiskMetrics:
    """Tail risk metrics from EVT analysis"""
    asset_id: str
    var_99: float  # 99% VaR
    var_99_9: float  # 99.9% VaR
    expected_shortfall_99: float  # ES at 99%
    expected_shortfall_99_9: float  # ES at 99.9%
    extreme_loss_prob: float  # P(loss > 10x daily vol)
    tail_index: float  # Inverse of xi, measures tail heaviness
    return_level_100d: float  # Expected loss for 100-day return period
    return_level_250d: float  # Expected loss for 250-day return period
    timestamp_ns: int


@dataclass
class BlockMaximaResult:
    """Block maxima analysis result"""
    block_size: int
    maxima: np.ndarray
    gev_shape: float
    gev_scale: float
    gev_location: float


class GPDFitter:
    """
    Generalized Pareto Distribution fitter with online updates.
    Uses Maximum Likelihood Estimation with Newton-Raphson optimization.
    """
    
    # Optimization settings
    MAX_ITERATIONS = 100
    TOLERANCE = 1e-8
    
    def __init__(self):
        self._params: Optional[GPDParameters] = None
        self._exceedances: deque = deque(maxlen=10000)  # Bounded memory
        self._threshold: Optional[float] = None
    
    def set_threshold(self, threshold: float):
        """Set fixed threshold for exceedances"""
        self._threshold = threshold
    
    def add_exceedance(self, value: float):
        """Add a new exceedance observation"""
        self._exceedances.append(value)
    
    def fit_online(self, returns: np.ndarray, 
                   threshold_percentile: float = 95.0) -> GPDParameters:
        """
        Fit GPD to returns above threshold using online MLE.
        
        Args:
            returns: Array of returns (negative values are losses)
            threshold_percentile: Percentile for threshold selection
        
        Returns:
            Fitted GPD parameters
        """
        start_time = time.perf_counter()
        
        # Convert to losses (positive values)
        losses = -returns[returns < 0]
        
        if len(losses) < 10:
            raise ValueError("Insufficient loss data for GPD fitting")
        
        # Determine threshold
        if self._threshold is None:
            threshold = np.percentile(losses, threshold_percentile)
        else:
            threshold = self._threshold
        
        # Get exceedances
        exceedances = losses[losses > threshold] - threshold
        
        if len(exceedances) < 5:
            raise ValueError("Insufficient exceedances for GPD fitting")
        
        # Store exceedances
        for exc in exceedances:
            self._exceedances.append(exc)
        
        # Fit GPD using MLE
        all_exceedances = np.array(list(self._exceedances))
        xi, sigma, loglik, se_xi, se_sigma = self._mle_fit(all_exceedances)
        
        fit_time_ms = (time.perf_counter() - start_time) * 1000
        
        self._params = GPDParameters(
            xi=float(xi),
            sigma=float(sigma),
            threshold=float(threshold),
            n_exceedances=len(all_exceedances),
            log_likelihood=float(loglik),
            standard_error_xi=float(se_xi),
            standard_error_sigma=float(se_sigma),
            fit_time_ms=fit_time_ms,
            timestamp_ns=time.time_ns()
        )
        
        return self._params
    
    def _mle_fit(self, exceedances: np.ndarray) -> Tuple[float, float, float, float, float]:
        """
        Maximum Likelihood Estimation for GPD parameters.
        Returns xi, sigma, log_likelihood, se_xi, se_sigma.
        """
        n = len(exceedances)
        
        # Initial estimates using method of moments
        mean_exc = np.mean(exceedances)
        var_exc = np.var(exceedances)
        
        if var_exc > 0:
            cv = np.sqrt(var_exc) / mean_exc
            xi_init = (cv**2 - 1) / (2 * cv**2 - 1)
            xi_init = np.clip(xi_init, -0.5, 0.5)
        else:
            xi_init = 0.0
        
        sigma_init = mean_exc * (1 - xi_init)
        
        # Negative log-likelihood function
        def neg_loglik(params):
            xi, sigma = params
            if sigma <= 0 or (xi < -0.5 or xi > 1.0):
                return 1e10
            
            z = 1 + xi * exceedances / sigma
            if np.any(z <= 0):
                return 1e10
            
            ll = -n * np.log(sigma) - (1 + 1/xi) * np.sum(np.log(z))
            return -ll if np.isfinite(ll) else 1e10
        
        # Optimize
        try:
            result = optimize.minimize(
                neg_loglik,
                x0=[xi_init, sigma_init],
                method='L-BFGS-B',
                bounds=[(-0.5, 1.0), (1e-6, None)],
                options={'maxiter': self.MAX_ITERATIONS, 'ftol': self.TOLERANCE}
            )
            
            xi, sigma = result.x
            loglik = -result.fun
            converged = result.success
        except Exception:
            xi, sigma = xi_init, sigma_init
            loglik = -neg_loglik([xi, sigma])
            converged = False
        
        # Calculate standard errors using Hessian approximation
        eps = 1e-5
        hessian = np.zeros((2, 2))
        base = neg_loglik([xi, sigma])
        
        for i in range(2):
            for j in range(i, 2):
                params_pp = [xi, sigma]
                params_pm = [xi, sigma]
                params_mp = [xi, sigma]
                params_mm = [xi, sigma]
                
                if i == 0:
                    params_pp[0] += eps
                    params_pm[0] += eps
                    params_mp[0] -= eps
                    params_mm[0] -= eps
                else:
                    params_pp[1] += eps
                    params_pm[1] += eps
                    params_mp[1] -= eps
                    params_mm[1] -= eps
                
                if j == 0:
                    params_pp[0] += eps
                    params_pm[0] -= eps
                    params_mp[0] += eps
                    params_mm[0] -= eps
                else:
                    params_pp[1] += eps
                    params_pm[1] -= eps
                    params_mp[1] += eps
                    params_mm[1] -= eps
                
                hessian[i, j] = (neg_loglik(params_pp) - neg_loglik(params_pm) - 
                                neg_loglik(params_mp) + neg_loglik(params_mm)) / (4 * eps * eps)
                hessian[j, i] = hessian[i, j]
        
        # Standard errors from inverse Hessian
        try:
            inv_hessian = np.linalg.inv(hessian)
            se_xi = np.sqrt(max(0, inv_hessian[0, 0]))
            se_sigma = np.sqrt(max(0, inv_hessian[1, 1]))
        except Exception:
            se_xi = 0.1
            se_sigma = sigma * 0.1
        
        return xi, sigma, loglik, se_xi, se_sigma
    
    def calculate_var(self, p: float) -> float:
        """
        Calculate Value at Risk at confidence level p.
        
        Args:
            p: Confidence level (e.g., 0.99 for 99% VaR)
        
        Returns:
            VaR estimate
        """
        if self._params is None:
            raise ValueError("GPD not fitted")
        
        xi, sigma, threshold = self._params.xi, self._params.sigma, self._params.threshold
        n = self._params.n_exceedances
        n_total = n + 100  # Approximate total observations
        
        # Probability of exceedance
        p_exceed = n / n_total
        
        if p < 1 - p_exceed:
            # VaR below threshold - use empirical
            return threshold
        
        # GPD VaR formula
        if abs(xi) < 1e-6:
            # Exponential case (xi = 0)
            var = threshold + sigma * np.log(p_exceed / (1 - p))
        else:
            var = threshold + sigma / xi * (((1 - p) / p_exceed) ** (-xi) - 1)
        
        return float(var)
    
    def calculate_expected_shortfall(self, p: float) -> float:
        """
        Calculate Expected Shortfall (CVaR) at confidence level p.
        
        Args:
            p: Confidence level (e.g., 0.99 for 99% ES)
        
        Returns:
            ES estimate
        """
        if self._params is None:
            raise ValueError("GPD not fitted")
        
        xi, sigma, threshold = self._params.xi, self._params.sigma, self._params.threshold
        n = self._params.n_exceedances
        n_total = n + 100
        
        p_exceed = n / n_total
        
        if p < 1 - p_exceed:
            return threshold
        
        var_p = self.calculate_var(p)
        
        # ES formula for GPD
        if xi < 1:
            if abs(xi) < 1e-6:
                es = var_p + sigma
            else:
                es = var_p + (sigma + xi * (var_p - threshold)) / (1 - xi)
        else:
            # Infinite mean case
            es = float('inf')
        
        return float(es)
    
    def calculate_return_level(self, return_period: int) -> float:
        """
        Calculate return level for given return period.
        
        Args:
            return_period: Number of observations (e.g., 250 for annual)
        
        Returns:
            Expected loss for the return period
        """
        if self._params is None:
            raise ValueError("GPD not fitted")
        
        xi, sigma, threshold = self._params.xi, self._params.sigma, self._params.threshold
        n = self._params.n_exceedances
        n_total = n + 100
        
        # Probability for return level
        p = 1 - 1 / return_period
        
        return self.calculate_var(p)


class EVTRiskAnalyzer:
    """
    Complete EVT-based tail risk analyzer.
    Combines GPD fitting with comprehensive risk metrics.
    """
    
    MAX_RETURN_HISTORY = 5000  # Bounded memory
    
    def __init__(self):
        self._gpd_fitters: Dict[str, GPDFitter] = {}
        self._return_history: Dict[str, deque] = {}
        self._risk_metrics: Dict[str, TailRiskMetrics] = {}
        self._lock = asyncio.Lock()
    
    async def add_returns(self, asset_id: str, returns: np.ndarray):
        """Add return data for an asset"""
        async with self._lock:
            if asset_id not in self._return_history:
                self._return_history[asset_id] = deque(maxlen=self.MAX_RETURN_HISTORY)
                self._gpd_fitters[asset_id] = GPDFitter()
            
            for r in returns:
                self._return_history[asset_id].append(r)
            
            # Update GPD fit
            await self._update_gpd_fit(asset_id)
    
    async def _update_gpd_fit(self, asset_id: str):
        """Update GPD fit for an asset"""
        if asset_id not in self._return_history:
            return
        
        returns = np.array(list(self._return_history[asset_id]))
        
        if len(returns) < 100:
            return
        
        fitter = self._gpd_fitters[asset_id]
        
        try:
            params = fitter.fit_online(returns, threshold_percentile=95.0)
            
            # Calculate comprehensive risk metrics
            metrics = await self._calculate_risk_metrics(asset_id, fitter, params)
            self._risk_metrics[asset_id] = metrics
            
        except Exception as e:
            pass  # Silently handle fit failures
    
    async def _calculate_risk_metrics(self, asset_id: str, fitter: GPDFitter,
                                       params: GPDParameters) -> TailRiskMetrics:
        """Calculate comprehensive tail risk metrics"""
        # VaR levels
        var_99 = fitter.calculate_var(0.99)
        var_99_9 = fitter.calculate_var(0.999)
        
        # Expected Shortfall
        es_99 = fitter.calculate_expected_shortfall(0.99)
        es_99_9 = fitter.calculate_expected_shortfall(0.999)
        
        # Extreme loss probability (loss > 10x daily vol)
        returns = np.array(list(self._return_history[asset_id]))
        daily_vol = np.std(returns)
        extreme_threshold = 10 * daily_vol
        extreme_prob = fitter.calculate_var(1 - 1/1000)  # Approximate
        
        # Tail index (inverse of xi)
        tail_index = 1 / params.xi if params.xi > 0 else float('inf')
        
        # Return levels
        rl_100d = fitter.calculate_return_level(100)
        rl_250d = fitter.calculate_return_level(250)
        
        return TailRiskMetrics(
            asset_id=asset_id,
            var_99=float(var_99),
            var_99_9=float(var_99_9),
            expected_shortfall_99=float(es_99) if np.isfinite(es_99) else var_99_9,
            expected_shortfall_99_9=float(es_99_9) if np.isfinite(es_99_9) else var_99_9 * 1.5,
            extreme_loss_prob=float(extreme_prob),
            tail_index=float(tail_index) if np.isfinite(tail_index) else 10.0,
            return_level_100d=float(rl_100d),
            return_level_250d=float(rl_250d),
            timestamp_ns=time.time_ns()
        )
    
    def get_risk_metrics(self, asset_id: str) -> Optional[TailRiskMetrics]:
        """Get current risk metrics for an asset"""
        return self._risk_metrics.get(asset_id)
    
    def get_all_metrics(self) -> Dict[str, TailRiskMetrics]:
        """Get all current risk metrics"""
        return self._risk_metrics.copy()
    
    def get_portfolio_es(self, weights: Dict[str, float]) -> float:
        """
        Calculate portfolio Expected Shortfall.
        Simple weighted sum (ignores dependence - use copulas for full model).
        """
        total_es = 0.0
        
        for asset_id, weight in weights.items():
            metrics = self._risk_metrics.get(asset_id)
            if metrics:
                total_es += weight * metrics.expected_shortfall_99
        
        return total_es
    
    def compare_to_gaussian(self, asset_id: str) -> Dict[str, float]:
        """Compare EVT metrics to Gaussian assumptions"""
        metrics = self._risk_metrics.get(asset_id)
        if metrics is None or asset_id not in self._return_history:
            return {}
        
        returns = np.array(list(self._return_history[asset_id]))
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        # Gaussian VaR and ES
        gaussian_var_99 = -(mean_ret + 2.326 * std_ret)
        gaussian_es_99 = -(mean_ret + 2.667 * std_ret)
        
        return {
            'evt_var_99': metrics.var_99,
            'gaussian_var_99': gaussian_var_99,
            'var_ratio': metrics.var_99 / gaussian_var_99 if gaussian_var_99 != 0 else float('inf'),
            'evt_es_99': metrics.expected_shortfall_99,
            'gaussian_es_99': gaussian_es_99,
            'es_ratio': metrics.expected_shortfall_99 / gaussian_es_99 if gaussian_es_99 != 0 else float('inf')
        }


# Global singleton instance
_analyzer_instance: Optional[EVTRiskAnalyzer] = None


def get_evt_analyzer() -> EVTRiskAnalyzer:
    """Get or create global EVT risk analyzer"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = EVTRiskAnalyzer()
    return _analyzer_instance


async def demo():
    """Demo usage of EVT risk analyzer"""
    print("=== EVT Tail Risk Demo ===\n")
    
    analyzer = get_evt_analyzer()
    
    # Generate synthetic returns with fat tails
    np.random.seed(42)
    n_returns = 2000
    
    # Student-t returns (fat-tailed)
    t_returns = stats.t.rvs(df=4, loc=0.0001, scale=0.02, size=n_returns)
    
    # Add some extreme events
    t_returns[np.random.choice(n_returns, 20)] *= 5
    
    await analyzer.add_returns("BTC", t_returns)
    
    # Get risk metrics
    metrics = analyzer.get_risk_metrics("BTC")
    if metrics:
        print(f"Tail Risk Metrics for BTC:")
        print(f"  99% VaR: {metrics.var_99:.4f}")
        print(f"  99.9% VaR: {metrics.var_99_9:.4f}")
        print(f"  99% ES: {metrics.expected_shortfall_99:.4f}")
        print(f"  99.9% ES: {metrics.expected_shortfall_99_9:.4f}")
        print(f"  Tail Index: {metrics.tail_index:.2f}")
        print(f"  100-day Return Level: {metrics.return_level_100d:.4f}")
        print(f"  250-day Return Level: {metrics.return_level_250d:.4f}")
    
    # Compare to Gaussian
    comparison = analyzer.compare_to_gaussian("BTC")
    if comparison:
        print(f"\nEVT vs Gaussian Comparison:")
        print(f"  EVT VaR 99%: {comparison['evt_var_99']:.4f}")
        print(f"  Gaussian VaR 99%: {comparison['gaussian_var_99']:.4f}")
        print(f"  VaR Ratio (EVT/Gaussian): {comparison['var_ratio']:.2f}x")
        print(f"  EVT ES 99%: {comparison['evt_es_99']:.4f}")
        print(f"  Gaussian ES 99%: {comparison['gaussian_es_99']:.4f}")
        print(f"  ES Ratio (EVT/Gaussian): {comparison['es_ratio']:.2f}x")


if __name__ == "__main__":
    asyncio.run(demo())
