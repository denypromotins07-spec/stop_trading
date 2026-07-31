"""
Black-Litterman Optimizer integrating ML ensemble alpha views with market equilibrium.
Dynamically shifts multi-asset weights based on confidence scores from SOUL.md.
Memory-bounded implementation for 3GB RAM limit - no Pandas in hot-path.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class InvestorView:
    """Represents a single investor view on asset returns."""
    assets: List[str]  # Assets involved in the view
    expected_return: float  # Expected excess return
    confidence: float  # Confidence level (0 to 1)
    view_type: str = "absolute"  # "absolute" or "relative"


class BlackLittermanOptimizer:
    """
    Black-Litterman portfolio optimizer combining market equilibrium with ML views.
    
    The model starts with market-implied equilibrium returns and adjusts them
    based on investor views weighted by their confidence levels.
    """
    
    def __init__(self, asset_ids: List[str], risk_aversion: float = 2.5,
                 tau: float = 0.05):
        """
        Initialize the Black-Litterman optimizer.
        
        Args:
            asset_ids: List of asset identifiers
            risk_aversion: Risk aversion coefficient (default 2.5)
            tau: Uncertainty scalar for prior (default 0.05)
        """
        self.asset_ids = asset_ids
        self.n_assets = len(asset_ids)
        self.risk_aversion = risk_aversion
        self.tau = tau
        
        # Cache for equilibrium returns
        self._equilibrium_returns: Optional[np.ndarray] = None
        self._last_market_caps_hash: int = 0
    
    def _compute_equilibrium_returns(self, market_caps: np.ndarray,
                                      cov_matrix: np.ndarray) -> np.ndarray:
        """
        Compute implied equilibrium returns from market capitalizations.
        
        Uses reverse optimization: given market weights and covariance,
        find the returns that would make these weights optimal.
        """
        # Market weights from capitalizations
        total_cap = np.sum(market_caps)
        market_weights = market_caps / (total_cap + 1e-10)
        
        # Implied equilibrium returns: Pi = delta * Sigma * w_mkt
        equilibrium = self.risk_aversion * np.dot(cov_matrix, market_weights)
        
        self._equilibrium_returns = equilibrium
        return equilibrium
    
    def _build_view_matrices(self, views: List[InvestorView]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the P (pick) and Q (view) matrices from investor views.
        
        Returns:
            P: K x N matrix mapping assets to views
            Q: K x 1 vector of view expected returns
        """
        n_views = len(views)
        P = np.zeros((n_views, self.n_assets))
        Q = np.zeros(n_views)
        
        for k, view in enumerate(views):
            if view.view_type == "absolute":
                # Absolute view: single asset expected return
                for asset in view.assets:
                    if asset in self.asset_ids:
                        idx = self.asset_ids.index(asset)
                        P[k, idx] = 1.0
                Q[k] = view.expected_return
            
            elif view.view_type == "relative":
                # Relative view: outperformance between assets
                if len(view.assets) >= 2:
                    long_asset = view.assets[0]
                    short_asset = view.assets[1]
                    
                    if long_asset in self.asset_ids:
                        P[k, self.asset_ids.index(long_asset)] = 1.0
                    if short_asset in self.asset_ids:
                        P[k, self.asset_ids.index(short_asset)] = -1.0
                    
                    Q[k] = view.expected_return
        
        return P, Q
    
    def _build_omega(self, P: np.ndarray, cov_matrix: np.ndarray,
                     confidences: np.ndarray) -> np.ndarray:
        """
        Build the Omega matrix representing uncertainty in views.
        
        Omega is diagonal with elements proportional to view variance.
        Higher confidence means lower uncertainty.
        """
        n_views = P.shape[0]
        omega = np.zeros((n_views, n_views))
        
        for k in range(n_views):
            # View variance: confidence-weighted projection of covariance
            p_k = P[k:k+1, :].T
            view_variance = np.dot(p_k.T, np.dot(cov_matrix, p_k))[0, 0]
            
            # Scale by confidence (higher confidence = lower uncertainty)
            confidence = confidences[k]
            uncertainty_scale = self.tau * view_variance * (1 - confidence + 0.01)
            
            omega[k, k] = max(uncertainty_scale, 1e-8)
        
        return omega
    
    def compute_bl_returns(self, market_caps: np.ndarray, cov_matrix: np.ndarray,
                           views: List[InvestorView]) -> np.ndarray:
        """
        Compute Black-Litterman posterior expected returns.
        
        Args:
            market_caps: Market capitalizations for each asset
            cov_matrix: Covariance matrix of asset returns
            views: List of investor views with confidence levels
            
        Returns:
            Posterior expected returns vector
        """
        if cov_matrix.shape != (self.n_assets, self.n_assets):
            raise ValueError("Covariance matrix shape mismatch")
        
        if len(market_caps) != self.n_assets:
            raise ValueError("Market caps length mismatch")
        
        # Compute equilibrium returns
        pi = self._compute_equilibrium_returns(market_caps, cov_matrix)
        
        if not views:
            return pi.copy()
        
        # Build view matrices
        P, Q = self._build_view_matrices(views)
        confidences = np.array([v.confidence for v in views])
        
        # Build Omega (view uncertainty)
        Omega = self._build_omega(P, cov_matrix, confidences)
        
        # Black-Litterman formula:
        # E[R] = [(tau * Sigma)^-1 + P' * Omega^-1 * P]^-1 * 
        #        [(tau * Sigma)^-1 * Pi + P' * Omega^-1 * Q]
        
        tau_sigma_inv = np.linalg.inv(self.tau * cov_matrix + 1e-10 * np.eye(self.n_assets))
        omega_inv = np.linalg.inv(Omega + 1e-10 * np.eye(len(views)))
        
        # Posterior precision
        posterior_precision = tau_sigma_inv + np.dot(P.T, np.dot(omega_inv, P))
        
        # Posterior mean
        posterior_mean = np.dot(
            np.linalg.inv(posterior_precision + 1e-10 * np.eye(self.n_assets)),
            np.dot(tau_sigma_inv, pi) + np.dot(P.T, np.dot(omega_inv, Q))
        )
        
        return posterior_mean
    
    def optimize(self, market_caps: np.ndarray, cov_matrix: np.ndarray,
                 views: List[InvestorView], target_vol: Optional[float] = None) -> Dict[str, float]:
        """
        Compute optimal Black-Litterman weights.
        
        Args:
            market_caps: Market capitalizations
            cov_matrix: Covariance matrix
            views: Investor views
            target_vol: Optional target volatility for scaling
            
        Returns:
            Dictionary mapping asset IDs to weights
        """
        # Get posterior expected returns
        bl_returns = self.compute_bl_returns(market_caps, cov_matrix, views)
        
        # Mean-variance optimization: w = (1/delta) * Sigma^-1 * mu
        cov_inv = np.linalg.inv(cov_matrix + 1e-10 * np.eye(self.n_assets))
        raw_weights = np.dot(cov_inv, bl_returns) / self.risk_aversion
        
        # Normalize to sum to 1
        weights = raw_weights / (np.sum(raw_weights) + 1e-10)
        
        # Scale to target volatility if specified
        if target_vol is not None:
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if port_vol > 1e-10:
                weights *= target_vol / port_vol
        
        # Clip extreme weights
        weights = np.clip(weights, -0.5, 1.0)
        weights = weights / (np.sum(weights) + 1e-10)
        
        return dict(zip(self.asset_ids, weights))
    
    def get_nautilus_allocation(self, market_caps: np.ndarray, cov_matrix: np.ndarray,
                                 views: List[InvestorView], 
                                 portfolio_value: float) -> List[Dict]:
        """
        Generate Nautilus-compatible allocation commands.
        
        Returns:
            List of allocation dictionaries for Nautilus portfolio manager
        """
        weights = self.optimize(market_caps, cov_matrix, views)
        
        allocations = []
        for asset_id, weight in weights.items():
            if abs(weight) > 1e-6:
                allocations.append({
                    "instrument_id": asset_id,
                    "target_weight": float(weight),
                    "target_value": float(weight * portfolio_value),
                    "rebalance_threshold": 0.03,  # 3% drift tolerance
                    "source": "black_litterman"
                })
        
        return allocations
    
    def blend_with_hrp(self, hrp_weights: Dict[str, float], bl_weights: Dict[str, float],
                       bl_confidence: float) -> Dict[str, float]:
        """
        Blend Black-Litterman weights with HRP weights based on confidence.
        
        Args:
            hrp_weights: Weights from HRP optimizer
            bl_weights: Weights from Black-Litterman optimizer
            bl_confidence: Overall confidence in BL views (0 to 1)
            
        Returns:
            Blended weight dictionary
        """
        blended = {}
        
        for asset_id in self.asset_ids:
            hrp_w = hrp_weights.get(asset_id, 0.0)
            bl_w = bl_weights.get(asset_id, 0.0)
            
            # Confidence-weighted blend
            blended[asset_id] = (1 - bl_confidence) * hrp_w + bl_confidence * bl_w
        
        # Renormalize
        total = sum(blended.values())
        if abs(total) > 1e-10:
            blended = {k: v / total for k, v in blended.items()}
        
        return blended


def parse_views_from_soul(soul_data: Dict) -> List[InvestorView]:
    """
    Parse investor views from SOUL.md confidence scores.
    
    Args:
        soul_data: Dictionary containing ML ensemble predictions and confidence scores
        
    Returns:
        List of InvestorView objects
    """
    views = []
    
    # Extract alpha signals and confidence from SOUL data
    alpha_signals = soul_data.get("alpha_signals", {})
    confidence_scores = soul_data.get("confidence_scores", {})
    
    for asset, signal in alpha_signals.items():
        confidence = confidence_scores.get(asset, 0.5)
        
        # Only create views for high-confidence signals
        if confidence > 0.6:
            # Convert signal to expected return (scaled by confidence)
            expected_return = signal * confidence * 0.02  # 2% max daily
            
            views.append(InvestorView(
                assets=[asset],
                expected_return=expected_return,
                confidence=confidence,
                view_type="absolute"
            ))
    
    # Add relative views for pairs with divergent signals
    assets_list = list(alpha_signals.keys())
    for i, asset1 in enumerate(assets_list):
        for asset2 in assets_list[i+1:]:
            sig1 = alpha_signals.get(asset1, 0)
            sig2 = alpha_signals.get(asset2, 0)
            
            # Significant divergence
            if abs(sig1 - sig2) > 0.3:
                conf1 = confidence_scores.get(asset1, 0.5)
                conf2 = confidence_scores.get(asset2, 0.5)
                combined_conf = (conf1 + conf2) / 2
                
                if combined_conf > 0.5:
                    views.append(InvestorView(
                        assets=[asset1, asset2],
                        expected_return=(sig1 - sig2) * 0.01,
                        confidence=combined_conf,
                        view_type="relative"
                    ))
    
    return views


if __name__ == "__main__":
    # Example usage
    assets = ["BTC", "ETH", "SOL"]
    
    # Market capitalizations (in billions)
    market_caps = np.array([500, 250, 50])
    
    # Covariance matrix (annualized)
    cov_matrix = np.array([
        [0.04, 0.02, 0.015],
        [0.02, 0.06, 0.025],
        [0.015, 0.025, 0.09]
    ])
    
    # Sample views from ML ensemble
    views = [
        InvestorView(assets=["BTC"], expected_return=0.05, confidence=0.7),
        InvestorView(assets=["ETH"], expected_return=0.08, confidence=0.8),
        InvestorView(assets=["ETH", "SOL"], expected_return=0.03, confidence=0.6, view_type="relative")
    ]
    
    optimizer = BlackLittermanOptimizer(assets)
    weights = optimizer.optimize(market_caps, cov_matrix, views, target_vol=0.20)
    
    print("Black-Litterman Weights:")
    for asset, weight in weights.items():
        print(f"  {asset}: {weight:.4f}")
    
    allocations = optimizer.get_nautilus_allocation(
        market_caps, cov_matrix, views, portfolio_value=100000
    )
    print("\nNautilus Allocations:")
    for alloc in allocations:
        print(f"  {alloc}")
