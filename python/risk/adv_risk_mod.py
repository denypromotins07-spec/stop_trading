"""
Advanced Risk Module Root.
Aggregates Copula and EVT metrics to calculate true portfolio Expected Shortfall (CVaR).
"""
from .copula_ray import (
    CopulaModel,
    CopulaType,
    CopulaFitResult,
    TailDependenceMetrics,
    PortfolioRiskMetrics,
    create_copula_model
)
from .evt_extremes import (
    GPDFitter,
    GPDParameters,
    EVTRiskAnalyzer,
    TailRiskMetrics,
    get_evt_analyzer
)

__all__ = [
    # Copula Models
    "CopulaModel",
    "CopulaType",
    "CopulaFitResult",
    "TailDependenceMetrics",
    "PortfolioRiskMetrics",
    "create_copula_model",
    
    # EVT Models
    "GPDFitter",
    "GPDParameters",
    "EVTRiskAnalyzer",
    "TailRiskMetrics",
    "get_evt_analyzer",
]
