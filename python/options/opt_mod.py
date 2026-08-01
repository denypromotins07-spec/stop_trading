"""
Options Module Root.
Manages options data ingestion and pushes vol-arb signals to the Nautilus alpha ensemble.
"""
from .vol_surface_tracker import (
    VolSurfaceTracker,
    SVIFitter,
    SVIParams,
    VolSurfacePoint,
    FittedSurface,
    get_tracker
)
from .gamma_scalper import (
    GammaScalper,
    BlackScholes,
    Greeks,
    OptionPosition,
    OptionType,
    HedgeAdjustment,
    get_scalper
)

__all__ = [
    # Volatility Surface
    "VolSurfaceTracker",
    "SVIFitter",
    "SVIParams",
    "VolSurfacePoint",
    "FittedSurface",
    "get_tracker",
    
    # Gamma Scalping
    "GammaScalper",
    "BlackScholes",
    "Greeks",
    "OptionPosition",
    "OptionType",
    "HedgeAdjustment",
    "get_scalper",
]
