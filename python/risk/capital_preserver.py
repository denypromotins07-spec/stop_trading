"""
Dynamic Capital Preservation Engine
Builds a dynamic capital preservation engine that scales down position sizes linearly during stress.
Integrates with drawdown predictions and real-time PnL to protect capital during adverse conditions.
"""

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class StressLevel(Enum):
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class CapitalPreservationState:
    """Current state of capital preservation system."""
    current_capital: float
    peak_capital: float
    current_drawdown: float
    max_drawdown_limit: float
    stress_level: StressLevel
    position_scale_factor: float
    daily_pnl: float
    volatility_adjustment: float


@dataclass
class PositionSizingRecommendation:
    """Recommended position sizing based on current conditions."""
    base_size: float
    adjusted_size: float
    scale_factor: float
    reason: str
    risk_flags: List[str]


class CapitalPreservationEngine:
    """
    Dynamic capital preservation engine that scales positions based on:
    - Current drawdown from peak
    - Volatility regime changes
    - Daily/weekly PnL performance
    - External risk signals (from drawdown predictor)
    """

    def __init__(
        self,
        initial_capital: float,
        max_drawdown_limit: float = 0.05,
        daily_loss_limit: float = 0.02,
        weekly_loss_limit: float = 0.04,
        stress_threshold_dd: float = 0.02,
        critical_threshold_dd: float = 0.035,
        recovery_hysteresis: float = 0.01,
    ):
        """
        Initialize the capital preservation engine.

        Args:
            initial_capital: Starting capital amount
            max_drawdown_limit: Maximum allowed drawdown before hard stop
            daily_loss_limit: Maximum daily loss as fraction of capital
            weekly_loss_limit: Maximum weekly loss as fraction of capital
            stress_threshold_dd: Drawdown level triggering stress mode
            critical_threshold_dd: Drawdown level triggering critical mode
            recovery_hysteresis: Buffer for recovering from stress mode
        """
        self.initial_capital = initial_capital
        self.max_drawdown_limit = max_drawdown_limit
        self.daily_loss_limit = daily_loss_limit
        self.weekly_loss_limit = weekly_loss_limit
        self.stress_threshold_dd = stress_threshold_dd
        self.critical_threshold_dd = critical_threshold_dd
        self.recovery_hysteresis = recovery_hysteresis

        # Capital tracking
        self._peak_capital: float = initial_capital
        self._current_capital: float = initial_capital
        self._daily_start_capital: float = initial_capital
        self._weekly_start_capital: float = initial_capital

        # Scaling factors
        self._base_scale: float = 1.0
        self._drawdown_scale: float = 1.0
        self._volatility_scale: float = 1.0
        self._pnl_scale: float = 1.0

        # Rolling volatility estimate
        self._returns_buffer: List[float] = []
        self._max_returns_history: int = 252  # ~1 trading year

        # Risk flags
        self._active_risk_flags: List[str] = []

        # Metrics
        self._scale_history: List[float] = []
        self._stress_events: int = 0

    def update_capital(self, current_capital: float) -> CapitalPreservationState:
        """
        Update capital and recalculate all scaling factors.

        Args:
            current_capital: Current total capital value

        Returns:
            Updated capital preservation state
        """
        # Track peak
        if current_capital > self._peak_capital:
            self._peak_capital = current_capital

        # Calculate current drawdown
        self._current_capital = current_capital
        current_drawdown = (self._peak_capital - current_capital) / self._peak_capital

        # Update rolling returns for volatility
        if len(self._returns_buffer) > 0:
            last_capital = self._returns_buffer[-1]
            ret = (current_capital - last_capital) / (last_capital + 1e-9)
            self._returns_buffer.append(ret)
            if len(self._returns_buffer) > self._max_returns_history:
                self._returns_buffer.pop(0)
        else:
            self._returns_buffer.append(0.0)

        # Calculate all scaling factors
        self._update_drawdown_scale(current_drawdown)
        self._update_volatility_scale()
        self._update_pnl_scale()

        # Determine stress level
        stress_level = self._determine_stress_level(current_drawdown)

        # Calculate composite scale factor
        composite_scale = self._calculate_composite_scale(stress_level)

        return CapitalPreservationState(
            current_capital=current_capital,
            peak_capital=self._peak_capital,
            current_drawdown=current_drawdown,
            max_drawdown_limit=self.max_drawdown_limit,
            stress_level=stress_level,
            position_scale_factor=composite_scale,
            daily_pnl=current_capital - self._daily_start_capital,
            volatility_adjustment=self._volatility_scale,
        )

    def _update_drawdown_scale(self, current_drawdown: float) -> None:
        """Update position scale based on current drawdown using linear scaling."""
        if current_drawdown <= self.stress_threshold_dd:
            # Normal operation - full size
            self._drawdown_scale = 1.0
        elif current_drawdown >= self.max_drawdown_limit:
            # Hard stop - no new positions
            self._drawdown_scale = 0.0
        else:
            # Linear scaling between threshold and limit
            range_start = self.stress_threshold_dd
            range_end = self.max_drawdown_limit

            # Linear interpolation: 1.0 at threshold, 0.0 at limit
            progress = (current_drawdown - range_start) / (range_end - range_start)
            self._drawdown_scale = 1.0 - progress

        self._drawdown_scale = max(0.0, min(1.0, self._drawdown_scale))

    def _update_volatility_scale(self) -> None:
        """Update position scale based on realized volatility."""
        if len(self._returns_buffer) < 20:
            self._volatility_scale = 1.0
            return

        # Calculate recent volatility (annualized)
        recent_returns = self._returns_buffer[-20:]
        volatility = np.std(recent_returns) * np.sqrt(252)

        # Define volatility regimes
        low_vol_threshold = 0.10  # 10% annualized
        high_vol_threshold = 0.30  # 30% annualized

        if volatility <= low_vol_threshold:
            self._volatility_scale = 1.0
        elif volatility >= high_vol_threshold:
            self._volatility_scale = 0.5
        else:
            # Linear scaling between thresholds
            progress = (volatility - low_vol_threshold) / (high_vol_threshold - low_vol_threshold)
            self._volatility_scale = 1.0 - progress * 0.5

        self._volatility_scale = max(0.5, min(1.0, self._volatility_scale))

    def _update_pnl_scale(self) -> None:
        """Update position scale based on daily/weekly PnL performance."""
        daily_pnl = (self._current_capital - self._daily_start_capital) / self._daily_start_capital
        weekly_pnl = (self._current_capital - self._weekly_start_capital) / self._weekly_start_capital

        # Start with full scale
        pnl_scale = 1.0
        self._active_risk_flags = []

        # Check daily loss limit
        if daily_pnl < -self.daily_loss_limit:
            pnl_scale *= 0.5
            self._active_risk_flags.append("DAILY_LOSS_LIMIT")

        # Check weekly loss limit
        if weekly_pnl < -self.weekly_loss_limit:
            pnl_scale *= 0.5
            self._active_risk_flags.append("WEEKLY_LOSS_LIMIT")

        # Consecutive losses penalty
        if len(self._returns_buffer) >= 5:
            recent = self._returns_buffer[-5:]
            if all(r < 0 for r in recent):
                pnl_scale *= 0.7
                self._active_risk_flags.append("CONSECUTIVE_LOSSES")

        self._pnl_scale = max(0.0, min(1.0, pnl_scale))

    def _determine_stress_level(self, current_drawdown: float) -> StressLevel:
        """Determine overall stress level based on drawdown."""
        if current_drawdown >= self.max_drawdown_limit:
            return StressLevel.CRITICAL
        elif current_drawdown >= self.critical_threshold_dd:
            return StressLevel.HIGH
        elif current_drawdown >= self.stress_threshold_dd:
            return StressLevel.ELEVATED
        else:
            return StressLevel.NORMAL

    def _calculate_composite_scale(self, stress_level: StressLevel) -> float:
        """Calculate composite position scale factor."""
        # Multiply all scaling factors
        composite = self._drawdown_scale * self._volatility_scale * self._pnl_scale

        # Apply additional stress level multipliers
        stress_multipliers = {
            StressLevel.NORMAL: 1.0,
            StressLevel.ELEVATED: 0.8,
            StressLevel.HIGH: 0.5,
            StressLevel.CRITICAL: 0.0,
        }

        composite *= stress_multipliers.get(stress_level, 1.0)

        # Track history
        self._scale_history.append(composite)
        if len(self._scale_history) > 1000:
            self._scale_history.pop(0)

        # Track stress events
        if stress_level != StressLevel.NORMAL and len(self._scale_history) >= 2:
            prev_stress = self._determine_stress_level(
                (self._peak_capital - self._current_capital) / self._peak_capital
                if len(self._scale_history) < 2 else
                0
            )
            if prev_stress == StressLevel.NORMAL:
                self._stress_events += 1

        return max(0.0, min(1.0, composite))

    def get_position_recommendation(
        self,
        base_size: float,
        instrument_name: str = "",
    ) -> PositionSizingRecommendation:
        """
        Get position sizing recommendation for a given base size.

        Args:
            base_size: The base position size before adjustments
            instrument_name: Name of the instrument (for reporting)

        Returns:
            Position sizing recommendation with adjusted size
        """
        # Get current state
        current_drawdown = (self._peak_capital - self._current_capital) / self._peak_capital
        stress_level = self._determine_stress_level(current_drawdown)
        scale_factor = self._calculate_composite_scale(stress_level)

        adjusted_size = base_size * scale_factor

        # Build reason string
        reasons = []
        if self._drawdown_scale < 1.0:
            reasons.append(f"Drawdown scaling: {self._drawdown_scale:.2f}")
        if self._volatility_scale < 1.0:
            reasons.append(f"Volatility scaling: {self._volatility_scale:.2f}")
        if self._pnl_scale < 1.0:
            reasons.append(f"PnL scaling: {self._pnl_scale:.2f}")
        if stress_level != StressLevel.NORMAL:
            reasons.append(f"Stress level: {stress_level.value}")

        reason = "; ".join(reasons) if reasons else "Normal operation"

        return PositionSizingRecommendation(
            base_size=base_size,
            adjusted_size=adjusted_size,
            scale_factor=scale_factor,
            reason=reason,
            risk_flags=self._active_risk_flags.copy(),
        )

    def reset_daily(self) -> None:
        """Reset daily PnL tracking (call at start of each trading day)."""
        self._daily_start_capital = self._current_capital

    def reset_weekly(self) -> None:
        """Reset weekly PnL tracking (call at start of each trading week)."""
        self._weekly_start_capital = self._current_capital

    def should_halt_trading(self) -> bool:
        """Check if trading should be halted due to risk limits."""
        current_drawdown = (self._peak_capital - self._current_capital) / self._peak_capital
        return current_drawdown >= self.max_drawdown_limit

    def get_metrics(self) -> Dict[str, Any]:
        """Get capital preservation metrics."""
        current_drawdown = (self._peak_capital - self._current_capital) / self._peak_capital

        return {
            "current_capital": self._current_capital,
            "peak_capital": self._peak_capital,
            "current_drawdown": current_drawdown,
            "max_drawdown_limit": self.max_drawdown_limit,
            "daily_pnl": self._current_capital - self._daily_start_capital,
            "weekly_pnl": self._current_capital - self._weekly_start_capital,
            "current_scale_factor": self._calculate_composite_scale(
                self._determine_stress_level(current_drawdown)
            ),
            "stress_events": self._stress_events,
            "active_risk_flags": self._active_risk_flags,
            "volatility_20d": np.std(self._returns_buffer[-20:]) * np.sqrt(252) if len(self._returns_buffer) >= 20 else 0.0,
        }

    def reset(self, new_initial_capital: Optional[float] = None) -> None:
        """Reset all state."""
        if new_initial_capital is not None:
            self.initial_capital = new_initial_capital
            self._peak_capital = new_initial_capital
            self._current_capital = new_initial_capital

        self._daily_start_capital = self._current_capital
        self._weekly_start_capital = self._current_capital
        self._returns_buffer.clear()
        self._active_risk_flags.clear()
        self._scale_history.clear()
        self._stress_events = 0

        self._base_scale = 1.0
        self._drawdown_scale = 1.0
        self._volatility_scale = 1.0
        self._pnl_scale = 1.0
