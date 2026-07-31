"""
ML calibrator for the Almgren-Chriss market impact model.
Updates non-linear impact parameters in real-time using TCA data.
Memory-bounded implementation for 3GB RAM limit - no Pandas in hot-path.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class MarketImpactParams:
    """Almgren-Chriss market impact parameters."""
    linear_impact: float = 1e-6  # Temporary impact coefficient
    permanent_impact: float = 5e-7  # Permanent impact coefficient
    decay_rate: float = 0.1  # Impact decay rate
    elasticity: float = 0.5  # Volume elasticity


class AlmgrenChrissModel:
    """
    Almgren-Chriss market impact model with ML-calibrated parameters.
    
    The model decomposes impact into temporary and permanent components:
    - Temporary impact: Linear in participation rate, decays after execution
    - Permanent impact: Proportional to total order size, persists
    """
    
    def __init__(self, params: MarketImpactParams = None):
        self.params = params or MarketImpactParams()
        
        # Historical data for calibration
        self._execution_history: List[Dict] = []
        self._max_history = 1000
        
        # Parameter estimation state
        self._sum_participation: float = 0
        self._sum_impact: float = 0
        self._sum_participation_sq: float = 0
        self._sum_cross: float = 0
        self._n_samples: int = 0
    
    def predict_impact(self, order_size: float, daily_volume: float,
                       volatility: float, execution_time_pct: float = 1.0) -> Dict:
        """
        Predict market impact for an order.
        
        Args:
            order_size: Order size in base currency
            daily_volume: Average daily volume
            volatility: Annualized volatility
            execution_time_pct: Fraction of day for execution
            
        Returns:
            Impact prediction dictionary
        """
        # Participation rate
        participation_rate = order_size / (daily_volume * execution_time_pct + 1e-10)
        
        # Temporary impact (bps)
        temp_impact = self.params.linear_impact * order_size * np.sqrt(volatility) * 10000
        
        # Add participation rate effect
        temp_impact *= (1 + np.log(1 + participation_rate * 10))
        
        # Permanent impact (bps)
        perm_impact = self.params.permanent_impact * order_size * 10000
        
        # Total expected impact
        total_impact = temp_impact + perm_impact
        
        # Expected slippage including timing risk
        timing_risk = volatility * np.sqrt(execution_time_pct) * 0.5
        
        return {
            "temporary_impact_bps": float(temp_impact),
            "permanent_impact_bps": float(perm_impact),
            "total_expected_impact_bps": float(total_impact),
            "participation_rate": float(participation_rate),
            "timing_risk_bps": float(timing_risk * 100),
            "recommended_pace": self._recommend_pace(order_size, daily_volume, volatility)
        }
    
    def _recommend_pace(self, order_size: float, daily_volume: float,
                        volatility: float) -> Dict:
        """Recommend optimal execution pace."""
        # Simple rule: participate at rate that keeps impact < 10 bps
        target_impact_bps = 10
        max_participation = target_impact_bps / (
            self.params.linear_impact * daily_volume * 10000 * np.sqrt(volatility) + 1e-10
        )
        max_participation = min(max_participation, 0.1)  # Cap at 10%
        
        execution_time = order_size / (daily_volume * max_participation + 1e-10)
        execution_time = min(execution_time, 1.0)  # Max 1 day
        
        return {
            "optimal_participation_rate": float(max_participation),
            "estimated_execution_time_days": float(execution_time),
            "twap_slices": max(4, int(execution_time * 24))  # At least 4 slices
        }
    
    def update_from_execution(self, order_size: float, realized_impact_bps: float,
                              daily_volume: float, volatility: float,
                              participation_rate: float):
        """
        Update model parameters from realized execution data.
        
        Args:
            order_size: Executed order size
            realized_impact_bps: Realized impact in basis points
            daily_volume: Daily volume at execution time
            volatility: Volatility at execution time
            participation_rate: Actual participation rate
        """
        # Store execution
        self._execution_history.append({
            "order_size": order_size,
            "realized_impact_bps": realized_impact_bps,
            "daily_volume": daily_volume,
            "volatility": volatility,
            "participation_rate": participation_rate,
            "timestamp": time.time()
        })
        
        # Keep history bounded
        if len(self._execution_history) > self._max_history:
            self._execution_history = self._execution_history[-self._max_history:]
        
        # Online parameter update using recursive least squares
        # Model: impact = a * participation + b * order_size
        x1 = participation_rate
        x2 = order_size * np.sqrt(volatility)
        y = realized_impact_bps
        
        # Update sufficient statistics
        self._sum_participation += x1
        self._sum_impact += y
        self._sum_participation_sq += x1 ** 2
        self._sum_cross += x1 * y
        self._n_samples += 1
        
        # Periodic re-estimation
        if self._n_samples % 50 == 0 and self._n_samples >= 100:
            self._reestimate_parameters()
    
    def _reestimate_parameters(self):
        """Re-estimate parameters using accumulated statistics."""
        if self._n_samples < 10:
            return
        
        # Simple OLS estimate for linear impact coefficient
        mean_x = self._sum_participation / self._n_samples
        mean_y = self._sum_impact / self._n_samples
        
        numerator = self._sum_cross - self._n_samples * mean_x * mean_y
        denominator = self._sum_participation_sq - self._n_samples * mean_x ** 2
        
        if abs(denominator) > 1e-10:
            new_linear_impact = numerator / denominator / 10000
            # Smooth update
            self.params.linear_impact = 0.8 * self.params.linear_impact + 0.2 * new_linear_impact
        
        # Estimate permanent impact from large orders
        large_orders = [e for e in self._execution_history if e["order_size"] > 1e6]
        if len(large_orders) >= 5:
            avg_impact = np.mean([e["realized_impact_bps"] for e in large_orders])
            avg_size = np.mean([e["order_size"] for e in large_orders])
            new_perm_impact = avg_impact / (avg_size * 10000 + 1e-10) * 0.5
            self.params.permanent_impact = 0.8 * self.params.permanent_impact + 0.2 * new_perm_impact
    
    def get_calibration_stats(self) -> Dict:
        """Get calibration statistics."""
        if not self._execution_history:
            return {"status": "no_data"}
        
        impacts = [e["realized_impact_bps"] for e in self._execution_history]
        
        return {
            "n_executions": len(self._execution_history),
            "mean_impact_bps": float(np.mean(impacts)),
            "std_impact_bps": float(np.std(impacts)),
            "linear_impact": self.params.linear_impact,
            "permanent_impact": self.params.permanent_impact,
            "decay_rate": self.params.decay_rate
        }


class ImpactCalibrator:
    """
    Real-time market impact calibrator managing models for multiple instruments.
    Integrates with TCA systems for continuous parameter updates.
    """
    
    def __init__(self, instruments: List[str]):
        self.instruments = instruments
        self.models: Dict[str, AlmgrenChrissModel] = {
            inst: AlmgrenChrissModel() for inst in instruments
        }
        
        # Initial parameter estimates based on asset class
        self._default_params: Dict[str, MarketImpactParams] = {
            "BTC": MarketImpactParams(
                linear_impact=2e-6,
                permanent_impact=1e-6,
                decay_rate=0.15
            ),
            "ETH": MarketImpactParams(
                linear_impact=3e-6,
                permanent_impact=1.5e-6,
                decay_rate=0.12
            ),
            "SOL": MarketImpactParams(
                linear_impact=5e-6,
                permanent_impact=2e-6,
                decay_rate=0.10
            )
        }
        
        # Initialize with defaults
        for inst, params in self._default_params.items():
            if inst in self.models:
                self.models[inst].params = params
    
    def calibrate_from_tca_data(self, tca_reports: List[Dict]):
        """
        Calibrate models from TCA (Transaction Cost Analysis) reports.
        
        Args:
            tca_reports: List of TCA report dictionaries
        """
        for report in tca_reports:
            instrument_id = report.get("instrument_id")
            if instrument_id not in self.models:
                continue
            
            model = self.models[instrument_id]
            
            # Extract data from TCA report
            order_size = report.get("order_size", 0)
            realized_slippage_bps = report.get("realized_slippage_bps", 0)
            daily_volume = report.get("daily_volume", 1e9)
            volatility = report.get("volatility", 0.5)
            participation_rate = report.get("participation_rate", 0.01)
            
            model.update_from_execution(
                order_size,
                realized_slippage_bps,
                daily_volume,
                volatility,
                participation_rate
            )
    
    def get_optimal_execution_strategy(self, instrument_id: str,
                                        order_size: float,
                                        daily_volume: float,
                                        volatility: float,
                                        urgency: float = 0.5) -> Dict:
        """
        Generate optimal execution strategy for an order.
        
        Args:
            instrument_id: Asset identifier
            order_size: Order size
            daily_volume: Average daily volume
            volatility: Volatility
            urgency: Urgency factor (0 = patient, 1 = urgent)
            
        Returns:
            Execution strategy dictionary
        """
        if instrument_id not in self.models:
            return {"error": "Instrument not found"}
        
        model = self.models[instrument_id]
        impact_pred = model.predict_impact(order_size, daily_volume, volatility)
        
        # Adjust for urgency
        base_pace = impact_pred["recommended_pace"]
        
        if urgency > 0.7:
            # Aggressive: faster execution, accept higher impact
            participation_rate = min(base_pace["optimal_participation_rate"] * 2, 0.2)
            execution_time = base_pace["estimated_execution_time_days"] * 0.5
        elif urgency < 0.3:
            # Passive: slower execution, minimize impact
            participation_rate = base_pace["optimal_participation_rate"] * 0.5
            execution_time = min(base_pace["estimated_execution_time_days"] * 1.5, 2.0)
        else:
            participation_rate = base_pace["optimal_participation_rate"]
            execution_time = base_pace["estimated_execution_time_days"]
        
        # Generate slice schedule
        n_slices = max(4, int(execution_time * 24))
        slice_size = order_size / n_slices
        
        return {
            "instrument_id": instrument_id,
            "order_size": order_size,
            "strategy": "twap" if urgency < 0.5 else "aggressive_twap",
            "n_slices": n_slices,
            "slice_size": float(slice_size),
            "participation_rate": float(participation_rate),
            "estimated_duration_hours": float(execution_time * 24),
            "expected_total_impact_bps": impact_pred["total_expected_impact_bps"],
            "urgency_adjustment": urgency
        }
    
    def get_all_model_stats(self) -> Dict[str, Dict]:
        """Get calibration stats for all instruments."""
        return {
            inst: model.get_calibration_stats()
            for inst, model in self.models.items()
        }


if __name__ == "__main__":
    # Example usage
    instruments = ["BTC", "ETH", "SOL"]
    
    calibrator = ImpactCalibrator(instruments)
    
    # Simulate TCA data
    np.random.seed(42)
    tca_reports = []
    
    for _ in range(100):
        inst = np.random.choice(instruments)
        order_size = np.random.lognormal(14, 1)
        daily_volume = np.random.lognormal(18, 0.5)
        volatility = np.random.uniform(0.3, 1.0)
        participation = order_size / daily_volume
        
        # Simulated realized impact
        base_impact = 5 + 100 * participation + 0.001 * order_size / 1e6
        realized_bps = base_impact * (1 + np.random.normal(0, 0.3))
        
        tca_reports.append({
            "instrument_id": inst,
            "order_size": order_size,
            "realized_slippage_bps": realized_bps,
            "daily_volume": daily_volume,
            "volatility": volatility,
            "participation_rate": participation
        })
    
    # Calibrate from TCA data
    calibrator.calibrate_from_tca_data(tca_reports)
    
    # Get stats
    stats = calibrator.get_all_model_stats()
    print("Calibration Statistics:")
    for inst, s in stats.items():
        print(f"\n{inst}:")
        print(f"  Executions: {s.get('n_executions', 0)}")
        print(f"  Mean Impact: {s.get('mean_impact_bps', 0):.2f} bps")
        print(f"  Linear Impact: {s.get('linear_impact', 0):.2e}")
    
    # Get execution strategy
    strategy = calibrator.get_optimal_execution_strategy(
        instrument_id="BTC",
        order_size=5e6,
        daily_volume=2e9,
        volatility=0.6,
        urgency=0.5
    )
    
    print(f"\nExecution Strategy for BTC $5M order:")
    print(f"  Strategy: {strategy['strategy']}")
    print(f"  Slices: {strategy['n_slices']}")
    print(f"  Slice Size: ${strategy['slice_size']:,.0f}")
    print(f"  Duration: {strategy['estimated_duration_hours']:.1f} hours")
    print(f"  Expected Impact: {strategy['expected_total_impact_bps']:.2f} bps")
