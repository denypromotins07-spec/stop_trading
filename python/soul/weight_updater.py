"""
Weight updater for hot-swapping adaptive weights from SOUL.md.
Parses updated weights and applies them to live Nautilus strategies without restart.
"""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import get_logger

logger = get_logger("weight_updater")


class WeightUpdateEvent:
    """Represents a weight update event."""
    
    def __init__(
        self,
        weight_name: str,
        old_value: float,
        new_value: float,
        timestamp: float,
        source: str = "SOUL.md",
    ):
        self.weight_name = weight_name
        self.old_value = old_value
        self.new_value = new_value
        self.timestamp = timestamp
        self.source = source
    
    def __repr__(self) -> str:
        return (
            f"WeightUpdateEvent({self.weight_name}: "
            f"{self.old_value:.4f} -> {self.new_value:.4f})"
        )


class AdaptiveWeightManager:
    """
    Manages adaptive weights for ML models and trading strategies.
    Supports hot-swapping weights without restarting the system.
    """
    
    def __init__(self):
        self._weights: Dict[str, float] = {}
        self._weight_history: List[Dict[str, Any]] = []
        self._update_callbacks: List[Callable[[WeightUpdateEvent], None]] = []
        self._max_history_size = 1000
    
    def set_weight(self, name: str, value: float) -> Optional[WeightUpdateEvent]:
        """
        Set or update a weight value.
        
        Args:
            name: Weight name
            value: New weight value
        
        Returns:
            WeightUpdateEvent if changed, None otherwise
        """
        old_value = self._weights.get(name, 0.0)
        
        # Only create event if value actually changed
        if abs(old_value - value) < 1e-9:
            return None
        
        self._weights[name] = value
        
        # Record history
        self._weight_history.append({
            'name': name,
            'old_value': old_value,
            'new_value': value,
            'timestamp': time.time(),
        })
        
        # Trim history if needed
        if len(self._weight_history) > self._max_history_size:
            self._weight_history = self._weight_history[-self._max_history_size:]
        
        # Create and dispatch event
        event = WeightUpdateEvent(
            weight_name=name,
            old_value=old_value,
            new_value=value,
            timestamp=time.time(),
        )
        
        self._dispatch_update(event)
        
        logger.info(f"Weight updated: {name} = {value:.6f} (was {old_value:.6f})")
        return event
    
    def get_weight(self, name: str, default: float = 0.0) -> float:
        """Get a weight value by name."""
        return self._weights.get(name, default)
    
    def get_all_weights(self) -> Dict[str, float]:
        """Get all current weights."""
        return dict(self._weights)
    
    def set_weights_batch(self, weights: Dict[str, float]) -> List[WeightUpdateEvent]:
        """
        Set multiple weights at once.
        
        Args:
            weights: Dictionary of weight name -> value
        
        Returns:
            List of WeightUpdateEvents for changed weights
        """
        events = []
        for name, value in weights.items():
            event = self.set_weight(name, value)
            if event:
                events.append(event)
        
        if events:
            logger.info(f"Batch update: {len(events)} weights changed")
        
        return events
    
    def register_update_callback(self, callback: Callable[[WeightUpdateEvent], None]) -> None:
        """Register a callback for weight updates."""
        self._update_callbacks.append(callback)
        logger.debug(f"Registered weight update callback: {callback.__name__}")
    
    def _dispatch_update(self, event: WeightUpdateEvent) -> None:
        """Dispatch update event to all callbacks."""
        for callback in self._update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(event))
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Weight update callback error: {e}")
    
    def get_weight_history(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get weight change history.
        
        Args:
            name: Optional weight name filter
        
        Returns:
            List of historical weight changes
        """
        if name is None:
            return list(self._weight_history)
        
        return [h for h in self._weight_history if h['name'] == name]


class NautilusStrategyWeightUpdater:
    """
    Integrates adaptive weights with Nautilus Trader strategies.
    Hot-swaps weights into live strategies without restart.
    """
    
    def __init__(self, weight_manager: AdaptiveWeightManager):
        self.weight_manager = weight_manager
        self._strategies: Dict[str, Any] = {}
        self._strategy_params: Dict[str, Dict[str, str]] = {}
    
    def register_strategy(
        self,
        strategy_id: str,
        strategy_instance: Any,
        param_mapping: Dict[str, str],
    ) -> None:
        """
        Register a Nautilus strategy for weight updates.
        
        Args:
            strategy_id: Unique strategy identifier
            strategy_instance: The strategy object
            param_mapping: Mapping of weight_name -> strategy_param_name
        """
        self._strategies[strategy_id] = strategy_instance
        self._strategy_params[strategy_id] = param_mapping
        logger.info(f"Registered strategy {strategy_id} for weight updates")
    
    def apply_weights_to_strategy(self, strategy_id: str) -> bool:
        """
        Apply current weights to a registered strategy.
        
        Args:
            strategy_id: Strategy to update
        
        Returns:
            True if successful
        """
        if strategy_id not in self._strategies:
            logger.error(f"Unknown strategy: {strategy_id}")
            return False
        
        strategy = self._strategies[strategy_id]
        param_mapping = self._strategy_params.get(strategy_id, {})
        
        if not param_mapping:
            logger.warning(f"No parameter mapping for strategy {strategy_id}")
            return False
        
        try:
            for weight_name, param_name in param_mapping.items():
                weight_value = self.weight_manager.get_weight(weight_name)
                
                # Try to set the parameter on the strategy
                if hasattr(strategy, param_name):
                    setattr(strategy, param_name, weight_value)
                    logger.debug(
                        f"Applied weight {weight_name}={weight_value:.4f} "
                        f"to strategy.{param_name}"
                    )
                else:
                    logger.warning(
                        f"Strategy {strategy_id} has no attribute {param_name}"
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply weights to strategy {strategy_id}: {e}")
            return False
    
    def apply_all_weights(self) -> Dict[str, bool]:
        """
        Apply all weights to all registered strategies.
        
        Returns:
            Dictionary of strategy_id -> success status
        """
        results = {}
        for strategy_id in self._strategies:
            results[strategy_id] = self.apply_weights_to_strategy(strategy_id)
        return results


class SOULWeightUpdater:
    """
    Main class that ties together SOUL.md parsing and weight updates.
    Monitors SOUL.md for weight changes and applies them live.
    """
    
    def __init__(self):
        self.weight_manager = AdaptiveWeightManager()
        self.nautilus_updater = NautilusStrategyWeightUpdater(self.weight_manager)
        self._current_soul_weights: Dict[str, float] = {}
    
    def update_from_soul_data(self, soul_data: Dict[str, Any]) -> List[WeightUpdateEvent]:
        """
        Update weights based on parsed SOUL.md data.
        
        Args:
            soul_data: Parsed data from ledger_parser
        
        Returns:
            List of weight update events
        """
        new_weights = soul_data.get('adaptive_weights', {})
        
        if not new_weights:
            return []
        
        # Compare with current weights
        events = []
        for name, value in new_weights.items():
            old_value = self._current_soul_weights.get(name)
            
            if old_value is None or abs(old_value - value) > 1e-9:
                event = self.weight_manager.set_weight(name, value)
                if event:
                    events.append(event)
        
        # Update cached weights
        self._current_soul_weights = dict(new_weights)
        
        # Apply to all registered strategies
        if events:
            self.nautilus_updater.apply_all_weights()
        
        return events
    
    def register_strategy(
        self,
        strategy_id: str,
        strategy_instance: Any,
        param_mapping: Dict[str, str],
    ) -> None:
        """Convenience method to register a strategy."""
        self.nautilus_updater.register_strategy(
            strategy_id, strategy_instance, param_mapping
        )
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get all current adaptive weights."""
        return self.weight_manager.get_all_weights()


# Global weight updater instance
_weight_updater_instance: Optional[SOULWeightUpdater] = None


def get_weight_updater() -> SOULWeightUpdater:
    """Get or create the global weight updater instance."""
    global _weight_updater_instance
    if _weight_updater_instance is None:
        _weight_updater_instance = SOULWeightUpdater()
    return _weight_updater_instance


def apply_soul_weights(soul_data: Dict[str, Any]) -> List[WeightUpdateEvent]:
    """
    Convenience function to apply weights from SOUL.md data.
    
    Args:
        soul_data: Parsed SOUL.md data
    
    Returns:
        List of weight update events
    """
    updater = get_weight_updater()
    return updater.update_from_soul_data(soul_data)
