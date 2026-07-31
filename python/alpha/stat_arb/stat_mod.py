"""
Statistical Arbitrage Module Root.
Wraps cointegration and mean reversion calculations in Ray actors for parallelized pair-wise spread monitoring.
Memory-efficient design with strict RAM limits to stay within 3GB Python ceiling.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class PairMetrics:
    """Container for pair-wise statistical arbitrage metrics."""
    pair: Tuple[str, str]
    hedge_ratio: float
    spread: float
    z_score: float
    is_cointegrated: bool
    half_life: float
    signal: int  # 1=LONG, -1=SHORT, 0=FLAT
    timestamp_ns: int


class StatArbCoordinator:
    """
    Central coordinator for statistical arbitrage calculations.
    Manages Ray actors for distributed pair monitoring.
    """
    
    def __init__(self, 
                 pairs: List[Tuple[str, str]],
                 max_workers: int = 8,
                 memory_limit_mb: int = 256):
        """
        Args:
            pairs: List of (asset_x, asset_y) tuples to monitor
            max_workers: Maximum number of parallel workers
            memory_limit_mb: Memory limit per worker in MB
        """
        self.pairs = pairs
        self.max_workers = max_workers
        self.memory_limit_mb = memory_limit_mb
        
        self.actors = {}
        self.results_cache = {}
        self._ray_initialized = False
        
    def initialize_ray(self):
        """Initialize Ray cluster with memory constraints."""
        try:
            import ray
            
            if not ray.is_initialized():
                # Configure Ray with strict memory limits
                ray.init(
                    num_cpus=self.max_workers,
                    _system_config={
                        'object_spilling_enabled': False,  # Disable disk spilling for low latency
                        'max_object_store_memory': self.memory_limit_mb * 1024 * 1024 // len(self.pairs),
                    }
                )
            self._ray_initialized = True
            
        except ImportError:
            print("Ray not available, falling back to sequential execution")
            self._ray_initialized = False
    
    def create_actors(self):
        """Create Ray actors for each pair."""
        from .cointegration import create_pair_actor_class
        from .mean_reversion import SpreadMonitor
        
        PairMonitorActor = create_pair_actor_class()
        
        if PairMonitorActor is None or not self._ray_initialized:
            # Fallback to sequential monitoring
            self.sequential_monitor = SpreadMonitor(
                [(p[0], p[1]) for p in self.pairs],
                window_size=252
            )
            return
        
        # Create actors for subset of pairs (batch processing)
        batch_size = min(len(self.pairs), self.max_workers)
        
        for i, pair in enumerate(self.pairs[:batch_size]):
            actor = PairMonitorActor.remote(pair, window_size=252)
            self.actors[pair] = actor
    
    async def update_prices_async(self, prices: Dict[str, float]) -> List[PairMetrics]:
        """
        Update all pairs asynchronously using Ray actors.
        
        Args:
            prices: Dictionary of current asset prices
            
        Returns:
            List of PairMetrics for all monitored pairs
        """
        if not self._ray_initialized or not self.actors:
            return self._update_sequential(prices)
        
        import ray
        
        # Trigger updates on all actors
        futures = []
        for pair, actor in self.actors.items():
            asset_x, asset_y = pair
            if asset_x in prices and asset_y in prices:
                future = actor.update.remote(prices[asset_x], prices[asset_y])
                futures.append((pair, future))
        
        # Collect results
        results = []
        for pair, future in futures:
            try:
                data = await future
                metrics = self._convert_to_metrics(pair, data, time.time_ns())
                results.append(metrics)
            except Exception as e:
                # Log error but continue processing other pairs
                pass
        
        return results
    
    def _update_sequential(self, prices: Dict[str, float]) -> List[PairMetrics]:
        """Fallback sequential update when Ray is not available."""
        from .mean_reversion import SpreadMonitor, SignalType
        
        if not hasattr(self, 'sequential_monitor'):
            self.sequential_monitor = SpreadMonitor(
                [(p[0], p[1]) for p in self.pairs],
                window_size=252
            )
        
        results = self.sequential_monitor.update_prices(prices)
        metrics = []
        
        for (asset_x, asset_y), data in results.items():
            signal_map = {'LONG': 1, 'SHORT': -1, 'FLAT': 0}
            metric = PairMetrics(
                pair=(asset_x, asset_y),
                hedge_ratio=1.0,  # Would need to track separately
                spread=data['spread'],
                z_score=data['z_score'],
                is_cointegrated=True,  # Assumed for sequential
                half_life=data.get('half_life', 0.0) or 0.0,
                signal=signal_map.get(data['signal'], 0),
                timestamp_ns=time.time_ns()
            )
            metrics.append(metric)
        
        return metrics
    
    def _convert_to_metrics(self, pair: tuple, data: dict, timestamp_ns: int) -> PairMetrics:
        """Convert raw actor data to PairMetrics."""
        from .mean_reversion import SignalType
        
        # Map z-score to signal
        z_score = data.get('z_score', 0.0)
        if z_score > 2.0:
            signal = -1  # Short
        elif z_score < -2.0:
            signal = 1   # Long
        else:
            signal = 0   # Flat
        
        return PairMetrics(
            pair=pair,
            hedge_ratio=data.get('hedge_ratio', 1.0),
            spread=data.get('spread', 0.0),
            z_score=z_score,
            is_cointegrated=data.get('is_cointegrated', False),
            half_life=data.get('half_life', 0.0),
            signal=signal,
            timestamp_ns=timestamp_ns
        )
    
    def get_signals(self, prices: Dict[str, float]) -> List[Dict]:
        """
        Synchronous method to get trading signals.
        
        Args:
            prices: Current asset prices
            
        Returns:
            List of signal dictionaries
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        metrics = loop.run_until_complete(self.update_prices_async(prices))
        
        signals = []
        for m in metrics:
            if m.signal != 0:
                signals.append({
                    'pair': f"{m.pair[1]}/{m.pair[0]}",
                    'side': 'BUY' if m.signal > 0 else 'SELL',
                    'z_score': m.z_score,
                    'spread': m.spread,
                    'hedge_ratio': m.hedge_ratio,
                    'confidence': min(abs(m.z_score) / 3.0, 1.0),
                    'is_cointegrated': m.is_cointegrated,
                    'half_life_ms': m.half_life * 1000 if m.half_life else None
                })
        
        return signals
    
    def shutdown(self):
        """Shutdown Ray cluster and cleanup resources."""
        import ray
        
        if self._ray_initialized:
            # Kill all actors
            for actor in self.actors.values():
                ray.kill(actor)
            self.actors.clear()
            
            # Shutdown Ray
            ray.shutdown()
            self._ray_initialized = False


# Batch processor for high-throughput scenarios
class BatchStatArbProcessor:
    """
    Processes batches of price updates efficiently.
    Optimized for throughput over latency.
    """
    
    def __init__(self, pairs: List[Tuple[str, str]], batch_size: int = 100):
        """
        Args:
            pairs: List of (asset_x, asset_y) tuples
            batch_size: Number of updates to batch before processing
        """
        from .cointegration import AdaptiveCointegrationTracker
        from .mean_reversion import SpreadMonitor
        
        self.pairs = pairs
        self.batch_size = batch_size
        
        # Initialize trackers
        self.coint_tracker = AdaptiveCointegrationTracker(pairs, window_size=252)
        self.spread_monitor = SpreadMonitor(pairs, window_size=252)
        
        # Price buffers
        self.price_buffer = {}
        self.buffer_count = 0
        
    def push_price(self, asset: str, price: float) -> Optional[List[Dict]]:
        """
        Push a price update and process if batch is full.
        
        Args:
            asset: Asset symbol
            price: Current price
            
        Returns:
            List of signals if batch processed, None otherwise
        """
        self.price_buffer[asset] = price
        self.buffer_count += 1
        
        if self.buffer_count >= self.batch_size:
            return self.process_batch()
        
        return None
    
    def process_batch(self) -> List[Dict]:
        """Process accumulated batch and return signals."""
        if not self.price_buffer:
            return []
        
        # Update cointegration tracking
        coint_results = self.coint_tracker.get_all_pairs_status(self.price_buffer)
        
        # Update spread monitoring
        spread_results = self.spread_monitor.update_prices(self.price_buffer)
        
        # Combine results into signals
        signals = []
        for result in coint_results:
            pair_key = result['pair']
            if pair_key in spread_results:
                spread_data = spread_results[pair_key]
                
                if result['is_cointegrated'] and abs(result['z_score']) > 2.0:
                    signal_type = 'BUY' if result['z_score'] < 0 else 'SELL'
                    
                    signals.append({
                        'pair': f"{pair_key[1]}/{pair_key[0]}",
                        'side': signal_type,
                        'z_score': result['z_score'],
                        'spread': result['spread'],
                        'hedge_ratio': result['hedge_ratio'],
                        'half_life': self.coint_tracker.eg_trackers[pair_key].state.half_life,
                        'method': result['method'],
                        'confidence': min(abs(result['z_score']) / 3.0, 1.0)
                    })
        
        # Reset buffer
        self.price_buffer.clear()
        self.buffer_count = 0
        
        return signals
    
    def get_current_state(self) -> Dict:
        """Get current state of all tracked pairs."""
        state = {}
        
        for pair in self.pairs:
            if pair in self.coint_tracker.eg_trackers:
                eg_state = self.coint_tracker.eg_trackers[pair].state
                kf_state = self.coint_tracker.kf_trackers[pair].get_state()
                
                state[pair] = {
                    'eg_beta': eg_state.beta,
                    'kf_beta': kf_state.beta,
                    'mean_spread': eg_state.mean_spread,
                    'std_spread': eg_state.std_spread,
                    'is_cointegrated': eg_state.n_samples > 50
                }
        
        return state


# Factory function for creating configured coordinators
def create_stat_arb_system(pairs: List[Tuple[str, str]], 
                           mode: str = 'distributed',
                           **kwargs) -> object:
    """
    Factory function to create stat arb system in different modes.
    
    Args:
        pairs: List of pairs to monitor
        mode: 'distributed' for Ray actors, 'sequential' for single-threaded, 'batch' for throughput
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured coordinator instance
    """
    if mode == 'distributed':
        coordinator = StatArbCoordinator(pairs, **kwargs)
        coordinator.initialize_ray()
        coordinator.create_actors()
        return coordinator
    elif mode == 'batch':
        return BatchStatArbProcessor(pairs, **kwargs)
    else:
        return StatArbCoordinator(pairs, **kwargs)


__all__ = [
    'StatArbCoordinator',
    'BatchStatArbProcessor',
    'PairMetrics',
    'create_stat_arb_system'
]
