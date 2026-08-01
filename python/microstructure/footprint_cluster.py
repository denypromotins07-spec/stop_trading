"""
Chapter 1: Advanced Order Flow & Footprint Analytics
footprint_cluster.py - Cluster trade ticks into price-node footprint charts using Numba JIT
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class FootprintNode:
    """Represents a single price node in the footprint chart"""
    price: float
    bid_volume: float
    ask_volume: float
    delta: float
    trade_count: int
    imbalance: float


@njit(cache=True, nogil=True)
def cluster_ticks_to_footprint(
    prices: np.ndarray,
    volumes: np.ndarray,
    sides: np.ndarray,  # 1 = ask (buy), -1 = bid (sell)
    price_tick_size: float,
    min_price: float,
    max_price: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cluster individual trade ticks into footprint nodes at exact price levels.
    
    Args:
        prices: Array of trade prices
        volumes: Array of trade volumes
        sides: Array of trade sides (1=ask, -1=bid)
        price_tick_size: Minimum price increment
        min_price: Minimum price level to track
        max_price: Maximum price level to track
    
    Returns:
        Tuple of (price_levels, bid_volumes, ask_volumes, deltas, trade_counts)
    """
    n_levels = int((max_price - min_price) / price_tick_size) + 1
    
    bid_vols = np.zeros(n_levels, dtype=np.float64)
    ask_vols = np.zeros(n_levels, dtype=np.float64)
    counts = np.zeros(n_levels, dtype=np.int64)
    
    for i in range(len(prices)):
        if prices[i] < min_price or prices[i] > max_price:
            continue
        
        level_idx = int((prices[i] - min_price) / price_tick_size)
        
        if sides[i] > 0:  # Ask side (aggressive buy)
            ask_vols[level_idx] += volumes[i]
        else:  # Bid side (aggressive sell)
            bid_vols[level_idx] += volumes[i]
        
        counts[level_idx] += 1
    
    deltas = ask_vols - bid_vols
    
    # Generate price levels
    price_levels = np.empty(n_levels, dtype=np.float64)
    for i in range(n_levels):
        price_levels[i] = min_price + (i * price_tick_size)
    
    return price_levels, bid_vols, ask_vols, deltas, counts


@njit(cache=True, nogil=True)
def calculate_imbalance(bid_vols: np.ndarray, ask_vols: np.ndarray) -> np.ndarray:
    """
    Calculate volume imbalance at each price level.
    Imbalance = (Ask - Bid) / (Ask + Bid) normalized to [-1, 1]
    """
    n = len(bid_vols)
    imbalances = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        total = ask_vols[i] + bid_vols[i]
        if total > 0:
            imbalances[i] = (ask_vols[i] - bid_vols[i]) / total
        else:
            imbalances[i] = 0.0
    
    return imbalances


@njit(cache=True, nogil=True)
def detect_poc(price_levels: np.ndarray, volumes: np.ndarray) -> int:
    """
    Detect Point of Control (POC) - price level with highest volume.
    Returns index of POC level.
    """
    max_vol = 0.0
    poc_idx = 0
    
    for i in range(len(volumes)):
        if volumes[i] > max_vol:
            max_vol = volumes[i]
            poc_idx = i
    
    return poc_idx


@njit(cache=True, nogil=True)
def find_value_area(
    price_levels: np.ndarray,
    volumes: np.ndarray,
    poc_idx: int,
    target_percentage: float = 0.70
) -> Tuple[int, int]:
    """
    Find Value Area (VAH/VAL) containing target_percentage of total volume.
    Starts from POC and expands outward.
    
    Returns:
        Tuple of (lower_idx, upper_idx) defining value area bounds
    """
    total_vol = 0.0
    for v in volumes:
        total_vol += v
    
    target_vol = total_vol * target_percentage
    
    if total_vol == 0:
        return 0, len(volumes) - 1
    
    current_vol = volumes[poc_idx]
    lower_idx = poc_idx
    upper_idx = poc_idx
    
    while current_vol < target_vol:
        left_vol = 0.0
        right_vol = 0.0
        
        if lower_idx > 0:
            left_vol = volumes[lower_idx - 1]
        
        if upper_idx < len(volumes) - 1:
            right_vol = volumes[upper_idx + 1]
        
        if left_vol >= right_vol:
            if lower_idx > 0:
                lower_idx -= 1
                current_vol += left_vol
            elif upper_idx < len(volumes) - 1:
                upper_idx += 1
                current_vol += right_vol
            else:
                break
        else:
            if upper_idx < len(volumes) - 1:
                upper_idx += 1
                current_vol += right_vol
            elif lower_idx > 0:
                lower_idx -= 1
                current_vol += left_vol
            else:
                break
    
    return lower_idx, upper_idx


@njit(cache=True, nogil=True)
def detect_stacked_imbalance(
    imbalances: np.ndarray,
    threshold: float = 0.8,
    min_stack: int = 3
) -> np.ndarray:
    """
    Detect stacked imbalances - consecutive price levels with extreme imbalance.
    These indicate potential support/resistance zones.
    
    Returns:
        Array of starting indices for stacked imbalance zones
    """
    n = len(imbalances)
    result = np.zeros(n, dtype=np.int64)
    count = 0
    
    i = 0
    while i < n:
        stack_start = -1
        stack_len = 0
        
        while i < n and abs(imbalances[i]) >= threshold:
            if stack_start == -1:
                stack_start = i
            stack_len += 1
            i += 1
        
        if stack_len >= min_stack:
            result[count] = stack_start
            count += 1
        
        i += 1
    
    return result[:count]


class FootprintClusterEngine:
    """
    High-performance footprint clustering engine using Numba JIT.
    Processes tick data into actionable footprint analytics.
    """
    
    def __init__(
        self,
        price_tick_size: float,
        lookback_levels: int = 50
    ):
        self.price_tick_size = price_tick_size
        self.lookback_levels = lookback_levels
        self._cache_min_price = None
        self._cache_max_price = None
    
    def process_ticks(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        sides: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Process raw tick data into complete footprint analytics.
        
        Args:
            prices: Trade prices
            volumes: Trade volumes
            sides: Trade sides (1=ask, -1=bid)
        
        Returns:
            Dictionary containing all footprint metrics
        """
        # Determine price range dynamically
        min_price = np.min(prices)
        max_price = np.max(prices)
        
        # Expand range slightly for context
        price_buffer = self.price_tick_size * self.lookback_levels
        min_price -= price_buffer
        max_price += price_buffer
        
        # Cluster ticks to footprint
        price_levels, bid_vols, ask_vols, deltas, counts = \
            cluster_ticks_to_footprint(
                prices, volumes, sides,
                self.price_tick_size, min_price, max_price
            )
        
        # Calculate derived metrics
        total_vols = bid_vols + ask_vols
        imbalances = calculate_imbalance(bid_vols, ask_vols)
        poc_idx = detect_poc(price_levels, total_vols)
        val_idx, vah_idx = find_value_area(price_levels, total_vols, poc_idx)
        
        # Detect stacked imbalances
        stacked_zones = detect_stacked_imbalance(imbalances)
        
        return {
            'price_levels': price_levels,
            'bid_volumes': bid_vols,
            'ask_volumes': ask_vols,
            'deltas': deltas,
            'trade_counts': counts,
            'total_volumes': total_vols,
            'imbalances': imbalances,
            'poc_idx': poc_idx,
            'poc_price': price_levels[poc_idx],
            'vah_idx': vah_idx,
            'val_idx': val_idx,
            'stacked_imbalances': stacked_zones
        }
    
    def get_active_nodes(
        self,
        footprint_data: Dict[str, np.ndarray]
    ) -> List[FootprintNode]:
        """Extract only active (non-zero volume) nodes."""
        nodes = []
        counts = footprint_data['trade_counts']
        
        for i in range(len(counts)):
            if counts[i] > 0:
                node = FootprintNode(
                    price=footprint_data['price_levels'][i],
                    bid_volume=footprint_data['bid_volumes'][i],
                    ask_volume=footprint_data['ask_volumes'][i],
                    delta=footprint_data['deltas'][i],
                    trade_count=int(counts[i]),
                    imbalance=footprint_data['imbalances'][i]
                )
                nodes.append(node)
        
        return nodes


# Module-level convenience function
def create_footprint_engine(
    tick_size: float,
    lookback: int = 50
) -> FootprintClusterEngine:
    """Factory function to create optimized footprint engine."""
    return FootprintClusterEngine(tick_size, lookback)
