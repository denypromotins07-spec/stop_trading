"""
Contextual Multi-Armed Bandit for Smart Order Routing (SOR).
Predicts highest fill probability across fragmented venues.
Uses bounded replay buffer to prevent unbounded memory growth.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import time


class ContextualBandit:
    """
    Linear Thompson Sampling contextual bandit for venue selection.
    Maintains bounded replay buffer for continuous learning.
    """
    
    def __init__(self, n_venues: int, n_features: int = 10,
                 lambda_reg: float = 1.0, sigma_noise: float = 0.1):
        """
        Initialize the contextual bandit.
        
        Args:
            n_venues: Number of available venues
            n_features: Number of context features
            lambda_reg: Regularization parameter
            sigma_noise: Noise variance for Thompson Sampling
        """
        self.n_venues = n_venues
        self.n_features = n_features
        
        # Model parameters per venue
        self.theta_mean = np.zeros((n_venues, n_features))
        self.theta_cov = np.eye(n_features) / lambda_reg
        
        # Sufficient statistics for online updates
        self.A = [np.eye(n_features) * lambda_reg for _ in range(n_venues)]
        self.b = [np.zeros(n_features) for _ in range(n_venues)]
        
        # Bounded replay buffer
        self.replay_buffer: deque = deque(maxlen=5000)
        
        # Tracking
        self._total_rewards: Dict[int, float] = {i: 0.0 for i in range(n_venues)}
        self._total_pulls: Dict[int, int] = {i: 0 for i in range(n_venues)}
        self._last_update_time: float = 0
    
    def _extract_context(self, venue_data: Dict) -> np.ndarray:
        """Extract feature vector from venue data."""
        features = []
        
        # Latency features
        features.append(venue_data.get("latency_ms", 50) / 100)
        features.append(venue_data.get("latency_std_ms", 10) / 50)
        
        # Fee features
        features.append(venue_data.get("maker_fee_bps", 10) / 100)
        features.append(venue_data.get("taker_fee_bps", 20) / 100)
        
        # Liquidity features
        features.append(venue_data.get("bid_depth_usd", 1e6) / 1e7)
        features.append(venue_data.get("ask_depth_usd", 1e6) / 1e7)
        features.append(venue_data.get("spread_bps", 5) / 50)
        
        # Queue position estimate
        features.append(venue_data.get("queue_position_pct", 0.5))
        
        # Historical fill rate
        features.append(venue_data.get("recent_fill_rate", 0.8))
        
        # Time-based features
        hour = time.localtime().tm_hour
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        
        # Ensure correct dimension
        while len(features) < self.n_features:
            features.append(0.0)
        
        return np.array(features[:self.n_features])
    
    def select_venue(self, venue_contexts: Dict[int, Dict],
                     exploration_bonus: float = 0.1) -> Tuple[int, Dict]:
        """
        Select optimal venue using Thompson Sampling.
        
        Args:
            venue_contexts: Dict mapping venue_id to context data
            exploration_bonus: Additional exploration bonus
            
        Returns:
            Tuple of (selected_venue_id, selection_info)
        """
        # Sample theta for each venue
        sampled_theta = {}
        ucb_values = {}
        
        for venue_id in venue_contexts.keys():
            context = self._extract_context(venue_contexts[venue_id])
            
            # Thompson Sampling: sample from posterior
            theta_sample = np.random.multivariate_normal(
                self.theta_mean[venue_id],
                self.theta_cov * sigma_noise if (sigma_noise := getattr(self, '_sigma', 0.1)) > 0 
                else self.theta_cov
            )
            sampled_theta[venue_id] = theta_sample
            
            # Compute UCB value
            predicted_reward = np.dot(theta_sample, context)
            
            # Add exploration bonus based on uncertainty
            uncertainty = np.sqrt(np.dot(context, np.dot(self.theta_cov, context)))
            ucb_values[venue_id] = predicted_reward + exploration_bonus * uncertainty
        
        # Select venue with highest UCB
        selected_venue = max(ucb_values, key=ucb_values.get)
        
        return selected_venue, {
            "ucb_values": ucb_values,
            "sampled_theta_norms": {k: float(np.linalg.norm(v)) for k, v in sampled_theta.items()},
            "timestamp": int(time.time() * 1e9)
        }
    
    def update(self, venue_id: int, context: np.ndarray, reward: float):
        """
        Update model with observed reward.
        
        Args:
            venue_id: Selected venue
            context: Context feature vector
            reward: Observed reward (fill rate, negative cost, etc.)
        """
        # Store in replay buffer
        self.replay_buffer.append((venue_id, context.copy(), reward))
        
        # Update sufficient statistics
        self.A[venue_id] += np.outer(context, context)
        self.b[venue_id] += reward * context
        
        # Update posterior mean
        self.theta_mean[venue_id] = np.linalg.solve(self.A[venue_id], self.b[venue_id])
        
        # Update tracking
        self._total_rewards[venue_id] += reward
        self._total_pulls[venue_id] += 1
        self._last_update_time = time.time()
    
    def batch_update_from_replay(self, batch_size: int = 32):
        """Perform batch update from replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        
        for idx in indices:
            venue_id, context, reward = self.replay_buffer[idx]
            # Small learning rate for stability
            lr = 0.01
            self.theta_mean[venue_id] += lr * (reward - np.dot(self.theta_mean[venue_id], context)) * context
    
    def get_venue_stats(self) -> Dict[int, Dict]:
        """Get statistics for all venues."""
        stats = {}
        for venue_id in range(self.n_venues):
            pulls = self._total_pulls[venue_id]
            avg_reward = self._total_rewards[venue_id] / (pulls + 1e-10)
            
            stats[venue_id] = {
                "total_pulls": pulls,
                "average_reward": float(avg_reward),
                "estimated_fill_rate": float(avg_reward),
                "theta_norm": float(np.linalg.norm(self.theta_mean[venue_id]))
            }
        
        return stats


class VenuePredictor:
    """
    Main venue prediction system using contextual bandits.
    Continuously updates routing policy based on real-time metrics.
    """
    
    def __init__(self, venues: List[Dict], instruments: List[str]):
        """
        Initialize venue predictor.
        
        Args:
            venues: List of venue configurations
            instruments: List of tradable instruments
        """
        self.venues = {v["id"]: v for v in venues}
        self.instruments = instruments
        self.n_venues = len(venues)
        
        # Create bandit per instrument
        self.bandits: Dict[str, ContextualBandit] = {
            inst: ContextualBandit(n_venues=self.n_venues, n_features=10)
            for inst in instruments
        }
        
        # Venue state tracking
        self._venue_states: Dict[str, Dict] = {}
        
        # Routing history
        self._routing_history: deque = deque(maxlen=1000)
    
    def update_venue_state(self, venue_id: str, state: Dict):
        """Update real-time state for a venue."""
        self._venue_states[venue_id] = {
            **self._venue_states.get(venue_id, {}),
            **state,
            "last_update": time.time()
        }
    
    def predict_best_venue(self, instrument_id: str, order_side: str,
                           order_size: float) -> Dict:
        """
        Predict best venue for an order.
        
        Args:
            instrument_id: Asset identifier
            order_side: "buy" or "sell"
            order_size: Order size
            
        Returns:
            Routing decision dictionary
        """
        if instrument_id not in self.bandits:
            return {"error": "Instrument not found"}
        
        bandit = self.bandits[instrument_id]
        
        # Build context for each venue
        venue_contexts = {}
        for venue_id, venue_info in self.venues.items():
            state = self._venue_states.get(venue_id, {})
            
            venue_contexts[venue_id] = {
                "latency_ms": state.get("latency_ms", venue_info.get("avg_latency_ms", 50)),
                "latency_std_ms": state.get("latency_std_ms", 10),
                "maker_fee_bps": venue_info.get("maker_fee_bps", 10),
                "taker_fee_bps": venue_info.get("taker_fee_bps", 20),
                "bid_depth_usd": state.get("bid_depth_usd", 1e6),
                "ask_depth_usd": state.get("ask_depth_usd", 1e6),
                "spread_bps": state.get("spread_bps", 5),
                "queue_position_pct": state.get("queue_position_pct", 0.5),
                "recent_fill_rate": state.get("recent_fill_rate", 0.8),
                "order_size_factor": order_size / 1e6
            }
        
        # Select venue
        selected_venue, selection_info = bandit.select_venue(venue_contexts)
        
        # Calculate expected metrics
        venue_state = self._venue_states.get(selected_venue, {})
        expected_fill_rate = venue_state.get("recent_fill_rate", 0.8)
        expected_cost_bps = (
            self.venues[selected_venue].get("taker_fee_bps", 20) +
            venue_state.get("spread_bps", 5) * 0.5
        )
        
        result = {
            "instrument_id": instrument_id,
            "order_side": order_side,
            "order_size": order_size,
            "selected_venue": selected_venue,
            "expected_fill_rate": expected_fill_rate,
            "expected_cost_bps": expected_cost_bps,
            "selection_confidence": selection_info["ucb_values"].get(selected_venue, 0),
            "alternative_venues": sorted(
                selection_info["ucb_values"].keys(),
                key=lambda x: selection_info["ucb_values"][x],
                reverse=True
            )[1:3],
            "timestamp": int(time.time() * 1e9)
        }
        
        # Record routing decision
        self._routing_history.append(result)
        
        return result
    
    def record_execution_outcome(self, instrument_id: str, venue_id: str,
                                  order_size: float, filled: bool,
                                  fill_rate: float, cost_bps: float):
        """
        Record execution outcome for learning.
        
        Args:
            instrument_id: Asset identifier
            venue_id: Executed venue
            order_size: Order size
            filled: Whether order was fully filled
            fill_rate: Percentage filled
            cost_bps: Total cost in basis points
        """
        if instrument_id not in self.bandits:
            return
        
        bandit = self.bandits[instrument_id]
        
        # Compute reward (combination of fill rate and cost efficiency)
        fill_reward = fill_rate if filled else fill_rate * 0.5
        cost_penalty = cost_bps / 100  # Normalize
        reward = fill_reward - cost_penalty * 0.1  # Weight cost less than fills
        
        # Get context that was used
        venue_state = self._venue_states.get(venue_id, {})
        context = bandit._extract_context({
            "latency_ms": venue_state.get("latency_ms", 50),
            "latency_std_ms": venue_state.get("latency_std_ms", 10),
            "maker_fee_bps": self.venues.get(venue_id, {}).get("maker_fee_bps", 10),
            "taker_fee_bps": self.venues.get(venue_id, {}).get("taker_fee_bps", 20),
            "bid_depth_usd": venue_state.get("bid_depth_usd", 1e6),
            "ask_depth_usd": venue_state.get("ask_depth_usd", 1e6),
            "spread_bps": venue_state.get("spread_bps", 5),
            "queue_position_pct": venue_state.get("queue_position_pct", 0.5),
            "recent_fill_rate": venue_state.get("recent_fill_rate", 0.8),
            "order_size_factor": order_size / 1e6
        })
        
        # Update bandit
        venue_idx = list(self.venues.keys()).index(venue_id) if venue_id in self.venues else 0
        bandit.update(venue_idx, context, reward)
        
        # Periodic batch update
        if len(bandit.replay_buffer) % 100 == 0:
            bandit.batch_update_from_replay(batch_size=32)
    
    def get_routing_analytics(self) -> Dict:
        """Get comprehensive routing analytics."""
        analytics = {
            "total_routings": len(self._routing_history),
            "by_instrument": {},
            "by_venue": {}
        }
        
        # Per-instrument stats
        for inst, bandit in self.bandits.items():
            analytics["by_instrument"][inst] = bandit.get_venue_stats()
        
        # Venue-level aggregation
        venue_counts = {}
        for routing in self._routing_history:
            venue = routing.get("selected_venue")
            venue_counts[venue] = venue_counts.get(venue, 0) + 1
        
        analytics["by_venue"] = venue_counts
        
        return analytics


if __name__ == "__main__":
    # Example usage
    venues = [
        {"id": "binance", "maker_fee_bps": 10, "taker_fee_bps": 10, "avg_latency_ms": 30},
        {"id": "coinbase", "maker_fee_bps": 5, "taker_fee_bps": 15, "avg_latency_ms": 50},
        {"id": "kraken", "maker_fee_bps": 8, "taker_fee_bps": 12, "avg_latency_ms": 40},
        {"id": "ftx", "maker_fee_bps": 2, "taker_fee_bps": 7, "avg_latency_ms": 35}
    ]
    
    instruments = ["BTC", "ETH", "SOL"]
    
    predictor = VenuePredictor(venues, instruments)
    
    # Simulate venue states
    np.random.seed(42)
    for venue in venues:
        predictor.update_venue_state(venue["id"], {
            "latency_ms": np.random.uniform(20, 60),
            "latency_std_ms": np.random.uniform(5, 15),
            "bid_depth_usd": np.random.lognormal(16, 0.5),
            "ask_depth_usd": np.random.lognormal(16, 0.5),
            "spread_bps": np.random.exponential(5),
            "recent_fill_rate": np.random.uniform(0.7, 0.95)
        })
    
    # Test routing predictions
    print("Venue Routing Predictions:\n")
    
    for inst in instruments:
        result = predictor.predict_best_venue(inst, "buy", order_size=1e5)
        print(f"{inst}:")
        print(f"  Selected Venue: {result['selected_venue']}")
        print(f"  Expected Fill Rate: {result['expected_fill_rate']:.1%}")
        print(f"  Expected Cost: {result['expected_cost_bps']:.2f} bps")
        print(f"  Alternatives: {result['alternative_venues']}")
        print()
        
        # Simulate execution outcome
        predictor.record_execution_outcome(
            instrument_id=inst,
            venue_id=result["selected_venue"],
            order_size=1e5,
            filled=np.random.random() > 0.2,
            fill_rate=np.random.uniform(0.8, 1.0),
            cost_bps=np.random.uniform(5, 20)
        )
    
    # Analytics
    analytics = predictor.get_routing_analytics()
    print(f"Total Routings: {analytics['total_routings']}")
    print(f"Venue Distribution: {analytics['by_venue']}")
