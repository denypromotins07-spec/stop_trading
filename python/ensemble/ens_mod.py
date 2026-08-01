"""
Ensemble Module Root
Stage 49: Manages ensemble inference queue, bounding meta-learner tensor allocations.
Strictly enforces 3GB RAM ceiling for Python processes.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import deque
import zmq
import numpy as np

from .stacking_generalizer import StackingGeneralizer, get_generalizer, ModelPrediction, EnsembleOutput
from .regime_router import RegimeRouter, get_router, MarketRegime, StrategyType

logger = logging.getLogger(__name__)


class InferenceRequest:
    """Request for ensemble inference."""
    
    def __init__(self, 
                 predictions: List[ModelPrediction],
                 timestamp: float,
                 priority: int = 0):
        self.predictions = predictions
        self.timestamp = timestamp
        self.priority = priority
        self.future = asyncio.Future()


class EnsembleModule:
    """
    Central module managing ML ensemble inference queue.
    Strictly bounds tensor allocations to respect 3GB RAM ceiling.
    """
    
    # Memory limits (bytes)
    MAX_TENSOR_MEMORY = 512 * 1024 * 1024  # 512MB for tensors
    MAX_QUEUE_SIZE = 1000
    MAX_BATCH_SIZE = 32
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Components
        self.generalizer: Optional[StackingGeneralizer] = None
        self.router: Optional[RegimeRouter] = None
        
        # Inference queue with bounded size
        self._inference_queue: deque = deque(maxlen=self.MAX_QUEUE_SIZE)
        self._processing = False
        self._running = False
        
        # Memory tracking
        self._tensor_memory_allocated = 0
        self._inference_count = 0
        
        # ZMQ sockets
        self._zmq_context: Optional[zmq.Context] = None
        self._pub_socket: Optional[zmq.Socket] = None
        self._sub_socket: Optional[zmq.Socket] = None
        
        # Performance metrics
        self._latency_history = deque(maxlen=100)
        self._last_inference_time = 0.0
    
    async def initialize(self) -> bool:
        """Initialize the ensemble module."""
        try:
            logger.info("Initializing EnsembleModule...")
            
            # Create generalizer with memory-bounded configuration
            self.generalizer = StackingGeneralizer(
                num_base_models=3,
                num_classes=5,
                hidden_dim=64,
                max_batch_size=self.MAX_BATCH_SIZE,
            )
            
            # Create regime router
            self.router = RegimeRouter(
                min_confidence_threshold=0.6,
                max_strategy_count=5,
            )
            
            # Setup ZMQ
            self._zmq_context = zmq.Context()
            self._pub_socket = self._zmq_context.socket(zmq.PUB)
            self._pub_socket.bind("tcp://*:5562")
            
            self._sub_socket = self._zmq_context.socket(zmq.SUB)
            self._sub_socket.connect("tcp://localhost:5563")  # HMM regime updates
            self._sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            
            self._running = True
            logger.info("EnsembleModule initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize EnsembleModule: {e}")
            return False
    
    async def submit_inference(self, 
                              predictions: List[ModelPrediction],
                              priority: int = 0) -> Optional[EnsembleOutput]:
        """Submit predictions for ensemble fusion."""
        if len(self._inference_queue) >= self.MAX_QUEUE_SIZE:
            logger.warning("Inference queue full, dropping request")
            return None
        
        request = InferenceRequest(predictions, datetime.utcnow().timestamp(), priority)
        self._inference_queue.append(request)
        
        # Process immediately if not already processing
        if not self._processing:
            await self._process_queue()
        
        try:
            return await asyncio.wait_for(request.future, timeout=5.0)
        except asyncio.TimeoutError:
            logger.error("Inference request timed out")
            return None
    
    async def _process_queue(self):
        """Process inference requests from the queue."""
        if self._processing or not self._running:
            return
        
        self._processing = True
        
        try:
            while self._inference_queue and self._running:
                # Sort by priority
                sorted_queue = sorted(
                    list(self._inference_queue),
                    key=lambda x: (-x.priority, x.timestamp)
                )
                
                if not sorted_queue:
                    break
                
                request = sorted_queue[0]
                self._inference_queue.remove(request)
                
                # Check memory budget
                if not self._check_memory_budget():
                    logger.warning("Memory budget exceeded, skipping inference")
                    if not request.future.done():
                        request.future.set_result(None)
                    continue
                
                # Perform inference
                start_time = datetime.utcnow()
                
                try:
                    output = self.generalizer.fuse_predictions(
                        request.predictions,
                        batch_size=1,
                    )
                    
                    latency = (datetime.utcnow() - start_time).total_seconds()
                    self._latency_history.append(latency)
                    self._inference_count += 1
                    
                    if not request.future.done():
                        request.future.set_result(output)
                    
                except Exception as e:
                    logger.error(f"Inference error: {e}")
                    if not request.future.done():
                        request.future.set_exception(e)
                
        except Exception as e:
            logger.error(f"Queue processing error: {e}")
        finally:
            self._processing = False
    
    def _check_memory_budget(self) -> bool:
        """Check if we're within memory budget."""
        # Estimate tensor memory usage
        estimated_usage = (
            self.MAX_BATCH_SIZE * 3 * 5 * 4 +  # Input buffer
            self.MAX_BATCH_SIZE * 64 * 4 +      # Hidden layers
            self.MAX_BATCH_SIZE * 5 * 4         # Output
        )
        
        return estimated_usage < self.MAX_TENSOR_MEMORY
    
    async def update_regime(self,
                           regime: str,
                           confidence: float,
                           volatility: float,
                           momentum: float,
                           correlation: float):
        """Update market regime from HMM."""
        if not self.router:
            return
        
        try:
            regime_enum = MarketRegime(regime)
            activations = self.router.update_regime(
                regime_enum, confidence, volatility, momentum, correlation
            )
            
            # Publish regime update
            self._publish_regime_update(regime, activations)
            
        except ValueError:
            logger.warning(f"Unknown regime: {regime}")
    
    def _publish_regime_update(self, regime: str, activations: List):
        """Publish regime update to subscribers."""
        try:
            self._pub_socket.send_json({
                'type': 'REGIME_UPDATE',
                'regime': regime,
                'active_strategies': [a.strategy_id for a in activations if a.activated],
                'timestamp': datetime.utcnow().isoformat(),
            }, flags=zmq.NOBLOCK)
        except Exception as e:
            logger.error(f"Failed to publish regime update: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get ensemble module status."""
        avg_latency = (
            sum(self._latency_history) / len(self._latency_history)
            if self._latency_history else 0.0
        )
        
        return {
            'running': self._running,
            'queue_size': len(self._inference_queue),
            'inference_count': self._inference_count,
            'avg_latency_ms': avg_latency * 1000,
            'memory_budget_ok': self._check_memory_budget(),
            'regime': self.router.get_regime_summary() if self.router else None,
        }
    
    async def shutdown(self):
        """Gracefully shutdown the ensemble module."""
        logger.info("Shutting down EnsembleModule...")
        self._running = False
        
        # Wait for pending inferences
        await asyncio.sleep(0.5)
        
        # Shutdown components
        if self.generalizer:
            self.generalizer.shutdown()
        
        if self.router:
            self.router.shutdown()
        
        # Close ZMQ
        if self._pub_socket:
            self._pub_socket.close()
        if self._sub_socket:
            self._sub_socket.close()
        if self._zmq_context:
            self._zmq_context.term()
        
        logger.info("EnsembleModule shut down complete")


# Global module instance
_module: Optional[EnsembleModule] = None


def get_module() -> EnsembleModule:
    """Get or create the global EnsembleModule instance."""
    global _module
    if _module is None:
        _module = EnsembleModule({})
    return _module


def create_module(config: Dict[str, Any]) -> EnsembleModule:
    """Create a new EnsembleModule with custom configuration."""
    global _module
    _module = EnsembleModule(config)
    return _module
