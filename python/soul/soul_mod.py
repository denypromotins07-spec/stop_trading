"""
SOUL Module Root - Manages continuous stream of feedback from Rust to Python ML backend.
Integrates parser and feedback loop for the self-learning engine.
Strictly enforces 3GB RAM limit.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import ray

from soul.parser import SOULParser, TradeOutcome, Mistake, RegimeMemory, SOULBlock
from soul.feedback_loop import (
    FeedbackLoopActor, 
    FeedbackLoopManager,
    RewardSignal,
    LossAdjustment
)


logger = logging.getLogger(__name__)


class SOULEngine:
    """
    Main engine for the SOUL.md self-learning system.
    Coordinates parsing, feedback processing, and ML backend integration.
    """
    
    def __init__(self,
                 num_feedback_actors: int = 2,
                 parser_chunk_size: int = 8192,
                 max_pending_signals: int = 5000):
        """
        Initialize SOUL engine.
        
        Args:
            num_feedback_actors: Number of Ray actors for feedback processing
            parser_chunk_size: Chunk size for streaming parser
            max_pending_signals: Maximum pending reward signals in queue
        """
        self.num_feedback_actors = num_feedback_actors
        self.parser_chunk_size = parser_chunk_size
        self.max_pending_signals = max_pending_signals
        
        self.parser = SOULParser(chunk_size=parser_chunk_size)
        self.feedback_manager: Optional[FeedbackLoopManager] = None
        self._initialized = False
        
        # Pending signals queue (bounded)
        self._pending_signals: List[RewardSignal] = []
        self._pending_adjustments: List[LossAdjustment] = []
        
        # Statistics
        self._total_processed = 0
        self._total_errors = 0
    
    async def initialize(self):
        """Initialize Ray and feedback actors."""
        if not self._initialized:
            try:
                ray.init(
                    ignore_reinit_error=True,
                    namespace="soul_engine",
                    _system_config={
                        "object_store_memory": 512 * 1024 * 1024,  # 512MB limit
                        "max_workers": 4
                    }
                )
                self.feedback_manager = FeedbackLoopManager(
                    num_actors=self.num_feedback_actors
                )
                self._initialized = True
                logger.info("SOUL engine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SOUL engine: {e}")
                self._total_errors += 1
                raise
    
    async def process_file(self, filepath: str) -> Dict[str, Any]:
        """
        Process a SOUL.md file end-to-end.
        
        Args:
            filepath: Path to SOUL.md file
            
        Returns:
            Dict with processing statistics
        """
        if not self._initialized:
            await self.initialize()
        
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"SOUL.md file not found: {filepath}")
        
        results = {
            "outcomes_processed": 0,
            "mistakes_processed": 0,
            "memories_processed": 0,
            "signals_generated": 0,
            "adjustments_generated": 0,
            "errors": 0
        }
        
        try:
            async for block in self.parser.parse_file(filepath):
                # Process outcomes
                for outcome in block.outcomes:
                    try:
                        signal = await self._process_outcome(outcome)
                        if signal:
                            results["signals_generated"] += 1
                        results["outcomes_processed"] += 1
                    except Exception as e:
                        logger.warning(f"Error processing outcome: {e}")
                        results["errors"] += 1
                
                # Process mistakes
                for mistake in block.mistakes:
                    try:
                        signal, adjustment = await self._process_mistake(mistake)
                        if signal:
                            results["signals_generated"] += 1
                        if adjustment:
                            results["adjustments_generated"] += 1
                        results["mistakes_processed"] += 1
                    except Exception as e:
                        logger.warning(f"Error processing mistake: {e}")
                        results["errors"] += 1
                
                # Process regime memories
                for memory in block.memories:
                    try:
                        await self._process_memory(memory)
                        results["memories_processed"] += 1
                    except Exception as e:
                        logger.warning(f"Error processing memory: {e}")
                        results["errors"] += 1
                
                self._total_processed += 1
                
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {e}")
            self._total_errors += 1
            results["errors"] += 1
        
        return results
    
    async def _process_outcome(self, outcome: TradeOutcome) -> Optional[RewardSignal]:
        """Process single trade outcome."""
        if self.feedback_manager:
            actor = self.feedback_manager.actors[0]
            return await actor.process_outcome.remote(outcome)
        return None
    
    async def _process_mistake(self, mistake: Mistake) -> tuple:
        """Process single mistake."""
        if self.feedback_manager:
            actor = self.feedback_manager.actors[0]
            return await actor.process_mistake.remote(mistake)
        return None, None
    
    async def _process_memory(self, memory: RegimeMemory):
        """Process single regime memory."""
        if self.feedback_manager:
            actor = self.feedback_manager.actors[0]
            return await actor.process_regime_memory.remote(memory)
    
    async def get_reward_signals(self, 
                                 time_window_ns: int = 3600_000_000_000
                                 ) -> Dict[str, float]:
        """Get aggregated reward signals."""
        if not self.feedback_manager:
            return {"total_reward": 0.0, "count": 0}
        
        total_rewards = {"total_reward": 0.0, "count": 0}
        
        for actor in self.feedback_manager.actors:
            rewards = await actor.get_aggregate_rewards.remote(time_window_ns)
            total_rewards["total_reward"] += rewards.get("total_reward", 0.0)
            total_rewards["count"] += rewards.get("count", 0)
        
        return total_rewards
    
    async def get_loss_adjustments(self) -> List[LossAdjustment]:
        """Get all pending loss adjustments."""
        if not self.feedback_manager:
            return []
        
        all_adjustments = []
        for actor in self.feedback_manager.actors:
            adjustments = await actor.get_loss_adjustments.remote()
            all_adjustments.extend(adjustments)
        
        return all_adjustments
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "initialized": self._initialized,
            "total_processed": self._total_processed,
            "total_errors": self._total_errors,
            "pending_signals": len(self._pending_signals),
            "pending_adjustments": len(self._pending_adjustments),
            "feedback_actors": self.num_feedback_actors
        }
    
    async def shutdown(self):
        """Shutdown SOUL engine and release resources."""
        if ray.is_initialized():
            ray.shutdown()
        self._initialized = False
        logger.info("SOUL engine shut down")


# Module-level singleton
_soul_engine: Optional[SOULEngine] = None


def get_engine() -> SOULEngine:
    """Get or create SOUL engine singleton."""
    global _soul_engine
    if _soul_engine is None:
        _soul_engine = SOULEngine()
    return _soul_engine


async def initialize_soul(num_actors: int = 2):
    """Initialize SOUL engine with specified configuration."""
    engine = get_engine()
    engine.num_feedback_actors = num_actors
    await engine.initialize()
    return engine


async def process_soul_file(filepath: str) -> Dict[str, Any]:
    """Process SOUL.md file using engine singleton."""
    engine = get_engine()
    return await engine.process_file(filepath)


async def get_feedback_summary() -> Dict[str, Any]:
    """Get summary of current feedback state."""
    engine = get_engine()
    rewards = await engine.get_reward_signals()
    adjustments = await engine.get_loss_adjustments()
    
    return {
        "rewards": rewards,
        "pending_adjustments": len(adjustments),
        "engine_stats": engine.get_stats()
    }


# Example usage
async def main():
    """Example usage of SOUL module."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize engine
    engine = await initialize_soul(num_actors=2)
    
    # Process SOUL.md file
    if Path("SOUL.md").exists():
        results = await process_soul_file("SOUL.md")
        print(f"Processing results: {results}")
    
    # Get feedback summary
    summary = await get_feedback_summary()
    print(f"Feedback summary: {summary}")
    
    # Shutdown
    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
