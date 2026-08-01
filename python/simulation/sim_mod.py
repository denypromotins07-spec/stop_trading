"""
Simulation Module Root.
Manages simulation vectorization and Ray rollout workers for distributed PPO training.
"""

import asyncio
from typing import Optional, Dict, Any, List
import logging
import numpy as np

try:
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None
    tune = None

from .portfolio_env import PortfolioEnv, register_portfolio_env
from .adversarial_env import AdversarialEnv, register_adversarial_env

logger = logging.getLogger(__name__)


class SimulationModule:
    """
    Central manager for simulation subsystem.
    Handles Ray cluster management and distributed RL training.
    """

    def __init__(
        self,
        ray_address: Optional[str] = None,
        num_workers: int = 4,
        training_iterations: int = 100,
    ):
        self.ray_address = ray_address
        self.num_workers = num_workers
        self.training_iterations = training_iterations

        self._ray_initialized = False
        self._training_running = False
        self._results: Dict[str, Any] = {}

    def initialize_ray(self) -> bool:
        """Initialize Ray cluster."""
        if not RAY_AVAILABLE:
            logger.warning("Ray not available, running in local mode")
            return False

        try:
            if not ray.is_initialized():
                if self.ray_address:
                    ray.init(address=self.ray_address)
                else:
                    ray.init(
                        num_cpus=self.num_workers + 1,
                        include_dashboard=False,
                    )
                self._ray_initialized = True
                logger.info(f"Ray initialized with {self.num_workers} workers")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            return False

    def shutdown_ray(self):
        """Shutdown Ray cluster."""
        if self._ray_initialized and ray:
            ray.shutdown()
            self._ray_initialized = False
            logger.info("Ray shutdown complete")

    def train_ppo(
        self,
        env_name: str = "PortfolioEnv-v0",
        config_overrides: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Train PPO agent on specified environment.

        Args:
            env_name: Environment name
            config_overrides: Optional config overrides

        Returns:
            Training results
        """
        if not RAY_AVAILABLE:
            logger.warning("Ray not available, skipping PPO training")
            return {"status": "ray_unavailable"}

        if not self._ray_initialized:
            self.initialize_ray()

        # Register environments
        register_portfolio_env()
        register_adversarial_env()

        # Default PPO config
        config = {
            "env": env_name,
            "num_workers": self.num_workers,
            "train_batch_size": 4000,
            "gamma": 0.99,
            "lr": 3e-4,
            "clip_param": 0.2,
            "grad_clip": 10.0,
            "kl_target": 0.01,
            "model": {
                "fcnet_hiddens": [256, 256, 128],
                "fcnet_activation": "relu",
            },
        }

        if config_overrides:
            config.update(config_overrides)

        try:
            # Build PPO config
            ppo_config = (
                PPOConfig()
                .environment(env=env_name)
                .rollouts(num_rollout_workers=self.num_workers)
                .training(**config)
            )

            # Create algorithm
            algo = ppo_config.build()

            # Train
            results = []
            for i in range(self.training_iterations):
                result = algo.train()
                results.append(result)

                if i % 10 == 0:
                    logger.info(f"Iteration {i}: reward={result.get('episode_reward_mean', 0):.2f}")

            self._training_running = False
            self._results = {
                "status": "complete",
                "final_reward": results[-1].get("episode_reward_mean", 0) if results else 0,
                "iterations": len(results),
            }

            # Cleanup
            algo.stop()

            return self._results

        except Exception as e:
            logger.error(f"PPO training failed: {e}")
            self._training_running = False
            return {"status": "error", "error": str(e)}

    async def train_async(
        self,
        env_name: str = "PortfolioEnv-v0",
        config_overrides: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Async wrapper for PPO training."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.train_ppo(env_name, config_overrides),
        )

    def run_stress_test(
        self,
        scenario: str = "flash_crash",
        n_episodes: int = 100,
    ) -> Dict[str, Any]:
        """
        Run stress test with adversarial scenarios.

        Args:
            scenario: Adversarial scenario name
            n_episodes: Number of test episodes

        Returns:
            Stress test results
        """
        register_adversarial_env()

        env = AdversarialEnv(
            adversarial_intensity=0.8,
        )

        results = {
            "scenario": scenario,
            "episodes": n_episodes,
            "circuit_breaker_triggers": 0,
            "avg_drawdown": 0.0,
            "max_drawdown": 0.0,
            "survival_rate": 0.0,
        }

        drawdowns = []
        survived = 0

        for _ in range(n_episodes):
            obs, info = env.reset(options={"scenario": scenario})
            done = False
            step = 0

            while not done and step < 500:
                action = env.action_space.sample()  # Random actions
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1

            if info.get("circuit_breaker_triggered"):
                results["circuit_breaker_triggers"] += 1
            else:
                survived += 1

            drawdowns.append(info.get("max_drawdown", 0))

        results["avg_drawdown"] = np.mean(drawdowns)
        results["max_drawdown"] = np.min(drawdowns)
        results["survival_rate"] = survived / n_episodes

        logger.info(f"Stress test complete: survival_rate={results['survival_rate']:.2%}")
        return results

    def vectorized_simulation(
        self,
        env_class,
        n_envs: int = 16,
        steps: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Run vectorized simulation across multiple environments.

        Args:
            env_class: Environment class
            n_envs: Number of parallel environments
            steps: Steps per environment

        Returns:
            List of results from each environment
        """
        envs = [env_class() for _ in range(n_envs)]
        results = []

        for env in envs:
            obs, info = env.reset()
            total_reward = 0.0

            for _ in range(steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    obs, info = env.reset()

            results.append({
                "total_reward": total_reward,
                "final_info": info,
            })

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get simulation module statistics."""
        stats = {
            "ray_initialized": self._ray_initialized,
            "ray_available": RAY_AVAILABLE,
            "num_workers": self.num_workers,
            "training_running": self._training_running,
        }

        if ray and self._ray_initialized:
            stats["ray_nodes"] = len(ray.nodes())
            stats["ray_resources"] = ray.available_resources()

        return stats


# Module singleton
_module: Optional[SimulationModule] = None


def get_simulation_module(
    ray_address: Optional[str] = None,
    num_workers: int = 4,
) -> SimulationModule:
    """Get or create the simulation module singleton."""
    global _module
    if _module is None:
        _module = SimulationModule(
            ray_address=ray_address,
            num_workers=num_workers,
        )
    return _module


async def initialize_simulation(
    ray_address: Optional[str] = None,
    num_workers: int = 4,
) -> SimulationModule:
    """Initialize the simulation module."""
    module = get_simulation_module(ray_address, num_workers)
    module.initialize_ray()
    return module


async def shutdown_simulation_module():
    """Gracefully shutdown the simulation module."""
    global _module
    if _module:
        _module.shutdown_ray()
        _module = None
