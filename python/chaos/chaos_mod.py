"""
Module root managing chaos routines.
Strictly disabled in live production via environment variables and compile-time flags.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger(__name__)


# Environment variable to disable chaos in production
CHAOS_ENABLED_ENV = "HFT_ENABLE_CHAOS"
PRODUCTION_ENV = "HFT_PRODUCTION_MODE"


@dataclass
class ChaosModuleConfig:
    """Configuration for chaos module."""
    # Enable/disable flags
    enabled: bool = True
    allow_production: bool = False
    
    # Component-specific settings
    enable_toxic_injection: bool = True
    enable_worker_kills: bool = True
    enable_latency_injection: bool = True
    
    # Rates
    toxic_injection_rate: float = 0.01
    worker_kill_probability: float = 0.005
    latency_spike_probability: float = 0.02
    
    # Safety limits
    max_concurrent_chaos: int = 3
    cooldown_seconds: float = 10.0


class ChaosOrchestrator:
    """
    Main orchestrator for all chaos engineering activities.
    Ensures safe operation with multiple kill switches.
    """
    
    def __init__(self, config: Optional[ChaosModuleConfig] = None):
        self.config = config or ChaosModuleConfig()
        
        # Check if chaos should be disabled
        self._production_mode = os.getenv(PRODUCTION_ENV, "false").lower() == "true"
        self._explicitly_enabled = os.getenv(CHAOS_ENABLED_ENV, "false").lower() == "true"
        
        # Final enabled state
        self.enabled = (
            self.config.enabled and 
            self._explicitly_enabled and 
            (not self._production_mode or self.config.allow_production)
        )
        
        # Active chaos routines
        self._active_routines: Dict[str, Any] = {}
        self._last_chaos_time: float = 0.0
        self._total_events = 0
        
        # Child components (lazy loaded)
        self._toxic_injector = None
        self._worker_killer = None
        
        if self.enabled:
            logger.warning("Chaos engineering ENABLED - system stability not guaranteed")
        else:
            logger.info("Chaos engineering disabled (production mode or not explicitly enabled)")
    
    def _check_enabled(self) -> bool:
        """Check if chaos is currently enabled."""
        if not self.enabled:
            return False
        
        # Check cooldown
        if time.time() - self._last_chaos_time < self.config.cooldown_seconds:
            return False
        
        # Check concurrent limit
        if len(self._active_routines) >= self.config.max_concurrent_chaos:
            return False
        
        return True
    
    def run_toxic_injection_test(
        self,
        data: Any,
        target_component: str = "inference"
    ) -> Dict[str, Any]:
        """Run toxic data injection test."""
        if not self._check_enabled() or not self.config.enable_toxic_injection:
            return {'status': 'skipped', 'reason': 'disabled'}
        
        try:
            from .toxic_ipc_injector import ToxicIPCMInjector, ToxicityConfig
            
            if self._toxic_injector is None:
                tox_config = ToxicityConfig(
                    nan_probability=self.config.toxic_injection_rate,
                    global_toxicity_rate=self.config.toxic_injection_rate
                )
                self._toxic_injector = ToxicIPCMInjector(tox_config)
            
            # Inject toxicity
            import numpy as np
            if isinstance(data, np.ndarray):
                modified_data, result = self._toxic_injector.inject_random_toxicity(data)
                
                self._last_chaos_time = time.time()
                self._total_events += 1
                
                return {
                    'status': 'executed',
                    'type': 'toxic_injection',
                    'target': target_component,
                    'result': {
                        'toxicity_type': result.toxicity_type,
                        'num_toxic_elements': result.num_toxic_elements
                    }
                }
            
            return {'status': 'skipped', 'reason': 'unsupported data type'}
            
        except Exception as e:
            logger.error(f"Toxic injection test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_worker_kill_test(
        self,
        worker_id: Optional[str] = None,
        pid: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run worker termination test."""
        if not self._check_enabled() or not self.config.enable_worker_kills:
            return {'status': 'skipped', 'reason': 'disabled'}
        
        try:
            from .ray_worker_killer import RayWorkerKiller, ChaosConfig
            
            if self._worker_killer is None:
                killer_config = ChaosConfig(
                    kill_probability=self.config.worker_kill_probability
                )
                self._worker_killer = RayWorkerKiller(killer_config)
            
            if worker_id and pid:
                event = self._worker_killer.kill_worker(worker_id, pid)
            else:
                event = self._worker_killer.kill_random_worker()
            
            if event:
                self._last_chaos_time = time.time()
                self._total_events += 1
                
                return {
                    'status': 'executed',
                    'type': 'worker_kill',
                    'result': {
                        'worker_id': event.worker_id,
                        'kill_method': event.kill_method
                    }
                }
            
            return {'status': 'skipped', 'reason': 'no workers available'}
            
        except Exception as e:
            logger.error(f"Worker kill test failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all chaos tests sequentially."""
        results = {
            'timestamp': time.time(),
            'tests': []
        }
        
        # Run toxic injection
        tox_result = self.run_toxic_injection_test(
            __import__('numpy').random.randn(100, 10),
            "test_component"
        )
        results['tests'].append(('toxic_injection', tox_result))
        
        # Note: Worker kill test requires actual workers to track
        
        results['summary'] = {
            'total_tests': len(results['tests']),
            'executed': sum(1 for _, r in results['tests'] if r.get('status') == 'executed'),
            'skipped': sum(1 for _, r in results['tests'] if r.get('status') == 'skipped'),
            'failed': sum(1 for _, r in results['tests'] if r.get('status') == 'failed')
        }
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get chaos module status."""
        return {
            'enabled': self.enabled,
            'production_mode': self._production_mode,
            'explicitly_enabled': self._explicitly_enabled,
            'active_routines': len(self._active_routines),
            'total_events': self._total_events,
            'last_event_time': self._last_chaos_time,
            'config': {
                'toxic_injection': self.config.enable_toxic_injection,
                'worker_kills': self.config.enable_worker_kills,
                'allow_production': self.config.allow_production
            }
        }
    
    def emergency_disable(self):
        """Emergency disable all chaos routines."""
        self.enabled = False
        self._active_routines.clear()
        logger.critical("Chaos module EMERGENCY DISABLED")
    
    def reset(self):
        """Reset module state."""
        self._active_routines.clear()
        self._last_chaos_time = 0.0
        if self._toxic_injector:
            self._toxic_injector = None
        if self._worker_killer:
            self._worker_killer = None


# Module singleton
_module_instance: Optional[ChaosOrchestrator] = None


def get_module() -> ChaosOrchestrator:
    """Get or create module singleton."""
    global _module_instance
    if _module_instance is None:
        _module_instance = ChaosOrchestrator()
    return _module_instance


def initialize_module(
    enabled: bool = True,
    allow_production: bool = False,
    **kwargs
) -> ChaosOrchestrator:
    """Initialize the chaos module."""
    global _module_instance
    
    config = ChaosModuleConfig(
        enabled=enabled,
        allow_production=allow_production,
        **kwargs
    )
    
    _module_instance = ChaosOrchestrator(config)
    return _module_instance


def is_safe_to_run() -> bool:
    """Check if it's safe to run chaos tests."""
    module = get_module()
    return module.enabled and not module._production_mode
