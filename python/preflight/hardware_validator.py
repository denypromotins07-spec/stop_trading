"""
Pre-Flight System Checks & Hardware Validation
Stage 49: Queries host OS via psutil to verify CPU, RAM, and shared memory limits.
Refuses to start if 6.5GB total system RAM constraint or hugepages requirements are violated.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import zmq

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


@dataclass
class HardwareSpec:
    """Hardware specification requirements."""
    min_cpu_cores: int = 6
    min_ram_gb: float = 4.0
    max_total_ram_gb: float = 6.5
    min_shm_mb: int = 512
    require_hugepages: bool = True
    hugepages_min_mb: int = 256


@dataclass
class ValidationResult:
    """Result of a hardware validation check."""
    check_name: str
    passed: bool
    actual_value: Any
    expected_value: Any
    message: str


class HardwareValidator:
    """
    Queries host OS to verify hardware meets strict HFT requirements.
    Enforces 6.5GB total system RAM constraint and hugepages requirements.
    """
    
    def __init__(self, spec: Optional[HardwareSpec] = None):
        self.spec = spec or HardwareSpec()
        self._results: list = []
        
        # ZMQ socket for Rust IPC
        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.PUSH)
        self._zmq_socket.connect("tcp://localhost:5570")
    
    def validate_all(self) -> Tuple[bool, list]:
        """
        Run all hardware validation checks.
        
        Returns:
            Tuple of (all_passed, list of ValidationResult)
        """
        self._results = []
        
        # Check 1: CPU cores
        cpu_result = self._check_cpu_cores()
        self._results.append(cpu_result)
        
        # Check 2: Available RAM
        ram_result = self._check_ram()
        self._results.append(ram_result)
        
        # Check 3: Total system RAM constraint
        total_ram_result = self._check_total_ram_constraint()
        self._results.append(total_ram_result)
        
        # Check 4: Shared memory (/dev/shm)
        shm_result = self._check_shared_memory()
        self._results.append(shm_result)
        
        # Check 5: Hugepages
        hp_result = self._check_hugepages()
        self._results.append(hp_result)
        
        # Check 6: Disk space
        disk_result = self._check_disk_space()
        self._results.append(disk_result)
        
        all_passed = all(r.passed for r in self._results)
        
        # Log results
        self._log_results()
        
        # Notify Rust
        self._notify_rust(all_passed, self._results)
        
        return all_passed, self._results
    
    def _check_cpu_cores(self) -> ValidationResult:
        """Check CPU core count."""
        if psutil is None:
            return ValidationResult(
                check_name="cpu_cores",
                passed=False,
                actual_value="psutil not installed",
                expected_value=f">= {self.spec.min_cpu_cores} cores",
                message="psutil library not available",
            )
        
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        
        passed = physical_cores >= self.spec.min_cpu_cores
        
        return ValidationResult(
            check_name="cpu_cores",
            passed=passed,
            actual_value=f"{physical_cores} physical, {logical_cores} logical",
            expected_value=f">= {self.spec.min_cpu_cores} physical cores",
            message=f"CPU: {physical_cores} physical cores detected",
        )
    
    def _check_ram(self) -> ValidationResult:
        """Check available RAM."""
        if psutil is None:
            return ValidationResult(
                check_name="available_ram",
                passed=False,
                actual_value="psutil not installed",
                expected_value=f">= {self.spec.min_ram_gb} GB",
                message="psutil library not available",
            )
        
        virtual_mem = psutil.virtual_memory()
        available_gb = virtual_mem.available / (1024 ** 3)
        total_gb = virtual_mem.total / (1024 ** 3)
        
        passed = available_gb >= self.spec.min_ram_gb
        
        return ValidationResult(
            check_name="available_ram",
            passed=passed,
            actual_value=f"{available_gb:.2f} GB available, {total_gb:.2f} GB total",
            expected_value=f">= {self.spec.min_ram_gb} GB available",
            message=f"RAM: {available_gb:.2f} GB available of {total_gb:.2f} GB total",
        )
    
    def _check_total_ram_constraint(self) -> ValidationResult:
        """Check that total system RAM doesn't exceed 6.5GB constraint."""
        if psutil is None:
            return ValidationResult(
                check_name="total_ram_constraint",
                passed=False,
                actual_value="psutil not installed",
                expected_value=f"<= {self.spec.max_total_ram_gb} GB",
                message="psutil library not available",
            )
        
        virtual_mem = psutil.virtual_memory()
        total_gb = virtual_mem.total / (1024 ** 3)
        
        # This is a HARD constraint - fail if exceeded
        passed = total_gb <= self.spec.max_total_ram_gb
        
        return ValidationResult(
            check_name="total_ram_constraint",
            passed=passed,
            actual_value=f"{total_gb:.2f} GB total system RAM",
            expected_value=f"<= {self.spec.max_total_ram_gb} GB (HARD CONSTRAINT)",
            message=f"{'PASS' if passed else 'FAIL'}: Total RAM {total_gb:.2f} GB",
        )
    
    def _check_shared_memory(self) -> ValidationResult:
        """Check /dev/shm shared memory availability."""
        try:
            # Try to get /dev/shm stats
            shm_path = "/dev/shm"
            if os.path.exists(shm_path):
                stat = os.statvfs(shm_path)
                shm_available_mb = (stat.f_bavail * stat.f_frsize) / (1024 ** 2)
                shm_total_mb = (stat.f_blocks * stat.f_frsize) / (1024 ** 2)
            else:
                shm_available_mb = 0
                shm_total_mb = 0
            
            passed = shm_available_mb >= self.spec.min_shm_mb
            
            return ValidationResult(
                check_name="shared_memory",
                passed=passed,
                actual_value=f"{shm_available_mb:.0f} MB available",
                expected_value=f">= {self.spec.min_shm_mb} MB",
                message=f"/dev/shm: {shm_available_mb:.0f} MB available",
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="shared_memory",
                passed=False,
                actual_value=str(e),
                expected_value=f">= {self.spec.min_shm_mb} MB",
                message=f"Failed to check /dev/shm: {e}",
            )
    
    def _check_hugepages(self) -> ValidationResult:
        """Check hugepages configuration."""
        if not self.spec.require_hugepages:
            return ValidationResult(
                check_name="hugepages",
                passed=True,
                actual_value="not required",
                expected_value="N/A",
                message="Hugepages not required by configuration",
            )
        
        try:
            # Read hugepages info from /proc/meminfo
            hugepages_total = 0
            hugepages_free = 0
            
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("HugePages_Total:"):
                        hugepages_total = int(line.split()[1])
                    elif line.startswith("HugePages_Free:"):
                        hugepages_free = int(line.split()[1])
            
            # Convert to MB (assuming 2MB hugepages)
            hugepages_total_mb = hugepages_total * 2
            hugepages_free_mb = hugepages_free * 2
            
            passed = hugepages_total_mb >= self.spec.hugepages_min_mb
            
            return ValidationResult(
                check_name="hugepages",
                passed=passed,
                actual_value=f"{hugepages_total_mb} MB total, {hugepages_free_mb} MB free",
                expected_value=f">= {self.spec.hugepages_min_mb} MB",
                message=f"Hugepages: {hugepages_total_mb} MB configured",
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="hugepages",
                passed=False,
                actual_value=str(e),
                expected_value=f">= {self.spec.hugepages_min_mb} MB",
                message=f"Failed to check hugepages: {e}",
            )
    
    def _check_disk_space(self) -> ValidationResult:
        """Check available disk space."""
        try:
            if psutil is None:
                disk_usage = os.statvfs("/")
                free_gb = (disk_usage.f_bavail * disk_usage.f_frsize) / (1024 ** 3)
            else:
                disk_usage = psutil.disk_usage("/")
                free_gb = disk_usage.free / (1024 ** 3)
            
            passed = free_gb >= 5.0  # At least 5GB free
            
            return ValidationResult(
                check_name="disk_space",
                passed=passed,
                actual_value=f"{free_gb:.2f} GB free",
                expected_value=">= 5.0 GB",
                message=f"Disk: {free_gb:.2f} GB free on root partition",
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="disk_space",
                passed=False,
                actual_value=str(e),
                expected_value=">= 5.0 GB",
                message=f"Failed to check disk space: {e}",
            )
    
    def _log_results(self):
        """Log validation results."""
        logger.info("=" * 60)
        logger.info("HARDWARE VALIDATION RESULTS")
        logger.info("=" * 60)
        
        for result in self._results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            logger.info(f"{status}: {result.check_name}")
            logger.info(f"    Actual:   {result.actual_value}")
            logger.info(f"    Expected: {result.expected_value}")
            logger.info(f"    Message:  {result.message}")
        
        all_passed = all(r.passed for r in self._results)
        logger.info("=" * 60)
        logger.info(f"OVERALL: {'ALL CHECKS PASSED' if all_passed else 'VALIDATION FAILED'}")
        logger.info("=" * 60)
    
    def _notify_rust(self, all_passed: bool, results: list):
        """Send validation results to Rust via ZMQ."""
        try:
            self._zmq_socket.send_json({
                'type': 'HARDWARE_VALIDATION',
                'passed': all_passed,
                'results': [
                    {
                        'check': r.check_name,
                        'passed': r.passed,
                        'actual': str(r.actual_value),
                        'expected': str(r.expected_value),
                        'message': r.message,
                    }
                    for r in results
                ],
                'timestamp': __import__('datetime').datetime.utcnow().isoformat(),
            }, flags=zmq.NOBLOCK)
        except Exception as e:
            logger.error(f"Failed to notify Rust: {e}")
    
    def shutdown(self):
        """Cleanup resources."""
        self._zmq_socket.close()
        self._zmq_context.term()
        logger.info("HardwareValidator shut down")


# Global instance
_validator: Optional[HardwareValidator] = None


def get_validator() -> HardwareValidator:
    """Get or create the global HardwareValidator instance."""
    global _validator
    if _validator is None:
        _validator = HardwareValidator()
    return _validator


def create_validator(spec: Optional[HardwareSpec] = None) -> HardwareValidator:
    """Create a new HardwareValidator with custom specification."""
    global _validator
    _validator = HardwareValidator(spec=spec)
    return _validator


def validate_hardware() -> Tuple[bool, list]:
    """Convenience function to run all hardware validations."""
    validator = get_validator()
    return validator.validate_all()
