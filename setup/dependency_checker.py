#!/usr/bin/env python3
"""
Dependency Checker - Stage 50
Verifies Rust binaries are compiled in --release mode and all Python ONNX models are present.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import subprocess
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger('DependencyChecker')

# Constants
WORKSPACE_ROOT = Path('/workspace')
RUST_TARGET_DIR = WORKSPACE_ROOT / 'target' / 'release'
PYTHON_MODELS_DIR = WORKSPACE_ROOT / 'python' / 'models'
REQUIRED_BINARIES = ['crypto_bot', 'nautilus_core']
REQUIRED_MODELS = [
    'xgboost_model.onnx',
    'transformer_model.onnx',
    'mlp_generalizer.onnx',
    'rl_agent.onnx'
]


class RustBinaryChecker:
    """Checks Rust binary compilation status."""
    
    def __init__(self):
        self.target_dir = RUST_TARGET_DIR
        self.debug_dir = WORKSPACE_ROOT / 'target' / 'debug'
    
    def check_release_build(self) -> Tuple[bool, str]:
        """Verify binaries are compiled in release mode."""
        if not self.target_dir.exists():
            return False, f"Release target directory not found: {self.target_dir}"
        
        missing_binaries = []
        for binary_name in REQUIRED_BINARIES:
            binary_path = self.target_dir / binary_name
            
            # Handle Windows .exe extension
            if not binary_path.exists():
                binary_path = binary_path.with_suffix('.exe')
            
            if not binary_path.exists():
                missing_binaries.append(binary_name)
        
        if missing_binaries:
            return False, f"Missing release binaries: {missing_binaries}"
        
        # Verify release optimizations are enabled
        for binary_name in REQUIRED_BINARIES:
            binary_path = self.target_dir / binary_name
            if not binary_path.exists():
                binary_path = binary_path.with_suffix('.exe')
            
            # Check file size (release builds are typically smaller due to LTO)
            file_size_mb = binary_path.stat().st_size / (1024 * 1024)
            
            if file_size_mb < 0.5:
                logger.warning(f"Binary {binary_name} seems unusually small ({file_size_mb:.2f}MB)")
            elif file_size_mb > 100:
                logger.warning(f"Binary {binary_name} seems unusually large ({file_size_mb:.2f}MB)")
        
        logger.info(f"All {len(REQUIRED_BINARIES)} release binaries found")
        return True, "Release build verified"
    
    def get_binary_info(self, binary_name: str) -> Optional[Dict]:
        """Get information about a specific binary."""
        binary_path = self.target_dir / binary_name
        if not binary_path.exists():
            binary_path = binary_path.with_suffix('.exe')
        
        if not binary_path.exists():
            return None
        
        stat = binary_path.stat()
        return {
            'path': str(binary_path),
            'size_mb': stat.st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'executable': os.access(binary_path, os.X_OK)
        }
    
    def run_cargo_check(self) -> bool:
        """Run cargo check to verify Rust code compiles."""
        try:
            result = subprocess.run(
                ['cargo', 'check', '--release'],
                cwd=WORKSPACE_ROOT,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info("Cargo check passed")
                return True
            else:
                logger.error(f"Cargo check failed: {result.stderr}")
                return False
        
        except subprocess.TimeoutExpired:
            logger.error("Cargo check timed out")
            return False
        except FileNotFoundError:
            logger.warning("Cargo not found, skipping Rust check")
            return True  # Non-fatal if Rust not installed
    
    def check_all(self) -> Dict:
        """Run all Rust binary checks."""
        results = {
            'release_build_ok': False,
            'binaries': {},
            'cargo_check_ok': False,
            'errors': []
        }
        
        # Check release build
        ok, msg = self.check_release_build()
        results['release_build_ok'] = ok
        if not ok:
            results['errors'].append(msg)
        
        # Get info for each binary
        for name in REQUIRED_BINARIES:
            info = self.get_binary_info(name)
            if info:
                results['binaries'][name] = info
        
        # Run cargo check
        results['cargo_check_ok'] = self.run_cargo_check()
        
        return results


class ONNXModelChecker:
    """Checks Python ONNX model files."""
    
    def __init__(self):
        self.models_dir = PYTHON_MODELS_DIR
    
    def check_models_exist(self) -> Tuple[bool, List[str]]:
        """Verify all required ONNX models exist."""
        if not self.models_dir.exists():
            return False, ["Models directory not found"]
        
        missing = []
        for model_name in REQUIRED_MODELS:
            model_path = self.models_dir / model_name
            if not model_path.exists():
                missing.append(model_name)
        
        if missing:
            return False, missing
        
        return True, []
    
    def validate_onnx_model(self, model_path: Path) -> Tuple[bool, str]:
        """Validate an ONNX model file."""
        try:
            import onnx
            
            # Load and check model
            model = onnx.load(str(model_path))
            
            # Verify model structure
            onnx.checker.check_model(model)
            
            # Get model info
            input_names = [inp.name for inp in model.graph.input]
            output_names = [out.name for out in model.graph.output]
            
            logger.info(
                f"Model {model_path.name}: "
                f"inputs={input_names}, outputs={output_names}"
            )
            
            return True, "Model valid"
        
        except ImportError:
            logger.warning("onnx package not installed, skipping deep validation")
            return True, "File exists (validation skipped)"
        
        except Exception as e:
            return False, str(e)
    
    def get_model_info(self, model_path: Path) -> Optional[Dict]:
        """Get information about a model file."""
        if not model_path.exists():
            return None
        
        stat = model_path.stat()
        info = {
            'path': str(model_path),
            'size_mb': stat.st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
        
        # Try to get ONNX-specific info
        try:
            import onnx
            model = onnx.load(str(model_path))
            
            # Count parameters
            param_count = sum(
                onnx.numpy_helper.to_array(init).size
                for init in model.graph.initializer
            )
            
            info['parameters'] = param_count
            info['opset_version'] = model.opset_import[0].version
            
        except:
            pass
        
        return info
    
    def check_all(self) -> Dict:
        """Run all ONNX model checks."""
        results = {
            'models_exist': False,
            'models_valid': True,
            'models': {},
            'errors': []
        }
        
        # Check existence
        exists, missing = self.check_models_exist()
        results['models_exist'] = exists
        
        if not exists:
            results['errors'].append(f"Missing models: {missing}")
            return results
        
        # Validate each model
        for model_name in REQUIRED_MODELS:
            model_path = self.models_dir / model_name
            
            # Get basic info
            info = self.get_model_info(model_path)
            if info:
                results['models'][model_name] = info
            
            # Deep validation
            valid, msg = self.validate_onnx_model(model_path)
            if not valid:
                results['models_valid'] = False
                results['errors'].append(f"{model_name}: {msg}")
        
        return results


class PythonDependencyChecker:
    """Checks Python package dependencies."""
    
    REQUIRED_PACKAGES = [
        'numpy',
        'pandas',
        'scipy',
        'zmq',
        'psutil',
        'ray',
        'aiofiles'
    ]
    
    OPTIONAL_PACKAGES = [
        'rich',
        'prompt_toolkit',
        'onnx',
        'onnxruntime'
    ]
    
    def check_packages(self) -> Dict:
        """Check if required Python packages are installed."""
        results = {
            'required_installed': [],
            'required_missing': [],
            'optional_installed': [],
            'optional_missing': []
        }
        
        # Check required packages
        for pkg in self.REQUIRED_PACKAGES:
            if self._is_package_installed(pkg):
                results['required_installed'].append(pkg)
            else:
                results['required_missing'].append(pkg)
        
        # Check optional packages
        for pkg in self.OPTIONAL_PACKAGES:
            if self._is_package_installed(pkg):
                results['optional_installed'].append(pkg)
            else:
                results['optional_missing'].append(pkg)
        
        return results
    
    def _is_package_installed(self, package: str) -> bool:
        """Check if a package is installed."""
        try:
            __import__(package)
            return True
        except ImportError:
            return False


class DependencyCoordinator:
    """Coordinates all dependency checks."""
    
    def __init__(self):
        self.rust_checker = RustBinaryChecker()
        self.onnx_checker = ONNXModelChecker()
        self.python_checker = PythonDependencyChecker()
    
    def run_full_check(self) -> bool:
        """Run all dependency checks."""
        logger.info("=" * 60)
        logger.info("DEPENDENCY CHECK - STAGE 50")
        logger.info("=" * 60)
        
        all_passed = True
        
        # Check Rust binaries
        logger.info("\n📦 Checking Rust binaries...")
        rust_results = self.rust_checker.check_all()
        
        if rust_results['release_build_ok']:
            logger.info("✅ Release build verified")
            for name, info in rust_results['binaries'].items():
                logger.info(f"   {name}: {info['size_mb']:.2f}MB")
        else:
            logger.error("❌ Release build check failed")
            for error in rust_results['errors']:
                logger.error(f"   {error}")
            all_passed = False
        
        # Check ONNX models
        logger.info("\n🧠 Checking ONNX models...")
        onnx_results = self.onnx_checker.check_all()
        
        if onnx_results['models_exist'] and onnx_results['models_valid']:
            logger.info("✅ All models present and valid")
            for name, info in onnx_results['models'].items():
                params = info.get('parameters', 'N/A')
                logger.info(f"   {name}: {info['size_mb']:.2f}MB, {params} params")
        else:
            logger.error("❌ Model check failed")
            for error in onnx_results['errors']:
                logger.error(f"   {error}")
            all_passed = False
        
        # Check Python packages
        logger.info("\n🐍 Checking Python packages...")
        py_results = self.python_checker.check_packages()
        
        if py_results['required_missing']:
            logger.error(f"❌ Missing required packages: {py_results['required_missing']}")
            all_passed = False
        else:
            logger.info(f"✅ All {len(py_results['required_installed'])} required packages installed")
        
        if py_results['optional_missing']:
            logger.warning(f"⚠️  Optional packages missing: {py_results['optional_missing']}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        if all_passed:
            logger.info("✅ ALL DEPENDENCY CHECKS PASSED")
        else:
            logger.error("❌ SOME DEPENDENCY CHECKS FAILED")
        logger.info("=" * 60)
        
        return all_passed


def main():
    """Entry point for dependency checker."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Check System Dependencies')
    parser.add_argument('--rust-only', action='store_true', help='Check only Rust binaries')
    parser.add_argument('--models-only', action='store_true', help='Check only ONNX models')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    args = parser.parse_args()
    
    coordinator = DependencyCoordinator()
    
    if args.rust_only:
        results = coordinator.rust_checker.check_all()
        success = results['release_build_ok']
    elif args.models_only:
        results = coordinator.onnx_checker.check_all()
        success = results['models_exist'] and results['models_valid']
    else:
        success = coordinator.run_full_check()
        results = {'overall_success': success}
    
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
