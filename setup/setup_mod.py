#!/usr/bin/env python3
"""
Setup Module Root - Stage 50
Manages one-time setup routines, SOUL.md initialization, and directory structure validation.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger('SetupMod')

# Constants
WORKSPACE_ROOT = Path('/workspace')
LOGS_DIR = WORKSPACE_ROOT / 'logs'
DATA_DIR = WORKSPACE_ROOT / 'data'
MODELS_DIR = WORKSPACE_ROOT / 'python' / 'models'
ENCRYPTED_KEYS_DIR = WORKSPACE_ROOT / 'encrypted_keys'
SOUL_MD_PATH = WORKSPACE_ROOT / 'SOUL.md'
SETUP_COMPLETE_MARKER = WORKSPACE_ROOT / '.setup_complete'


class DirectoryStructureValidator:
    """Validates and creates required directory structure."""
    
    REQUIRED_DIRS = [
        LOGS_DIR,
        DATA_DIR,
        MODELS_DIR,
        ENCRYPTED_KEYS_DIR,
        WORKSPACE_ROOT / 'python' / 'orchestration',
        WORKSPACE_ROOT / 'python' / 'ensemble',
        WORKSPACE_ROOT / 'python' / 'safety',
        WORKSPACE_ROOT / 'python' / 'audit',
        WORKSPACE_ROOT / 'python' / 'preflight',
        WORKSPACE_ROOT / 'launch',
        WORKSPACE_ROOT / 'cli',
        WORKSPACE_ROOT / 'telemetry',
        WORKSPACE_ROOT / 'daemon',
    ]
    
    def __init__(self):
        self.created_dirs: List[Path] = []
        self.existing_dirs: List[Path] = []
    
    def validate_and_create(self) -> bool:
        """Validate existing dirs and create missing ones."""
        logger.info("Validating directory structure...")
        
        for dir_path in self.REQUIRED_DIRS:
            if dir_path.exists():
                self.existing_dirs.append(dir_path)
                logger.debug(f"✓ Existing: {dir_path}")
            else:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.created_dirs.append(dir_path)
                    logger.info(f"✓ Created: {dir_path}")
                except Exception as e:
                    logger.error(f"✗ Failed to create {dir_path}: {e}")
                    return False
        
        logger.info(
            f"Directory validation complete: "
            f"{len(self.existing_dirs)} existing, {len(self.created_dirs)} created"
        )
        return True
    
    def set_permissions(self):
        """Set appropriate permissions on directories."""
        # Logs should be readable only by owner
        for log_file in LOGS_DIR.glob('*.log'):
            try:
                os.chmod(log_file, 0o640)
            except:
                pass
        
        # Encrypted keys should be highly restricted
        if ENCRYPTED_KEYS_DIR.exists():
            try:
                os.chmod(ENCRYPTED_KEYS_DIR, 0o700)
                for key_file in ENCRYPTED_KEYS_DIR.glob('*'):
                    os.chmod(key_file, 0o600)
            except Exception as e:
                logger.warning(f"Could not set key file permissions: {e}")


class SOULMdInitializer:
    """Initializes and manages the SOUL.md system journal."""
    
    def __init__(self):
        self.soul_path = SOUL_MD_PATH
        self.entries: List[Dict] = []
    
    def initialize(self) -> bool:
        """Initialize SOUL.md if it doesn't exist."""
        if self.soul_path.exists():
            logger.info(f"SOUL.md already exists: {self.soul_path}")
            self._load_existing()
            return True
        
        try:
            initial_content = self._generate_initial_content()
            self.soul_path.write_text(initial_content)
            logger.info(f"Initialized SOUL.md: {self.soul_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize SOUL.md: {e}")
            return False
    
    def _generate_initial_content(self) -> str:
        """Generate initial SOUL.md content."""
        now = datetime.now().isoformat()
        
        content = f"""# SOUL.md - System Operations & Unification Log

## System Identity
- **Bot Name**: CRYPTO_MEDIUM_FREQUENCY_TRADING_BOT
- **Stage**: 50 (Ultimate Handoff)
- **Initialized**: {now}
- **Version**: 1.0.0

## Architecture Overview

### Core Components
1. **Rust Ultra-Low-Latency Core** - Market data ingestion, order execution
2. **Python ML Backend** - Ray/Nautilus strategies, ensemble models
3. **Message Bus** - ZMQ-based IPC between Rust and Python
4. **Global Kill Switch** - Emergency shutdown system

### Strategy Modules
- StatArb (Statistical Arbitrage)
- Trend Following
- Market Making
- RL Execution Agent

## Trading Parameters
- **Window**: 4 hours continuous
- **Capital**: Dynamic Fractional Kelly
- **Risk Limit**: Per-position and portfolio-level
- **Target**: 3,000 → 50,000 INR

## Safety Systems
- Python-side circuit breakers
- Correlation eigenvalue monitoring
- ML hallucination detection
- Hardware validators

## Session Log

"""
        return content
    
    def _load_existing(self):
        """Load existing SOUL.md entries."""
        try:
            content = self.soul_path.read_text()
            # Parse entries from markdown (simplified)
            self.entries = []
        except:
            pass
    
    def append_entry(self, entry_type: str, message: str, metadata: Dict = None):
        """Append an entry to SOUL.md."""
        now = datetime.now().isoformat()
        
        entry = f"\n### [{now}] {entry_type}\n{message}\n"
        
        if metadata:
            entry += f"```json\n{json.dumps(metadata, indent=2)}\n```\n"
        
        try:
            with open(self.soul_path, 'a') as f:
                f.write(entry)
            
            self.entries.append({
                'timestamp': now,
                'type': entry_type,
                'message': message,
                'metadata': metadata
            })
        except Exception as e:
            logger.error(f"Failed to append to SOUL.md: {e}")
    
    def log_startup(self):
        """Log system startup event."""
        self.append_entry(
            "STARTUP",
            "System initialized and starting trading session",
            {
                'stage': 50,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def log_shutdown(self, reason: str):
        """Log system shutdown event."""
        self.append_entry(
            "SHUTDOWN",
            f"System shutting down: {reason}",
            {
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def log_trade(self, symbol: str, side: str, quantity: float, price: float):
        """Log a trade event."""
        self.append_entry(
            "TRADE",
            f"{side} {quantity} {symbol} @ {price}",
            {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price
            }
        )


class SetupCompletionMarker:
    """Manages setup completion marker file."""
    
    def __init__(self):
        self.marker_path = SETUP_COMPLETE_MARKER
    
    def mark_complete(self, setup_data: Dict = None):
        """Mark setup as complete."""
        data = {
            'completed': True,
            'timestamp': datetime.now().isoformat(),
            'stage': 50,
            'setup_data': setup_data or {}
        }
        
        try:
            self.marker_path.write_text(json.dumps(data, indent=2))
            logger.info(f"Setup completion marker written: {self.marker_path}")
        except Exception as e:
            logger.error(f"Failed to write completion marker: {e}")
    
    def is_complete(self) -> bool:
        """Check if setup is marked complete."""
        if not self.marker_path.exists():
            return False
        
        try:
            data = json.loads(self.marker_path.read_text())
            return data.get('completed', False)
        except:
            return False
    
    def get_setup_data(self) -> Dict:
        """Get setup data from marker file."""
        if not self.marker_path.exists():
            return {}
        
        try:
            return json.loads(self.marker_path.read_text())
        except:
            return {}


class SetupModuleCoordinator:
    """Coordinates all setup operations."""
    
    def __init__(self):
        self.dir_validator = DirectoryStructureValidator()
        self.soul_initializer = SOULMdInitializer()
        self.completion_marker = SetupCompletionMarker()
    
    def run_full_setup(self) -> bool:
        """Run complete setup sequence."""
        logger.info("=" * 60)
        logger.info("SETUP MODULE - STAGE 50")
        logger.info("=" * 60)
        
        all_passed = True
        
        # Step 1: Directory structure
        logger.info("\n📁 Creating directory structure...")
        if not self.dir_validator.validate_and_create():
            logger.error("Directory structure creation failed")
            all_passed = False
        else:
            self.dir_validator.set_permissions()
        
        # Step 2: SOUL.md initialization
        logger.info("\n📖 Initializing SOUL.md...")
        if not self.soul_initializer.initialize():
            logger.error("SOUL.md initialization failed")
            all_passed = False
        else:
            self.soul_initializer.log_startup()
        
        # Step 3: Validate dependencies (import check)
        logger.info("\n🔍 Validating dependencies...")
        from .dependency_checker import DependencyCoordinator
        dep_coordinator = DependencyCoordinator()
        if not dep_coordinator.run_full_check():
            logger.warning("Some dependency checks failed (may be non-fatal)")
        
        # Step 4: Mark setup complete
        if all_passed:
            logger.info("\n✅ Marking setup as complete...")
            self.completion_marker.mark_complete({
                'directories_created': len(self.dir_validator.created_dirs),
                'soul_initialized': True
            })
        
        # Summary
        logger.info("\n" + "=" * 60)
        if all_passed:
            logger.info("✅ SETUP COMPLETE - System ready for launch")
        else:
            logger.error("❌ SETUP INCOMPLETE - Review errors above")
        logger.info("=" * 60)
        
        return all_passed
    
    def verify_setup(self) -> bool:
        """Verify setup is complete without re-running."""
        if not self.completion_marker.is_complete():
            logger.warning("Setup not marked complete")
            return False
        
        # Verify critical components
        checks = [
            (SOUL_MD_PATH.exists(), "SOUL.md exists"),
            (LOGS_DIR.exists(), "Logs directory exists"),
            (MODELS_DIR.exists(), "Models directory exists"),
        ]
        
        all_ok = True
        for ok, msg in checks:
            if ok:
                logger.debug(f"✓ {msg}")
            else:
                logger.warning(f"✗ {msg}")
                all_ok = False
        
        return all_ok


def run_setup():
    """Convenience function to run setup."""
    coordinator = SetupModuleCoordinator()
    success = coordinator.run_full_setup()
    return success


def main():
    """Entry point for setup module."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup Module')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing setup')
    parser.add_argument('--force', action='store_true',
                       help='Force re-run setup even if complete')
    args = parser.parse_args()
    
    coordinator = SetupModuleCoordinator()
    
    if args.verify_only:
        success = coordinator.verify_setup()
    elif args.force or not coordinator.completion_marker.is_complete():
        success = coordinator.run_full_setup()
    else:
        logger.info("Setup already complete. Use --force to re-run.")
        success = True
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
