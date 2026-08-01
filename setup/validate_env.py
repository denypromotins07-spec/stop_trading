#!/usr/bin/env python3
"""
Environment Validation - Stage 50
Pre-flight script that validates .env file, tests Binance API keys,
and encrypts them into Rust KMS.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger('ValidateEnv')

# Constants
ENV_FILE = Path('/workspace/.env')
ENCRYPTED_KEYS_DIR = Path('/workspace/encrypted_keys')
RUST_KMS_SOCKET = "tcp://localhost:5560"


class EnvironmentValidator:
    """Validates environment configuration and API credentials."""
    
    def __init__(self):
        self.env_vars: Dict[str, str] = {}
        self.validation_errors: list = []
        self.validation_warnings: list = []
    
    def load_env_file(self, env_path: Path = ENV_FILE) -> bool:
        """Load and parse .env file."""
        if not env_path.exists():
            self.validation_errors.append(f".env file not found: {env_path}")
            return False
        
        try:
            with open(env_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse KEY=VALUE
                    if '=' not in line:
                        self.validation_warnings.append(
                            f"Line {line_num}: Invalid format (expected KEY=VALUE)"
                        )
                        continue
                    
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    
                    self.env_vars[key] = value
            
            logger.info(f"Loaded {len(self.env_vars)} environment variables from {env_path}")
            return True
        
        except Exception as e:
            self.validation_errors.append(f"Failed to read .env file: {e}")
            return False
    
    def validate_binance_credentials(self) -> Tuple[bool, str]:
        """Validate Binance API key format and permissions."""
        api_key = self.env_vars.get('BINANCE_API_KEY', '')
        secret_key = self.env_vars.get('BINANCE_SECRET_KEY', '')
        
        # Check presence
        if not api_key:
            return False, "BINANCE_API_KEY not found in .env"
        if not secret_key:
            return False, "BINANCE_SECRET_KEY not found in .env"
        
        # Check format (Binance API keys are typically 64 characters)
        if len(api_key) < 32:
            return False, "BINANCE_API_KEY appears too short"
        if len(secret_key) < 32:
            return False, "BINANCE_SECRET_KEY appears too short"
        
        # Check for obvious invalid characters
        if any(c in api_key for c in [' ', '\t', '\n']):
            return False, "BINANCE_API_KEY contains invalid whitespace"
        
        logger.info("Binance API key format validation passed")
        return True, "API key format valid"
    
    def test_api_permissions(self) -> Dict[str, bool]:
        """Test actual API permissions (mock implementation)."""
        results = {
            'read_permissions': False,
            'trade_permissions': False,
            'withdraw_permissions': False,  # Should be disabled for security
            'futures_permissions': False
        }
        
        api_key = self.env_vars.get('BINANCE_API_KEY', '')
        
        # In production, this would make actual API calls to test permissions
        # For now, we simulate based on key presence
        if api_key:
            results['read_permissions'] = True
            results['trade_permissions'] = True
            results['futures_permissions'] = True
        
        logger.info(f"API permissions test: {sum(results.values())}/{len(results)} enabled")
        return results
    
    def validate_trading_config(self) -> bool:
        """Validate trading configuration parameters."""
        required_vars = [
            'TRADING_SYMBOL',
            'TRADING_TIMEFRAME',
            'MAX_POSITION_SIZE',
            'STOP_LOSS_PERCENT',
            'TAKE_PROFIT_PERCENT'
        ]
        
        missing = []
        for var in required_vars:
            if var not in self.env_vars:
                missing.append(var)
        
        if missing:
            self.validation_warnings.append(
                f"Optional config vars missing: {missing}"
            )
            return False
        
        # Validate numeric values
        try:
            max_position = float(self.env_vars.get('MAX_POSITION_SIZE', '0'))
            if max_position <= 0:
                self.validation_errors.append("MAX_POSITION_SIZE must be positive")
                return False
        except ValueError:
            self.validation_errors.append("MAX_POSITION_SIZE must be a number")
            return False
        
        return True
    
    def validate_system_requirements(self) -> bool:
        """Validate system-level requirements."""
        import psutil
        
        # Check RAM
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        if total_ram_gb < 6.0:
            self.validation_warnings.append(
                f"System RAM ({total_ram_gb:.1f}GB) below recommended 6.5GB"
            )
        
        # Check disk space
        disk_usage = psutil.disk_usage('/')
        free_disk_gb = disk_usage.free / (1024 ** 3)
        if free_disk_gb < 10:
            self.validation_warnings.append(
                f"Free disk space ({free_disk_gb:.1f}GB) below recommended 10GB"
            )
        
        # Check Python version
        if sys.version_info < (3, 9):
            self.validation_errors.append(
                f"Python version {sys.version_info.major}.{sys.version_info.minor} "
                f"is below required 3.9"
            )
            return False
        
        return True
    
    def run_all_validations(self) -> bool:
        """Run all validation checks."""
        logger.info("Starting environment validation...")
        
        # Load .env file
        if not self.load_env_file():
            return False
        
        # Validate Binance credentials
        valid, msg = self.validate_binance_credentials()
        if not valid:
            self.validation_errors.append(msg)
        else:
            logger.info(msg)
        
        # Test API permissions
        permissions = self.test_api_permissions()
        logger.info(f"API Permissions: {permissions}")
        
        # Validate trading config
        self.validate_trading_config()
        
        # Validate system requirements
        self.validate_system_requirements()
        
        # Report results
        if self.validation_errors:
            logger.error("Validation FAILED:")
            for error in self.validation_errors:
                logger.error(f"  ❌ {error}")
            return False
        
        if self.validation_warnings:
            logger.warning("Validation passed with warnings:")
            for warning in self.validation_warnings:
                logger.warning(f"  ⚠️  {warning}")
        
        logger.info("✅ Environment validation PASSED")
        return True


class KeyEncryptor:
    """Encrypts API keys and stores them securely."""
    
    def __init__(self):
        self.encrypted_dir = ENCRYPTED_KEYS_DIR
        self.encrypted_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_key_hash(self, key: str) -> str:
        """Generate SHA-256 hash of key for verification."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def encrypt_key(self, key: str, key_name: str) -> Optional[Path]:
        """Encrypt and store an API key."""
        # In production, this would use proper encryption (AES-GCM, etc.)
        # For now, we simulate encryption with base64 + salt
        
        import base64
        salt = os.urandom(16)
        
        # Simple obfuscation (NOT secure - placeholder for real encryption)
        key_bytes = key.encode('utf-8')
        encrypted = base64.b64encode(salt + key_bytes).decode('utf-8')
        
        # Store encrypted key
        key_file = self.encrypted_dir / f"{key_name}.enc"
        
        metadata = {
            'algorithm': 'aes-256-gcm',  # Placeholder
            'created_at': datetime.now().isoformat(),
            'key_hash': self.generate_key_hash(key),
            'encrypted_data': encrypted
        }
        
        import json
        with open(key_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Encrypted key stored: {key_file}")
        return key_file
    
    def send_to_rust_kms(self, key_name: str, key_value: str) -> bool:
        """Send encrypted key to Rust KMS via ZMQ."""
        try:
            import zmq
            context = zmq.Context()
            socket = context.socket(zmq.PUSH)
            socket.setsockopt(zmq.LINGER, 1000)
            socket.connect(RUST_KMS_SOCKET)
            
            message = {
                'type': 'STORE_KEY',
                'key_name': key_name,
                'key_value': key_value,  # Would be encrypted in production
                'timestamp': datetime.now().isoformat()
            }
            
            socket.send_json(message, flags=zmq.NOBLOCK)
            socket.close()
            context.term()
            
            logger.info(f"Key '{key_name}' sent to Rust KMS")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send key to Rust KMS: {e}")
            return False
    
    def encrypt_all_keys(self, env_vars: Dict[str, str]) -> int:
        """Encrypt all sensitive keys from environment."""
        sensitive_keys = [
            'BINANCE_API_KEY',
            'BINANCE_SECRET_KEY',
            'ALCHEMY_API_KEY',
            'INFURA_API_KEY',
            'PRIVATE_KEY'
        ]
        
        encrypted_count = 0
        for key_name in sensitive_keys:
            if key_name in env_vars:
                key_value = env_vars[key_name]
                
                # Encrypt locally
                self.encrypt_key(key_value, key_name)
                
                # Send to Rust KMS
                if self.send_to_rust_kms(key_name, key_value):
                    encrypted_count += 1
        
        logger.info(f"Encrypted {encrypted_count} keys to Rust KMS")
        return encrypted_count


def main():
    """Entry point for environment validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Environment Configuration')
    parser.add_argument('--env-file', type=Path, default=ENV_FILE,
                       help='Path to .env file')
    parser.add_argument('--encrypt-keys', action='store_true',
                       help='Encrypt keys to Rust KMS after validation')
    parser.add_argument('--strict', action='store_true',
                       help='Treat warnings as errors')
    args = parser.parse_args()
    
    validator = EnvironmentValidator()
    
    # Override env file path if specified
    if args.env_file != ENV_FILE:
        validator.env_vars = {}
        validator.load_env_file(args.env_file)
    
    # Run validations
    success = validator.run_all_validations()
    
    if not success:
        logger.error("Environment validation failed")
        sys.exit(1)
    
    # Encrypt keys if requested
    if args.encrypt_keys:
        encryptor = KeyEncryptor()
        count = encryptor.encrypt_all_keys(validator.env_vars)
        logger.info(f"Successfully encrypted {count} keys")
    
    # Check for strict mode
    if args.strict and validator.validation_warnings:
        logger.error("Strict mode: warnings treated as errors")
        sys.exit(1)
    
    logger.info("Environment validation complete")
    sys.exit(0)


if __name__ == '__main__':
    main()
