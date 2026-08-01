#!/usr/bin/env python3
"""
Systemd Installer - Stage 50
Generates and installs systemd service files for Ubuntu 24/7 execution.
Respects 6.5GB RAM limit and auto-starts on boot.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger('SystemdInstaller')

# Constants
SYSTEMD_DIR = Path('/etc/systemd/system')
SERVICE_NAME = 'crypto_bot'
WORKSPACE_ROOT = Path('/workspace')
PYTHON_BIN = sys.executable
MASTER_ORCHESTRATOR = WORKSPACE_ROOT / 'launch' / 'master_orchestrator.py'

# Memory limits in bytes (6.5GB total, split between Rust and Python)
MEMORY_LIMIT_BYTES = 6 * 1024 * 1024 * 1024  # 6GB soft limit


class SystemdServiceGenerator:
    """Generates systemd service file content."""
    
    def __init__(self):
        self.service_name = SERVICE_NAME
        self.workspace = WORKSPACE_ROOT
        self.python_bin = PYTHON_BIN
        self.orchestrator = MASTER_ORCHESTRATOR
    
    def generate_service_content(self) -> str:
        """Generate the systemd service file content."""
        now = datetime.now().isoformat()
        
        content = f"""[Unit]
Description=Crypto Medium Frequency Trading Bot (Stage 50)
Documentation=https://github.com/crypto-bot
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory={self.workspace}

# Environment setup
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"
Environment="RAY_DISABLE_DOCKER_CPU_WARNING=1"
Environment="TRADING_MODE=live"
Environment="RUST_LOG=info"

# Memory limits (6.5GB total system constraint)
MemoryLimit={MEMORY_LIMIT_BYTES}
MemoryHigh={MEMORY_LIMIT_BYTES * 90 // 100}

# CPU limits
CPUQuota=80%

# Restart policy
Restart=on-failure
RestartSec=10
StartLimitInterval=300
StartLimitBurst=5

# Main execution command
ExecStart={self.python_bin} {self.orchestrator}

# Graceful shutdown
TimeoutStopSec=30
KillMode=mixed
KillSignal=SIGTERM
SendSIGKILL=yes

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths={self.workspace} /tmp /var/log
PrivateTmp=true

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=crypto_bot

[Install]
WantedBy=multi-user.target
DefaultInstance=

# Generated at: {now}
# Stage 50 - Ultimate Handoff Infrastructure
"""
        return content
    
    def write_service_file(self) -> Path:
        """Write service file to systemd directory."""
        service_path = SYSTEMD_DIR / f"{self.service_name}.service"
        
        try:
            content = self.generate_service_content()
            service_path.write_text(content)
            
            # Set proper permissions
            os.chmod(service_path, 0o644)
            
            logger.info(f"Service file written: {service_path}")
            return service_path
        
        except Exception as e:
            logger.error(f"Failed to write service file: {e}")
            raise


class SystemdManager:
    """Manages systemd service operations."""
    
    def __init__(self):
        self.generator = SystemdServiceGenerator()
        self.service_name = SERVICE_NAME
    
    def install(self) -> bool:
        """Install and enable the service."""
        logger.info("Installing systemd service...")
        
        try:
            # Write service file
            service_path = self.generator.write_service_file()
            
            # Reload systemd daemon
            self._run_systemctl('daemon-reload', check=False)
            
            # Enable service (start on boot)
            self._run_systemctl('enable', self.service_name)
            
            logger.info(f"Service {self.service_name} installed and enabled")
            return True
        
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            return False
    
    def start(self) -> bool:
        """Start the service."""
        logger.info(f"Starting service {self.service_name}...")
        return self._run_systemctl('start', self.service_name)
    
    def stop(self) -> bool:
        """Stop the service."""
        logger.info(f"Stopping service {self.service_name}...")
        return self._run_systemctl('stop', self.service_name)
    
    def restart(self) -> bool:
        """Restart the service."""
        logger.info(f"Restarting service {self.service_name}...")
        return self._run_systemctl('restart', self.service_name)
    
    def status(self) -> str:
        """Get service status."""
        try:
            result = subprocess.run(
                ['systemctl', 'status', self.service_name],
                capture_output=True,
                text=True
            )
            return result.stdout
        except Exception as e:
            return f"Error getting status: {e}"
    
    def is_active(self) -> bool:
        """Check if service is active."""
        try:
            result = subprocess.run(
                ['systemctl', 'is-active', '--quiet', self.service_name],
                capture_output=True
            )
            return result.returncode == 0
        except:
            return False
    
    def uninstall(self) -> bool:
        """Uninstall the service."""
        logger.info(f"Uninstalling service {self.service_name}...")
        
        try:
            # Stop and disable
            self._run_systemctl('stop', self.service_name, check=False)
            self._run_systemctl('disable', self.service_name, check=False)
            
            # Remove service file
            service_path = SYSTEMD_DIR / f"{self.service_name}.service"
            if service_path.exists():
                service_path.unlink()
            
            # Reload daemon
            self._run_systemctl('daemon-reload', check=False)
            
            logger.info(f"Service {self.service_name} uninstalled")
            return True
        
        except Exception as e:
            logger.error(f"Uninstallation failed: {e}")
            return False
    
    def _run_systemctl(self, command: str, *args, check: bool = True) -> bool:
        """Run systemctl command."""
        try:
            cmd = ['systemctl', command] + list(args)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=check
            )
            
            if result.returncode == 0 or not check:
                return True
            
            logger.error(f"systemctl {command} failed: {result.stderr}")
            return False
        
        except subprocess.CalledProcessError as e:
            if check:
                logger.error(f"systemctl {command} error: {e}")
            return False
        except FileNotFoundError:
            logger.error("systemctl not found - is systemd installed?")
            return False


def check_systemd_available() -> bool:
    """Check if systemd is available on this system."""
    return Path('/run/systemd/system').exists()


def main():
    """Entry point for systemd installer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Systemd Service Installer')
    parser.add_argument('action', choices=['install', 'start', 'stop', 'restart', 'status', 'uninstall'],
                       help='Action to perform')
    parser.add_argument('--force', action='store_true', help='Force operation')
    args = parser.parse_args()
    
    # Check if running as root
    if os.geteuid() != 0 and not args.force:
        logger.error("This script must be run as root (use sudo)")
        sys.exit(1)
    
    # Check systemd availability
    if not check_systemd_available() and not args.force:
        logger.warning("Systemd may not be available (running in container?)")
        if not args.force:
            logger.info("Use --force to proceed anyway")
            sys.exit(1)
    
    manager = SystemdManager()
    
    success = False
    
    if args.action == 'install':
        success = manager.install()
    elif args.action == 'start':
        success = manager.start()
    elif args.action == 'stop':
        success = manager.stop()
    elif args.action == 'restart':
        success = manager.restart()
    elif args.action == 'status':
        print(manager.status())
        success = True
    elif args.action == 'uninstall':
        success = manager.uninstall()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
