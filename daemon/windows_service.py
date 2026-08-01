#!/usr/bin/env python3
"""
Windows Service - Stage 50
Generates Windows Task Scheduler or NSSM scripts for 24/7 background execution.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger('WindowsService')

# Constants
WORKSPACE_ROOT = Path('/workspace')
PYTHON_BIN = sys.executable if not sys.platform.startswith('win') else 'python.exe'
MASTER_ORCHESTRATOR = WORKSPACE_ROOT / 'launch' / 'master_orchestrator.py'
SCRIPTS_DIR = WORKSPACE_ROOT / 'daemon' / 'windows_scripts'


class TaskSchedulerGenerator:
    """Generates Windows Task Scheduler XML and PowerShell scripts."""
    
    def __init__(self):
        self.task_name = "CryptoBot"
        self.workspace = str(WORKSPACE_ROOT)
        self.python_bin = PYTHON_BIN
        self.orchestrator = str(MASTER_ORCHESTRATOR)
        self.scripts_dir = SCRIPTS_DIR
    
    def generate_powershell_script(self) -> Path:
        """Generate PowerShell script to create scheduled task."""
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        
        script_path = self.scripts_dir / 'install_task.ps1'
        now = datetime.now().isoformat()
        
        # Escape paths for PowerShell
        workspace_escaped = self.workspace.replace('\\', '\\\\')
        orchestrator_escaped = self.orchestrator.replace('\\', '\\\\')
        
        content = f'''# Crypto Bot Task Scheduler Installation Script
# Generated: {now}
# Stage 50 - Ultimate Handoff Infrastructure

Write-Host "Installing Crypto Bot Scheduled Task..." -ForegroundColor Cyan

$taskName = "{self.task_name}"
$taskPath = "\\Crypto Bot\\"
$userId = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

# Action settings
$action = New-ScheduledTaskAction -Execute "{self.python_bin}" `
    -Argument "{self.orchestrator}" `
    -WorkingDirectory "{self.workspace}"

# Trigger settings (run at startup)
$trigger = New-ScheduledTaskTrigger -AtStartup

# Settings
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 5 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 5) `
    -Priority 7

# Principal (run with highest privileges)
$principal = New-ScheduledTaskPrincipal -UserId $userId `
    -LogonType S4U `
    -RunLevel Highest

# Register the task
try {{
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    
    Register-ScheduledTask `
        -TaskName $taskName `
        -TaskPath $taskPath `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description "Crypto Medium Frequency Trading Bot - Stage 50"
    
    Write-Host "✅ Task '{self.task_name}' created successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To start the bot:" -ForegroundColor Yellow
    Write-Host "  Start-ScheduledTask -TaskName '{self.task_name}' -TaskPath '{self.taskPath}'" -ForegroundColor White
    Write-Host ""
    Write-Host "To check status:" -ForegroundColor Yellow
    Write-Host "  Get-ScheduledTask -TaskName '{self.task_name}' | Select-Object State" -ForegroundColor White
    Write-Host ""
    Write-Host "To stop the bot:" -ForegroundColor Yellow
    Write-Host "  Stop-ScheduledTask -TaskName '{self.task_name}' -TaskPath '{self.taskPath}'" -ForegroundColor White
}}
catch {{
    Write-Host "❌ Error creating task: $_" -ForegroundColor Red
    exit 1
}}
'''
        
        script_path.write_text(content)
        logger.info(f"PowerShell script generated: {script_path}")
        return script_path
    
    def generate_batch_start(self) -> Path:
        """Generate batch file to start the bot manually."""
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        
        batch_path = self.scripts_dir / 'start_bot.bat'
        
        content = f'''@echo off
REM Crypto Bot Manual Start Script
REM Generated: {datetime.now().isoformat()}

echo Starting Crypto Bot...
cd /d "{self.workspace}"
"{self.python_bin}" "{self.orchestrator}"

if errorlevel 1 (
    echo Bot exited with error code %errorlevel%
    pause
)
'''
        
        batch_path.write_text(content)
        logger.info(f"Batch script generated: {batch_path}")
        return batch_path
    
    def generate_uninstall_script(self) -> Path:
        """Generate PowerShell script to remove scheduled task."""
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        
        script_path = self.scripts_dir / 'uninstall_task.ps1'
        
        content = f'''# Crypto Bot Task Scheduler Removal Script

Write-Host "Removing Crypto Bot Scheduled Task..." -ForegroundColor Cyan

$taskName = "{self.task_name}"
$taskPath = "\\Crypto Bot\\"

try {{
    Unregister-ScheduledTask -TaskName $taskName -TaskPath $taskPath -Confirm:$false
    Write-Host "✅ Task removed successfully!" -ForegroundColor Green
}}
catch {{
    Write-Host "⚠️  Task not found or error: $_" -ForegroundColor Yellow
}}
'''
        
        script_path.write_text(content)
        return script_path


class NSSMGenerator:
    """Generates NSSM (Non-Sucking Service Manager) installation scripts."""
    
    def __init__(self):
        self.service_name = "CryptoBot"
        self.workspace = str(WORKSPACE_ROOT)
        self.python_bin = PYTHON_BIN
        self.orchestrator = str(MASTER_ORCHESTRATOR)
        self.scripts_dir = SCRIPTS_DIR
    
    def generate_nssm_install_script(self) -> Path:
        """Generate script to install using NSSM."""
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        
        script_path = self.scripts_dir / 'install_nssm.bat'
        
        content = f'''@echo off
REM NSSM Installation Script for Crypto Bot
REM Download NSSM from: https://nssm.cc/download
REM Generated: {datetime.now().isoformat()}

set SERVICE_NAME={self.service_name}
set APP_PATH={self.python_bin}
set APP_DIR={self.workspace}
set APP_ARGS={self.orchestrator}

echo Installing {self.service_name} as Windows Service using NSSM...

REM Check if NSSM is available
where nssm >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: NSSM not found in PATH
    echo Please download NSSM from https://nssm.cc/download
    echo Extract and add to PATH, or copy nssm.exe to this directory
    pause
    exit /b 1
)

REM Remove existing service if present
nssm remove %SERVICE_NAME% confirm >nul 2>nul

REM Install new service
nssm install %SERVICE_NAME% "%APP_PATH%" "%APP_ARGS%"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install service
    pause
    exit /b 1
)

REM Configure service
nssm set %SERVICE_NAME% DisplayName "Crypto Bot Trading System"
nssm set %SERVICE_NAME% Description "Medium Frequency Trading Bot - Stage 50"
nssm set %SERVICE_NAME% Start SERVICE_AUTO_START
nssm set %SERVICE_NAME% AppDirectory "%APP_DIR%"
nssm set %SERVICE_NAME% AppStdout "%APP_DIR%\\logs\\nssm_stdout.log"
nssm set %SERVICE_NAME% AppStderr "%APP_DIR%\\logs\\nssm_stderr.log"
nssm set %SERVICE_NAME% AppRotateFiles 1
nssm set %SERVICE_NAME% AppRotateBytes 10485760

REM Set memory limits (optional, requires Windows Enterprise)
REM nssm set %SERVICE_NAME% MemoryLimit 6442450944

echo.
echo ✅ Service installed successfully!
echo.
echo To start: net start %SERVICE_NAME%
echo To stop:  net stop %SERVICE_NAME%
echo To view:  sc query %SERVICE_NAME%
echo.

pause
'''
        
        script_path.write_text(content)
        logger.info(f"NSSM install script generated: {script_path}")
        return script_path
    
    def generate_nssm_uninstall_script(self) -> Path:
        """Generate script to uninstall NSSM service."""
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        
        script_path = self.scripts_dir / 'uninstall_nssm.bat'
        
        content = f'''@echo off
REM NSSM Uninstallation Script for Crypto Bot

set SERVICE_NAME={self.service_name}

echo Stopping and removing {self.service_name} service...

net stop %SERVICE_NAME% >nul 2>nul
nssm remove %SERVICE_NAME% confirm

if %ERRORLEVEL% EQU 0 (
    echo ✅ Service removed successfully!
) else (
    echo ⚠️  Service may not have been installed
)

pause
'''
        
        script_path.write_text(content)
        return script_path


class WindowsServiceCoordinator:
    """Coordinates Windows service installation."""
    
    def __init__(self, use_nssm: bool = False):
        self.use_nssm = use_nssm
        self.task_gen = TaskSchedulerGenerator()
        self.nssm_gen = NSSMGenerator()
    
    def install(self) -> bool:
        """Install the Windows service/task."""
        logger.info("Generating Windows service scripts...")
        
        if self.use_nssm:
            script = self.nssm_gen.generate_nssm_install_script()
            logger.info(f"To complete installation, run: {script}")
        else:
            script = self.task_gen.generate_powershell_script()
            self.task_gen.generate_batch_start()
            self.task_gen.generate_uninstall_script()
            logger.info(f"To complete installation, run PowerShell as Admin:")
            logger.info(f"  powershell -ExecutionPolicy Bypass -File {script}")
        
        return True
    
    def uninstall(self) -> bool:
        """Uninstall the Windows service/task."""
        if self.use_nssm:
            script = self.nssm_gen.generate_nssm_uninstall_script()
        else:
            script = self.task_gen.generate_uninstall_script()
        
        logger.info(f"To uninstall, run: {script}")
        return True


def is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform.startswith('win')


def main():
    """Entry point for Windows service installer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Windows Service Installer')
    parser.add_argument('action', choices=['install', 'uninstall'],
                       help='Action to perform')
    parser.add_argument('--nssm', action='store_true', 
                       help='Use NSSM instead of Task Scheduler')
    args = parser.parse_args()
    
    if not is_windows():
        logger.warning("This script is designed for Windows")
        logger.info("Scripts will be generated but may need path adjustments")
    
    coordinator = WindowsServiceCoordinator(use_nssm=args.nssm)
    
    if args.action == 'install':
        success = coordinator.install()
    elif args.action == 'uninstall':
        success = coordinator.uninstall()
    else:
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
