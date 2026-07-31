"""
Centralized settings parser for HFT Nautilus ML backend.
Reads root .env file and enforces strict 3GB RAM ceiling for Python processes.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from root .env file
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE = ROOT_DIR / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

# Hardware constraint validation
TOTAL_SYSTEM_RAM_GB = float(os.getenv("TOTAL_SYSTEM_RAM_GB", "6.5"))
PYTHON_RAM_CEILING_MB = int(os.getenv("PYTHON_RAM_CEILING_MB", "3072"))
RUST_RAM_RESERVED_MB = int(os.getenv("RUST_RAM_RESERVED_MB", "2500"))
OS_RAM_BUFFER_MB = int(os.getenv("OS_RAM_BUFFER_MB", "1000"))

# Validate total RAM allocation does not exceed system limit
TOTAL_ALLOCATED_MB = PYTHON_RAM_CEILING_MB + RUST_RAM_RESERVED_MB + OS_RAM_BUFFER_MB
if TOTAL_ALLOCATED_MB > (TOTAL_SYSTEM_RAM_GB * 1024):
    raise RuntimeError(
        f"Total allocated RAM ({TOTAL_ALLOCATED_MB}MB) exceeds system limit "
        f"({TOTAL_SYSTEM_RAM_GB * 1024}MB). Adjust PYTHON_RAM_CEILING_MB, "
        f"RUST_RAM_RESERVED_MB, or OS_RAM_BUFFER_MB in .env file."
    )

# Ray cluster configuration
RAY_NUM_CPUS = int(os.getenv("RAY_NUM_CPUS", "4"))
RAY_MEMORY_BYTES = PYTHON_RAM_CEILING_MB * 1024 * 1024  # Convert MB to bytes
RAY_DASHBOARD_HOST = os.getenv("RAY_DASHBOARD_HOST", "127.0.0.1")
RAY_DASHBOARD_PORT = int(os.getenv("RAY_DASHBOARD_PORT", "8265"))

# Nautilus Trader configuration
NAUTILUS_LOG_LEVEL = os.getenv("NAUTILUS_LOG_LEVEL", "WARNING")
NAUTILUS_BYPASS_RECONCILIATION = os.getenv("NAUTILUS_BYPASS_RECONCILIATION", "false").lower() == "true"
NAUTILUS_FLUSH_CACHE_INTERVAL = int(os.getenv("NAUTILUS_FLUSH_CACHE_INTERVAL", "60"))

# Binance API configuration (securely passed from Rust KMS)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

# Shared memory configuration
SHM_SEGMENT_NAME = os.getenv("SHM_SEGMENT_NAME", "/hft_feature_shm")
SHM_SEGMENT_SIZE = int(os.getenv("SHM_SEGMENT_SIZE", "67108864"))  # 64MB default

# ZeroMQ configuration
ZMQ_SIGNAL_PORT = int(os.getenv("ZMQ_SIGNAL_PORT", "5555"))
ZMQ_STATE_PORT = int(os.getenv("ZMQ_STATE_PORT", "5556"))
ZMQ_HOST = os.getenv("ZMQ_HOST", "127.0.0.1")

# SOUL.md integration
SOUL_LEDGER_PATH = ROOT_DIR / "SOUL.md"
SOUL_WATCHDOG_INTERVAL = float(os.getenv("SOUL_WATCHDOG_INTERVAL", "0.5"))

# CPU core pinning (AMD Ryzen specific)
# Cores 0-3 reserved for Rust engine, Cores 4-7 for Python ML workers
ML_WORKER_CPU_CORES = [int(x) for x in os.getenv("ML_WORKER_CPU_CORES", "4,5,6,7").split(",")]
RUST_ENGINE_CPU_CORES = [int(x) for x in os.getenv("RUST_ENGINE_CPU_CORES", "0,1,2,3").split(",")]


def get_python_memory_limit_bytes() -> int:
    """Return the strict memory limit for Python processes in bytes."""
    return PYTHON_RAM_CEILING_MB * 1024 * 1024


def get_ray_init_kwargs() -> dict:
    """Return keyword arguments for Ray initialization with strict resource bounds."""
    return {
        "num_cpus": RAY_NUM_CPUS,
        "_memory": RAY_MEMORY_BYTES,
        "_redis_max_memory": RAY_MEMORY_BYTES // 4,  # Reserve 25% for Redis
        "dashboard_host": RAY_DASHBOARD_HOST,
        "dashboard_port": RAY_DASHBOARD_PORT,
        "include_dashboard": True,
        "log_to_driver": False,  # Disable driver logging for performance
        "logging_level": "warning",
        "_temp_dir": str(ROOT_DIR / ".ray_temp"),
    }


def validate_environment() -> None:
    """Validate all required environment variables are set correctly."""
    if not BINANCE_API_KEY and not BINANCE_TESTNET:
        raise ValueError("BINANCE_API_KEY is required for live trading")
    if not BINANCE_API_SECRET and not BINANCE_TESTNET:
        raise ValueError("BINANCE_API_SECRET is required for live trading")
    
    if len(ML_WORKER_CPU_CORES) == 0:
        raise ValueError("At least one CPU core must be assigned to ML workers")
    
    # Check for overlapping CPU cores between Rust and Python
    overlap = set(ML_WORKER_CPU_CORES) & set(RUST_ENGINE_CPU_CORES)
    if overlap:
        raise ValueError(f"CPU cores {overlap} are assigned to both Rust and Python workers")
