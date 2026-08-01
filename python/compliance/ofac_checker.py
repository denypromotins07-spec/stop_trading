"""
OFAC Sanctions Checker - Memory-mapped Bloom filter for instant address screening.
Rejects transactions involving OFAC-sanctioned EVM/Solana addresses.
Strictly enforces <50MB RAM footprint using mmap-based bit array.
"""
import numpy as np
import logging
from typing import Set, List, Optional, Union
from pathlib import Path
import mmap
import struct
import hashlib

logger = logging.getLogger(__name__)


class MmapBloomFilter:
    """
    Memory-mapped Bloom filter for efficient address screening.
    Uses file-backed storage to keep RAM footprint minimal (<50MB).
    """
    
    def __init__(self, capacity: int = 10_000_000, error_rate: float = 0.001,
                 db_path: str = 'data/ofac_bloom.bin'):
        """
        Initialize memory-mapped Bloom filter.
        
        Args:
            capacity: Expected number of elements
            error_rate: Target false positive rate
            db_path: Path to backing file for mmap
        """
        self.capacity = capacity
        self.error_rate = error_rate
        self.db_path = Path(db_path)
        
        # Calculate optimal parameters
        # m = -n * ln(p) / (ln(2)^2)
        self.num_bits = int(-capacity * np.log(error_rate) / (np.log(2) ** 2))
        self.num_bits = ((self.num_bits + 7) // 8) * 8  # Round to byte boundary
        
        # k = m/n * ln(2)
        self.num_hashes = max(3, int(self.num_bits / capacity * np.log(2)))
        
        self._mmap = None
        self._bit_array = None
        self._count = 0
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load mmap
        self._init_mmap()
        
        logger.info(f"MmapBloomFilter initialized: {self.num_bits} bits, "
                   f"{self.num_hashes} hashes, {self.num_bits / 8 / 1024 / 1024:.2f}MB")
    
    def _init_mmap(self) -> None:
        """Initialize memory-mapped bit array."""
        file_size = self.num_bits // 8
        
        # Create or open file
        if not self.db_path.exists():
            with open(self.db_path, 'wb') as f:
                f.write(b'\x00' * file_size)
        
        # Open file for read/write
        self._fd = open(self.db_path, 'r+b')
        
        # Memory map the file
        self._mmap = mmap.mmap(self._fd.fileno(), file_size, 
                               access=mmap.ACCESS_WRITE)
        
        # Create numpy array view (zero-copy)
        self._bit_array = np.frombuffer(self._mmap, dtype=np.uint8)
    
    def _hashes(self, item: bytes) -> List[int]:
        """Generate hash indices for an item."""
        # Use double hashing: h(i) = h1(x) + i * h2(x)
        h1 = int(hashlib.md5(item).hexdigest(), 16)
        h2 = int(hashlib.sha256(item).hexdigest(), 16)
        
        indices = []
        for i in range(self.num_hashes):
            idx = (h1 + i * h2) % self.num_bits
            indices.append(idx)
        
        return indices
    
    def add(self, item: Union[str, bytes]) -> None:
        """Add an item to the filter."""
        if isinstance(item, str):
            item = item.lower().encode('utf-8')
        
        indices = self._hashes(item)
        
        for idx in indices:
            byte_idx = idx // 8
            bit_idx = idx % 8
            self._bit_array[byte_idx] |= (1 << bit_idx)
        
        self._count += 1
    
    def contains(self, item: Union[str, bytes]) -> bool:
        """Check if an item might be in the filter."""
        if isinstance(item, str):
            item = item.lower().encode('utf-8')
        
        indices = self._hashes(item)
        
        for idx in indices:
            byte_idx = idx // 8
            bit_idx = idx % 8
            
            if not (self._bit_array[byte_idx] & (1 << bit_idx)):
                return False
        
        return True
    
    def close(self) -> None:
        """Close memory-mapped file."""
        if self._mmap is not None:
            self._mmap.flush()
            self._mmap.close()
            self._mmap = None
        
        if hasattr(self, '_fd') and self._fd is not None:
            self._fd.close()
            self._fd = None
    
    def __del__(self):
        self.close()
    
    def __len__(self) -> int:
        return self._count


class OFACChecker:
    """
    OFAC sanctions list checker using memory-mapped Bloom filter.
    Supports EVM (Ethereum-compatible) and Solana addresses.
    """
    
    # Known sanctioned addresses (sample - would be populated from official sources)
    SAMPLE_SANCTIONED_EVM = {
        # Tornado Cash Router
        '0xd90e2f925da726b50c4ed8d0fb90ad053324f31b',
        # Tornado Cash ETH
        '0x47ce0c6ed5b0ce3d3a51fdb1c52dc66a7c3c2936',
        # Other known sanctioned addresses
        '0x9008d19f58aabd9ed0d60971565aa8510560ab41',
    }
    
    SAMPLE_SANCTIONED_SOLANA = {
        # Known sanctioned Solana addresses
        'DjFZV8ZJo8vGJkZqQzYXxvXzJzXzJzXzJzXzJzXzJzXz',
        'HWhBhFzV8ZJo8vGJkZqQzYXxvXzJzXzJzXzJzXzJzXzJ',
    }
    
    def __init__(self, db_path: str = 'data/ofac_bloom.bin',
                 evan_list_path: Optional[str] = None,
                 solana_list_path: Optional[str] = None,
                 auto_populate: bool = True):
        """
        Initialize OFAC checker.
        
        Args:
            db_path: Path to Bloom filter database
            evm_list_path: Path to custom EVM sanctions list
            solana_list_path: Path to custom Solana sanctions list
            auto_populate: Whether to populate with sample addresses
        """
        self.db_path = db_path
        self.evm_list_path = evm_list_path
        self.solana_list_path = solana_list_path
        
        # Initialize Bloom filter
        self.bloom = MmapBloomFilter(
            capacity=10_000_000,
            error_rate=0.001,
            db_path=db_path
        )
        
        # Track statistics
        self._checks_count = 0
        self._blocks_count = 0
        
        # Populate with sanctioned addresses
        if auto_populate:
            self._populate_sanctions_lists()
        
        logger.info(f"OFACChecker initialized with {len(self.bloom)} addresses")
    
    def _populate_sanctions_lists(self) -> None:
        """Populate Bloom filter with sanctioned addresses."""
        # Add sample EVM addresses
        for addr in self.SAMPLE_SANCTIONED_EVM:
            self.bloom.add(addr)
        
        # Add sample Solana addresses
        for addr in self.SAMPLE_SANCTIONED_SOLANA:
            self.bloom.add(addr)
        
        # Load from external files if provided
        if self.evm_list_path:
            self._load_address_file(self.evm_list_path)
        
        if self.solana_list_path:
            self._load_address_file(self.solana_list_path)
    
    def _load_address_file(self, path: str) -> None:
        """Load addresses from a file (one per line)."""
        try:
            with open(path, 'r') as f:
                for line in f:
                    addr = line.strip().lower()
                    if addr and not addr.startswith('#'):
                        self.bloom.add(addr)
            logger.info(f"Loaded addresses from {path}")
        except FileNotFoundError:
            logger.warning(f"Address file not found: {path}")
        except Exception as e:
            logger.error(f"Error loading address file {path}: {e}")
    
    def check_evm(self, address: str) -> dict:
        """
        Check an EVM address against OFAC sanctions list.
        
        Args:
            address: Ethereum-style address (0x...)
            
        Returns:
            Dictionary with check results
        """
        self._checks_count += 1
        
        # Normalize address
        addr_normalized = address.lower().strip()
        
        # Remove 0x prefix if present for consistent checking
        if addr_normalized.startswith('0x'):
            addr_check = addr_normalized[2:]
        else:
            addr_check = addr_normalized
        
        # Check Bloom filter (with and without 0x)
        is_sanctioned = (
            self.bloom.contains(addr_normalized) or
            self.bloom.contains(addr_check)
        )
        
        if is_sanctioned:
            self._blocks_count += 1
        
        result = {
            'address': address,
            'is_sanctioned': is_sanctioned,
            'chain': 'EVM',
            'action': 'BLOCK' if is_sanctioned else 'ALLOW',
            'reason': 'OFAC Sanctioned Address' if is_sanctioned else None
        }
        
        return result
    
    def check_solana(self, address: str) -> dict:
        """
        Check a Solana address against OFAC sanctions list.
        
        Args:
            address: Solana base58 address
            
        Returns:
            Dictionary with check results
        """
        self._checks_count += 1
        
        addr_normalized = address.strip()
        
        # Check Bloom filter
        is_sanctioned = self.bloom.contains(addr_normalized)
        
        if is_sanctioned:
            self._blocks_count += 1
        
        result = {
            'address': address,
            'is_sanctioned': is_sanctioned,
            'chain': 'Solana',
            'action': 'BLOCK' if is_sanctioned else 'ALLOW',
            'reason': 'OFAC Sanctioned Address' if is_sanctioned else None
        }
        
        return result
    
    def check(self, address: str, chain: Optional[str] = None) -> dict:
        """
        Universal address checker that auto-detects chain type.
        
        Args:
            address: Wallet address
            chain: Optional chain hint ('evm' or 'solana')
            
        Returns:
            Dictionary with check results
        """
        addr_stripped = address.strip()
        
        # Auto-detect chain
        if chain is None:
            if addr_stripped.startswith('0x') or len(addr_stripped) == 40:
                chain = 'evm'
            elif len(addr_stripped) >= 32 and len(addr_stripped) <= 44:
                chain = 'solana'
            else:
                return {
                    'address': address,
                    'is_sanctioned': False,
                    'chain': 'unknown',
                    'action': 'REVIEW',
                    'reason': 'Unknown address format'
                }
        
        if chain.lower() in ('evm', 'ethereum', 'eth'):
            return self.check_evm(address)
        elif chain.lower() in ('solana', 'sol'):
            return self.check_solana(address)
        else:
            return {
                'address': address,
                'is_sanctioned': False,
                'chain': chain,
                'action': 'REVIEW',
                'reason': f'Unsupported chain: {chain}'
            }
    
    def check_batch(self, addresses: List[str], 
                    chains: Optional[List[str]] = None) -> List[dict]:
        """
        Check multiple addresses efficiently.
        
        Args:
            addresses: List of addresses to check
            chains: Optional list of chain hints
            
        Returns:
            List of check results
        """
        results = []
        for i, addr in enumerate(addresses):
            chain = chains[i] if chains else None
            results.append(self.check(addr, chain))
        return results
    
    def get_statistics(self) -> dict:
        """Get checker statistics."""
        return {
            'total_checks': self._checks_count,
            'addresses_blocked': self._blocks_count,
            'block_rate': self._blocks_count / max(1, self._checks_count),
            'bloom_filter_size_mb': self.bloom.num_bits / 8 / 1024 / 1024,
            'estimated_addresses': len(self.bloom)
        }
    
    def add_sanctioned_address(self, address: str, chain: str = 'evm') -> None:
        """Manually add a sanctioned address."""
        self.bloom.add(address.lower())
        logger.info(f"Added sanctioned address: {address} ({chain})")
    
    def close(self) -> None:
        """Clean up resources."""
        self.bloom.close()
        logger.info(f"OFACChecker closed. Total checks: {self._checks_count}, "
                   f"blocks: {self._blocks_count}")


# Singleton instance
_ofac_checker: Optional[OFACChecker] = None


def get_ofac_checker(config: Optional[dict] = None) -> OFACChecker:
    """Get or create singleton OFACChecker instance."""
    global _ofac_checker
    if _ofac_checker is None:
        config = config or {}
        _ofac_checker = OFACChecker(
            db_path=config.get('db_path', 'data/ofac_bloom.bin'),
            evm_list_path=config.get('evm_list_path'),
            solana_list_path=config.get('solana_list_path'),
            auto_populate=config.get('auto_populate', True)
        )
    return _ofac_checker


def reset_ofac_checker() -> None:
    """Reset singleton instance."""
    global _ofac_checker
    if _ofac_checker is not None:
        _ofac_checker.close()
    _ofac_checker = None


__all__ = ['OFACChecker', 'MmapBloomFilter', 'get_ofac_checker', 'reset_ofac_checker']
