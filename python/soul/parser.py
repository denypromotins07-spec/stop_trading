"""
SOUL.md Parser - Asynchronous, low-memory parser for Rust-generated SOUL.md.
Extracts trade outcomes, mistakes, and regime memories using chunked streaming.
Strictly enforces 3GB RAM limit by avoiding full file loads.
"""
import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, List, Optional, Any
from pathlib import Path
import aiofiles


@dataclass
class TradeOutcome:
    """Represents a single trade outcome extracted from SOUL.md."""
    trade_id: str
    instrument: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    slippage_bps: float
    timestamp_ns: int
    regime_id: str
    alpha_signal: str
    execution_quality: float


@dataclass
class Mistake:
    """Represents an identified mistake from SOUL.md."""
    mistake_id: str
    trade_id: Optional[str]
    category: str  # e.g., "timing", "sizing", "regime_misclassification"
    severity: float  # 0.0 to 1.0
    description: str
    penalty_weight: float
    timestamp_ns: int
    suggested_correction: str


@dataclass
class RegimeMemory:
    """Represents a regime memory block from SOUL.md."""
    regime_id: str
    start_timestamp_ns: int
    end_timestamp_ns: int
    characteristics: Dict[str, Any]
    dominant_alpha: str
    volatility_state: str
    liquidity_state: str
    lessons_learned: List[str] = field(default_factory=list)


@dataclass
class SOULBlock:
    """Container for parsed SOUL.md data."""
    outcomes: List[TradeOutcome] = field(default_factory=list)
    mistakes: List[Mistake] = field(default_factory=list)
    memories: List[RegimeMemory] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SOULParser:
    """
    Asynchronous, chunked streaming parser for SOUL.md files.
    Designed for low-memory operation on AMD Ryzen laptops with 3GB Python limit.
    """
    
    # Regex patterns for parsing SOUL.md sections
    OUTCOME_PATTERN = re.compile(
        r'### Outcome:\s*(\S+)\s*\n'
        r'Instrument:\s*(\S+)\s*\n'
        r'Side:\s*(\S+)\s*\n'
        r'Entry:\s*([\d.]+)\s*\n'
        r'Exit:\s*([\d.]+)\s*\n'
        r'Quantity:\s*([\d.]+)\s*\n'
        r'PnL:\s*([\d.-]+)\s*\n'
        r'Slippage:\s*([\d.-]+)\s*bps\s*\n'
        r'Timestamp:\s*(\d+)\s*\n'
        r'Regime:\s*(\S+)\s*\n'
        r'Signal:\s*(\S+)\s*\n'
        r'Quality:\s*([\d.]+)',
        re.MULTILINE
    )
    
    MISTAKE_PATTERN = re.compile(
        r'### Mistake:\s*(\S+)\s*\n'
        r'(?:Trade:\s*(\S+)\s*\n)?'
        r'Category:\s*(\S+)\s*\n'
        r'Severity:\s*([\d.]+)\s*\n'
        r'Description:\s*(.+?)\s*\n'
        r'Penalty:\s*([\d.]+)\s*\n'
        r'Timestamp:\s*(\d+)\s*\n'
        r'Correction:\s*(.+?)\s*\n',
        re.MULTILINE | re.DOTALL
    )
    
    REGIME_PATTERN = re.compile(
        r'## Regime Memory:\s*(\S+)\s*\n'
        r'Start:\s*(\d+)\s*\n'
        r'End:\s*(\d+)\s*\n'
        r'Characteristics:\s*\n(.+?)\n'
        r'Dominant Alpha:\s*(\S+)\s*\n'
        r'Volatility:\s*(\S+)\s*\n'
        r'Liquidity:\s*(\S+)\s*\n'
        r'Lessons:\s*\n(.+?)\n(?=##|\Z)',
        re.MULTILINE | re.DOTALL
    )

    def __init__(self, chunk_size: int = 8192):
        """
        Initialize parser with chunk size for streaming.
        
        Args:
            chunk_size: Bytes per chunk for streaming (default 8KB)
        """
        self.chunk_size = chunk_size
        self._buffer = ""
        self._parsed_count = 0
    
    async def parse_file(self, filepath: str) -> AsyncGenerator[SOULBlock, None]:
        """
        Parse SOUL.md file asynchronously using chunked streaming.
        
        Args:
            filepath: Path to SOUL.md file
            
        Yields:
            SOULBlock objects containing parsed data
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"SOUL.md file not found: {filepath}")
        
        self._buffer = ""
        async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
            while True:
                chunk = await f.read(self.chunk_size)
                if not chunk:
                    break
                self._buffer += chunk
                
                # Process complete blocks in buffer
                while self._has_complete_block():
                    block = self._extract_block()
                    if block:
                        self._parsed_count += 1
                        yield block
        
        # Process remaining buffer
        if self._buffer.strip():
            block = self._extract_block()
            if block:
                self._parsed_count += 1
                yield block
    
    def _has_complete_block(self) -> bool:
        """Check if buffer contains a complete SOUL block."""
        # Look for section delimiters
        return '## ' in self._buffer and '\n## ' in self._buffer
    
    def _extract_block(self) -> Optional[SOULBlock]:
        """Extract and parse a single SOUL block from buffer."""
        # Find the first two section headers
        first_header = self._buffer.find('## ')
        if first_header == -1:
            return None
        
        second_header = self._buffer.find('\n## ', first_header + 3)
        if second_header == -1:
            # Check for end of file marker
            if len(self._buffer) < self.chunk_size:
                second_header = len(self._buffer)
            else:
                return None
        
        block_text = self._buffer[first_header:second_header].strip()
        self._buffer = self._buffer[second_header + 1:] if second_header < len(self._buffer) else ""
        
        return self._parse_block_text(block_text)
    
    def _parse_block_text(self, text: str) -> SOULBlock:
        """Parse raw block text into structured data."""
        block = SOULBlock()
        
        # Parse trade outcomes
        for match in self.OUTCOME_PATTERN.finditer(text):
            outcome = TradeOutcome(
                trade_id=match.group(1),
                instrument=match.group(2),
                side=match.group(3),
                entry_price=float(match.group(4)),
                exit_price=float(match.group(5)),
                quantity=float(match.group(6)),
                pnl=float(match.group(7)),
                slippage_bps=float(match.group(8)),
                timestamp_ns=int(match.group(9)),
                regime_id=match.group(10),
                alpha_signal=match.group(11),
                execution_quality=float(match.group(12))
            )
            block.outcomes.append(outcome)
        
        # Parse mistakes
        for match in self.MISTAKE_PATTERN.finditer(text):
            mistake = Mistake(
                mistake_id=match.group(1),
                trade_id=match.group(2) if match.group(2) else None,
                category=match.group(3),
                severity=float(match.group(4)),
                description=match.group(5).strip(),
                penalty_weight=float(match.group(6)),
                timestamp_ns=int(match.group(7)),
                suggested_correction=match.group(8).strip()
            )
            block.mistakes.append(mistake)
        
        # Parse regime memories
        for match in self.REGIME_PATTERN.finditer(text):
            try:
                chars_json = "{" + match.group(4).strip() + "}"
                characteristics = json.loads(chars_json.replace("'", '"'))
            except (json.JSONDecodeError, Exception):
                characteristics = {"raw": match.group(4).strip()}
            
            lessons = [l.strip().lstrip('- ').lstrip('* ') 
                      for l in match.group(9).strip().split('\n') if l.strip()]
            
            memory = RegimeMemory(
                regime_id=match.group(1),
                start_timestamp_ns=int(match.group(2)),
                end_timestamp_ns=int(match.group(3)),
                characteristics=characteristics,
                dominant_alpha=match.group(5),
                volatility_state=match.group(6),
                liquidity_state=match.group(7),
                lessons_learned=lessons
            )
            block.memories.append(memory)
        
        # Extract metadata
        block.metadata = {
            "block_type": self._detect_block_type(text),
            "timestamp_range": self._extract_timestamp_range(text)
        }
        
        return block
    
    def _detect_block_type(self, text: str) -> str:
        """Detect the type of SOUL block."""
        if '### Outcome:' in text:
            return "outcomes"
        elif '### Mistake:' in text:
            return "mistakes"
        elif '## Regime Memory:' in text:
            return "memories"
        return "unknown"
    
    def _extract_timestamp_range(self, text: str) -> tuple:
        """Extract timestamp range from block text."""
        timestamps = re.findall(r'Timestamp:\s*(\d+)', text)
        if timestamps:
            return (int(min(timestamps)), int(max(timestamps)))
        return (0, 0)
    
    @property
    def parsed_count(self) -> int:
        """Return number of blocks parsed."""
        return self._parsed_count
    
    async def parse_outcomes_only(self, filepath: str) -> AsyncGenerator[TradeOutcome, None]:
        """Stream only trade outcomes for memory-efficient processing."""
        async for block in self.parse_file(filepath):
            for outcome in block.outcomes:
                yield outcome
    
    async def parse_mistakes_only(self, filepath: str) -> AsyncGenerator[Mistake, None]:
        """Stream only mistakes for memory-efficient processing."""
        async for block in self.parse_file(filepath):
            for mistake in block.mistakes:
                yield mistake


async def main():
    """Example usage of SOULParser."""
    parser = SOULParser(chunk_size=8192)
    
    # Example: Parse and print summary
    async for block in parser.parse_file("SOUL.md"):
        print(f"Parsed block: {block.metadata['block_type']}")
        print(f"  Outcomes: {len(block.outcomes)}")
        print(f"  Mistakes: {len(block.mistakes)}")
        print(f"  Memories: {len(block.memories)}")


if __name__ == "__main__":
    asyncio.run(main())
