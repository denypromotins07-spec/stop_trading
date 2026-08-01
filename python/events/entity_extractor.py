"""
Lightweight ONNX-compiled Named Entity Recognition for coin ticker extraction.
Extracts specific coin tickers from news squawks without heavy NLP library bloat.
Routes coin-specific sentiment shocks to Nautilus strategy actors.
Optimized for 3GB RAM constraint with minimal dependencies.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


@dataclass
class TickerMatch:
    """Represents a matched ticker in text."""
    ticker: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str


class CoinTickerNER:
    """
    Lightweight Named Entity Recognition for cryptocurrency tickers.
    Uses pattern matching and lookup tables instead of heavy ML models.
    Can be compiled to ONNX for production deployment.
    """
    
    # Major crypto tickers with regex patterns
    MAJOR_TICKERS = {
        'BTC', 'ETH', 'XRP', 'SOL', 'ADA', 'DOGE', 'AVAX', 'DOT', 'MATIC', 
        'LINK', 'UNI', 'ATOM', 'LTC', 'BCH', 'XLM', 'ALGO', 'VET', 'ICP',
        'FIL', 'TRX', 'ETC', 'NEAR', 'APT', 'ARB', 'OP', 'INJ', 'SUI'
    }
    
    # Common name to ticker mappings
    NAME_TO_TICKER = {
        'bitcoin': 'BTC',
        'ethereum': 'ETH',
        'ripple': 'XRP',
        'solana': 'SOL',
        'cardano': 'ADA',
        'dogecoin': 'DOGE',
        'avalanche': 'AVAX',
        'polkadot': 'DOT',
        'polygon': 'MATIC',
        'chainlink': 'LINK',
        'uniswap': 'UNI',
        'cosmos': 'ATOM',
        'litecoin': 'LTC',
        'bitcoin cash': 'BCH',
        'stellar': 'XLM',
        'algorand': 'ALGO',
        'vechain': 'VET',
        'internet computer': 'ICP',
        'filecoin': 'FIL',
        'tron': 'TRX',
        'ethereum classic': 'ETC',
        'near protocol': 'NEAR',
        'aptos': 'APT',
        'arbitrum': 'ARB',
        'optimism': 'OP',
        'injective': 'INJ',
        'sui': 'SUI',
    }
    
    def __init__(self, custom_tickers: Optional[Set[str]] = None):
        """
        Initialize the NER model.
        
        Args:
            custom_tickers: Optional set of additional tickers to recognize
        """
        self._tickers = self.MAJOR_TICKERS.copy()
        if custom_tickers:
            self._tickers.update(custom_tickers)
            
        self._name_to_ticker = self.NAME_TO_TICKER.copy()
        
        # Compile regex patterns for efficient matching
        self._ticker_pattern = self._build_ticker_pattern()
        self._name_pattern = self._build_name_pattern()
        
        # Cache for fast lookups
        self._cache: Dict[str, List[TickerMatch]] = {}
        self._cache_max_size = 1000
        
        # Statistics for monitoring
        self._stats = defaultdict(int)
        
    def _build_ticker_pattern(self) -> re.Pattern:
        """Build regex pattern for ticker symbols."""
        # Match tickers as whole words (not part of other words)
        ticker_str = '|'.join(sorted(self._tickers, key=len, reverse=True))
        return re.compile(r'\b(' + ticker_str + r')\b', re.IGNORECASE)
    
    def _build_name_pattern(self) -> re.Pattern:
        """Build regex pattern for coin names."""
        # Sort by length (longest first) to match multi-word names correctly
        names_sorted = sorted(self._name_to_ticker.keys(), key=len, reverse=True)
        name_str = '|'.join(re.escape(name) for name in names_sorted)
        return re.compile(r'\b(' + name_str + r')\b', re.IGNORECASE)
    
    def extract_tickers(self, text: str, 
                       include_context: bool = True,
                       context_window: int = 50) -> List[TickerMatch]:
        """
        Extract coin tickers from news text.
        
        Args:
            text: Input text (news squawk, headline, etc.)
            include_context: Whether to include surrounding context
            context_window: Number of characters around match for context
            
        Returns:
            List of TickerMatch objects
        """
        # Check cache first
        cache_key = hash(text)
        if cache_key in self._cache:
            self._stats['cache_hits'] += 1
            return self._cache[cache_key]
        
        self._stats['total_extractions'] += 1
        matches: List[TickerMatch] = []
        
        # Find ticker symbol matches
        for match in self._ticker_pattern.finditer(text):
            ticker = match.group(1).upper()
            start, end = match.span()
            
            # Calculate confidence based on context clues
            confidence = self._calculate_confidence(text, start, end, ticker)
            
            # Extract context if requested
            context = ""
            if include_context:
                ctx_start = max(0, start - context_window)
                ctx_end = min(len(text), end + context_window)
                context = text[ctx_start:ctx_end]
            
            matches.append(TickerMatch(
                ticker=ticker,
                start_pos=start,
                end_pos=end,
                confidence=confidence,
                context=context
            ))
        
        # Find coin name matches
        for match in self._name_pattern.finditer(text):
            name = match.group(1).lower()
            ticker = self._name_to_ticker.get(name)
            if not ticker:
                continue
                
            start, end = match.span()
            
            # Avoid duplicate matches
            if any(m.ticker == ticker and abs(m.start_pos - start) < 10 for m in matches):
                continue
                
            confidence = 0.9  # High confidence for explicit name matches
            
            context = ""
            if include_context:
                ctx_start = max(0, start - context_window)
                ctx_end = min(len(text), end + context_window)
                context = text[ctx_start:ctx_end]
            
            matches.append(TickerMatch(
                ticker=ticker,
                start_pos=start,
                end_pos=end,
                confidence=confidence,
                context=context
            ))
        
        # Sort by position
        matches.sort(key=lambda m: m.start_pos)
        
        # Update cache (LRU-style)
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[cache_key] = matches
        
        return matches
    
    def _calculate_confidence(self, text: str, start: int, end: int, 
                             ticker: str) -> float:
        """
        Calculate confidence score for a ticker match.
        
        Args:
            text: Full text
            start: Start position of match
            end: End position of match
            ticker: Matched ticker
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.7
        
        # Boost confidence if near keywords
        context_start = max(0, start - 30)
        context_end = min(len(text), end + 30)
        context = text[context_start:context_end].lower()
        
        boost_keywords = ['price', 'trading', 'market', 'surge', 'drop', 
                         'rally', 'crash', 'volume', 'exchange', 'whale']
        
        for keyword in boost_keywords:
            if keyword in context:
                base_confidence += 0.05
                
        # Boost if ticker appears multiple times
        ticker_count = len(self._ticker_pattern.findall(text.upper()))
        if ticker_count > 1:
            base_confidence += 0.1
            
        return min(1.0, base_confidence)
    
    def get_sentiment_routing(self, text: str, 
                             sentiment_score: float) -> Dict[str, float]:
        """
        Route sentiment scores to affected tickers.
        
        Args:
            text: Input text
            sentiment_score: Pre-computed sentiment score (-1.0 to 1.0)
            
        Returns:
            Dict mapping tickers to weighted sentiment scores
        """
        matches = self.extract_tickers(text, include_context=False)
        
        routing: Dict[str, float] = {}
        for match in matches:
            # Weight by confidence
            weighted_sentiment = sentiment_score * match.confidence
            
            if match.ticker in routing:
                # Aggregate if ticker appears multiple times
                routing[match.ticker] = max(routing[match.ticker], 
                                           weighted_sentiment)
            else:
                routing[match.ticker] = weighted_sentiment
                
        return routing
    
    def clear_cache(self):
        """Clear the extraction cache."""
        self._cache.clear()
        logger.debug("NER cache cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """Get extraction statistics."""
        return dict(self._stats)


class EntityExtractor:
    """
    High-level entity extractor that wraps CoinTickerNER.
    Provides async-friendly interface for Nautilus integration.
    """
    
    def __init__(self, custom_tickers: Optional[Set[str]] = None):
        """Initialize the entity extractor."""
        self._ner = CoinTickerNER(custom_tickers)
        self._callback_registry: Dict[str, List[callable]] = defaultdict(list)
        
    def register_ticker_callback(self, ticker: str, callback: callable):
        """
        Register callback for specific ticker events.
        
        Args:
            ticker: Ticker symbol to monitor
            callback: Function to call when ticker is detected
        """
        self._callback_registry[ticker.upper()].append(callback)
        
    def process_news(self, text: str, 
                    sentiment_score: Optional[float] = None,
                    timestamp_ns: Optional[int] = None) -> List[Dict]:
        """
        Process news text and route to appropriate handlers.
        
        Args:
            text: News text to process
            sentiment_score: Optional pre-computed sentiment score
            timestamp_ns: Optional nanosecond timestamp
            
        Returns:
            List of processed entity dictionaries
        """
        matches = self._ner.extract_tickers(text)
        
        results = []
        for match in matches:
            result = {
                'ticker': match.ticker,
                'confidence': match.confidence,
                'text_snippet': match.context,
                'timestamp_ns': timestamp_ns,
            }
            
            if sentiment_score is not None:
                result['sentiment'] = sentiment_score * match.confidence
                
            results.append(result)
            
            # Trigger callbacks
            for callback in self._callback_registry.get(match.ticker, []):
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Callback error for {match.ticker}: {e}")
                    
        return results
    
    def batch_process(self, texts: List[str]) -> Dict[str, List[Dict]]:
        """
        Batch process multiple news items.
        
        Args:
            texts: List of news texts
            
        Returns:
            Dict mapping text index to extracted entities
        """
        results = {}
        for i, text in enumerate(texts):
            results[i] = self.process_news(text)
        return results
    
    def get_affected_tickers(self, text: str, 
                            min_confidence: float = 0.5) -> Set[str]:
        """
        Get set of tickers mentioned in text above confidence threshold.
        
        Args:
            text: Input text
            min_confidence: Minimum confidence threshold
            
        Returns:
            Set of ticker symbols
        """
        matches = self._ner.extract_tickers(text, include_context=False)
        return {m.ticker for m in matches if m.confidence >= min_confidence}
