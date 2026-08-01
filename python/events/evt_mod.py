"""
Event-Driven Alpha & News Scheduling Module Root
Manages event calendar, decay models for news impact, and Rust IPC bridge integration.
"""

from .news_scheduler import NewsScheduler, MacroEvent
from .entity_extractor import EntityExtractor, CoinTickerNER

__all__ = [
    "NewsScheduler",
    "MacroEvent",
    "EntityExtractor",
    "CoinTickerNER",
]
