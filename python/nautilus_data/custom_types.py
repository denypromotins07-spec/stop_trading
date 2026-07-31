# Nautilus Custom Data Types for HFT System
# Inherits from nautilus_trader.core.data.Data for MessageBus routing

from __future__ import annotations
from typing import Any
from decimal import Decimal

from nautilus_trader.core.data import Data
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity


class OrderFlowData(Data):
    """
    Custom data type representing order flow events.
    Captures aggressive buy/sell volume, trade direction, and micro-price deviations.
    """
    
    def __init__(
        self,
        instrument_id: InstrumentId,
        ts_event: int,
        ts_init: int,
        aggressor_side: str,  # 'BUY' or 'SELL'
        volume: float,
        price: float,
        micro_price_deviation: float,
        trade_count: int,
    ) -> None:
        super().__init__(instrument_id=instrument_id, ts_event=ts_event, ts_init=ts_init)
        self.aggressor_side = aggressor_side
        self.volume = volume
        self.price = price
        self.micro_price_deviation = micro_price_deviation
        self.trade_count = trade_count

    def __repr__(self) -> str:
        return f"OrderFlowData({self.instrument_id}, {self.aggressor_side}, {self.volume})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "OrderFlowData",
            "instrument_id": str(self.instrument_id),
            "ts_event": self.ts_event,
            "ts_init": self.ts_init,
            "aggressor_side": self.aggressor_side,
            "volume": self.volume,
            "price": self.price,
            "micro_price_deviation": self.micro_price_deviation,
            "trade_count": self.trade_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrderFlowData:
        return cls(
            instrument_id=InstrumentId.from_str(data["instrument_id"]),
            ts_event=data["ts_event"],
            ts_init=data["ts_init"],
            aggressor_side=data["aggressor_side"],
            volume=data["volume"],
            price=data["price"],
            micro_price_deviation=data["micro_price_deviation"],
            trade_count=data["trade_count"],
        )


class SMCBlockData(Data):
    """
    Custom data type representing Smart Money Concept (SMC) blocks.
    Includes Order Blocks, Fair Value Gaps, and Liquidity Pools.
    """
    
    def __init__(
        self,
        instrument_id: InstrumentId,
        ts_event: int,
        ts_init: int,
        block_type: str,  # 'BULLISH_OB', 'BEARISH_OB', 'FVG', 'LIQUIDITY'
        start_price: float,
        end_price: float,
        strength: float,  # 0.0 to 1.0
        touched: bool,
    ) -> None:
        super().__init__(instrument_id=instrument_id, ts_event=ts_event, ts_init=ts_init)
        self.block_type = block_type
        self.start_price = start_price
        self.end_price = end_price
        self.strength = strength
        self.touched = touched

    def __repr__(self) -> str:
        return f"SMCBlockData({self.instrument_id}, {self.block_type}, {self.strength})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "SMCBlockData",
            "instrument_id": str(self.instrument_id),
            "ts_event": self.ts_event,
            "ts_init": self.ts_init,
            "block_type": self.block_type,
            "start_price": self.start_price,
            "end_price": self.end_price,
            "strength": self.strength,
            "touched": self.touched,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SMCBlockData:
        return cls(
            instrument_id=InstrumentId.from_str(data["instrument_id"]),
            ts_event=data["ts_event"],
            ts_init=data["ts_init"],
            block_type=data["block_type"],
            start_price=data["start_price"],
            end_price=data["end_price"],
            strength=data["strength"],
            touched=data["touched"],
        )


class RegimeStateData(Data):
    """
    Custom data type representing market regime states.
    Used for regime-switching models and RL context conditioning.
    """
    
    def __init__(
        self,
        instrument_id: InstrumentId,
        ts_event: int,
        ts_init: int,
        regime_type: str,  # 'TRENDING_UP', 'TRENDING_DOWN', 'MEAN_REVERTING', 'HIGH_VOL', 'LOW_VOL'
        confidence: float,
        volatility: float,
        trend_strength: float,
        mean_reversion_score: float,
    ) -> None:
        super().__init__(instrument_id=instrument_id, ts_event=ts_event, ts_init=ts_init)
        self.regime_type = regime_type
        self.confidence = confidence
        self.volatility = volatility
        self.trend_strength = trend_strength
        self.mean_reversion_score = mean_reversion_score

    def __repr__(self) -> str:
        return f"RegimeStateData({self.instrument_id}, {self.regime_type}, {self.confidence})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "RegimeStateData",
            "instrument_id": str(self.instrument_id),
            "ts_event": self.ts_event,
            "ts_init": self.ts_init,
            "regime_type": self.regime_type,
            "confidence": self.confidence,
            "volatility": self.volatility,
            "trend_strength": self.trend_strength,
            "mean_reversion_score": self.mean_reversion_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegimeStateData:
        return cls(
            instrument_id=InstrumentId.from_str(data["instrument_id"]),
            ts_event=data["ts_event"],
            ts_init=data["ts_init"],
            regime_type=data["regime_type"],
            confidence=data["confidence"],
            volatility=data["volatility"],
            trend_strength=data["trend_strength"],
            mean_reversion_score=data["mean_reversion_score"],
        )


# Registry for custom data types
CUSTOM_DATA_TYPES = [
    OrderFlowData,
    SMCBlockData,
    RegimeStateData,
]
