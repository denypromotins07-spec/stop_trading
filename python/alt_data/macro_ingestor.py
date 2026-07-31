# Macro Data Ingestor for Alternative Data
# Nautilus DataClient ingesting macro events, funding rates, and whale alerts

from __future__ import annotations
import logging
import asyncio
from typing import Optional, Dict, Any, List
from decimal import Decimal

from nautilus_trader.adapters.base import LiveDataClient
from nautilus_trader.core.data import Data
from nautilus_trader.model.identifiers import InstrumentId, Venue, Symbol
from nautilus_trader.msgbus.bus import MessageBus
from nautilus_trader.common.component import Clock

log = logging.getLogger(__name__)


class MacroEventData(Data):
    """Custom data type for macroeconomic events."""
    
    def __init__(
        self,
        ts_event: int,
        ts_init: int,
        event_type: str,  # 'CPI', 'NFP', 'FOMC', 'GDP', etc.
        country: str,
        actual: Optional[float],
        forecast: Optional[float],
        previous: Optional[float],
        impact: str,  # 'LOW', 'MEDIUM', 'HIGH'
        description: str,
    ) -> None:
        super().__init__(instrument_id=None, ts_event=ts_event, ts_init=ts_init)
        self.event_type = event_type
        self.country = country
        self.actual = actual
        self.forecast = forecast
        self.previous = previous
        self.impact = impact
        self.description = description

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "MacroEventData",
            "ts_event": self.ts_event,
            "ts_init": self.ts_init,
            "event_type": self.event_type,
            "country": self.country,
            "actual": self.actual,
            "forecast": self.forecast,
            "previous": self.previous,
            "impact": self.impact,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MacroEventData":
        return cls(
            ts_event=data["ts_event"],
            ts_init=data["ts_init"],
            event_type=data["event_type"],
            country=data["country"],
            actual=data.get("actual"),
            forecast=data.get("forecast"),
            previous=data.get("previous"),
            impact=data["impact"],
            description=data["description"],
        )


class FundingRateData(Data):
    """Custom data type for perpetual swap funding rates."""
    
    def __init__(
        self,
        instrument_id: InstrumentId,
        ts_event: int,
        ts_init: int,
        funding_rate: float,
        predicted_rate: float,
        time_to_next: int,  # Seconds until next funding
        annualized_rate: float,
    ) -> None:
        super().__init__(instrument_id=instrument_id, ts_event=ts_event, ts_init=ts_init)
        self.funding_rate = funding_rate
        self.predicted_rate = predicted_rate
        self.time_to_next = time_to_next
        self.annualized_rate = annualized_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "FundingRateData",
            "instrument_id": str(self.instrument_id),
            "ts_event": self.ts_event,
            "ts_init": self.ts_init,
            "funding_rate": self.funding_rate,
            "predicted_rate": self.predicted_rate,
            "time_to_next": self.time_to_next,
            "annualized_rate": self.annualized_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FundingRateData":
        return cls(
            instrument_id=InstrumentId.from_str(data["instrument_id"]),
            ts_event=data["ts_event"],
            ts_init=data["ts_init"],
            funding_rate=data["funding_rate"],
            predicted_rate=data["predicted_rate"],
            time_to_next=data["time_to_next"],
            annualized_rate=data["annualized_rate"],
        )


class WhaleAlertData(Data):
    """Custom data type for large transaction alerts."""
    
    def __init__(
        self,
        ts_event: int,
        ts_init: int,
        symbol: str,
        amount: float,
        value_usd: float,
        from_address: str,
        to_address: str,
        transaction_type: str,  # 'TRANSFER', 'EXCHANGE_IN', 'EXCHANGE_OUT', 'MINING'
        exchange: Optional[str],
    ) -> None:
        super().__init__(instrument_id=None, ts_event=ts_event, ts_init=ts_init)
        self.symbol = symbol
        self.amount = amount
        self.value_usd = value_usd
        self.from_address = from_address
        self.to_address = to_address
        self.transaction_type = transaction_type
        self.exchange = exchange

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "WhaleAlertData",
            "ts_event": self.ts_event,
            "ts_init": self.ts_init,
            "symbol": self.symbol,
            "amount": self.amount,
            "value_usd": self.value_usd,
            "from_address": self.from_address,
            "to_address": self.to_address,
            "transaction_type": self.transaction_type,
            "exchange": self.exchange,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WhaleAlertData":
        return cls(
            ts_event=data["ts_event"],
            ts_init=data["ts_init"],
            symbol=data["symbol"],
            amount=data["amount"],
            value_usd=data["value_usd"],
            from_address=data["from_address"],
            to_address=data["to_address"],
            transaction_type=data["transaction_type"],
            exchange=data.get("exchange"),
        )


class MacroDataIngestor(LiveDataClient):
    """
    Nautilus DataClient for ingesting alternative data from Rust.
    Aligns asynchronous events to Nautilus global clock.
    """

    def __init__(
        self,
        msgbus: MessageBus,
        clock: Clock,
        instruments: Optional[List[InstrumentId]] = None,
    ) -> None:
        super().__init__(msgbus=msgbus, clock=clock)
        self.instruments = instruments or []
        self._running = False
        
        # Event queues (pre-allocated)
        self._macro_events: List[MacroEventData] = []
        self._funding_rates: Dict[InstrumentId, FundingRateData] = {}
        self._whale_alerts: List[WhaleAlertData] = []

    async def _connect(self) -> None:
        """Initialize connection to Rust data source."""
        log.info("MacroDataIngestor connecting...")
        # Connection handled via Rust IPC bridge
        self._running = True

    async def _disconnect(self) -> None:
        """Cleanup connection."""
        self._running = False
        log.info("MacroDataIngestor disconnected")

    def process_macro_event(
        self,
        event_type: str,
        country: str,
        actual: Optional[float],
        forecast: Optional[float],
        previous: Optional[float],
        impact: str,
        description: str,
    ) -> None:
        """Process incoming macro event from Rust."""
        ts_now = self._clock.timestamp_ns()
        
        event = MacroEventData(
            ts_event=ts_now,
            ts_init=ts_now,
            event_type=event_type,
            country=country,
            actual=actual,
            forecast=forecast,
            previous=previous,
            impact=impact,
            description=description,
        )
        
        self._macro_events.append(event)
        
        # Publish to MessageBus
        self._msgbus.publish(topic="data.macro", msg=event)
        log.debug(f"Macro event published: {event_type} ({country})")

    def process_funding_rate(
        self,
        instrument_id: InstrumentId,
        funding_rate: float,
        predicted_rate: float,
        time_to_next: int,
    ) -> None:
        """Process funding rate update from Rust."""
        ts_now = self._clock.timestamp_ns()
        annualized = funding_rate * 3 * 365  # Approximate annualization
        
        data = FundingRateData(
            instrument_id=instrument_id,
            ts_event=ts_now,
            ts_init=ts_now,
            funding_rate=funding_rate,
            predicted_rate=predicted_rate,
            time_to_next=time_to_next,
            annualized_rate=annualized,
        )
        
        self._funding_rates[instrument_id] = data
        
        # Publish to MessageBus
        self._msgbus.publish(topic="data.funding", msg=data)
        log.debug(f"Funding rate published: {instrument_id} = {funding_rate}")

    def process_whale_alert(
        self,
        symbol: str,
        amount: float,
        value_usd: float,
        from_address: str,
        to_address: str,
        transaction_type: str,
        exchange: Optional[str] = None,
    ) -> None:
        """Process whale alert from Rust."""
        ts_now = self._clock.timestamp_ns()
        
        alert = WhaleAlertData(
            ts_event=ts_now,
            ts_init=ts_now,
            symbol=symbol,
            amount=amount,
            value_usd=value_usd,
            from_address=from_address,
            to_address=to_address,
            transaction_type=transaction_type,
            exchange=exchange,
        )
        
        self._whale_alerts.append(alert)
        
        # Keep only recent alerts
        if len(self._whale_alerts) > 1000:
            self._whale_alerts = self._whale_alerts[-500:]
        
        # Publish to MessageBus
        self._msgbus.publish(topic="data.whale", msg=alert)
        log.debug(f"Whale alert published: {amount} {symbol} (${value_usd:,.0f})")

    async def _subscribe(self, topics: list[str]) -> None:
        """Subscribe to data topics."""
        await self._connect()
        log.info(f"Subscribed to macro data topics: {topics}")

    async def _unsubscribe(self, topics: list[str]) -> None:
        """Unsubscribe from data topics."""
        await self._disconnect()
        log.info(f"Unsubscribed from macro data topics: {topics}")

    def get_recent_macro_events(self, limit: int = 50) -> List[MacroEventData]:
        """Get recent macro events."""
        return self._macro_events[-limit:]

    def get_funding_rate(self, instrument_id: InstrumentId) -> Optional[FundingRateData]:
        """Get latest funding rate for instrument."""
        return self._funding_rates.get(instrument_id)

    def get_recent_whale_alerts(self, limit: int = 20) -> List[WhaleAlertData]:
        """Get recent whale alerts."""
        return self._whale_alerts[-limit:]
