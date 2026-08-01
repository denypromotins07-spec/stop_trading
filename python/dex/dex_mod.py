"""
DEX Module Root.
Normalizes DEX quotes into Nautilus OrderBookDeltas and integrates with Smart Order Router.
"""

import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import logging
import time

from .jupiter_client import JupiterClient, Quote as JupiterQuote
from .oneinch_client import OneInchClient, OneInchQuote

logger = logging.getLogger(__name__)


class DEXType(Enum):
    """Supported DEX types."""
    JUPITER = "jupiter"
    ONEINCH = "oneinch"


class ChainType(Enum):
    """Supported blockchain types."""
    SOLANA = "solana"
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"


@dataclass
class NormalizedQuote:
    """Normalized quote across all DEXes."""
    dex_type: DEXType
    chain: ChainType
    input_token: str
    output_token: str
    input_amount: int
    output_amount: int
    price_impact_pct: float
    fee_estimate: int
    route_path: List[str]
    raw_quote: Any
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())

    @property
    def effective_price(self) -> Decimal:
        """Calculate effective execution price."""
        if self.input_amount == 0:
            return Decimal(0)
        return Decimal(self.output_amount) / Decimal(self.input_amount)

    @property
    def slippage_tolerance(self) -> float:
        """Estimate slippage tolerance based on price impact."""
        return max(0.005, self.price_impact_pct * 2)


@dataclass
class OrderBookDelta:
    """Nautilus-compatible order book delta."""
    instrument_id: str
    action: str  # "update", "insert", "delete"
    side: str  # "buy", "sell"
    price: Decimal
    quantity: Decimal
    order_id: Optional[str] = None
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())


class SmartOrderRouter:
    """
    Smart Order Router (SOR) that aggregates quotes from multiple DEXes
    and routes orders to the optimal execution venue.
    """

    def __init__(
        self,
        jupiter_api_key: Optional[str] = None,
        oneinch_api_key: Optional[str] = None,
    ):
        self.jupiter_client = JupiterClient()
        self.oneinch_client = (
            OneInchClient(api_key=oneinch_api_key or "")
            if oneinch_api_key
            else None
        )
        self._quote_cache: Dict[str, NormalizedQuote] = {}
        self._cache_ttl_ns = 500_000_000  # 500ms cache

    async def close(self):
        """Close all client connections."""
        await self.jupiter_client.close()
        if self.oneinch_client:
            await self.oneinch_client.close()

    async def get_best_route(
        self,
        input_token: str,
        output_token: str,
        amount: int,
        chains: Optional[List[ChainType]] = None,
    ) -> Optional[NormalizedQuote]:
        """
        Find the best execution route across all available DEXes.

        Args:
            input_token: Input token address/mint
            output_token: Output token address/mint
            amount: Amount to swap
            chains: List of chains to consider

        Returns:
            Best NormalizedQuote or None
        """
        chains = chains or [ChainType.SOLANA, ChainType.ETHEREUM]
        quotes: List[NormalizedQuote] = []

        # Fetch quotes from Jupiter (Solana)
        if ChainType.SOLANA in chains:
            try:
                jup_quote = await self.jupiter_client.get_quote(
                    input_mint=input_token,
                    output_mint=output_token,
                    amount=amount,
                )
                if jup_quote:
                    normalized = self._normalize_jupiter_quote(jup_quote)
                    quotes.append(normalized)
            except Exception as e:
                logger.error(f"Jupiter quote error: {e}")

        # Fetch quotes from 1inch (EVM)
        if self.oneinch_client and ChainType.ETHEREUM in chains:
            try:
                oneinch_quote = await self.oneinch_client.get_quote(
                    src_token=input_token,
                    dst_token=output_token,
                    amount=amount,
                )
                if oneinch_quote:
                    normalized = self._normalize_oneinch_quote(oneinch_quote)
                    quotes.append(normalized)
            except Exception as e:
                logger.error(f"1inch quote error: {e}")

        if not quotes:
            return None

        # Select best quote by output amount
        best_quote = max(quotes, key=lambda q: q.output_amount)
        return best_quote

    def _normalize_jupiter_quote(
        self,
        quote: JupiterQuote,
    ) -> NormalizedQuote:
        """Convert Jupiter quote to normalized format."""
        route_path = []
        for hop in quote.route_plan:
            if "swapInfo" in hop:
                route_path.append(hop["swapInfo"].get("label", "unknown"))

        return NormalizedQuote(
            dex_type=DEXType.JUPITER,
            chain=ChainType.SOLANA,
            input_token=quote.input_mint,
            output_token=quote.output_mint,
            input_amount=quote.in_amount,
            output_amount=quote.out_amount,
            price_impact_pct=quote.price_impact_pct,
            fee_estimate=quote.priority_fee_lamports,
            route_path=route_path,
            raw_quote=quote,
        )

    def _normalize_oneinch_quote(
        self,
        quote: OneInchQuote,
    ) -> NormalizedQuote:
        """Convert 1inch quote to normalized format."""
        route_path = []
        for protocol_list in quote.protocols:
            for protocol in protocol_list:
                route_path.append(protocol.get("name", "unknown"))

        # Convert gas to fee estimate (simplified)
        fee_estimate = quote.estimated_gas * quote.gas_price

        return NormalizedQuote(
            dex_type=DEXType.ONEINCH,
            chain=ChainType.ETHEREUM,
            input_token=quote.src_token,
            output_token=quote.dst_token,
            input_amount=quote.amount,
            output_amount=quote.to_amount,
            price_impact_pct=0.0,  # 1inch doesn't provide this directly
            fee_estimate=fee_estimate,
            route_path=route_path,
            raw_quote=quote,
        )

    def quote_to_orderbook_deltas(
        self,
        quote: NormalizedQuote,
        instrument_id: str,
    ) -> List[OrderBookDelta]:
        """
        Convert a DEX quote into Nautilus OrderBookDeltas for simulation.

        Args:
            quote: Normalized quote
            instrument_id: Nautilus instrument ID

        Returns:
            List of OrderBookDelta objects
        """
        deltas = []

        # Create synthetic bid/ask levels based on quote
        mid_price = quote.effective_price
        spread = Decimal(str(quote.slippage_tolerance))

        # Bid side (buying input token)
        bid_price = mid_price * (Decimal(1) - spread)
        bid_qty = Decimal(quote.input_amount) / Decimal(10 ** 18)  # Normalize decimals

        deltas.append(OrderBookDelta(
            instrument_id=instrument_id,
            action="update",
            side="buy",
            price=bid_price,
            quantity=bid_qty,
        ))

        # Ask side (selling output token)
        ask_price = mid_price * (Decimal(1) + spread)
        ask_qty = Decimal(quote.output_amount) / Decimal(10 ** 18)

        deltas.append(OrderBookDelta(
            instrument_id=instrument_id,
            action="update",
            side="sell",
            price=ask_price,
            quantity=ask_qty,
        ))

        return deltas

    async def execute_split_route(
        self,
        input_token: str,
        output_token: str,
        total_amount: int,
        num_splits: int = 3,
    ) -> List[NormalizedQuote]:
        """
        Split large order across multiple routes for better execution.

        Args:
            input_token: Input token
            output_token: Output token
            total_amount: Total amount to swap
            num_splits: Number of splits

        Returns:
            List of executed quotes
        """
        split_amount = total_amount // num_splits
        executed_quotes = []

        for i in range(num_splits):
            remaining = total_amount - (split_amount * i)
            current_amount = min(split_amount, remaining)

            if current_amount <= 0:
                break

            best_quote = await self.get_best_route(
                input_token=input_token,
                output_token=output_token,
                amount=current_amount,
            )

            if best_quote:
                executed_quotes.append(best_quote)
                # Small delay to avoid rate limits
                await asyncio.sleep(0.1)

        return executed_quotes

    def calculate_vwap(
        self,
        quotes: List[NormalizedQuote],
    ) -> Decimal:
        """
        Calculate volume-weighted average price from executed quotes.

        Args:
            quotes: List of executed quotes

        Returns:
            VWAP as Decimal
        """
        if not quotes:
            return Decimal(0)

        total_output = sum(q.output_amount for q in quotes)
        total_input = sum(q.input_amount for q in quotes)

        if total_input == 0:
            return Decimal(0)

        return Decimal(total_output) / Decimal(total_input)


# Module-level singleton instance
_sor_instance: Optional[SmartOrderRouter] = None


def get_smart_order_router(
    jupiter_api_key: Optional[str] = None,
    oneinch_api_key: Optional[str] = None,
) -> SmartOrderRouter:
    """Get or create the SmartOrderRouter singleton."""
    global _sor_instance
    if _sor_instance is None:
        _sor_instance = SmartOrderRouter(
            jupiter_api_key=jupiter_api_key,
            oneinch_api_key=oneinch_api_key,
        )
    return _sor_instance


async def shutdown_dex_module():
    """Gracefully shutdown the DEX module."""
    global _sor_instance
    if _sor_instance:
        await _sor_instance.close()
        _sor_instance = None
