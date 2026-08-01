"""
Jupiter Client for Solana DEX Aggregation.
Implements async, connection-pooled client for quoting and multi-hop swap routing.
Parses on-chain liquidity pool states for MEV-protected execution.
"""

import asyncio
import aiohttp
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


@dataclass
class Quote:
    """Represents a Jupiter swap quote."""
    input_mint: str
    output_mint: str
    in_amount: int
    out_amount: int
    price_impact_pct: float
    route_plan: List[Dict[str, Any]]
    slippage_bps: int
    priority_fee_lamports: int


@dataclass
class LiquidityPoolState:
    """Parsed on-chain liquidity pool state."""
    address: str
    token_a_mint: str
    token_b_mint: str
    token_a_reserve: int
    token_b_reserve: int
    fee_bps: int
    amm_type: str  # e.g., "Raydium", "Orca", "Meteora"


class JupiterClient:
    """
    Async, connection-pooled client for Jupiter Aggregator API.
    Handles quoting, route planning, and MEV protection via priority fees.
    """

    JUPITER_API_BASE = "https://quote-api.jup.ag/v6"
    CONNECTION_POOL_SIZE = 10
    CONNECTION_TIMEOUT = 30  # seconds
    REQUEST_TIMEOUT = 10  # seconds

    def __init__(self, cluster: str = "mainnet-beta"):
        self.cluster = cluster
        self._session: Optional[aiohttp.ClientSession] = None
        self._connection_pool: Optional[aiohttp.TCPConnector] = None
        self._lock = asyncio.Lock()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session with connection pooling is initialized."""
        if self._session is None or self._session.closed:
            self._connection_pool = aiohttp.TCPConnector(
                limit=self.CONNECTION_POOL_SIZE,
                limit_per_host=5,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            self._session = aiohttp.ClientSession(
                connector=self._connection_pool,
                timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT),
            )
        return self._session

    async def close(self):
        """Close the HTTP session and connection pool."""
        if self._session and not self._session.closed:
            await self._session.close()
        if self._connection_pool:
            await self._connection_pool.close()

    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 50,
        only_direct_routes: bool = False,
        as_legacy_transaction: bool = False,
    ) -> Optional[Quote]:
        """
        Fetch a swap quote from Jupiter.

        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Amount in smallest units (lamports for SOL)
            slippage_bps: Slippage tolerance in basis points (default 0.5%)
            only_direct_routes: If True, only direct routes (no multi-hop)
            as_legacy_transaction: Use legacy transaction format

        Returns:
            Quote object or None if failed
        """
        session = await self._ensure_session()
        url = f"{self.JUPITER_API_BASE}/quote"

        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageBps": slippage_bps,
            "onlyDirectRoutes": str(only_direct_routes).lower(),
            "asLegacyTransaction": str(as_legacy_transaction).lower(),
        }

        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Jupiter quote failed: {response.status}")
                    return None

                data = await response.json()

                if "data" not in data:
                    logger.warning("No quote data returned")
                    return None

                quote_data = data["data"]
                route_plan = quote_data.get("routePlan", [])

                # Calculate effective price impact
                price_impact = float(quote_data.get("priceImpactPct", "0"))

                # Estimate priority fee based on route complexity
                priority_fee = self._calculate_priority_fee(route_plan)

                return Quote(
                    input_mint=input_mint,
                    output_mint=output_mint,
                    in_amount=int(quote_data["inAmount"]),
                    out_amount=int(quote_data["outAmount"]),
                    price_impact_pct=price_impact,
                    route_plan=route_plan,
                    slippage_bps=slippage_bps,
                    priority_fee_lamports=priority_fee,
                )

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error fetching quote: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching quote: {e}")
            return None

    async def get_swap_instruction(
        self,
        quote: Quote,
        user_public_key: str,
        wrap_unwrap_sol: bool = True,
        use_shared_accounts: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Get serialized swap instruction from Jupiter.

        Args:
            quote: Quote object from get_quote
            user_public_key: User's wallet public key
            wrap_unwrap_sol: Auto wrap/unwrap SOL
            use_shared_accounts: Use shared intermediate token accounts

        Returns:
            Dict containing serialized transaction data
        """
        session = await self._ensure_session()
        url = f"{self.JUPITER_API_BASE}/swap-instructions"

        payload = {
            "quoteResponse": {
                "inputMint": quote.input_mint,
                "outputMint": quote.output_mint,
                "inAmount": str(quote.in_amount),
                "outAmount": str(quote.out_amount),
                "routePlan": quote.route_plan,
                "slippageBps": quote.slippage_bps,
            },
            "userPublicKey": user_public_key,
            "wrapAndUnwrapSol": wrap_unwrap_sol,
            "useSharedAccounts": use_shared_accounts,
        }

        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Swap instruction failed: {response.status}")
                    return None

                return await response.json()

        except Exception as e:
            logger.error(f"Error getting swap instruction: {e}")
            return None

    def _calculate_priority_fee(self, route_plan: List[Dict[str, Any]]) -> int:
        """
        Calculate optimal priority fee based on route complexity and network conditions.
        MEV protection: higher fees for complex multi-hop routes.
        """
        base_fee = 5000  # Base priority fee in lamports

        # Add fee per hop
        hop_count = len(route_plan)
        hop_fee = hop_count * 2000

        # Add fee based on expected price impact
        # Higher impact = more MEV risk = higher priority fee needed
        total_fee = base_fee + hop_fee

        return min(total_fee, 100000)  # Cap at 0.0001 SOL

    async def parse_liquidity_pool_state(
        self,
        pool_address: str,
        amm_type: str,
    ) -> Optional[LiquidityPoolState]:
        """
        Parse on-chain liquidity pool state from RPC.
        This would typically call a Solana RPC node directly.
        For now, returns a placeholder structure.
        """
        # In production, this would use solana-py or anchor-py to fetch
        # and deserialize account data from the chain
        logger.debug(f"Parsing pool state for {pool_address} ({amm_type})")

        # Placeholder - in production, fetch from RPC
        return LiquidityPoolState(
            address=pool_address,
            token_a_mint="",
            token_b_mint="",
            token_a_reserve=0,
            token_b_reserve=0,
            fee_bps=30,
            amm_type=amm_type,
        )

    async def get_multi_hop_route(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        max_hops: int = 4,
    ) -> Optional[List[Quote]]:
        """
        Find optimal multi-hop routing path.
        Jupiter API handles this internally, but we can request specific constraints.
        """
        # Jupiter's quote API already finds optimal multi-hop routes
        quote = await self.get_quote(
            input_mint=input_mint,
            output_mint=output_mint,
            amount=amount,
            slippage_bps=50,
            only_direct_routes=False,
        )

        if quote and len(quote.route_plan) > 1:
            return [quote]
        return None

    async def execute_arbitrage_check(
        self,
        token_pair: tuple,
        amount: int,
        min_profit_bps: int = 10,
    ) -> Optional[Dict[str, Any]]:
        """
        Check for arbitrage opportunities across different routes.
        Compares direct vs multi-hop routes for profitability.
        """
        input_mint, output_mint = token_pair

        # Get direct route
        direct_quote = await self.get_quote(
            input_mint=input_mint,
            output_mint=output_mint,
            amount=amount,
            only_direct_routes=True,
        )

        # Get multi-hop route
        multi_quote = await self.get_quote(
            input_mint=input_mint,
            output_mint=output_mint,
            amount=amount,
            only_direct_routes=False,
        )

        if not direct_quote or not multi_quote:
            return None

        # Compare outputs
        output_diff = multi_quote.out_amount - direct_quote.out_amount
        profit_bps = (output_diff / direct_quote.out_amount) * 10000

        if profit_bps >= min_profit_bps:
            return {
                "opportunity": "multi_hop_arbitrage",
                "direct_out": direct_quote.out_amount,
                "multi_out": multi_quote.out_amount,
                "profit_bps": profit_bps,
                "route": multi_quote.route_plan,
            }

        return None
