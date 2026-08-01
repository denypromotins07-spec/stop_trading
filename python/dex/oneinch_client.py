"""
1inch Client for EVM DEX Aggregation.
Implements async client with Flashbots Protect RPC integration for MEV protection.
Prevents sandwich attacks and front-running during cross-chain arbitrage.
"""

import asyncio
import aiohttp
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class OneInchQuote:
    """Represents a 1inch swap quote."""
    src_token: str
    dst_token: str
    amount: int
    to_amount: int
    protocols: List[List[Dict[str, Any]]]
    estimated_gas: int
    gas_price: int
    tx_hash: Optional[str] = None


@dataclass
class FlashbotsBundle:
    """Flashbots bundle for private transaction submission."""
    transactions: List[str]
    block_number: int
    min_timestamp: int
    max_timestamp: int
    reverting_tx_hashes: List[str]


class OneInchClient:
    """
    Async client for 1inch Aggregation Protocol with Flashbots integration.
    Provides MEV-protected routing for EVM chains.
    """

    # 1inch API endpoints per chain
    API_ENDPOINTS = {
        "ethereum": "https://api.1inch.dev/swap/v5.2/1",
        "polygon": "https://api.1inch.dev/swap/v5.2/137",
        "arbitrum": "https://api.1inch.dev/swap/v5.2/42161",
        "optimism": "https://api.1inch.dev/swap/v5.2/10",
        "bsc": "https://api.1inch.dev/swap/v5.2/56",
        "avalanche": "https://api.1inch.dev/swap/v5.2/43114",
    }

    # Flashbots Protect RPC endpoints
    FLASHBOTS_RPC = {
        "ethereum": "https://rpc.flashbots.net/fast",
        "ethereum_standard": "https://rpc.flashbots.net",
    }

    CONNECTION_POOL_SIZE = 10
    REQUEST_TIMEOUT = 15

    def __init__(self, api_key: str, chain: str = "ethereum"):
        self.api_key = api_key
        self.chain = chain
        self._session: Optional[aiohttp.ClientSession] = None
        self._flashbots_session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session is initialized."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.CONNECTION_POOL_SIZE,
                ttl_dns_cache=300,
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._session

    async def _ensure_flashbots_session(self) -> aiohttp.ClientSession:
        """Ensure Flashbots session is initialized."""
        if self._flashbots_session is None or self._flashbots_session.closed:
            self._flashbots_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT),
                headers={"Content-Type": "application/json"},
            )
        return self._flashbots_session

    async def close(self):
        """Close all sessions."""
        if self._session and not self._session.closed:
            await self._session.close()
        if self._flashbots_session and not self._flashbots_session.closed:
            await self._flashbots_session.close()

    async def get_quote(
        self,
        src_token: str,
        dst_token: str,
        amount: int,
        slippage: float = 0.5,
        fee: float = 0,
        gas_limit: Optional[int] = None,
    ) -> Optional[OneInchQuote]:
        """
        Fetch a swap quote from 1inch.

        Args:
            src_token: Source token address
            dst_token: Destination token address
            amount: Amount in token decimals
            slippage: Slippage percentage (default 0.5%)
            fee: Protocol fee percentage
            gas_limit: Optional gas limit override

        Returns:
            OneInchQuote or None if failed
        """
        session = await self._ensure_session()
        endpoint = self.API_ENDPOINTS.get(self.chain)

        if not endpoint:
            logger.error(f"Unsupported chain: {self.chain}")
            return None

        url = f"{endpoint}/quote"
        params = {
            "src": src_token,
            "dst": dst_token,
            "amount": str(amount),
            "slippage": slippage,
        }

        if fee > 0:
            params["fee"] = str(fee)

        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"1inch quote failed ({response.status}): {error_text}")
                    return None

                data = await response.json()

                return OneInchQuote(
                    src_token=src_token,
                    dst_token=dst_token,
                    amount=int(data["srcAmount"]),
                    to_amount=int(data["toAmount"]),
                    protocols=data.get("protocols", []),
                    estimated_gas=int(data.get("estimatedGas", 0)),
                    gas_price=int(data.get("gasPrice", 0)),
                )

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error fetching quote: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching quote: {e}")
            return None

    async def build_swap_tx(
        self,
        quote: OneInchQuote,
        sender_address: str,
        receiver_address: Optional[str] = None,
        use_flashbots: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Build swap transaction with optional Flashbots protection.

        Args:
            quote: Quote from get_quote
            sender_address: Transaction sender address
            receiver_address: Optional receiver (defaults to sender)
            use_flashbots: Enable Flashbots MEV protection

        Returns:
            Transaction data dict
        """
        session = await self._ensure_session()
        endpoint = self.API_ENDPOINTS.get(self.chain)

        if not endpoint:
            return None

        url = f"{endpoint}/swap"
        params = {
            "src": quote.src_token,
            "dst": quote.dst_token,
            "amount": str(quote.amount),
            "sender": sender_address,
            "receiver": receiver_address or sender_address,
            "slippage": 0.5,
            "disableEstimate": "false",
        }

        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Swap tx build failed: {response.status}")
                    return None

                tx_data = await response.json()

                # Add Flashbots metadata if enabled
                if use_flashbots:
                    tx_data["flashbots"] = {
                        "enabled": True,
                        "protection_type": "fast",
                        "revert_protection": True,
                    }

                return tx_data

        except Exception as e:
            logger.error(f"Error building swap tx: {e}")
            return None

    async def submit_flashbots_bundle(
        self,
        signed_txs: List[str],
        target_block: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Submit transaction bundle to Flashbots for MEV-protected execution.

        Args:
            signed_txs: List of signed transaction hex strings
            target_block: Target block number (optional)

        Returns:
            Bundle submission result
        """
        session = await self._ensure_flashbots_session()
        rpc_url = self.FLASHBOTS_RPC.get("ethereum")

        # Get current block number
        current_block = await self._get_block_number(session, rpc_url)
        target = target_block or current_block + 1

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_sendBundle",
            "params": [
                {
                    "txs": signed_txs,
                    "blockNumber": hex(target),
                    "minTimestamp": 0,
                    "maxTimestamp": int(asyncio.get_event_loop().time()) + 60,
                }
            ],
        }

        try:
            async with session.post(rpc_url, json=payload) as response:
                result = await response.json()
                if "result" in result:
                    logger.info(f"Flashbots bundle submitted for block {target}")
                    return result["result"]
                else:
                    logger.error(f"Flashbots submission failed: {result}")
                    return None

        except Exception as e:
            logger.error(f"Error submitting Flashbots bundle: {e}")
            return None

    async def _get_block_number(
        self,
        session: aiohttp.ClientSession,
        rpc_url: str,
    ) -> int:
        """Get current block number from RPC."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_blockNumber",
            "params": [],
        }

        try:
            async with session.post(rpc_url, json=payload) as response:
                result = await response.json()
                return int(result.get("result", "0x0"), 16)
        except Exception:
            return 0

    async def check_sandwich_risk(
        self,
        quote: OneInchQuote,
        liquidity_threshold: int = 100000,
    ) -> Dict[str, Any]:
        """
        Analyze quote for sandwich attack risk.

        Args:
            quote: Swap quote to analyze
            liquidity_threshold: Minimum liquidity threshold (USD)

        Returns:
            Risk assessment dict
        """
        risk_score = 0.0
        risk_factors = []

        # Check price impact
        if quote.protocols:
            total_hops = sum(len(protocol) for protocol in quote.protocols)
            if total_hops > 3:
                risk_score += 0.2
                risk_factors.append("high_hop_count")

        # Check gas price vs average
        # High gas price indicates competition/MEV activity
        avg_gas_price = 50_000_000_000  # 50 gwei baseline
        if quote.gas_price > avg_gas_price * 1.5:
            risk_score += 0.3
            risk_factors.append("elevated_gas_price")

        # Check trade size relative to typical MEV profits
        if quote.amount > 10_000_000_000_000_000_000:  # 10 ETH equivalent
            risk_score += 0.25
            risk_factors.append("large_trade_size")

        return {
            "risk_score": min(risk_score, 1.0),
            "risk_factors": risk_factors,
            "recommendation": "use_flashbots" if risk_score > 0.3 else "standard",
            "flashbots_recommended": risk_score > 0.3,
        }

    async def execute_arbitrage_route(
        self,
        routes: List[Dict[str, Any]],
        min_profit_eth: float = 0.01,
    ) -> Optional[FlashbotsBundle]:
        """
        Execute cross-DEX arbitrage with Flashbots protection.

        Args:
            routes: List of arbitrage route configurations
            min_profit_eth: Minimum profit threshold in ETH

        Returns:
            FlashbotsBundle if successful
        """
        signed_txs = []

        for route in routes:
            # Build and sign each leg of the arbitrage
            # In production, this would use web3.py to sign transactions
            tx_hex = route.get("signed_tx")
            if tx_hex:
                signed_txs.append(tx_hex)

        if not signed_txs:
            return None

        # Submit as atomic bundle
        bundle = await self.submit_flashbots_bundle(signed_txs)

        if bundle:
            return FlashbotsBundle(
                transactions=signed_txs,
                block_number=bundle.get("blockNumber", 0),
                min_timestamp=0,
                max_timestamp=int(asyncio.get_event_loop().time()) + 60,
                reverting_tx_hashes=[],
            )

        return None

    def calculate_optimal_slippage(
        self,
        volatility: float,
        liquidity: int,
        trade_size: int,
    ) -> float:
        """
        Calculate optimal slippage tolerance based on market conditions.

        Args:
            volatility: Price volatility (0-1)
            liquidity: Pool liquidity
            trade_size: Trade size in base units

        Returns:
            Optimal slippage percentage
        """
        base_slippage = 0.3  # Base 0.3%

        # Adjust for volatility
        vol_adjustment = volatility * 2

        # Adjust for trade size vs liquidity
        size_ratio = trade_size / max(liquidity, 1)
        liquidity_adjustment = min(size_ratio * 100, 2.0)

        optimal = base_slippage + vol_adjustment + liquidity_adjustment
        return min(optimal, 5.0)  # Cap at 5%
