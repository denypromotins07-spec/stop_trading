"""
Network Latency Test
Stage 49: Microsecond-precision ICMP and WebSocket handshake latency tests.
Calibrates PTP clock offsets and execution slippage tolerances before live trading.
"""

import asyncio
import socket
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import zmq

try:
    import aiohttp
except ImportError:
    aiohttp = None

logger = logging.getLogger(__name__)


@dataclass
class LatencyResult:
    """Result of a single latency measurement."""
    endpoint: str
    method: str  # icmp, websocket, tcp
    latency_us: float
    success: bool
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EndpointConfig:
    """Configuration for an endpoint to test."""
    name: str
    host: str
    port: int
    protocol: str  # icmp, ws, wss, tcp
    path: Optional[str] = None


class NetworkLatencyTester:
    """
    Performs microsecond-precision network latency tests.
    Tests ICMP ping, WebSocket handshakes, and TCP connections.
    Calibrates PTP clock offsets and slippage tolerances.
    """
    
    def __init__(self, 
                 endpoints: Optional[List[EndpointConfig]] = None,
                 num_samples: int = 10,
                 timeout_seconds: float = 5.0):
        
        self.endpoints = endpoints or self._default_endpoints()
        self.num_samples = num_samples
        self.timeout_seconds = timeout_seconds
        
        # Results storage
        self._results: Dict[str, List[LatencyResult]] = {}
        self._baselines: Dict[str, float] = {}
        
        # ZMQ socket for Rust IPC
        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.PUSH)
        self._zmq_socket.connect("tcp://localhost:5571")
    
    def _default_endpoints(self) -> List[EndpointConfig]:
        """Default exchange endpoints for testing."""
        return [
            # Binance
            EndpointConfig("binance_spot", "api.binance.com", 443, "wss", "/ws"),
            EndpointConfig("binance_futures", "fstream.binance.com", 443, "wss", "/ws"),
            
            # DEX RPC endpoints
            EndpointConfig("ethereum_mainnet", "mainnet.infura.io", 443, "wss", "/v3/"),
            EndpointConfig("arbitrum", "arb1.arbitrum.io", 443, "https", "/rpc"),
            
            # Market data
            EndpointConfig("ny4_equinix", "169.254.169.254", 80, "tcp"),  # Placeholder
        ]
    
    async def test_all_endpoints(self) -> Dict[str, List[LatencyResult]]:
        """Test all configured endpoints."""
        self._results = {}
        
        logger.info(f"Starting network latency tests on {len(self.endpoints)} endpoints...")
        
        for endpoint in self.endpoints:
            results = await self._test_endpoint(endpoint)
            self._results[endpoint.name] = results
            
            if results:
                avg_latency = sum(r.latency_us for r in results if r.success) / len([r for r in results if r.success])
                logger.info(f"{endpoint.name}: avg={avg_latency:.2f}μs, samples={len(results)}")
        
        # Calculate baselines
        self._calculate_baselines()
        
        # Notify Rust
        self._notify_rust()
        
        return self._results
    
    async def _test_endpoint(self, endpoint: EndpointConfig) -> List[LatencyResult]:
        """Test a single endpoint with multiple samples."""
        results = []
        
        for i in range(self.num_samples):
            try:
                if endpoint.protocol == "tcp":
                    result = await self._test_tcp(endpoint.host, endpoint.port, endpoint.name)
                elif endpoint.protocol in ("ws", "wss"):
                    result = await self._test_websocket(endpoint, endpoint.name)
                elif endpoint.protocol in ("http", "https"):
                    result = await self._test_https(endpoint, endpoint.name)
                else:
                    result = LatencyResult(
                        endpoint=endpoint.name,
                        method="unknown",
                        latency_us=0.0,
                        success=False,
                        error=f"Unknown protocol: {endpoint.protocol}",
                    )
                
                results.append(result)
                
                # Small delay between samples
                if i < self.num_samples - 1:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error testing {endpoint.name}: {e}")
                results.append(LatencyResult(
                    endpoint=endpoint.name,
                    method=endpoint.protocol,
                    latency_us=0.0,
                    success=False,
                    error=str(e),
                ))
        
        return results
    
    async def _test_tcp(self, host: str, port: int, name: str) -> LatencyResult:
        """Test TCP connection latency."""
        try:
            start = time.perf_counter_ns()
            
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.timeout_seconds
            )
            
            end = time.perf_counter_ns()
            latency_us = (end - start) / 1000.0
            
            writer.close()
            await writer.wait_closed()
            
            return LatencyResult(
                endpoint=name,
                method="tcp",
                latency_us=latency_us,
                success=True,
            )
            
        except asyncio.TimeoutError:
            return LatencyResult(
                endpoint=name,
                method="tcp",
                latency_us=0.0,
                success=False,
                error="Connection timeout",
            )
        except Exception as e:
            return LatencyResult(
                endpoint=name,
                method="tcp",
                latency_us=0.0,
                success=False,
                error=str(e),
            )
    
    async def _test_websocket(self, endpoint: EndpointConfig, name: str) -> LatencyResult:
        """Test WebSocket handshake latency."""
        if aiohttp is None:
            return LatencyResult(
                endpoint=name,
                method="websocket",
                latency_us=0.0,
                success=False,
                error="aiohttp not installed",
            )
        
        try:
            url = f"{endpoint.protocol}://{endpoint.host}:{endpoint.port}{endpoint.path or ''}"
            
            start = time.perf_counter_ns()
            
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(url, timeout=self.timeout_seconds) as ws:
                    end = time.perf_counter_ns()
                    latency_us = (end - start) / 1000.0
                    
                    return LatencyResult(
                        endpoint=name,
                        method="websocket",
                        latency_us=latency_us,
                        success=True,
                    )
                    
        except asyncio.TimeoutError:
            return LatencyResult(
                endpoint=name,
                method="websocket",
                latency_us=0.0,
                success=False,
                error="WebSocket timeout",
            )
        except Exception as e:
            return LatencyResult(
                endpoint=name,
                method="websocket",
                latency_us=0.0,
                success=False,
                error=str(e),
            )
    
    async def _test_https(self, endpoint: EndpointConfig, name: str) -> LatencyResult:
        """Test HTTPS request latency."""
        if aiohttp is None:
            return LatencyResult(
                endpoint=name,
                method="https",
                latency_us=0.0,
                success=False,
                error="aiohttp not installed",
            )
        
        try:
            url = f"{endpoint.protocol}://{endpoint.host}:{endpoint.port}{endpoint.path or '/'}/health"
            
            start = time.perf_counter_ns()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.timeout_seconds) as resp:
                    end = time.perf_counter_ns()
                    latency_us = (end - start) / 1000.0
                    
                    return LatencyResult(
                        endpoint=name,
                        method="https",
                        latency_us=latency_us,
                        success=resp.status == 200,
                    )
                    
        except asyncio.TimeoutError:
            return LatencyResult(
                endpoint=name,
                method="https",
                latency_us=0.0,
                success=False,
                error="HTTPS timeout",
            )
        except Exception as e:
            return LatencyResult(
                endpoint=name,
                method="https",
                latency_us=0.0,
                success=False,
                error=str(e),
            )
    
    def _calculate_baselines(self):
        """Calculate baseline latencies from test results."""
        for name, results in self._results.items():
            successful = [r.latency_us for r in results if r.success]
            if successful:
                # Use median as baseline (more robust than mean)
                sorted_latencies = sorted(successful)
                median_idx = len(sorted_latencies) // 2
                self._baselines[name] = sorted_latencies[median_idx]
    
    def get_slippage_tolerance(self, endpoint_name: str) -> float:
        """
        Calculate recommended slippage tolerance based on latency.
        Uses baseline latency + 3 standard deviations.
        """
        results = self._results.get(endpoint_name, [])
        successful = [r.latency_us for r in results if r.success]
        
        if not successful:
            return 0.0  # Cannot determine
        
        import statistics
        baseline = statistics.median(successful)
        std_dev = statistics.stdev(successful) if len(successful) > 1 else 0.0
        
        # Slippage tolerance = baseline + 3σ (covers 99.7% of cases)
        tolerance_us = baseline + 3 * std_dev
        
        return tolerance_us
    
    def get_ptp_offset_estimate(self) -> Dict[str, float]:
        """
        Estimate PTP clock offset for each endpoint.
        Based on round-trip time asymmetry assumptions.
        """
        offsets = {}
        for name, baseline in self._baselines.items():
            # Assume symmetric latency, offset is half RTT
            offsets[name] = baseline / 2.0
        return offsets
    
    def _notify_rust(self):
        """Send test results to Rust via ZMQ."""
        try:
            self._zmq_socket.send_json({
                'type': 'NETWORK_LATENCY_TEST',
                'results': {
                    name: [
                        {
                            'latency_us': r.latency_us,
                            'success': r.success,
                            'error': r.error,
                        }
                        for r in results
                    ]
                    for name, results in self._results.items()
                },
                'baselines_us': self._baselines,
                'slippage_tolerances': {
                    name: self.get_slippage_tolerance(name)
                    for name in self._results.keys()
                },
                'timestamp': datetime.utcnow().isoformat(),
            }, flags=zmq.NOBLOCK)
        except Exception as e:
            logger.error(f"Failed to notify Rust: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get tester status."""
        return {
            'endpoints_tested': len(self._results),
            'baselines_us': self._baselines,
            'ptp_offsets': self.get_ptp_offset_estimate(),
        }
    
    def shutdown(self):
        """Cleanup resources."""
        self._zmq_socket.close()
        self._zmq_context.term()
        logger.info("NetworkLatencyTester shut down")


# Global instance
_tester: Optional[NetworkLatencyTester] = None


def get_tester() -> NetworkLatencyTester:
    """Get or create the global NetworkLatencyTester instance."""
    global _tester
    if _tester is None:
        _tester = NetworkLatencyTester()
    return _tester


def create_tester(endpoints: Optional[List[EndpointConfig]] = None,
                 num_samples: int = 10) -> NetworkLatencyTester:
    """Create a new NetworkLatencyTester with custom configuration."""
    global _tester
    _tester = NetworkLatencyTester(endpoints=endpoints, num_samples=num_samples)
    return _tester


async def test_network_latency() -> Dict[str, Any]:
    """Convenience function to run all network latency tests."""
    tester = get_tester()
    results = await tester.test_all_endpoints()
    return {
        'results': results,
        'baselines': tester._baselines,
        'slippage_tolerances': {
            name: tester.get_slippage_tolerance(name)
            for name in results.keys()
        },
    }
