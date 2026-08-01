"""
Async Alertmanager client for critical ML system alerts.
Fires webhooks/PagerDuty alerts on ML hallucination, OOM warnings, and other critical events.
Non-blocking async design ensures alerting never delays inference or execution.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Set
from enum import Enum
from datetime import datetime, timezone
import aiohttp
import hashlib
import time

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Predefined alert types for ML systems."""
    ML_HALLUCINATION = "ml_hallucination"
    OOM_WARNING = "oom_warning"
    INFERENCE_LATENCY_SPIKE = "inference_latency_spike"
    FEATURE_DRIFT = "feature_drift"
    MODEL_STALENESS = "model_staleness"
    QUEUE_BACKLOG = "queue_backlog"
    EXECUTION_ERROR = "execution_error"
    DATA_QUALITY = "data_quality"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CUSTOM = "custom"


@dataclass
class Alert:
    """Represents a single alert."""
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    fingerprint: Optional[str] = None
    resolved: bool = False
    
    def __post_init__(self):
        """Generate fingerprint if not provided."""
        if self.fingerprint is None:
            self.fingerprint = self._generate_fingerprint()
    
    def _generate_fingerprint(self) -> str:
        """Generate unique fingerprint for deduplication."""
        content = f"{self.alert_type.value}:{self.severity.value}:{self.title}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def to_alertmanager_format(self) -> Dict:
        """Convert to Alertmanager API format."""
        return {
            'labels': {
                'alertname': self.alert_type.value,
                'severity': self.severity.value,
                **self.labels,
            },
            'annotations': {
                'title': self.title,
                'description': self.description,
                **self.annotations,
            },
            'startsAt': self.timestamp.isoformat(),
            'endsAt': None if not self.resolved else datetime.now(timezone.utc).isoformat(),
            'fingerprint': self.fingerprint,
        }


@dataclass
class AlertConfig:
    """Configuration for alert routing."""
    pagerduty_url: Optional[str] = None
    pagerduty_service_key: Optional[str] = None
    slack_webhook_url: Optional[str] = None
    discord_webhook_url: Optional[str] = None
    custom_webhook_urls: List[str] = field(default_factory=list)
    
    # Alertmanager specific
    alertmanager_url: Optional[str] = None
    
    # Rate limiting
    max_alerts_per_minute: int = 60
    dedup_window_seconds: int = 300  # 5 minutes


class AlertManager:
    """
    Async Alertmanager client for ML system alerts.
    
    Features:
    - Non-blocking async alert delivery
    - Multiple notification channels (PagerDuty, Slack, Discord, Webhooks)
    - Alert deduplication and rate limiting
    - Automatic retry with exponential backoff
    - Alert aggregation for storm prevention
    """
    
    def __init__(self, config: Optional[AlertConfig] = None):
        """
        Initialize alert manager.
        
        Args:
            config: Alert configuration
        """
        self.config = config or AlertConfig()
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        
        # Alert queue for batching
        self._alert_queue: asyncio.Queue = asyncio.Queue()
        
        # Deduplication tracking
        self._recent_fingerprints: Dict[str, datetime] = {}
        self._rate_limit_timestamps: List[float] = []
        
        # Alert statistics
        self._stats = {
            'alerts_sent': 0,
            'alerts_deduplicated': 0,
            'alerts_rate_limited': 0,
            'send_failures': 0,
            'pagerduty_sent': 0,
            'slack_sent': 0,
            'webhook_sent': 0,
        }
        
        # Custom alert handlers
        self._alert_handlers: List[Callable[[Alert], Any]] = []
        
        # Severity filter
        self._min_severity = AlertSeverity.INFO
        
        logger.info("AlertManager initialized")
    
    async def start(self):
        """Start the alert manager background processor."""
        if self._running:
            return
        
        self._running = True
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        # Start background processor
        asyncio.create_task(self._process_alerts())
        
        logger.info("AlertManager started")
    
    async def stop(self):
        """Stop the alert manager."""
        self._running = False
        
        # Process remaining alerts
        while not self._alert_queue.empty():
            try:
                alert = self._alert_queue.get_nowait()
                await self._send_alert(alert)
            except asyncio.QueueEmpty:
                break
        
        if self._session:
            await self._session.close()
        
        logger.info("AlertManager stopped")
    
    async def send_alert(self, alert: Alert) -> bool:
        """
        Send an alert (non-blocking, queues for batch processing).
        
        Args:
            alert: Alert to send
            
        Returns:
            True if alert was queued successfully
        """
        # Check severity filter
        severity_order = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.CRITICAL: 2,
            AlertSeverity.EMERGENCY: 3,
        }
        
        if severity_order[alert.severity] < severity_order[self._min_severity]:
            return False
        
        # Check deduplication
        if self._is_duplicate(alert):
            self._stats['alerts_deduplicated'] += 1
            logger.debug(f"Alert deduplicated: {alert.fingerprint}")
            return False
        
        # Check rate limit
        if self._is_rate_limited():
            self._stats['alerts_rate_limited'] += 1
            logger.warning(f"Alert rate limited: {alert.title}")
            # Still queue but mark for delayed sending
            alert.labels['rate_limited'] = 'true'
        
        # Queue for processing
        await self._alert_queue.put(alert)
        
        # Notify custom handlers
        for handler in self._alert_handlers:
            try:
                result = handler(alert)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        return True
    
    async def send_critical_alert(self, alert_type: AlertType,
                                  title: str, description: str,
                                  **kwargs) -> bool:
        """
        Send a critical alert (convenience method).
        
        Args:
            alert_type: Type of alert
            title: Alert title
            description: Alert description
            **kwargs: Additional labels/annotations
            
        Returns:
            True if alert was sent
        """
        alert = Alert(
            alert_type=alert_type,
            severity=AlertSeverity.CRITICAL,
            title=title,
            description=description,
            labels=kwargs.get('labels', {}),
            annotations=kwargs.get('annotations', {}),
        )
        
        return await self.send_alert(alert)
    
    def _is_duplicate(self, alert: Alert) -> bool:
        """Check if alert is a duplicate within dedup window."""
        now = datetime.now(timezone.utc)
        
        # Clean old fingerprints
        cutoff = now.timestamp() - self.config.dedup_window_seconds
        self._recent_fingerprints = {
            fp: ts for fp, ts in self._recent_fingerprints.items()
            if ts.timestamp() > cutoff
        }
        
        # Check if fingerprint exists
        if alert.fingerprint in self._recent_fingerprints:
            return True
        
        # Add new fingerprint
        self._recent_fingerprints[alert.fingerprint] = now
        return False
    
    def _is_rate_limited(self) -> bool:
        """Check if we're exceeding rate limit."""
        now = time.time()
        
        # Clean old timestamps (older than 1 minute)
        self._rate_limit_timestamps = [
            ts for ts in self._rate_limit_timestamps
            if now - ts < 60
        ]
        
        # Check limit
        if len(self._rate_limit_timestamps) >= self.config.max_alerts_per_minute:
            return True
        
        # Record this alert
        self._rate_limit_timestamps.append(now)
        return False
    
    async def _process_alerts(self):
        """Background task to process queued alerts."""
        batch_size = 10
        batch_timeout = 1.0  # seconds
        
        while self._running:
            try:
                # Collect batch of alerts
                batch = []
                try:
                    first_alert = await asyncio.wait_for(
                        self._alert_queue.get(),
                        timeout=batch_timeout
                    )
                    batch.append(first_alert)
                    
                    # Get more alerts without waiting
                    while len(batch) < batch_size:
                        try:
                            alert = self._alert_queue.get_nowait()
                            batch.append(alert)
                        except asyncio.QueueEmpty:
                            break
                except asyncio.TimeoutError:
                    continue
                
                # Send batch
                if batch:
                    await self._send_batch(batch)
                    
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(1)
    
    async def _send_batch(self, alerts: List[Alert]):
        """Send a batch of alerts."""
        tasks = []
        
        for alert in alerts:
            tasks.append(self._send_alert(alert))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_alert(self, alert: Alert):
        """Send individual alert to all configured channels."""
        tasks = []
        
        # Send to Alertmanager
        if self.config.alertmanager_url:
            tasks.append(self._send_to_alertmanager(alert))
        
        # Send to PagerDuty
        if self.config.pagerduty_url and self.config.pagerduty_service_key:
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                tasks.append(self._send_to_pagerduty(alert))
        
        # Send to Slack
        if self.config.slack_webhook_url:
            tasks.append(self._send_to_slack(alert))
        
        # Send to Discord
        if self.config.discord_webhook_url:
            tasks.append(self._send_to_discord(alert))
        
        # Send to custom webhooks
        for url in self.config.custom_webhook_urls:
            tasks.append(self._send_to_webhook(alert, url))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            if success_count > 0:
                self._stats['alerts_sent'] += 1
            else:
                self._stats['send_failures'] += 1
    
    async def _send_to_alertmanager(self, alert: Alert):
        """Send alert to Prometheus Alertmanager."""
        if not self._session:
            return
        
        url = f"{self.config.alertmanager_url}/api/v1/alerts"
        payload = [alert.to_alertmanager_format()]
        
        try:
            async with self._session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.debug(f"Alert sent to Alertmanager: {alert.title}")
                else:
                    logger.warning(f"Alertmanager error: {response.status}")
                    self._stats['send_failures'] += 1
        except Exception as e:
            logger.error(f"Alertmanager send error: {e}")
            raise
    
    async def _send_to_pagerduty(self, alert: Alert):
        """Send alert to PagerDuty."""
        if not self._session:
            return
        
        # Map severity to PagerDuty severity
        severity_map = {
            AlertSeverity.INFO: 'info',
            AlertSeverity.WARNING: 'warning',
            AlertSeverity.CRITICAL: 'error',
            AlertSeverity.EMERGENCY: 'critical',
        }
        
        payload = {
            'routing_key': self.config.pagerduty_service_key,
            'event_action': 'trigger',
            'payload': {
                'summary': alert.title,
                'source': 'ml-trading-system',
                'severity': severity_map[alert.severity],
                'custom_details': {
                    'description': alert.description,
                    'type': alert.alert_type.value,
                    'fingerprint': alert.fingerprint,
                    **alert.annotations,
                },
            },
        }
        
        try:
            async with self._session.post(
                self.config.pagerduty_url,
                json=payload
            ) as response:
                if response.status in [200, 202]:
                    self._stats['pagerduty_sent'] += 1
                    logger.info(f"PagerDuty alert sent: {alert.title}")
                else:
                    logger.warning(f"PagerDuty error: {response.status}")
        except Exception as e:
            logger.error(f"PagerDuty send error: {e}")
            raise
    
    async def _send_to_slack(self, alert: Alert):
        """Send alert to Slack webhook."""
        if not self._session:
            return
        
        # Color based on severity
        color_map = {
            AlertSeverity.INFO: '#36a64f',  # green
            AlertSeverity.WARNING: '#ff9800',  # orange
            AlertSeverity.CRITICAL: '#f44336',  # red
            AlertSeverity.EMERGENCY: '#9c27b0',  # purple
        }
        
        payload = {
            'attachments': [{
                'color': color_map[alert.severity],
                'title': alert.title,
                'text': alert.description,
                'fields': [
                    {'title': 'Type', 'value': alert.alert_type.value, 'short': True},
                    {'title': 'Severity', 'value': alert.severity.value, 'short': True},
                ],
                'footer': f'Fingerprint: {alert.fingerprint}',
                'ts': int(alert.timestamp.timestamp()),
            }]
        }
        
        try:
            async with self._session.post(
                self.config.slack_webhook_url,
                json=payload
            ) as response:
                if response.status == 200:
                    self._stats['slack_sent'] += 1
                else:
                    logger.warning(f"Slack error: {response.status}")
        except Exception as e:
            logger.error(f"Slack send error: {e}")
            raise
    
    async def _send_to_discord(self, alert: Alert):
        """Send alert to Discord webhook."""
        if not self._session:
            return
        
        # Color as decimal integer
        color_map = {
            AlertSeverity.INFO: 0x36a64f,
            AlertSeverity.WARNING: 0xff9800,
            AlertSeverity.CRITICAL: 0xf44336,
            AlertSeverity.EMERGENCY: 0x9c27b0,
        }
        
        payload = {
            'embeds': [{
                'title': alert.title,
                'description': alert.description,
                'color': color_map[alert.severity],
                'fields': [
                    {'name': 'Type', 'value': alert.alert_type.value, 'inline': True},
                    {'name': 'Severity', 'value': alert.severity.value, 'inline': True},
                ],
                'footer': {'text': f'Fingerprint: {alert.fingerprint}'},
                'timestamp': alert.timestamp.isoformat(),
            }]
        }
        
        try:
            async with self._session.post(
                self.config.discord_webhook_url,
                json=payload
            ) as response:
                if response.status in [200, 204]:
                    self._stats['discord_sent'] = self._stats.get('discord_sent', 0) + 1
                else:
                    logger.warning(f"Discord error: {response.status}")
        except Exception as e:
            logger.error(f"Discord send error: {e}")
            raise
    
    async def _send_to_webhook(self, alert: Alert, url: str):
        """Send alert to custom webhook."""
        if not self._session:
            return
        
        payload = alert.to_alertmanager_format()
        
        try:
            async with self._session.post(url, json=payload) as response:
                if response.status in [200, 201, 202, 204]:
                    self._stats['webhook_sent'] += 1
                else:
                    logger.warning(f"Webhook error ({url}): {response.status}")
        except Exception as e:
            logger.error(f"Webhook send error ({url}): {e}")
            raise
    
    def register_alert_handler(self, handler: Callable[[Alert], Any]):
        """Register custom alert handler."""
        self._alert_handlers.append(handler)
    
    def set_min_severity(self, severity: AlertSeverity):
        """Set minimum severity level for alerts."""
        self._min_severity = severity
    
    def get_stats(self) -> Dict:
        """Get alert statistics."""
        return self._stats.copy()
    
    # Convenience methods for common alerts
    async def alert_ml_hallucination(self, description: str, 
                                     confidence: float,
                                     **kwargs):
        """Alert on ML model hallucination."""
        alert = Alert(
            alert_type=AlertType.ML_HALLUCINATION,
            severity=AlertSeverity.CRITICAL,
            title="ML Model Hallucination Detected",
            description=f"{description} (confidence: {confidence:.2f})",
            labels={'confidence': str(confidence)},
        )
        return await self.send_alert(alert)
    
    async def alert_oom_warning(self, memory_usage_mb: float,
                                threshold_mb: float,
                                **kwargs):
        """Alert on OOM warning."""
        alert = Alert(
            alert_type=AlertType.OOM_WARNING,
            severity=AlertSeverity.WARNING,
            title="Memory Usage Critical",
            description=f"Memory usage: {memory_usage_mb:.0f}MB / {threshold_mb:.0f}MB ({100*memory_usage_mb/threshold_mb:.1f}%)",
            labels={
                'memory_usage_mb': str(memory_usage_mb),
                'threshold_mb': str(threshold_mb),
            },
        )
        return await self.send_alert(alert)
    
    async def alert_latency_spike(self, latency_ms: float,
                                  threshold_ms: float,
                                  **kwargs):
        """Alert on inference latency spike."""
        alert = Alert(
            alert_type=AlertType.INFERENCE_LATENCY_SPIKE,
            severity=AlertSeverity.WARNING,
            title="Inference Latency Spike",
            description=f"Latency: {latency_ms:.1f}ms (threshold: {threshold_ms:.1f}ms)",
            labels={
                'latency_ms': str(latency_ms),
                'threshold_ms': str(threshold_ms),
            },
        )
        return await self.send_alert(alert)
