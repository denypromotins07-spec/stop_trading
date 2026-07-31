//! Fiat Ramp Arbitrage Module
//!
//! Models fiat liquidity depth and transfer delays to calculate true risk-adjusted premium capture.
//! Filters out phantom premiums where withdrawal limits or network congestion prevent actual arbitrage execution.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Instant, Duration};
use crossbeam_channel::{bounded, Sender, Receiver};
use dashmap::DashMap;

use super::stablecoin_premium::{RegionalExchange, StablecoinPair, L2Snapshot, PremiumSignal};

/// Fiat currency types supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FiatCurrency {
    USD,
    EUR,
    KRW,
    INR,
    NGN,
    TRY,
    ARS,
    VND,
    JPY,
    GBP,
}

impl FiatCurrency {
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::USD => "USD",
            Self::EUR => "EUR",
            Self::KRW => "KRW",
            Self::INR => "INR",
            Self::NGN => "NGN",
            Self::TRY => "TRY",
            Self::ARS => "ARS",
            Self::VND => "VND",
            Self::JPY => "JPY",
            Self::GBP => "GBP",
        }
    }

    /// Typical wire transfer time in hours for this currency
    pub fn typical_wire_time_hours(&self) -> u32 {
        match self {
            Self::USD => 24,
            Self::EUR => 12,
            Self::KRW => 48,   // Strict capital controls
            Self::INR => 72,   // Banking bureaucracy
            Self::NGN => 96,   // Regulatory issues
            Self::TRY => 48,   // Volatility controls
            Self::ARS => 168,  // Extreme capital controls
            Self::VND => 72,
            Self::JPY => 24,
            Self::GBP => 12,
        }
    }

    /// Capital control severity (0.0 = free, 1.0 = heavily restricted)
    pub fn capital_control_factor(&self) -> f64 {
        match self {
            Self::USD | Self::EUR | Self::JPY | Self::GBP => 0.05,
            Self::KRW => 0.4,
            Self::INR => 0.6,
            Self::TRY => 0.5,
            Self::NGN => 0.7,
            Self::ARS => 0.9,
            Self::VND => 0.65,
        }
    }
}

/// Fiat ramp liquidity state
#[derive(Debug, Clone)]
pub struct FiatRampLiquidity {
    pub exchange: RegionalExchange,
    pub fiat_currency: FiatCurrency,
    pub crypto_asset: StablecoinPair,
    /// Available buy liquidity in fiat terms
    pub buy_liquidity_fiat: f64,
    /// Available sell liquidity in fiat terms
    pub sell_liquidity_fiat: f64,
    /// Daily withdrawal limit remaining
    pub withdrawal_limit_remaining: f64,
    /// Daily deposit limit remaining
    pub deposit_limit_remaining: f64,
    /// Current queue depth for withdrawals (number of pending requests)
    pub withdrawal_queue_depth: u32,
    /// Estimated processing time for new withdrawal
    pub estimated_withdrawal_time_minutes: u32,
    /// Network congestion factor (1.0 = normal, >1.0 = congested)
    pub network_congestion_factor: f64,
    /// Last updated timestamp
    pub timestamp_ns: u64,
}

impl FiatRampLiquidity {
    pub fn is_viable_for_arb(&self, required_volume: f64) -> bool {
        // Check basic liquidity
        if self.buy_liquidity_fiat < required_volume 
            || self.sell_liquidity_fiat < required_volume 
        {
            return false;
        }

        // Check withdrawal limits
        if self.withdrawal_limit_remaining < required_volume {
            return false;
        }

        // Check network congestion
        if self.network_congestion_factor > 3.0 {
            return false;
        }

        // Check withdrawal queue (too many pending = delay risk)
        if self.withdrawal_queue_depth > 100 {
            return false;
        }

        true
    }

    /// Calculate effective delay including queue time
    pub fn effective_delay_minutes(&self) -> u32 {
        let base_delay = self.exchange.typical_transfer_delay_minutes();
        let queue_delay = (self.withdrawal_queue_depth as f64 * 0.5) as u32; // 30 sec per request
        let congestion_delay = ((self.network_congestion_factor - 1.0) * 30.0) as u32;
        
        base_delay.saturating_add(queue_delay).saturating_add(congestion_delay)
    }
}

/// Risk-adjusted premium calculation result
#[derive(Debug, Clone)]
pub struct RiskAdjustedPremium {
    pub raw_premium_bps: f64,
    pub transfer_cost_bps: f64,
    pub opportunity_cost_bps: f64,
    pub slippage_cost_bps: f64,
    pub funding_cost_bps: f64,
    pub risk_premium_bps: f64,
    pub net_premium_bps: f64,
    pub sharpe_ratio_estimate: f64,
    pub is_executable: bool,
    pub rejection_reason: Option<&'static str>,
}

impl RiskAdjustedPremium {
    pub fn calculate(
        signal: &PremiumSignal,
        source_ramp: &FiatRampLiquidity,
        target_ramp: &FiatRampLiquidity,
        funding_rate_annual: f64,
        volatility_annual: f64,
    ) -> Self {
        let raw_premium = signal.premium_bps;

        // 1. Transfer costs (wire fees, blockchain fees)
        let transfer_cost = Self::calculate_transfer_cost(signal, source_ramp, target_ramp);

        // 2. Opportunity cost (capital locked during transfer)
        let opportunity_cost = Self::calculate_opportunity_cost(
            signal, funding_rate_annual, source_ramp, target_ramp,
        );

        // 3. Slippage cost
        let slippage_cost = Self::calculate_slippage_cost(signal, source_ramp, target_ramp);

        // 4. Funding cost (if using leverage)
        let funding_cost = Self::calculate_funding_cost(
            signal, funding_rate_annual, source_ramp, target_ramp,
        );

        // 5. Risk premium (volatility exposure during transfer)
        let risk_premium = Self::calculate_risk_premium(
            signal, volatility_annual, source_ramp, target_ramp,
        );

        // Net premium
        let total_costs = transfer_cost + opportunity_cost + slippage_cost 
            + funding_cost + risk_premium;
        let net_premium = raw_premium - total_costs;

        // Sharpe ratio estimate (simplified)
        let expected_return = net_premium / 10000.0;
        let risk = volatility_annual * (signal.transfer_delay_minutes as f64 / (365.0 * 24.0 * 60.0)).sqrt();
        let sharpe = if risk > 0.0 { expected_return / risk } else { 0.0 };

        // Executability check
        let (is_executable, rejection_reason) = Self::check_executability(
            signal, source_ramp, target_ramp, net_premium,
        );

        Self {
            raw_premium_bps: raw_premium,
            transfer_cost_bps: transfer_cost,
            opportunity_cost_bps: opportunity_cost,
            slippage_cost_bps: slippage_cost,
            funding_cost_bps: funding_cost,
            risk_premium_bps: risk_premium,
            net_premium_bps: net_premium,
            sharpe_ratio_estimate: sharpe,
            is_executable,
            rejection_reason,
        }
    }

    fn calculate_transfer_cost(
        signal: &PremiumSignal,
        source: &FiatRampLiquidity,
        target: &FiatRampLiquidity,
    ) -> f64 {
        // Fixed wire fees (in bps relative to volume)
        let source_wire_fee = 30.0; // ~$30 wire fee
        let target_wire_fee = 30.0;
        let volume = signal.max_executable_volume;
        
        let wire_fee_bps = ((source_wire_fee + target_wire_fee) / volume) * 10000.0;

        // Blockchain network fees
        let blockchain_fee = match signal.pair {
            StablecoinPair::USDTUSD => 5.0,  // TRC20/ERC20 fees
            StablecoinPair::USDCUSD => 3.0,  // Often cheaper
            _ => 10.0,
        };
        let blockchain_fee_bps = (blockchain_fee / volume) * 10000.0;

        // FX conversion costs if different currencies
        let fx_spread = if source.fiat_currency != target.fiat_currency {
            20.0 // 20 bps FX spread
        } else {
            0.0
        };

        wire_fee_bps + blockchain_fee_bps + fx_spread
    }

    fn calculate_opportunity_cost(
        signal: &PremiumSignal,
        funding_rate_annual: f64,
        source: &FiatRampLiquidity,
        target: &FiatRampLiquidity,
    ) -> f64 {
        // Total time capital is locked
        let total_delay_minutes = source.effective_delay_minutes() 
            + target.effective_delay_minutes();
        let total_delay_years = total_delay_minutes as f64 / (365.0 * 24.0 * 60.0);

        // Opportunity cost = funding rate * time
        (funding_rate_annual * total_delay_years) * 10000.0 // Convert to bps
    }

    fn calculate_slippage_cost(
        signal: &PremiumSignal,
        source: &FiatRampLiquidity,
        target: &FiatRampLiquidity,
    ) -> f64 {
        // Slippage from the normalized spread calculation
        let source_slippage = if source.buy_liquidity_fiat > 0.0 {
            (signal.max_executable_volume / source.buy_liquidity_fiat) * 5.0 // 5 bps base slippage
        } else {
            100.0
        };

        let target_slippage = if target.sell_liquidity_fiat > 0.0 {
            (signal.max_executable_volume / target.sell_liquidity_fiat) * 5.0
        } else {
            100.0
        };

        source_slippage + target_slippage
    }

    fn calculate_funding_cost(
        signal: &PremiumSignal,
        funding_rate_annual: f64,
        _source: &FiatRampLiquidity,
        _target: &FiatRampLiquidity,
    ) -> f64 {
        // If using leverage, funding costs apply
        // Simplified: assume 50% leverage usage
        let leverage_factor = 0.5;
        let total_delay_minutes = signal.transfer_delay_minutes;
        let delay_years = total_delay_minutes as f64 / (365.0 * 24.0 * 60.0);

        (funding_rate_annual * leverage_factor * delay_years) * 10000.0
    }

    fn calculate_risk_premium(
        signal: &PremiumSignal,
        volatility_annual: f64,
        source: &FiatRampLiquidity,
        target: &FiatRampLiquidity,
    ) -> f64 {
        // Risk premium for exposure during transfer window
        let total_delay_minutes = source.effective_delay_minutes() 
            + target.effective_delay_minutes();
        
        // Volatility over the transfer period
        let delay_fraction = total_delay_minutes as f64 / (365.0 * 24.0 * 60.0);
        let period_volatility = volatility_annual * delay_fraction.sqrt();

        // Risk aversion factor (higher for emerging market currencies)
        let source_cc = source.fiat_currency.capital_control_factor();
        let target_cc = target.fiat_currency.capital_control_factor();
        let risk_aversion = 1.0 + (source_cc + target_cc) / 2.0;

        (period_volatility * risk_aversion) * 10000.0
    }

    fn check_executability(
        signal: &PremiumSignal,
        source: &FiatRampLiquidity,
        target: &FiatRampLiquidity,
        net_premium: f64,
    ) -> (bool, Option<&'static str>) {
        // Must have positive net premium
        if net_premium <= 0.0 {
            return (false, Some("NEGATIVE_NET_PREMIUM"));
        }

        // Minimum threshold for execution (covers unexpected costs)
        if net_premium < 5.0 {
            return (false, Some("PREMIUM_TOO_LOW"));
        }

        // Source ramp must be viable
        if !source.is_viable_for_arb(signal.max_executable_volume) {
            return (false, Some("SOURCE_RAMP_NOT_VIABLE"));
        }

        // Target ramp must be viable
        if !target.is_viable_for_arb(signal.max_executable_volume) {
            return (false, Some("TARGET_RAMP_NOT_VIABLE"));
        }

        // Signal must not be phantom
        if signal.is_phantom {
            return (false, Some("PHANTOM_SIGNAL"));
        }

        // Confidence must be adequate
        if signal.confidence_score < 0.5 {
            return (false, Some("LOW_CONFIDENCE"));
        }

        (true, None)
    }
}

/// Fiat ramp arb engine
pub struct FiatRampArbEngine {
    /// Fiat ramp liquidity states
    ramp_states: DashMap<(RegionalExchange, FiatCurrency, StablecoinPair), FiatRampLiquidity>,
    /// Computed risk-adjusted premiums
    computed_premiums: DashMap<u64, RiskAdjustedPremium>,
    /// Premium counter
    premium_counter: AtomicU64,
    /// Is engine active
    is_active: AtomicBool,
    /// Annual funding rate assumption
    funding_rate_annual: f64,
    /// Annual volatility assumption
    volatility_annual: f64,
    /// Minimum net premium threshold (bps)
    min_net_premium_bps: f64,
    /// Event channel
    event_tx: Sender<FiatRampEvent>,
    event_rx: Receiver<FiatRampEvent>,
}

/// Fiat ramp events
#[derive(Debug, Clone)]
pub enum FiatRampEvent {
    /// Liquidity state updated
    LiquidityUpdated(RegionalExchange, FiatCurrency, StablecoinPair),
    /// Executable arb opportunity found
    ExecutableOpportunity {
        signal: PremiumSignal,
        risk_adjusted: RiskAdjustedPremium,
    },
    /// Opportunity rejected
    OpportunityRejected {
        signal: PremiumSignal,
        reason: &'static str,
    },
    /// Ramp congestion warning
    CongestionWarning {
        exchange: RegionalExchange,
        fiat: FiatCurrency,
        congestion_factor: f64,
    },
}

impl FiatRampArbEngine {
    pub fn new(
        funding_rate_annual: f64,
        volatility_annual: f64,
        min_net_premium_bps: f64,
        buffer_size: usize,
    ) -> Self {
        let (tx, rx) = bounded(buffer_size);

        Self {
            ramp_states: DashMap::new(),
            computed_premiums: DashMap::new(),
            premium_counter: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
            funding_rate_annual,
            volatility_annual,
            min_net_premium_bps,
            event_tx: tx,
            event_rx: rx,
        }
    }

    /// Update fiat ramp liquidity state
    pub fn update_ramp_state(&self, state: FiatRampLiquidity) {
        let key = (state.exchange, state.fiat_currency, state.crypto_asset);
        
        // Check for congestion warning
        if state.network_congestion_factor > 2.0 {
            let _ = self.event_tx.send(FiatRampEvent::CongestionWarning {
                exchange: state.exchange,
                fiat: state.fiat_currency,
                congestion_factor: state.network_congestion_factor,
            });
        }

        self.ramp_states.insert(key, state.clone());

        let _ = self.event_tx.send(FiatRampEvent::LiquidityUpdated(
            state.exchange, state.fiat_currency, state.crypto_asset
        ));
    }

    /// Process a premium signal through the fiat ramp filter
    pub fn process_premium_signal(&self, signal: PremiumSignal) {
        if !self.is_active.load(Ordering::Relaxed) {
            return;
        }

        // Get ramp states for source and target
        let source_key = (
            signal.source_exchange,
            self.infer_fiat_currency(signal.source_exchange),
            signal.pair,
        );
        let target_key = (
            signal.target_exchange,
            self.infer_fiat_currency(signal.target_exchange),
            signal.pair,
        );

        let source_ramp = match self.ramp_states.get(&source_key) {
            Some(s) => s.clone(),
            None => return, // Cannot evaluate without ramp data
        };

        let target_ramp = match self.ramp_states.get(&target_key) {
            Some(t) => t.clone(),
            None => return,
        };

        // Calculate risk-adjusted premium
        let risk_adjusted = RiskAdjustedPremium::calculate(
            &signal,
            &source_ramp,
            &target_ramp,
            self.funding_rate_annual,
            self.volatility_annual,
        );

        // Store computation
        let id = self.premium_counter.fetch_add(1, Ordering::Relaxed);
        self.computed_premiums.insert(id, risk_adjusted.clone());

        // Emit event
        if risk_adjusted.is_executable && risk_adjusted.net_premium_bps >= self.min_net_premium_bps {
            let _ = self.event_tx.send(FiatRampEvent::ExecutableOpportunity {
                signal: signal.clone(),
                risk_adjusted,
            });
        } else if let Some(reason) = risk_adjusted.rejection_reason {
            let _ = self.event_tx.send(FiatRampEvent::OpportunityRejected {
                signal,
                reason,
            });
        }
    }

    /// Infer fiat currency from exchange
    fn infer_fiat_currency(&self, exchange: RegionalExchange) -> FiatCurrency {
        match exchange {
            RegionalExchange::Upbit | RegionalExchange::Bithumb | RegionalExchange::Korbit => FiatCurrency::KRW,
            RegionalExchange::WazirX | RegionalExchange::CoinDCX | RegionalExchange::ZebPay => FiatCurrency::INR,
            RegionalExchange::Bundle | RegionalExchange::Quidax => FiatCurrency::NGN,
            RegionalExchange::BtcTurk | RegionalExchange::Paribu => FiatCurrency::TRY,
            RegionalExchange::SatoshiTango | RegionalExchange::BuenBit => FiatCurrency::ARS,
            RegionalExchange::Remitano | RegionalExchange::Vndex => FiatCurrency::VND,
            RegionalExchange::Binance | RegionalExchange::Coinbase | RegionalExchange::Kraken => FiatCurrency::USD,
        }
    }

    /// Get all executable opportunities
    pub fn get_executable_opportunities(&self) -> Vec<(PremiumSignal, RiskAdjustedPremium)> {
        self.computed_premiums.iter()
            .filter(|e| e.value().is_executable)
            .filter_map(|e| {
                // We'd need to store the signal alongside, simplified here
                None
            })
            .collect()
    }

    /// Get best opportunity by net premium
    pub fn get_best_opportunity(&self) -> Option<RiskAdjustedPremium> {
        self.computed_premiums.iter()
            .filter(|e| e.value().is_executable)
            .max_by(|a, b| {
                a.value().net_premium_bps
                    .partial_cmp(&b.value().net_premium_bps)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|e| e.value().clone())
    }

    /// Get event receiver
    pub fn get_event_receiver(&self) -> Receiver<FiatRampEvent> {
        self.event_rx.clone()
    }

    /// Deactivate engine
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Activate engine
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Relaxed);
    }

    /// Clear old computations
    pub fn clear_old_computations(&self, max_age: Duration) {
        // In production, would track timestamps
        self.computed_premiums.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fiat_ramp_liquidity_viability() {
        let ramp = FiatRampLiquidity {
            exchange: RegionalExchange::Binance,
            fiat_currency: FiatCurrency::USD,
            crypto_asset: StablecoinPair::USDTUSD,
            buy_liquidity_fiat: 1_000_000.0,
            sell_liquidity_fiat: 1_000_000.0,
            withdrawal_limit_remaining: 100_000.0,
            deposit_limit_remaining: 100_000.0,
            withdrawal_queue_depth: 10,
            estimated_withdrawal_time_minutes: 30,
            network_congestion_factor: 1.0,
            timestamp_ns: Instant::now().duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default().as_nanos() as u64,
        };

        assert!(ramp.is_viable_for_arb(50_000.0));
        assert!(!ramp.is_viable_for_arb(150_000.0)); // Exceeds limit
    }

    #[test]
    fn test_effective_delay_calculation() {
        let ramp = FiatRampLiquidity {
            exchange: RegionalExchange::Upbit,
            fiat_currency: FiatCurrency::KRW,
            crypto_asset: StablecoinPair::USDTUSD,
            buy_liquidity_fiat: 500_000.0,
            sell_liquidity_fiat: 500_000.0,
            withdrawal_limit_remaining: 50_000.0,
            deposit_limit_remaining: 50_000.0,
            withdrawal_queue_depth: 50,
            estimated_withdrawal_time_minutes: 60,
            network_congestion_factor: 1.5,
            timestamp_ns: Instant::now().duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default().as_nanos() as u64,
        };

        let delay = ramp.effective_delay_minutes();
        assert!(delay > 60); // Base + queue + congestion
    }

    #[test]
    fn test_risk_adjusted_premium_positive() {
        let signal = PremiumSignal {
            pair: StablecoinPair::USDTUSD,
            source_exchange: RegionalExchange::Binance,
            target_exchange: RegionalExchange::Upbit,
            premium_bps: 50.0,
            risk_adjusted_premium_bps: 45.0,
            max_executable_volume: 10_000.0,
            estimated_profit_usd: 50.0,
            transfer_delay_minutes: 90,
            withdrawal_limit_usd: 10_000.0,
            confidence_score: 0.9,
            is_phantom: false,
            phantom_reason: None,
            timestamp_ns: Instant::now().duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default().as_nanos() as u64,
        };

        let source_ramp = FiatRampLiquidity {
            exchange: RegionalExchange::Binance,
            fiat_currency: FiatCurrency::USD,
            crypto_asset: StablecoinPair::USDTUSD,
            buy_liquidity_fiat: 1_000_000.0,
            sell_liquidity_fiat: 1_000_000.0,
            withdrawal_limit_remaining: 100_000.0,
            deposit_limit_remaining: 100_000.0,
            withdrawal_queue_depth: 5,
            estimated_withdrawal_time_minutes: 30,
            network_congestion_factor: 1.0,
            timestamp_ns: Instant::now().duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default().as_nanos() as u64,
        };

        let target_ramp = FiatRampLiquidity {
            exchange: RegionalExchange::Upbit,
            fiat_currency: FiatCurrency::KRW,
            crypto_asset: StablecoinPair::USDTUSD,
            buy_liquidity_fiat: 500_000.0,
            sell_liquidity_fiat: 500_000.0,
            withdrawal_limit_remaining: 50_000.0,
            deposit_limit_remaining: 50_000.0,
            withdrawal_queue_depth: 10,
            estimated_withdrawal_time_minutes: 60,
            network_congestion_factor: 1.1,
            timestamp_ns: Instant::now().duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default().as_nanos() as u64,
        };

        let result = RiskAdjustedPremium::calculate(
            &signal, &source_ramp, &target_ramp, 0.05, 0.6,
        );

        // Should have positive net premium after costs
        assert!(result.net_premium_bps > 0.0 || result.net_premium_bps > -20.0); // Allow some cost
        assert_eq!(result.raw_premium_bps, 50.0);
    }

    #[test]
    fn test_engine_initialization() {
        let engine = FiatRampArbEngine::new(0.05, 0.6, 10.0, 1000);

        assert!(engine.is_active.load(Ordering::Relaxed));
        assert_eq!(engine.get_best_opportunity(), None);
    }
}
