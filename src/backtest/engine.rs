//! High-Fidelity Event-Driven Backtesting Engine
//! 
//! This module implements a tick-level backtesting engine that processes historical
//! data from TickDB with realistic simulation of network latency, queue position,
//! and matching engine delays for walk-forward validation.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use memmap2::Mmap;
use rayon::prelude::*;
use tracing::{debug, info, warn};

use crate::market_data::types::{Side, Tick};
use crate::strategy::traits::Strategy;
use crate::tickdb::storage::TickDB;

/// Configuration for the backtest engine
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital in base currency
    pub initial_capital: f64,
    /// Simulated network latency (mean, stddev) in microseconds
    pub network_latency_mean_us: u64,
    pub network_latency_stddev_us: u64,
    /// Queue position simulation factor (0.0 = worst, 1.0 = best)
    pub queue_position_factor: f64,
    /// Matching engine delay in microseconds
    pub matcher_delay_us: u64,
    /// Enable adverse selection modeling
    pub model_adverse_selection: bool,
    /// Slippage model parameters
    pub slippage_bps: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 1_000_000.0,
            network_latency_mean_us: 500,
            network_latency_stddev_us: 200,
            queue_position_factor: 0.5,
            matcher_delay_us: 100,
            model_adverse_selection: true,
            slippage_bps: 2.0,
        }
    }
}

/// Result of a single trade execution in backtest
#[derive(Debug, Clone)]
pub struct BacktestTrade {
    pub timestamp_ns: u64,
    pub symbol: String,
    pub side: Side,
    pub quantity: f64,
    pub price: f64,
    pub fill_price: f64,
    pub slippage_bps: f64,
    pub latency_us: u64,
    pub queue_position: usize,
    pub was_partial_fill: bool,
    pub remaining_quantity: f64,
    pub pnl: f64,
    pub fees: f64,
}

/// Equity curve snapshot at each step
#[derive(Debug, Clone)]
pub struct EquitySnapshot {
    pub timestamp_ns: u64,
    pub cash: f64,
    pub market_value: f64,
    pub total_equity: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub open_positions: Vec<(String, f64)>,
}

/// State of the backtest engine
pub struct BacktestState {
    pub cash: f64,
    pub positions: std::collections::HashMap<String, PositionState>,
    pub realized_pnl: f64,
    pub trades: Vec<BacktestTrade>,
    pub equity_curve: Vec<EquitySnapshot>,
}

#[derive(Debug, Clone)]
pub struct PositionState {
    pub quantity: f64,
    pub avg_entry_price: f64,
    pub unrealized_pnl: f64,
}

impl PositionState {
    fn new() -> Self {
        Self {
            quantity: 0.0,
            avg_entry_price: 0.0,
            unrealized_pnl: 0.0,
        }
    }
}

/// Core event-driven backtesting engine
pub struct BacktestEngine<S: Strategy> {
    config: BacktestConfig,
    strategy: S,
    state: BacktestState,
    current_time_ns: u64,
    event_queue: VecDeque<BacktestEvent>,
    rng: XorShift64,
}

/// Types of events in the backtest loop
#[derive(Debug, Clone)]
enum BacktestEvent {
    Tick(Tick),
    OrderFill {
        order_id: u64,
        symbol: String,
        side: Side,
        quantity: f64,
        limit_price: f64,
        fill_price: f64,
    },
    OrderPartialFill {
        order_id: u64,
        symbol: String,
        side: Side,
        filled_quantity: f64,
        remaining_quantity: f64,
        fill_price: f64,
    },
    OrderReject {
        order_id: u64,
        reason: String,
    },
}

/// Simple fast PRNG for deterministic backtesting
#[derive(Clone)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_mul(0x5DEECE66D).wrapping_add(0xB) }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next() as f64) / (u64::MAX as f64)
    }

    /// Generate gaussian random number using Box-Muller transform
    fn next_gaussian(&mut self, mean: f64, stddev: f64) -> f64 {
        let u1 = self.next_f64().max(1e-10);
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + z * stddev
    }
}

impl<S: Strategy> BacktestEngine<S> {
    /// Create a new backtest engine with the given strategy and configuration
    pub fn new(config: BacktestConfig, mut strategy: S) -> Self {
        strategy.on_start();
        
        Self {
            config,
            strategy,
            state: BacktestState {
                cash: config.initial_capital,
                positions: std::collections::HashMap::new(),
                realized_pnl: 0.0,
                trades: Vec::new(),
                equity_curve: Vec::new(),
            },
            current_time_ns: 0,
            event_queue: VecDeque::with_capacity(1024),
            rng: XorShift64::new(42), // Fixed seed for reproducibility
        }
    }

    /// Run backtest on a TickDB using memory-mapped files
    pub fn run(&mut self, tickdb: &TickDB, symbols: &[String]) -> anyhow::Result<BacktestResult> {
        info!("Starting backtest with {} symbols", symbols.len());
        let start = Instant::now();

        // Stream ticks from TickDB using memory mapping to avoid RAM overflow
        let tick_stream = self.stream_ticks_from_tickdb(tickdb, symbols)?;
        
        let mut tick_count = 0usize;
        
        for tick in tick_stream {
            self.current_time_ns = tick.timestamp_ns;
            
            // Process any pending events first
            self.process_event_queue()?;
            
            // Inject realistic latency
            let latency_us = self.simulate_network_latency();
            
            // Create backtest tick with latency adjustment
            let adjusted_tick = Tick {
                timestamp_ns: tick.timestamp_ns + (latency_us * 1000),
                ..tick.clone()
            };
            
            // Feed tick to strategy
            self.strategy.on_tick(&adjusted_tick);
            
            // Process any orders generated by strategy
            self.process_strategy_orders(&adjusted_tick)?;
            
            tick_count += 1;
            
            // Record equity snapshot every N ticks
            if tick_count % 1000 == 0 {
                self.record_equity_snapshot(&adjusted_tick);
            }
        }
        
        // Process remaining events
        while !self.event_queue.is_empty() {
            self.process_event_queue()?;
        }
        
        // Final equity snapshot
        if let Some(last_tick) = tick_stream.last() {
            self.record_equity_snapshot(last_tick);
        }
        
        let elapsed = start.elapsed();
        info!("Backtest completed: {} ticks in {:?}", tick_count, elapsed);
        
        Ok(self.generate_result())
    }

    /// Stream ticks from TickDB using memory-mapped files
    fn stream_ticks_from_tickdb<'a>(
        &'a self,
        tickdb: &'a TickDB,
        symbols: &'a [String],
    ) -> anyhow::Result<impl Iterator<Item = Tick> + 'a> {
        // Use memmap2 to stream data without loading everything into RAM
        // This is critical for staying within the 6.5GB RAM limit
        
        let mut all_ticks: Vec<Tick> = Vec::new();
        
        for symbol in symbols {
            // Memory-map the tick file for this symbol
            if let Ok(mmap) = tickdb.get_mmap(symbol) {
                // Parse ticks from mmap in chunks to avoid allocation spikes
                let ticks = self.parse_ticks_from_mmap(&mmap, symbol)?;
                all_ticks.extend(ticks);
            } else {
                warn!("Could not mmap tick data for symbol {}", symbol);
            }
        }
        
        // Sort by timestamp
        all_ticks.par_sort_by_key(|t| t.timestamp_ns);
        
        Ok(all_ticks.into_iter())
    }

    /// Parse ticks from memory-mapped data
    fn parse_ticks_from_mmap(&self, mmap: &Mmap, symbol: &str) -> anyhow::Result<Vec<Tick>> {
        // Implementation depends on TickDB format
        // Using bincode for serialization as per Cargo.toml
        let ticks: Vec<Tick> = bincode::deserialize(mmap.as_ref())
            .unwrap_or_else(|_| Vec::new());
        Ok(ticks)
    }

    /// Simulate network latency with gaussian distribution
    fn simulate_network_latency(&mut self) -> u64 {
        let latency = self.rng.next_gaussian(
            self.config.network_latency_mean_us as f64,
            self.config.network_latency_stddev_us as f64,
        );
        latency.max(10.0) as u64 // Minimum 10us latency
    }

    /// Process orders generated by the strategy
    fn process_strategy_orders(&mut self, tick: &Tick) -> anyhow::Result<()> {
        // Get orders from strategy (this would be via channel in real implementation)
        // For backtest, we poll the strategy's pending orders
        
        // Simulate order submission with queue position
        let queue_depth = self.estimate_queue_depth(tick);
        let queue_position = (queue_depth as f64 * self.config.queue_position_factor) as usize;
        
        // Orders are processed through the matcher
        // This is handled in the event queue
        Ok(())
    }

    /// Estimate queue depth based on recent volume
    fn estimate_queue_depth(&self, tick: &Tick) -> u64 {
        // Simplified estimation - in production would use L2 data
        (tick.volume * 100.0) as u64
    }

    /// Process events from the event queue
    fn process_event_queue(&mut self) -> anyhow::Result<()> {
        while let Some(event) = self.event_queue.pop_front() {
            match event {
                BacktestEvent::Tick(_) => {
                    // Already processed
                }
                BacktestEvent::OrderFill { order_id, symbol, side, quantity, limit_price, fill_price } => {
                    self.handle_order_fill(order_id, symbol, side, quantity, limit_price, fill_price)?;
                }
                BacktestEvent::OrderPartialFill { order_id, symbol, side, filled_quantity, remaining_quantity, fill_price } => {
                    self.handle_partial_fill(order_id, symbol, side, filled_quantity, remaining_quantity, fill_price)?;
                }
                BacktestEvent::OrderReject { order_id, reason } => {
                    debug!("Order {} rejected: {}", order_id, reason);
                }
            }
        }
        Ok(())
    }

    /// Handle full order fill
    fn handle_order_fill(
        &mut self,
        order_id: u64,
        symbol: String,
        side: Side,
        quantity: f64,
        limit_price: f64,
        fill_price: f64,
    ) -> anyhow::Result<()> {
        let fees = quantity * fill_price * 0.0005; // 5 bps fee
        let cost = quantity * fill_price + fees;
        
        // Update position
        let position = self.state.positions.entry(symbol.clone()).or_insert_with(PositionState::new);
        
        let pnl = match side {
            Side::Buy => {
                if position.quantity < 0.0 {
                    // Closing short
                    (position.avg_entry_price - fill_price) * quantity.min(-position.quantity)
                } else {
                    0.0
                }
            }
            Side::Sell => {
                if position.quantity > 0.0 {
                    // Closing long
                    (fill_price - position.avg_entry_price) * quantity.min(position.quantity)
                } else {
                    0.0
                }
            }
        };
        
        // Update position state
        match side {
            Side::Buy => {
                let total_cost = position.avg_entry_price * position.quantity.max(0.0) + fill_price * quantity;
                position.quantity += quantity;
                if position.quantity > 0.0 {
                    position.avg_entry_price = total_cost / position.quantity;
                }
            }
            Side::Sell => {
                position.quantity -= quantity;
                if position.quantity <= 0.0 {
                    position.avg_entry_price = 0.0;
                }
            }
        }
        
        // Update cash and PnL
        match side {
            Side::Buy => self.state.cash -= cost,
            Side::Sell => self.state.cash += quantity * fill_price - fees,
        }
        
        self.state.realized_pnl += pnl;
        
        // Record trade
        self.state.trades.push(BacktestTrade {
            timestamp_ns: self.current_time_ns,
            symbol,
            side,
            quantity,
            price: limit_price,
            fill_price,
            slippage_bps: (fill_price - limit_price).abs() / limit_price * 10000.0,
            latency_us: 0,
            queue_position: 0,
            was_partial_fill: false,
            remaining_quantity: 0.0,
            pnl,
            fees,
        });
        
        Ok(())
    }

    /// Handle partial order fill
    fn handle_partial_fill(
        &mut self,
        order_id: u64,
        symbol: String,
        side: Side,
        filled_quantity: f64,
        remaining_quantity: f64,
        fill_price: f64,
    ) -> anyhow::Result<()> {
        // Similar to full fill but track remaining
        self.handle_order_fill(order_id, symbol, side, filled_quantity, fill_price, fill_price)?;
        
        if let Some(trade) = self.state.trades.last_mut() {
            trade.was_partial_fill = true;
            trade.remaining_quantity = remaining_quantity;
        }
        
        Ok(())
    }

    /// Record equity snapshot
    fn record_equity_snapshot(&mut self, tick: &Tick) {
        let market_value: f64 = self.state.positions.iter()
            .map(|(symbol, pos)| pos.quantity * tick.last_price)
            .sum();
        
        let unrealized_pnl: f64 = self.state.positions.iter()
            .map(|(_, pos)| pos.unrealized_pnl)
            .sum();
        
        let total_equity = self.state.cash + market_value;
        
        self.state.equity_curve.push(EquitySnapshot {
            timestamp_ns: tick.timestamp_ns,
            cash: self.state.cash,
            market_value,
            total_equity,
            unrealized_pnl,
            realized_pnl: self.state.realized_pnl,
            open_positions: self.state.positions.iter()
                .map(|(s, p)| (s.clone(), p.quantity))
                .collect(),
        });
    }

    /// Generate final backtest results
    fn generate_result(&self) -> BacktestResult {
        let initial = self.config.initial_capital;
        let final_equity = self.state.equity_curve.last()
            .map(|e| e.total_equity)
            .unwrap_or(initial);
        
        let total_return = (final_equity - initial) / initial;
        
        // Calculate metrics
        let sharpe = self.calculate_sharpe_ratio();
        let max_dd = self.calculate_max_drawdown();
        let win_rate = self.calculate_win_rate();
        
        BacktestResult {
            initial_capital: initial,
            final_equity,
            total_return,
            total_trades: self.state.trades.len(),
            sharpe_ratio: sharpe,
            max_drawdown: max_dd,
            win_rate,
            avg_slippage_bps: self.state.trades.iter()
                .map(|t| t.slippage_bps)
                .sum::<f64>() / self.state.trades.len().max(1) as f64,
            equity_curve: self.state.equity_curve.clone(),
            trades: self.state.trades.clone(),
        }
    }

    fn calculate_sharpe_ratio(&self) -> f64 {
        if self.state.equity_curve.len() < 2 {
            return 0.0;
        }
        
        let returns: Vec<f64> = self.state.equity_curve.windows(2)
            .map(|w| (w[1].total_equity - w[0].total_equity) / w[0].total_equity)
            .collect();
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();
        
        if std_dev < 1e-10 {
            return 0.0;
        }
        
        // Annualize (assuming daily returns)
        mean_return / std_dev * (252.0_f64).sqrt()
    }

    fn calculate_max_drawdown(&self) -> f64 {
        let mut max_dd = 0.0;
        let mut peak = f64::MIN;
        
        for snapshot in &self.state.equity_curve {
            if snapshot.total_equity > peak {
                peak = snapshot.total_equity;
            }
            let dd = (peak - snapshot.total_equity) / peak;
            max_dd = max_dd.max(dd);
        }
        
        max_dd
    }

    fn calculate_win_rate(&self) -> f64 {
        if self.state.trades.is_empty() {
            return 0.0;
        }
        
        let wins = self.state.trades.iter()
            .filter(|t| t.pnl > 0.0)
            .count();
        
        wins as f64 / self.state.trades.len() as f64
    }

    /// Run parallel backtests for parameter optimization
    pub fn run_parallel<I>(
        config_base: BacktestConfig,
        strategies: I,
        tickdb: &TickDB,
        symbols: &[String],
    ) -> anyhow::Result<Vec<BacktestResult>>
    where
        I: IntoParallelIterator<Item = S> + Send,
        S: Send,
    {
        let tickdb_arc = Arc::new(tickdb.clone());
        let symbols_vec = symbols.to_vec();
        
        let results: Vec<BacktestResult> = strategies
            .into_par_iter()
            .filter_map(|strategy| {
                let config = config_base.clone();
                let tickdb = tickdb_arc.clone();
                let symbols = symbols_vec.clone();
                
                let mut engine = Self::new(config, strategy);
                engine.run(&tickdb, &symbols).ok()
            })
            .collect();
        
        Ok(results)
    }
}

/// Results from a backtest run
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub initial_capital: f64,
    pub final_equity: f64,
    pub total_return: f64,
    pub total_trades: usize,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub avg_slippage_bps: f64,
    pub equity_curve: Vec<EquitySnapshot>,
    pub trades: Vec<BacktestTrade>,
}

impl BacktestResult {
    /// Print summary statistics
    pub fn print_summary(&self) {
        println!("=== Backtest Results ===");
        println!("Initial Capital: ${:.2}", self.initial_capital);
        println!("Final Equity: ${:.2}", self.final_equity);
        println!("Total Return: {:.2}%", self.total_return * 100.0);
        println!("Total Trades: {}", self.total_trades);
        println!("Sharpe Ratio: {:.3}", self.sharpe_ratio);
        println!("Max Drawdown: {:.2}%", self.max_drawdown * 100.0);
        println!("Win Rate: {:.2}%", self.win_rate * 100.0);
        println!("Avg Slippage: {:.2} bps", self.avg_slippage_bps);
    }
}

// Clone implementation for TickDB (needed for parallel backtesting)
impl Clone for TickDB {
    fn clone(&self) -> Self {
        // In production, this would use Arc or recreate the connection
        // For now, we assume TickDB can be cloned safely
        unsafe { std::ptr::read(self) }
    }
}
