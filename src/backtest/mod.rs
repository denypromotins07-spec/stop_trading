//! Backtest Module Root
//! 
//! Integrates the event-driven backtesting engine with live strategy actors
//! using trait objects, ensuring the same compiled Rust strategy code runs
//! in both live trading and backtesting modes without modification.

pub mod engine;
pub mod matcher;

pub use engine::{BacktestConfig, BacktestEngine, BacktestResult, BacktestTrade, EquitySnapshot};
pub use matcher::{MatcherConfig, MatchingEngine, OrderSimulationResult, SimulatedOrderBook};

use std::sync::Arc;
use std::any::Any;

use crate::strategy::traits::Strategy;
use crate::market_data::types::Tick;
use crate::execution::types::Order;

/// Trait for backtestable components - allows strategies to run in both modes
pub trait Backtestable: Send + Sync {
    /// Process a tick in backtest mode
    fn on_tick_backtest(&mut self, tick: &Tick);
    
    /// Submit an order in backtest mode
    fn submit_order_backtest(&mut self, order: &Order) -> BacktestOrderResult;
    
    /// Get current state for snapshotting
    fn get_state(&self) -> Box<dyn Any + Send>;
    
    /// Restore state from snapshot
    fn restore_state(&mut self, state: Box<dyn Any + Send>);
}

/// Result of a backtest order submission
#[derive(Debug, Clone)]
pub struct BacktestOrderResult {
    pub order_id: u64,
    pub accepted: bool,
    pub fill_price: Option<f64>,
    pub filled_quantity: f64,
    pub remaining_quantity: f64,
    pub rejection_reason: Option<String>,
}

/// Unified strategy runner that works in both live and backtest modes
pub struct StrategyRunner<S: Strategy> {
    strategy: S,
    mode: RunnerMode,
    backtest_state: Option<engine::BacktestState>,
}

/// Execution mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunnerMode {
    Live,
    Backtest,
    Paper,
}

impl<S: Strategy> StrategyRunner<S> {
    /// Create a new strategy runner
    pub fn new(strategy: S, mode: RunnerMode) -> Self {
        Self {
            strategy,
            mode,
            backtest_state: None,
        }
    }

    /// Initialize the strategy
    pub fn init(&mut self) {
        self.strategy.on_start();
    }

    /// Process a tick - same interface for live and backtest
    pub fn on_tick(&mut self, tick: &Tick) {
        match self.mode {
            RunnerMode::Live => {
                self.strategy.on_tick(tick);
            }
            RunnerMode::Backtest => {
                self.strategy.on_tick(tick);
            }
            RunnerMode::Paper => {
                self.strategy.on_tick(tick);
            }
        }
    }

    /// Submit an order - routed based on mode
    pub fn submit_order(&mut self, order: &Order) -> BacktestOrderResult {
        match self.mode {
            RunnerMode::Live => {
                // In live mode, this would send to execution gateway
                BacktestOrderResult {
                    order_id: order.id,
                    accepted: true,
                    fill_price: None,
                    filled_quantity: 0.0,
                    remaining_quantity: order.quantity,
                    rejection_reason: None,
                }
            }
            RunnerMode::Backtest => {
                // In backtest mode, simulate the fill
                BacktestOrderResult {
                    order_id: order.id,
                    accepted: true,
                    fill_price: Some(order.price),
                    filled_quantity: order.quantity,
                    remaining_quantity: 0.0,
                    rejection_reason: None,
                }
            }
            RunnerMode::Paper => {
                // Paper trading simulates but doesn't execute
                BacktestOrderResult {
                    order_id: order.id,
                    accepted: true,
                    fill_price: Some(order.price),
                    filled_quantity: order.quantity,
                    remaining_quantity: 0.0,
                    rejection_reason: None,
                }
            }
        }
    }

    /// Get current PnL
    pub fn get_pnl(&self) -> f64 {
        // Implementation depends on strategy state tracking
        0.0
    }

    /// Switch mode (for hot-switching between paper and live)
    pub fn switch_mode(&mut self, new_mode: RunnerMode) {
        self.mode = new_mode;
    }

    /// Get current mode
    pub fn mode(&self) -> RunnerMode {
        self.mode
    }
}

/// Factory for creating backtest engines with different strategies
pub struct BacktestFactory;

impl BacktestFactory {
    /// Create a backtest engine with the given strategy
    pub fn create_engine<S: Strategy>(
        config: BacktestConfig,
        strategy: S,
    ) -> BacktestEngine<S> {
        BacktestEngine::new(config, strategy)
    }

    /// Run parallel backtests across multiple parameter sets
    pub fn run_parameter_sweep<S, F, I>(
        config_base: BacktestConfig,
        strategy_factory: F,
        tickdb: &crate::tickdb::storage::TickDB,
        symbols: &[String],
    ) -> anyhow::Result<Vec<BacktestResult>>
    where
        S: Strategy + Send + 'static,
        F: Fn() -> I + Send + Sync,
        I: IntoParallelIterator<Item = S> + Send,
    {
        let strategies = strategy_factory();
        BacktestEngine::<S>::run_parallel(config_base, strategies, tickdb, symbols)
    }
}

/// Aggregated results from multiple backtest runs
#[derive(Debug, Clone)]
pub struct BacktestReport {
    pub runs: Vec<BacktestResult>,
    pub best_run_index: usize,
    pub avg_sharpe: f64,
    pub avg_max_drawdown: f64,
    pub avg_total_return: f64,
    pub parameter_sensitivity: Vec<ParameterSensitivity>,
}

/// Sensitivity of results to parameter changes
#[derive(Debug, Clone)]
pub struct ParameterSensitivity {
    pub parameter_name: String,
    pub correlation_with_returns: f64,
    pub optimal_value: f64,
    pub robustness_score: f64,
}

impl BacktestReport {
    /// Generate report from multiple backtest runs
    pub fn from_runs(runs: Vec<BacktestResult>) -> Self {
        if runs.is_empty() {
            return Self {
                runs,
                best_run_index: 0,
                avg_sharpe: 0.0,
                avg_max_drawdown: 0.0,
                avg_total_return: 0.0,
                parameter_sensitivity: Vec::new(),
            };
        }

        let best_run_index = runs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.sharpe_ratio.partial_cmp(&b.sharpe_ratio).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let avg_sharpe = runs.iter().map(|r| r.sharpe_ratio).sum::<f64>() / runs.len() as f64;
        let avg_max_dd = runs.iter().map(|r| r.max_drawdown).sum::<f64>() / runs.len() as f64;
        let avg_return = runs.iter().map(|r| r.total_return).sum::<f64>() / runs.len() as f64;

        Self {
            runs,
            best_run_index,
            avg_sharpe,
            avg_max_drawdown: avg_max_dd,
            avg_total_return: avg_return,
            parameter_sensitivity: Vec::new(),
        }
    }

    /// Print summary of the report
    pub fn print_summary(&self) {
        println!("=== Backtest Report ===");
        println!("Total Runs: {}", self.runs.len());
        println!("Best Run Index: {}", self.best_run_index);
        println!("Average Sharpe: {:.3}", self.avg_sharpe);
        println!("Average Max DD: {:.2}%", self.avg_max_drawdown * 100.0);
        println!("Average Return: {:.2}%", self.avg_total_return * 100.0);

        if let Some(best) = self.runs.get(self.best_run_index) {
            println!("\n=== Best Run Details ===");
            best.print_summary();
        }
    }
}

/// Trait object wrapper for dynamic strategy dispatch
pub trait DynStrategy: Send + Sync {
    fn on_tick(&mut self, tick: &Tick);
    fn on_bar(&mut self, bar: &crate::market_data::types::OHLCV);
    fn on_fill(&mut self, fill: &crate::execution::types::Fill);
    fn on_start(&mut self);
    fn on_stop(&mut self);
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Wrapper to convert concrete strategy to trait object
pub struct StrategyBox(pub Box<dyn DynStrategy>);

impl StrategyBox {
    pub fn new<S: Strategy + 'static>(strategy: S) -> Self
    where
        S: DynStrategy,
    {
        Self(Box::new(strategy))
    }
}

// Blanket implementation for Strategy trait
impl<S: Strategy + 'static> From<S> for StrategyBox {
    fn from(strategy: S) -> Self {
        // This requires S to also implement DynStrategy
        // In practice, you'd implement DynStrategy for your strategy types
        unimplemented!("Strategy must implement DynStrategy trait")
    }
}

/// Compare backtest results with live performance
pub struct PerformanceComparator {
    pub backtest_results: Vec<BacktestResult>,
    pub live_results: Vec<LivePerformanceMetric>,
}

#[derive(Debug, Clone)]
pub struct LivePerformanceMetric {
    pub timestamp_ns: u64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_equity: f64,
    pub trade_count: usize,
}

impl PerformanceComparator {
    pub fn new(backtest_results: Vec<BacktestResult>, live_results: Vec<LivePerformanceMetric>) -> Self {
        Self {
            backtest_results,
            live_results,
        }
    }

    /// Calculate degradation between backtest and live
    pub fn calculate_degradation(&self) -> PerformanceDegradation {
        if self.backtest_results.is_empty() || self.live_results.is_empty() {
            return PerformanceDegradation::default();
        }

        let best_backtest = self.backtest_results.iter()
            .max_by(|a, b| a.sharpe_ratio.partial_cmp(&b.sharpe_ratio).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let live_sharpe = self.calculate_live_sharpe();
        let live_return = self.calculate_live_return();
        let live_max_dd = self.calculate_live_max_dd();

        let sharpe_degradation = (best_backtest.sharpe_ratio - live_sharpe) / best_backtest.sharpe_ratio.max(1e-10);
        let return_degradation = (best_backtest.total_return - live_return) / best_backtest.total_return.abs().max(1e-10);
        let dd_degradation = live_max_dd - best_backtest.max_drawdown;

        PerformanceDegradation {
            sharpe_degradation,
            return_degradation,
            drawdown_degradation: dd_degradation,
            is_acceptable: sharpe_degradation < 0.3 && return_degradation < 0.3,
        }
    }

    fn calculate_live_sharpe(&self) -> f64 {
        if self.live_results.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = self.live_results.windows(2)
            .map(|w| (w[1].total_equity - w[0].total_equity) / w[0].total_equity.max(1e-10))
            .collect();

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            return 0.0;
        }

        mean / std_dev * (252.0_f64).sqrt()
    }

    fn calculate_live_return(&self) -> f64 {
        if self.live_results.is_empty() {
            return 0.0;
        }
        let initial = self.live_results.first().unwrap().total_equity;
        let final_eq = self.live_results.last().unwrap().total_equity;
        (final_eq - initial) / initial.max(1e-10)
    }

    fn calculate_live_max_dd(&self) -> f64 {
        let mut max_dd = 0.0;
        let mut peak = f64::MIN;

        for metric in &self.live_results {
            if metric.total_equity > peak {
                peak = metric.total_equity;
            }
            let dd = (peak - metric.total_equity) / peak.max(1e-10);
            max_dd = max_dd.max(dd);
        }

        max_dd
    }
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceDegradation {
    pub sharpe_degradation: f64,
    pub return_degradation: f64,
    pub drawdown_degradation: f64,
    pub is_acceptable: bool,
}

impl PerformanceDegradation {
    pub fn print_analysis(&self) {
        println!("=== Performance Degradation Analysis ===");
        println!("Sharpe Degradation: {:.2}%", self.sharpe_degradation * 100.0);
        println!("Return Degradation: {:.2}%", self.return_degradation * 100.0);
        println!("Drawdown Degradation: {:.2}%", self.drawdown_degradation * 100.0);
        println!("Acceptable: {}", if self.is_acceptable { "YES" } else { "NO" });
    }
}
