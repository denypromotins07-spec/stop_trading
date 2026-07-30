//! Monte Carlo Simulation Engine for Trade Sequence Analysis
//! 
//! Runs parallel Monte Carlo simulations on trade sequences using rayon
//! to calculate Probability of Ruin, Expected Shortfall, and stress test
//! strategies under extreme market conditions.

use std::sync::Arc;
use std::collections::VecDeque;

use rayon::prelude::*;
use tracing::{debug, info, warn};

use crate::backtest::engine::{BacktestResult, BacktestTrade, EquitySnapshot};

/// Configuration for Monte Carlo simulation
#[derive(Debug, Clone)]
pub struct MonteCarloConfig {
    /// Number of simulation runs
    pub num_simulations: usize,
    /// Number of trades per simulation
    pub trades_per_simulation: usize,
    /// Randomize trade order
    pub randomize_order: bool,
    /// Apply synthetic slippage shocks
    pub apply_slippage_shocks: bool,
    /// Slippage shock magnitude in basis points (std dev)
    pub slippage_shock_std_bps: f64,
    /// Enable probability of ruin calculation
    pub calculate_ruin_probability: bool,
    /// Ruin threshold (fraction of initial capital)
    pub ruin_threshold: f64,
    /// Confidence level for VaR/ES calculations
    pub confidence_level: f64,
    /// Maximum memory usage in MB (strictly bounded for 6.5GB RAM limit)
    pub max_memory_mb: usize,
    /// Use thread-local storage for RNG to prevent heap contention
    pub use_tls_rng: bool,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            num_simulations: 10_000,
            trades_per_simulation: 1000,
            randomize_order: true,
            apply_slippage_shocks: true,
            slippage_shock_std_bps: 5.0,
            calculate_ruin_probability: true,
            ruin_threshold: 0.5, // 50% drawdown = ruin
            confidence_level: 0.95,
            max_memory_mb: 2048, // 2GB limit for MC simulations
            use_tls_rng: true,
        }
    }
}

/// Result from a single Monte Carlo simulation run
#[derive(Debug, Clone)]
pub struct SimulationRun {
    pub run_id: usize,
    pub final_equity: f64,
    pub total_return: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub is_ruined: bool,
    pub equity_curve: Vec<f64>,
    pub trade_sequence: Vec<usize>,
}

/// Aggregated Monte Carlo results
#[derive(Debug, Clone)]
pub struct MonteCarloReport {
    pub config: MonteCarloConfig,
    pub num_runs: usize,
    pub probability_of_ruin: f64,
    pub expected_shortfall: f64,
    pub value_at_risk: f64,
    pub median_return: f64,
    pub mean_return: f64,
    pub std_dev_return: f64,
    pub best_case_return: f64,
    pub worst_case_return: f64,
    pub percentile_5: f64,
    pub percentile_25: f64,
    pub percentile_75: f64,
    pub percentile_95: f64,
    pub avg_max_drawdown: f64,
    pub worst_max_drawdown: f64,
    pub avg_sharpe: f64,
    pub runs: Vec<SimulationRun>,
    pub is_acceptable: bool,
}

/// Thread-local RNG wrapper for parallel simulations
struct TlsRng {
    state: u64,
}

impl TlsRng {
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

    /// Shuffle a slice in place (Fisher-Yates)
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.next() as usize % (i + 1);
            slice.swap(i, j);
        }
    }
}

/// Monte Carlo Simulation Engine
pub struct MonteCarloEngine {
    config: MonteCarloConfig,
    base_trades: Vec<BacktestTrade>,
    initial_capital: f64,
}

impl MonteCarloEngine {
    /// Create a new Monte Carlo engine with historical trades
    pub fn new(config: MonteCarloConfig, trades: Vec<BacktestTrade>, initial_capital: f64) -> Self {
        Self {
            config,
            base_trades: trades,
            initial_capital,
        }
    }

    /// Run Monte Carlo simulations in parallel
    pub fn run(&self) -> anyhow::Result<MonteCarloReport> {
        info!(
            "Starting Monte Carlo simulation: {} runs, {} trades each",
            self.config.num_simulations,
            self.config.trades_per_simulation
        );

        if self.base_trades.is_empty() {
            warn!("No trades available for Monte Carlo simulation");
            return Ok(self.empty_report());
        }

        let start = std::time::Instant::now();

        // Run simulations in parallel using rayon
        let runs: Vec<SimulationRun> = (0..self.config.num_simulations)
            .into_par_iter()
            .map(|run_id| self.run_single_simulation(run_id))
            .collect();

        let elapsed = start.elapsed();
        info!("Monte Carlo completed in {:?}", elapsed);

        // Calculate aggregate statistics
        let report = self.calculate_report(runs);

        info!(
            "Probability of Ruin: {:.2}%, Expected Shortfall: {:.2}%",
            report.probability_of_ruin * 100.0,
            report.expected_shortfall * 100.0
        );

        Ok(report)
    }

    /// Run a single simulation
    fn run_single_simulation(&self, run_id: usize) -> SimulationRun {
        // Use thread-local RNG based on run_id for reproducibility
        let mut rng = TlsRng::new(run_id as u64 * 12345);

        // Sample trades with replacement and optional randomization
        let mut trade_indices: Vec<usize> = if self.config.randomize_order {
            let mut indices: Vec<usize> = (0..self.base_trades.len()).collect();
            rng.shuffle(&mut indices);
            indices
        } else {
            (0..self.base_trades.len()).collect()
        };

        // Limit to configured number of trades
        trade_indices.truncate(self.config.trades_per_simulation);

        // Run simulation
        let mut equity = self.initial_capital;
        let mut peak_equity = equity;
        let mut max_drawdown = 0.0;
        let mut equity_curve = Vec::with_capacity(self.config.trades_per_simulation + 1);
        equity_curve.push(equity);

        let mut returns = Vec::new();
        let mut prev_equity = equity;

        for &trade_idx in &trade_indices {
            let base_trade = &self.base_trades[trade_idx];
            
            // Apply slippage shock if enabled
            let slippage_adjustment = if self.config.apply_slippage_shocks {
                let shock = rng.next_gaussian(0.0, self.config.slippage_shock_std_bps / 10000.0);
                1.0 + shock
            } else {
                1.0
            };

            // Calculate PnL with adjustment
            let pnl = base_trade.pnl * slippage_adjustment;
            equity += pnl;

            // Track drawdown
            if equity > peak_equity {
                peak_equity = equity;
            }
            let dd = (peak_equity - equity) / peak_equity;
            max_drawdown = max_drawdown.max(dd);

            // Record equity
            equity_curve.push(equity);

            // Track returns for Sharpe calculation
            let ret = (equity - prev_equity) / prev_equity;
            returns.push(ret);
            prev_equity = equity;
        }

        // Calculate metrics
        let total_return = (equity - self.initial_capital) / self.initial_capital;
        let is_ruined = equity < self.initial_capital * self.config.ruin_threshold;

        // Calculate Sharpe ratio
        let sharpe = if returns.len() > 1 {
            let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter()
                .map(|r| (r - mean_ret).powi(2))
                .sum::<f64>() / returns.len() as f64;
            let std_dev = variance.sqrt();
            if std_dev > 1e-10 {
                mean_ret / std_dev * (252.0_f64).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        SimulationRun {
            run_id,
            final_equity: equity,
            total_return,
            max_drawdown,
            sharpe_ratio: sharpe,
            is_ruined,
            equity_curve,
            trade_sequence: trade_indices,
        }
    }

    /// Calculate aggregated report from simulation runs
    fn calculate_report(&self, runs: Vec<SimulationRun>) -> MonteCarloReport {
        if runs.is_empty() {
            return self.empty_report();
        }

        let num_runs = runs.len();

        // Collect returns for statistical analysis
        let mut returns: Vec<f64> = runs.iter().map(|r| r.total_return).collect();
        returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate percentiles
        let get_percentile = |p: f64| -> f64 {
            let idx = ((p * (num_runs - 1) as f64) as usize).min(num_runs - 1);
            returns[idx]
        };

        let percentile_5 = get_percentile(0.05);
        let percentile_25 = get_percentile(0.25);
        let median = get_percentile(0.50);
        let percentile_75 = get_percentile(0.75);
        let percentile_95 = get_percentile(0.95);

        // Calculate probability of ruin
        let ruined_count = runs.iter().filter(|r| r.is_ruined).count();
        let probability_of_ruin = ruined_count as f64 / num_runs as f64;

        // Calculate Expected Shortfall (average of worst cases beyond VaR)
        let var_threshold = get_percentile(1.0 - self.config.confidence_level);
        let tail_losses: Vec<f64> = returns.iter()
            .filter(|&&r| r <= var_threshold)
            .copied()
            .collect();
        
        let expected_shortfall = if tail_losses.is_empty() {
            var_threshold
        } else {
            tail_losses.iter().sum::<f64>() / tail_losses.len() as f64
        };

        // Calculate Value at Risk
        let value_at_risk = var_threshold;

        // Calculate mean and std dev
        let mean_return = returns.iter().sum::<f64>() / num_runs as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / num_runs as f64;
        let std_dev_return = variance.sqrt();

        // Calculate average and worst max drawdown
        let avg_max_dd = runs.iter().map(|r| r.max_drawdown).sum::<f64>() / num_runs as f64;
        let worst_max_dd = runs.iter().map(|r| r.max_drawdown).fold(0.0_f64, f64::max);

        // Calculate average Sharpe
        let avg_sharpe = runs.iter().map(|r| r.sharpe_ratio).sum::<f64>() / num_runs as f64;

        // Determine if strategy is acceptable
        let is_acceptable = probability_of_ruin < 0.1 
            && avg_max_dd < 0.2 
            && median > 0.0
            && percentile_5 > -0.3;

        MonteCarloReport {
            config: self.config.clone(),
            num_runs,
            probability_of_ruin,
            expected_shortfall,
            value_at_risk,
            median_return: median,
            mean_return,
            std_dev_return,
            best_case_return: returns.last().copied().unwrap_or(0.0),
            worst_case_return: returns.first().copied().unwrap_or(0.0),
            percentile_5,
            percentile_25,
            percentile_75,
            percentile_95,
            avg_max_drawdown: avg_max_dd,
            worst_max_drawdown: worst_max_dd,
            avg_sharpe,
            runs,
            is_acceptable,
        }
    }

    /// Create empty report for error cases
    fn empty_report(&self) -> MonteCarloReport {
        MonteCarloReport {
            config: self.config.clone(),
            num_runs: 0,
            probability_of_ruin: 0.0,
            expected_shortfall: 0.0,
            value_at_risk: 0.0,
            median_return: 0.0,
            mean_return: 0.0,
            std_dev_return: 0.0,
            best_case_return: 0.0,
            worst_case_return: 0.0,
            percentile_5: 0.0,
            percentile_25: 0.0,
            percentile_75: 0.0,
            percentile_95: 0.0,
            avg_max_drawdown: 0.0,
            worst_max_drawdown: 0.0,
            avg_sharpe: 0.0,
            runs: Vec::new(),
            is_acceptable: false,
        }
    }

    /// Run stress test with extreme parameters
    pub fn run_stress_test(&self, shock_multiplier: f64) -> anyhow::Result<MonteCarloReport> {
        let mut stress_config = self.config.clone();
        stress_config.slippage_shock_std_bps *= shock_multiplier;
        stress_config.apply_slippage_shocks = true;

        let stress_engine = Self {
            config: stress_config,
            base_trades: self.base_trades.clone(),
            initial_capital: self.initial_capital,
        };

        stress_engine.run()
    }
}

impl MonteCarloReport {
    /// Print detailed analysis
    pub fn print_analysis(&self) {
        println!("=== Monte Carlo Simulation Report ===");
        println!("Number of Runs: {}", self.num_runs);
        println!();
        println!("=== Risk Metrics ===");
        println!("Probability of Ruin: {:.2}%", self.probability_of_ruin * 100.0);
        println!("Value at Risk ({}%): {:.2}%", 
            self.config.confidence_level * 100.0, 
            self.value_at_risk * 100.0);
        println!("Expected Shortfall: {:.2}%", self.expected_shortfall * 100.0);
        println!();
        println!("=== Return Distribution ===");
        println!("Mean Return: {:.2}%", self.mean_return * 100.0);
        println!("Median Return: {:.2}%", self.median_return * 100.0);
        println!("Std Dev: {:.2}%", self.std_dev_return * 100.0);
        println!("Best Case: {:.2}%", self.best_case_return * 100.0);
        println!("Worst Case: {:.2}%", self.worst_case_return * 100.0);
        println!();
        println!("=== Percentiles ===");
        println!("5th: {:.2}%", self.percentile_5 * 100.0);
        println!("25th: {:.2}%", self.percentile_25 * 100.0);
        println!("75th: {:.2}%", self.percentile_75 * 100.0);
        println!("95th: {:.2}%", self.percentile_95 * 100.0);
        println!();
        println!("=== Drawdown Analysis ===");
        println!("Avg Max DD: {:.2}%", self.avg_max_drawdown * 100.0);
        println!("Worst Max DD: {:.2}%", self.worst_max_drawdown * 100.0);
        println!("Avg Sharpe: {:.3}", self.avg_sharpe);
        println!();
        println!("Is Acceptable: {}", if self.is_acceptable { "YES ✓" } else { "NO ✗" });
    }
}

/// Bootstrap resampling for confidence intervals
pub struct BootstrapAnalyzer {
    pub num_resamples: usize,
}

impl BootstrapAnalyzer {
    pub fn new(num_resamples: usize) -> Self {
        Self { num_resamples }
    }

    /// Calculate bootstrap confidence interval for a metric
    pub fn confidence_interval(&self, data: &[f64], confidence: f64) -> (f64, f64) {
        let mut rng = TlsRng::new(99999);
        let mut bootstrap_means = Vec::with_capacity(self.num_resamples);

        for _ in 0..self.num_resamples {
            let mut sample_sum = 0.0;
            for _ in 0..data.len() {
                let idx = rng.next() as usize % data.len();
                sample_sum += data[idx];
            }
            bootstrap_means.push(sample_sum / data.len() as f64);
        }

        bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let lower_idx = ((1.0 - confidence) / 2.0 * (self.num_resamples - 1) as f64) as usize;
        let upper_idx = ((1.0 - (1.0 - confidence) / 2.0) * (self.num_resamples - 1) as f64) as usize;

        (bootstrap_means[lower_idx], bootstrap_means[upper_idx.min(self.num_resamples - 1)])
    }
}
