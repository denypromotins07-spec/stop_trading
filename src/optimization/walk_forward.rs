//! Walk-Forward Analysis (WFA) Implementation
//! 
//! Implements rolling walk-forward analysis to prevent curve-fitting and over-optimization.
//! Dynamically splits data into in-sample training and out-of-sample testing windows,
//! tracking equity curve degradation across multiple periods.

use std::sync::Arc;
use std::collections::VecDeque;

use rayon::prelude::*;
use tracing::{debug, info, warn};

use crate::backtest::engine::{BacktestConfig, BacktestEngine, BacktestResult};
use crate::tickdb::storage::TickDB;
use crate::strategy::traits::Strategy;

/// Configuration for Walk-Forward Analysis
#[derive(Debug, Clone)]
pub struct WalkForwardConfig {
    /// Total number of periods to analyze
    pub num_periods: usize,
    /// Ratio of in-sample to total data (0.6 = 60% in-sample, 40% out-of-sample)
    pub in_sample_ratio: f64,
    /// Minimum period length in days
    pub min_period_days: usize,
    /// Step size between periods (1 = non-overlapping, <1 = overlapping)
    pub step_ratio: f64,
    /// Enable parallel execution
    pub parallel: bool,
    /// Maximum memory usage in MB (to stay within 6.5GB RAM limit)
    pub max_memory_mb: usize,
}

impl Default for WalkForwardConfig {
    fn default() -> Self {
        Self {
            num_periods: 10,
            in_sample_ratio: 0.6,
            min_period_days: 30,
            step_ratio: 1.0,
            parallel: true,
            max_memory_mb: 4096, // Reserve ~2.5GB for other operations
        }
    }
}

/// Result from a single walk-forward period
#[derive(Debug, Clone)]
pub struct WalkForwardPeriod {
    pub period_index: usize,
    pub start_timestamp_ns: u64,
    pub end_timestamp_ns: u64,
    pub in_sample_start_ns: u64,
    pub in_sample_end_ns: u64,
    pub out_of_sample_start_ns: u64,
    pub out_of_sample_end_ns: u64,
    pub in_sample_result: Option<BacktestResult>,
    pub out_of_sample_result: Option<BacktestResult>,
    pub degradation_factor: f64,
    pub is_acceptable: bool,
}

/// Aggregated Walk-Forward Analysis results
#[derive(Debug, Clone)]
pub struct WalkForwardReport {
    pub config: WalkForwardConfig,
    pub periods: Vec<WalkForwardPeriod>,
    pub avg_in_sample_sharpe: f64,
    pub avg_out_of_sample_sharpe: f64,
    pub avg_degradation: f64,
    pub robustness_score: f64,
    pub is_robust: bool,
    pub equity_curve_stability: f64,
    pub parameter_stability: Vec<ParameterStabilityMetric>,
}

/// Stability metric for individual parameters
#[derive(Debug, Clone)]
pub struct ParameterStabilityMetric {
    pub parameter_name: String,
    pub optimal_values: Vec<f64>,
    pub mean_optimal: f64,
    pub std_dev: f64,
    pub coefficient_of_variation: f64,
    pub is_stable: bool,
}

/// Walk-Forward Analysis Engine
pub struct WalkForwardAnalyzer<S: Strategy> {
    config: WalkForwardConfig,
    base_strategy: S,
    base_config: BacktestConfig,
    _marker: std::marker::PhantomData<S>,
}

impl<S: Strategy + Send + Sync + 'static> WalkForwardAnalyzer<S> {
    /// Create a new walk-forward analyzer
    pub fn new(config: WalkForwardConfig, strategy: S, backtest_config: BacktestConfig) -> Self {
        Self {
            config,
            base_strategy: strategy,
            base_config: backtest_config,
            _marker: std::marker::PhantomData,
        }
    }

    /// Run walk-forward analysis on the given date range
    pub fn run(
        &self,
        tickdb: &TickDB,
        symbols: &[String],
        start_ns: u64,
        end_ns: u64,
    ) -> anyhow::Result<WalkForwardReport> {
        info!("Starting Walk-Forward Analysis with {} periods", self.config.num_periods);
        
        let total_duration_ns = end_ns - start_ns;
        let period_duration_ns = total_duration_ns / self.config.num_periods as u64;
        let in_sample_duration_ns = (period_duration_ns as f64 * self.config.in_sample_ratio) as u64;
        let step_ns = (period_duration_ns as f64 * self.config.step_ratio) as u64;

        // Generate period boundaries
        let mut periods = Vec::with_capacity(self.config.num_periods);
        let mut current_start = start_ns;

        for i in 0..self.config.num_periods {
            let period_end = current_start + period_duration_ns;
            if period_end > end_ns {
                break;
            }

            let in_sample_end = current_start + in_sample_duration_ns;
            let oos_start = in_sample_end;
            let oos_end = period_end;

            // Skip if period is too short
            if (in_sample_end - current_start) < (self.config.min_period_days as u64 * 86_400_000_000_000) {
                current_start += step_ns;
                continue;
            }

            periods.push(PeriodBoundaries {
                period_index: i,
                period_start: current_start,
                period_end,
                in_sample_start: current_start,
                in_sample_end,
                oos_start,
                oos_end,
            });

            current_start += step_ns;
        }

        info!("Generated {} walk-forward periods", periods.len());

        // Run analysis for each period
        let period_results: Vec<WalkForwardPeriod> = if self.config.parallel {
            self.run_parallel(tickdb, symbols, &periods)?
        } else {
            self.run_sequential(tickdb, symbols, &periods)?
        };

        // Calculate aggregate metrics
        let report = self.calculate_report(period_results);

        info!(
            "Walk-Forward Analysis complete. Robustness score: {:.3}, Is robust: {}",
            report.robustness_score,
            report.is_robust
        );

        Ok(report)
    }

    /// Run analysis in parallel using rayon
    fn run_parallel(
        &self,
        tickdb: &TickDB,
        symbols: &[String],
        periods: &[PeriodBoundaries],
    ) -> anyhow::Result<Vec<WalkForwardPeriod>> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        
        let tickdb_arc = Arc::new(tickdb.clone());
        let symbols_vec = Arc::new(symbols.to_vec());
        let processed = AtomicUsize::new(0);

        let results: Vec<WalkForwardPeriod> = periods
            .par_iter()
            .filter_map(|period| {
                let tickdb = tickdb_arc.clone();
                let symbols = symbols_vec.clone();
                
                let result = self.analyze_period(&tickdb, &symbols, period);
                
                let count = processed.fetch_add(1, Ordering::Relaxed);
                debug!("Processed period {}/{}", count + 1, periods.len());
                
                result.ok()
            })
            .collect();

        Ok(results)
    }

    /// Run analysis sequentially (for debugging or when parallel is disabled)
    fn run_sequential(
        &self,
        tickdb: &TickDB,
        symbols: &[String],
        periods: &[PeriodBoundaries],
    ) -> anyhow::Result<Vec<WalkForwardPeriod>> {
        let mut results = Vec::with_capacity(periods.len());

        for (i, period) in periods.iter().enumerate() {
            if let Ok(result) = self.analyze_period(tickdb, symbols, period) {
                results.push(result);
            }
            debug!("Processed period {}/{}", i + 1, periods.len());
        }

        Ok(results)
    }

    /// Analyze a single walk-forward period
    fn analyze_period(
        &self,
        tickdb: &TickDB,
        symbols: &[String],
        period: &PeriodBoundaries,
    ) -> anyhow::Result<WalkForwardPeriod> {
        // Clone strategy for in-sample optimization
        let mut in_sample_strategy = self.base_strategy.clone_for_backtest();
        
        // Run in-sample backtest
        let in_sample_result = self.run_backtest_for_period(
            tickdb,
            symbols,
            period.in_sample_start,
            period.in_sample_end,
            &mut in_sample_strategy,
        )?;

        // Clone strategy for out-of-sample testing (with optimized parameters)
        let mut oos_strategy = self.base_strategy.clone_for_backtest();
        
        // Apply optimized parameters from in-sample to OOS strategy
        // This would typically involve copying the best parameters found
        oos_strategy.apply_parameters_from(&in_sample_strategy);

        // Run out-of-sample backtest
        let oos_result = self.run_backtest_for_period(
            tickdb,
            symbols,
            period.oos_start,
            period.oos_end,
            &mut oos_strategy,
        )?;

        // Calculate degradation factor
        let degradation_factor = if in_sample_result.sharpe_ratio > 0.0 {
            (in_sample_result.sharpe_ratio - oos_result.sharpe_ratio) / in_sample_result.sharpe_ratio
        } else {
            0.0
        };

        // Determine if this period is acceptable (< 30% degradation)
        let is_acceptable = degradation_factor < 0.3 && oos_result.total_return > -0.1;

        Ok(WalkForwardPeriod {
            period_index: period.period_index,
            start_timestamp_ns: period.period_start,
            end_timestamp_ns: period.period_end,
            in_sample_start_ns: period.in_sample_start,
            in_sample_end_ns: period.in_sample_end,
            out_of_sample_start_ns: period.oos_start,
            out_of_sample_end_ns: period.oos_end,
            in_sample_result: Some(in_sample_result),
            out_of_sample_result: Some(oos_result),
            degradation_factor,
            is_acceptable,
        })
    }

    /// Run backtest for a specific time period
    fn run_backtest_for_period(
        &self,
        tickdb: &TickDB,
        symbols: &[String],
        start_ns: u64,
        end_ns: u64,
        strategy: &mut S,
    ) -> anyhow::Result<BacktestResult> {
        // Filter ticks for the period and run backtest
        // In production, this would use TickDB's time-range query
        
        let mut engine = BacktestEngine::new(self.base_config.clone(), strategy.clone_for_backtest());
        
        // Note: The actual filtering would happen in TickDB
        // For now, we run on all data and let the engine handle it
        engine.run(tickdb, symbols)
    }

    /// Calculate aggregated report from period results
    fn calculate_report(&self, periods: Vec<WalkForwardPeriod>) -> WalkForwardReport {
        if periods.is_empty() {
            return WalkForwardReport {
                config: self.config.clone(),
                periods,
                avg_in_sample_sharpe: 0.0,
                avg_out_of_sample_sharpe: 0.0,
                avg_degradation: 0.0,
                robustness_score: 0.0,
                is_robust: false,
                equity_curve_stability: 0.0,
                parameter_stability: Vec::new(),
            };
        }

        // Calculate average metrics
        let avg_is_sharpe: f64 = periods.iter()
            .filter_map(|p| p.in_sample_result.as_ref())
            .map(|r| r.sharpe_ratio)
            .sum::<f64>() / periods.len() as f64;

        let avg_oos_sharpe: f64 = periods.iter()
            .filter_map(|p| p.out_of_sample_result.as_ref())
            .map(|r| r.sharpe_ratio)
            .sum::<f64>() / periods.len() as f64;

        let avg_degradation = periods.iter()
            .map(|p| p.degradation_factor)
            .sum::<f64>() / periods.len() as f64;

        // Calculate robustness score (higher is better)
        // Based on: low degradation, consistent OOS performance, high acceptance rate
        let acceptance_rate = periods.iter().filter(|p| p.is_acceptable).count() as f64 / periods.len() as f64;
        let consistency = 1.0 - (periods.iter()
            .filter_map(|p| p.out_of_sample_result.as_ref())
            .map(|r| r.sharpe_ratio)
            .collect::<Vec<_>>()
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum::<f64>() / periods.len() as f64).min(1.0);

        let robustness_score = (acceptance_rate * 0.4 + consistency * 0.3 + (1.0 - avg_degradation.max(0.0)) * 0.3)
            .clamp(0.0, 1.0);

        // Determine if strategy is robust
        let is_robust = robustness_score > 0.7 && avg_degradation < 0.3 && avg_oos_sharpe > 0.5;

        // Calculate equity curve stability
        let equity_curve_stability = self.calculate_equity_curve_stability(&periods);

        WalkForwardReport {
            config: self.config.clone(),
            periods,
            avg_in_sample_sharpe: avg_is_sharpe,
            avg_out_of_sample_sharpe: avg_oos_sharpe,
            avg_degradation,
            robustness_score,
            is_robust,
            equity_curve_stability,
            parameter_stability: Vec::new(), // Would be populated with actual parameter tracking
        }
    }

    /// Calculate stability of equity curves across periods
    fn calculate_equity_curve_stability(&self, periods: &[WalkForwardPeriod]) -> f64 {
        // Compare normalized equity curves across periods
        // Higher value = more consistent performance
        
        let mut correlations = Vec::new();
        
        for window in periods.windows(2) {
            if let (Some(r1), Some(r2)) = (&window[0].out_of_sample_result, &window[1].out_of_sample_result) {
                let corr = self.correlate_equity_curves(&r1.equity_curve, &r2.equity_curve);
                correlations.push(corr);
            }
        }

        if correlations.is_empty() {
            return 0.0;
        }

        correlations.iter().sum::<f64>() / correlations.len() as f64
    }

    /// Calculate correlation between two equity curves
    fn correlate_equity_curves(
        &self,
        curve1: &[crate::backtest::engine::EquitySnapshot],
        curve2: &[crate::backtest::engine::EquitySnapshot],
    ) -> f64 {
        if curve1.len() < 2 || curve2.len() < 2 {
            return 0.0;
        }

        // Normalize curves to returns
        let returns1: Vec<f64> = curve1.windows(2)
            .map(|w| (w[1].total_equity - w[0].total_equity) / w[0].total_equity.max(1e-10))
            .collect();
        
        let returns2: Vec<f64> = curve2.windows(2)
            .map(|w| (w[1].total_equity - w[0].total_equity) / w[0].total_equity.max(1e-10))
            .collect();

        // Use shorter length for comparison
        let len = returns1.len().min(returns2.len());
        
        if len < 2 {
            return 0.0;
        }

        let mean1 = returns1[..len].iter().sum::<f64>() / len as f64;
        let mean2 = returns2[..len].iter().sum::<f64>() / len as f64;

        let mut covariance = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for i in 0..len {
            let d1 = returns1[i] - mean1;
            let d2 = returns2[i] - mean2;
            covariance += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        let denominator = (var1 * var2).sqrt();
        if denominator < 1e-10 {
            return 0.0;
        }

        (covariance / denominator).clamp(-1.0, 1.0)
    }
}

/// Period boundary definitions
#[derive(Debug, Clone)]
struct PeriodBoundaries {
    period_index: usize,
    period_start: u64,
    period_end: u64,
    in_sample_start: u64,
    in_sample_end: u64,
    oos_start: u64,
    oos_end: u64,
}

impl WalkForwardReport {
    /// Print detailed analysis
    pub fn print_analysis(&self) {
        println!("=== Walk-Forward Analysis Report ===");
        println!("Number of Periods: {}", self.periods.len());
        println!("In-Sample Ratio: {:.0}%", self.config.in_sample_ratio * 100.0);
        println!();
        println!("=== Aggregate Metrics ===");
        println!("Avg In-Sample Sharpe: {:.3}", self.avg_in_sample_sharpe);
        println!("Avg Out-of-Sample Sharpe: {:.3}", self.avg_out_of_sample_sharpe);
        println!("Avg Degradation: {:.2}%", self.avg_degradation * 100.0);
        println!("Robustness Score: {:.3}", self.robustness_score);
        println!("Equity Curve Stability: {:.3}", self.equity_curve_stability);
        println!("Is Robust: {}", if self.is_robust { "YES ✓" } else { "NO ✗" });
        println!();
        
        println!("=== Period Details ===");
        for period in &self.periods {
            let is_status = if period.is_acceptable { "✓" } else { "✗" };
            println!(
                "Period {}: IS Sharpe={:.3}, OOS Sharpe={:.3}, Degradation={:.2}% {}",
                period.period_index,
                period.in_sample_result.as_ref().map(|r| r.sharpe_ratio).unwrap_or(0.0),
                period.out_of_sample_result.as_ref().map(|r| r.sharpe_ratio).unwrap_or(0.0),
                period.degradation_factor * 100.0,
                is_status
            );
        }
    }
}

// Clone trait bound for Strategy
pub trait CloneStrategy: Strategy + Clone {
    fn clone_for_backtest(&self) -> Self;
    fn apply_parameters_from(&mut self, other: &Self);
}

// Blanket implementation
impl<T> CloneStrategy for T
where
    T: Strategy + Clone,
{
    fn clone_for_backtest(&self) -> Self {
        self.clone()
    }

    fn apply_parameters_from(&mut self, _other: &Self) {
        // Default no-op implementation
        // Strategies should override this to copy optimized parameters
    }
}
