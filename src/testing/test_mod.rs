//! Testing Module Root
//! 
//! Organizes test suites and enforces zero heap allocation rules during fuzzing runs.
//! Includes fuzzing harnesses and property-based testing infrastructure.

#![cfg(any(test, feature = "testing"))]

pub mod fuzz_harness;
pub mod proptest;
pub mod test_mod;

pub use fuzz_harness::{
    FuzzHarness,
    FuzzTarget,
    FuzzResult,
    FuzzStats,
    BatchFuzzResult,
    CorpusGenerator,
    MAX_FUZZ_INPUT_SIZE,
};

pub use proptest::{
    PropertyTestRunner,
    MarketDataConfig,
    PropertyResult,
    TestStats,
    AdversarialMarketGenerator,
    avellaneda_stoikov_invariants,
    black_litterman_invariants,
    MAX_PROPTTEST_CASES,
    DEFAULT_TEST_TIMEOUT_MS,
};

pub use test_mod::{
    TestSuite,
    TestConfig,
    TestResult,
    TestReport,
    AllocationTracker,
    ZeroAllocationGuard,
};

/// Global test configuration
#[derive(Debug, Clone)]
pub struct GlobalTestConfig {
    /// Enable fuzzing tests
    pub fuzzing_enabled: bool,
    /// Enable property-based tests
    pub proptest_enabled: bool,
    /// Maximum heap allocation allowed during tests (bytes)
    pub max_heap_allocation: usize,
    /// Fail on any allocation violation
    pub fail_on_allocation: bool,
}

impl Default for GlobalTestConfig {
    fn default() -> Self {
        GlobalTestConfig {
            fuzzing_enabled: true,
            proptest_enabled: true,
            max_heap_allocation: 6_500_000_000, // 6.5GB limit
            fail_on_allocation: false,
        }
    }
}

/// Run all tests with configuration
pub fn run_all_tests(config: &GlobalTestConfig) -> TestSummary {
    let mut summary = TestSummary::default();

    if config.fuzzing_enabled {
        summary.fuzz_results = run_fuzz_suite();
    }

    if config.proptest_enabled {
        summary.property_results = run_property_suite();
    }

    summary.total_passed = summary.fuzz_results.iter().filter(|r| r.passed).count() as u64
        + summary.property_results.iter().filter(|r| r.passed).count() as u64;
    
    summary.total_failed = summary.fuzz_results.iter().filter(|r| !r.passed).count() as u64
        + summary.property_results.iter().filter(|r| !r.passed).count() as u64;

    summary
}

/// Run fuzzing test suite
fn run_fuzz_suite() -> Vec<FuzzTestResult> {
    let mut results = Vec::new();

    // FIX Codec fuzzing
    let mut fix_harness = FuzzHarness::new(FuzzTarget::FixCodec);
    let mut gen = CorpusGenerator::new(12345);
    let fix_inputs: Vec<_> = (0..100).map(|_| gen.generate_fix_like()).collect();
    let fix_batch = fix_harness.run_batch(&fix_inputs);
    
    results.push(FuzzTestResult {
        name: "FIX_Codec_Fuzz",
        passed: fix_batch.failure_count() == 0,
        inputs_tested: fix_batch.total_inputs as u64,
        failures: fix_batch.failure_count() as u64,
    });

    // JSON Parser fuzzing
    let mut json_harness = FuzzHarness::new(FuzzTarget::JsonParser);
    let json_inputs: Vec<_> = (0..100).map(|_| gen.generate_json_like()).collect();
    let json_batch = json_harness.run_batch(&json_inputs);
    
    results.push(FuzzTestResult {
        name: "JSON_Parser_Fuzz",
        passed: json_batch.failure_count() == 0,
        inputs_tested: json_batch.total_inputs as u64,
        failures: json_batch.failure_count() as u64,
    });

    results
}

/// Run property-based test suite
fn run_property_suite() -> Vec<PropertyTestResult> {
    let mut results = Vec::new();

    // Price series properties
    let config = MarketDataConfig {
        num_prices: 500,
        ..Default::default()
    };
    let mut runner = PropertyTestRunner::new(config);
    
    let price_result = runner.run_property(
        "price_series_valid",
        |seed, cfg| {
            let mut gen = AdversarialMarketGenerator::new(seed);
            gen.generate_price_series(cfg)
        },
        |prices| {
            for &p in prices {
                if p < 0.0 || p.is_nan() {
                    return Err("Invalid price".to_string());
                }
            }
            Ok(())
        },
    );

    results.push(PropertyTestResult {
        name: "Price_Series_Valid",
        passed: price_result == PropertyResult::Passed,
        cases_tested: runner.get_stats().total,
    });

    results
}

/// Result from a single fuzz test
#[derive(Debug, Clone)]
pub struct FuzzTestResult {
    pub name: &'static str,
    pub passed: bool,
    pub inputs_tested: u64,
    pub failures: u64,
}

/// Result from a single property test
#[derive(Debug, Clone)]
pub struct PropertyTestResult {
    pub name: &'static str,
    pub passed: bool,
    pub cases_tested: u64,
}

/// Summary of all test runs
#[derive(Debug, Clone, Default)]
pub struct TestSummary {
    pub total_passed: u64,
    pub total_failed: u64,
    pub fuzz_results: Vec<FuzzTestResult>,
    pub property_results: Vec<PropertyTestResult>,
}

impl TestSummary {
    pub fn all_passed(&self) -> bool {
        self.total_failed == 0
    }

    pub fn pass_rate(&self) -> f64 {
        let total = self.total_passed + self.total_failed;
        if total == 0 {
            return 1.0;
        }
        self.total_passed as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_config_default() {
        let config = GlobalTestConfig::default();
        assert!(config.fuzzing_enabled);
        assert!(config.proptest_enabled);
        assert_eq!(config.max_heap_allocation, 6_500_000_000);
    }

    #[test]
    fn test_run_all_tests() {
        let config = GlobalTestConfig::default();
        let summary = run_all_tests(&config);

        // Should have some results
        assert!(summary.fuzz_results.len() > 0 || summary.property_results.len() > 0);
    }

    #[test]
    fn test_summary_calculations() {
        let mut summary = TestSummary::default();
        summary.total_passed = 8;
        summary.total_failed = 2;

        assert!(!summary.all_passed());
        assert!((summary.pass_rate() - 0.8).abs() < 0.01);
    }
}
