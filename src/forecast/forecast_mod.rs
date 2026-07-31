//! Online Time-Series Forecasting Module
//!
//! Implements ARMA/ARFIMA models with Recursive Least Squares for
//! O(1) per-tick updates without storing massive design matrices.

pub mod arma;
pub mod arima;

pub use arma::OnlineARMA;
pub use arima::ARFIMA;
