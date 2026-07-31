//! Extreme Value Theory (EVT) Module
//! 
//! Implements Generalized Pareto Distribution (GPD) and Peaks-Over-Threshold (POT)
//! modeling for extreme tail risk in crypto markets.

pub mod evt;
pub mod copula;

pub use evt::ExtremeValueTheory;
pub use copula::{Copula, GaussianCopula, StudentTCopula};
