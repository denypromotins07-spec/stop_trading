//! Multi-Path Network Routing Module
//!
//! Software-defined multi-path TCP/QUIC manager for bandwidth aggregation
//! and latency variance reduction in HFT infrastructure.

pub mod multipath;
pub mod anycast_sim;

pub use multipath::MultiPathManager;
pub use anycast_sim::AnycastRouter;
