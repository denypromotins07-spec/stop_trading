//! Advanced Terminal UI Module
//!
//! High-performance L2 order book heatmap and sparkline rendering
//! using pre-allocated buffers for zero-allocation 60FPS TUI.

pub mod heatmap;
pub mod sparklines;

pub use heatmap::OrderBookHeatmap;
pub use sparklines::SparklineRenderer;
