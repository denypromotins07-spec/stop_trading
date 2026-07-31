//! Queue-Aware Market Making Module
//!
//! Implements queue-aware quoting logic and adverse selection modeling
//! for high-frequency market making strategies.

pub mod queue_aware;
pub mod adverse_selection;

pub use queue_aware::QueueAwareMarketMaker;
pub use adverse_selection::AdverseSelectionModel;
