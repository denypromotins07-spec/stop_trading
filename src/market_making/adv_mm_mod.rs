//! Advanced Market Making Module Root
//!
//! Integrates stochastic quotes with L3 queue-position tracking.

pub mod avellaneda;
pub mod inventory_penalty;

use std::sync::Arc;
use avellaneda::{AvellanedaStoikovMM, HJBLookupTable, ASParameters, StochVolParams};
use inventory_penalty::InventoryPenaltyCalculator;

/// Combined market making engine
pub struct AdvancedMMEngine {
    /// Avellaneda-Stoikov MM instance
    pub as_mm: Arc<AvellanedaStoikovMM>,
    /// Inventory penalty calculator
    pub penalty_calc: Arc<InventoryPenaltyCalculator>,
}

impl AdvancedMMEngine {
    pub fn new(
        as_params: ASParameters,
        stoch_vol_params: StochVolParams,
        lookup_table: Arc<HJBLookupTable>,
        buffer_size: usize,
    ) -> Self {
        let as_mm = Arc::new(AvellanedaStoikovMM::new(
            as_params,
            stoch_vol_params,
            lookup_table,
            buffer_size,
        ));

        let penalty_calc = Arc::new(InventoryPenaltyCalculator::new(
            Default::default(),
            Default::default(),
            buffer_size,
        ));

        Self {
            as_mm,
            penalty_calc,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let table = Arc::new(HJBLookupTable::new(100, 100, 50, 100.0, 300.0, 2.0));
        let engine = AdvancedMMEngine::new(
            ASParameters::default(),
            StochVolParams::default(),
            table,
            1000,
        );

        assert!(engine.as_mm.get_inventory() == 0);
    }
}
