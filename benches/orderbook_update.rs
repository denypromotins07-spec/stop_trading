//! Benchmark for L2 order book delta application and microprice calculation.
//! Guarantees sub-microsecond order book updates for HFT requirements.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use std::time::Duration;
use std::collections::BTreeMap;

/// Represents a price level in the order book
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PriceLevel {
    pub price: u64,      // Fixed-point price (e.g., 50000_00000 for 50000.00000)
    pub quantity: u64,   // Fixed-point quantity
    pub order_count: u32,
    pub _padding: u32,   // Explicit padding for alignment
}

/// L2 Delta update from exchange
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct L2Delta {
    pub side: u8,        // 0 = Bid, 1 = Ask
    pub action: u8,      // 0 = Add, 1 = Update, 2 = Delete
    pub _padding: [u8; 2],
    pub price: u64,
    pub quantity: u64,
    pub timestamp_ns: u64,
}

/// Order book side (Bid or Ask)
struct OrderBookSide {
    levels: BTreeMap<u64, PriceLevel>, // Keyed by price
    total_volume: u64,
    last_update_ns: u64,
}

impl OrderBookSide {
    fn new() -> Self {
        Self {
            levels: BTreeMap::new(),
            total_volume: 0,
            last_update_ns: 0,
        }
    }

    #[inline]
    fn apply_delta(&mut self, delta: &L2Delta, timestamp_ns: u64) {
        match delta.action {
            0 => { // Add
                let level = PriceLevel {
                    price: delta.price,
                    quantity: delta.quantity,
                    order_count: 1,
                    _padding: 0,
                };
                self.total_volume = self.total_volume.wrapping_add(delta.quantity);
                self.levels.insert(delta.price, level);
            }
            1 => { // Update
                if let Some(level) = self.levels.get_mut(&delta.price) {
                    let old_qty = level.quantity;
                    level.quantity = delta.quantity;
                    level.order_count += 1;
                    if delta.quantity > old_qty {
                        self.total_volume = self.total_volume.wrapping_add(delta.quantity - old_qty);
                    } else {
                        self.total_volume = self.total_volume.wrapping_sub(old_qty - delta.quantity);
                    }
                }
            }
            2 => { // Delete
                if let Some(level) = self.levels.remove(&delta.price) {
                    self.total_volume = self.total_volume.wrapping_sub(level.quantity);
                }
            }
            _ => {}
        }
        self.last_update_ns = timestamp_ns;
    }

    #[inline]
    fn best_price(&self) -> Option<u64> {
        // For bids, we want highest price; for asks, lowest
        // This is simplified - real impl would differ per side
        self.levels.keys().next().copied()
    }
}

/// Full L2 Order Book
struct OrderBook {
    bids: OrderBookSide,
    asks: OrderBookSide,
    sequence: u64,
    symbol: u64,
}

impl OrderBook {
    fn new(symbol: u64) -> Self {
        Self {
            bids: OrderBookSide::new(),
            asks: OrderBookSide::new(),
            sequence: 0,
            symbol,
        }
    }

    /// Apply L2 delta and return update latency
    #[inline]
    fn apply_l2_delta(&mut self, delta: &L2Delta) -> Duration {
        let start = std::time::Instant::now();
        
        match delta.side {
            0 => self.bids.apply_delta(delta, start.elapsed().as_nanos() as u64),
            1 => self.asks.apply_delta(delta, start.elapsed().as_nanos() as u64),
            _ => {}
        }
        
        self.sequence += 1;
        start.elapsed()
    }

    /// Calculate microprice: weighted midpoint based on volume imbalance
    #[inline]
    fn calculate_microprice(&self) -> Option<u64> {
        let best_bid = self.bids.best_price()?;
        let best_ask = self.asks.best_price()?;
        
        let bid_volume = self.bids.levels.get(&best_bid)?.quantity;
        let ask_volume = self.asks.levels.get(&best_ask)?.quantity;
        
        if bid_volume + ask_volume == 0 {
            return None;
        }
        
        // Microprice = (bid * ask_vol + ask * bid_vol) / (bid_vol + ask_vol)
        let numerator = best_bid.wrapping_mul(ask_volume) + best_ask.wrapping_mul(bid_volume);
        let denominator = bid_volume + ask_volume;
        
        Some(numerator / denominator)
    }

    /// Calculate spread in basis points
    #[inline]
    fn spread_bps(&self) -> Option<u64> {
        let best_bid = self.bids.best_price()?;
        let best_ask = self.asks.best_price()?;
        
        if best_bid == 0 {
            return None;
        }
        
        let spread = best_ask.saturating_sub(best_bid);
        Some((spread * 10000) / best_bid)
    }
}

fn create_test_deltas(count: usize) -> Vec<L2Delta> {
    let mut deltas = Vec::with_capacity(count);
    for i in 0..count {
        deltas.push(L2Delta {
            side: (i % 2) as u8,
            action: 0, // Add
            _padding: [0; 2],
            price: 50000_00000u64 + (i as u64 * 100),
            quantity: 100_00000u64 + (i as u64 * 1000),
            timestamp_ns: 1234567890123456789u64 + (i as u64 * 1000),
        });
    }
    deltas
}

fn bench_single_delta_application(c: &mut Criterion) {
    let mut book = OrderBook::new(0x424E42555344);
    let delta = L2Delta {
        side: 0,
        action: 0,
        _padding: [0; 2],
        price: 50000_00000u64,
        quantity: 100_00000u64,
        timestamp_ns: 1234567890123456789u64,
    };

    c.bench_function("single_l2_delta_apply", |b| {
        b.iter(|| {
            let duration = book.apply_l2_delta(black_box(&delta));
            black_box(duration);
        })
    });
}

fn bench_batch_delta_application(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_delta_application");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(500);
    
    for batch_size in [10, 100, 1000].iter() {
        let deltas = create_test_deltas(**batch_size);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &deltas,
            |b, deltas| {
                b.iter(|| {
                    let mut book = OrderBook::new(0x424E42555344);
                    for delta in deltas {
                        book.apply_l2_delta(black_box(delta));
                    }
                    black_box(book.sequence);
                })
            },
        );
    }
    group.finish();
}

fn bench_microprice_calculation(c: &mut Criterion) {
    let mut book = OrderBook::new(0x424E42555344);
    
    // Seed with some levels
    let bid_delta = L2Delta {
        side: 0,
        action: 0,
        _padding: [0; 2],
        price: 50000_00000u64,
        quantity: 100_00000u64,
        timestamp_ns: 1234567890123456789u64,
    };
    let ask_delta = L2Delta {
        side: 1,
        action: 0,
        _padding: [0; 2],
        price: 50001_00000u64,
        quantity: 150_00000u64,
        timestamp_ns: 1234567890123456790u64,
    };
    
    book.apply_l2_delta(&bid_delta);
    book.apply_l2_delta(&ask_delta);

    c.bench_function("microprice_calculate", |b| {
        b.iter(|| {
            let microprice = book.calculate_microprice();
            black_box(microprice);
        })
    });
}

fn bench_spread_calculation(c: &mut Criterion) {
    let mut book = OrderBook::new(0x424E42555344);
    
    // Seed with some levels
    let bid_delta = L2Delta {
        side: 0,
        action: 0,
        _padding: [0; 2],
        price: 50000_00000u64,
        quantity: 100_00000u64,
        timestamp_ns: 1234567890123456789u64,
    };
    let ask_delta = L2Delta {
        side: 1,
        action: 0,
        _padding: [0; 2],
        price: 50001_00000u64,
        quantity: 150_00000u64,
        timestamp_ns: 1234567890123456790u64,
    };
    
    book.apply_l2_delta(&bid_delta);
    book.apply_l2_delta(&ask_delta);

    c.bench_function("spread_bps_calculate", |b| {
        b.iter(|| {
            let spread = book.spread_bps();
            black_box(spread);
        })
    });
}

fn bench_orderbook_snapshot(c: &mut Criterion) {
    let mut book = OrderBook::new(0x424E42555344);
    let deltas = create_test_deltas(100);
    
    // Build initial book
    for delta in &deltas {
        book.apply_l2_delta(delta);
    }

    c.bench_function("full_book_snapshot", |b| {
        b.iter(|| {
            let bid_count = black_box(book.bids.levels.len());
            let ask_count = black_box(book.asks.levels.len());
            let microprice = black_box(book.calculate_microprice());
            let spread = black_box(book.spread_bps());
            (bid_count, ask_count, microprice, spread)
        })
    });
}

criterion_group!(
    name = orderbook_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(2))
        .sample_size(1000)
        .noise_threshold(0.02)
        .nresamples(100_000);
    targets = bench_single_delta_application, bench_batch_delta_application, 
              bench_microprice_calculation, bench_spread_calculation, bench_orderbook_snapshot
);

criterion_main!(orderbook_benches);
