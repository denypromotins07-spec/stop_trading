//! Criterion benchmarks for measuring tick-to-trade latency.
//! Measures the full Disruptor loop from market data ingestion to order submission.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};

// Mock structures simulating the core engine components
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Tick {
    pub symbol: u64,
    pub price: u64,
    pub quantity: u64,
    pub timestamp_ns: u64,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Order {
    pub symbol: u64,
    pub side: u8, // 0 = Buy, 1 = Sell
    pub price: u64,
    pub quantity: u64,
    pub order_id: u64,
    pub timestamp_ns: u64,
}

/// Simulates the Disruptor ring buffer slot
struct DisruptorSlot {
    data: AtomicU64,
    sequence: AtomicU64,
}

impl DisruptorSlot {
    fn new() -> Self {
        Self {
            data: AtomicU64::new(0),
            sequence: AtomicU64::new(0),
        }
    }
}

/// Simulates the full tick-to-trade pipeline
struct TradingPipeline {
    slots: Vec<DisruptorSlot>,
    buffer_size: usize,
    write_seq: AtomicU64,
    read_seq: AtomicU64,
}

impl TradingPipeline {
    fn new(size: usize) -> Self {
        let mut slots = Vec::with_capacity(size);
        for _ in 0..size {
            slots.push(DisruptorSlot::new());
        }
        Self {
            slots,
            buffer_size: size,
            write_seq: AtomicU64::new(0),
            read_seq: AtomicU64::new(0),
        }
    }

    #[inline]
    fn publish_tick(&self, tick: &Tick) -> u64 {
        let seq = self.write_seq.fetch_add(1, Ordering::Relaxed);
        let idx = (seq as usize) % self.buffer_size;
        
        // Write tick data to slot (simulated as u64 chunks)
        let tick_ptr = tick as *const Tick as *const u64;
        unsafe {
            for i in 0..4 {
                self.slots[idx].data.store(tick_ptr.add(i).read(), Ordering::Release);
            }
        }
        self.slots[idx].sequence.store(seq, Ordering::Release);
        seq
    }

    #[inline]
    fn process_tick(&self, seq: u64) -> Option<Order> {
        let idx = (seq as usize) % self.buffer_size;
        
        // Wait for sequence (spinlock simulation)
        while self.slots[idx].sequence.load(Ordering::Acquire) != seq {
            std::hint::spin_loop();
        }

        // Read tick data
        let tick_ptr = &self.slots[idx].data as *const AtomicU64 as *const u64;
        let tick = unsafe {
            Tick {
                symbol: tick_ptr.add(0).read(),
                price: tick_ptr.add(1).read(),
                quantity: tick_ptr.add(2).read(),
                timestamp_ns: tick_ptr.add(3).read(),
                flags: 0,
            }
        };

        // Simulate strategy decision (microprice calculation)
        let decision_time = Instant::now();
        let _microprice = tick.price.wrapping_mul(1001) / 1000;
        
        // Generate order
        Some(Order {
            symbol: tick.symbol,
            side: if tick.price % 2 == 0 { 0 } else { 1 },
            price: tick.price,
            quantity: tick.quantity,
            order_id: seq,
            timestamp_ns: decision_time.elapsed().as_nanos() as u64,
        })
    }

    #[inline]
    fn submit_order(&self, order: &Order) {
        // Simulate order submission to exchange gateway
        black_box(order);
    }

    /// Full tick-to-trade loop
    #[inline]
    fn tick_to_trade(&self, tick: &Tick) -> Duration {
        let start = Instant::now();
        let seq = self.publish_tick(tick);
        if let Some(order) = self.process_tick(seq) {
            self.submit_order(&order);
        }
        start.elapsed()
    }
}

fn create_test_tick() -> Tick {
    Tick {
        symbol: 0x424E42555344, // "BNBUSDT" as u64
        price: 50000_00000, // 50000.00000
        quantity: 100_00000,
        timestamp_ns: 1234567890123456789,
        flags: 0x01,
    }
}

fn bench_full_tick_to_trade(c: &mut Criterion) {
    let pipeline = TradingPipeline::new(1024);
    let tick = create_test_tick();

    c.bench_function("tick_to_trade_full_loop", |b| {
        b.iter(|| {
            let duration = pipeline.tick_to_trade(black_box(&tick));
            black_box(duration);
        })
    });
}

fn bench_disruptor_publish(c: &mut Criterion) {
    let pipeline = TradingPipeline::new(1024);
    let tick = create_test_tick();

    c.bench_function("disruptor_publish_only", |b| {
        b.iter(|| {
            let seq = pipeline.publish_tick(black_box(&tick));
            black_box(seq);
        })
    });
}

fn bench_disruptor_consume(c: &mut Criterion) {
    let pipeline = TradingPipeline::new(1024);
    let tick = create_test_tick();
    
    // Pre-publish a tick
    let seq = pipeline.publish_tick(&tick);

    c.bench_function("disruptor_consume_only", |b| {
        b.iter(|| {
            let order = pipeline.process_tick(black_box(seq));
            black_box(order);
        })
    });
}

fn bench_orderbook_microprice(c: &mut Criterion) {
    c.bench_function("microprice_calculation", |b| {
        b.iter(|| {
            let bid_price = black_box(50000_00000u64);
            let ask_price = black_box(50001_00000u64);
            let bid_qty = black_box(100_00000u64);
            let ask_qty = black_box(150_00000u64);
            
            // Microprice = (bid_price * ask_qty + ask_price * bid_qty) / (bid_qty + ask_qty)
            let numerator = bid_price.wrapping_mul(ask_qty) + ask_price.wrapping_mul(bid_qty);
            let denominator = bid_qty + ask_qty;
            let microprice = numerator / denominator;
            
            black_box(microprice);
        })
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(3))
        .sample_size(1000)
        .noise_threshold(0.05)
        .nresamples(100_000);
    targets = bench_full_tick_to_trade, bench_disruptor_publish, bench_disruptor_consume, bench_orderbook_microprice
);

criterion_main!(benches);
