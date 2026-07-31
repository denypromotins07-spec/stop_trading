//! Ichimoku Cloud Implementation
//! Implements Tenkan, Kijun, Senkou, Chikou using cache-line padded rolling window buffers.
//! Stores only minimal required high/low/close primitives - no full candle history.

use core::sync::atomic::{AtomicUsize, Ordering};

/// Maximum lookback periods (9, 26, 52 are standard)
const MAX_LOOKBACK: usize = 52;

/// Cache-line padding size for optimal CPU cache utilization
const CACHE_LINE_SIZE: usize = 64;

/// Rolling window buffer for high/low tracking
#[repr(C)]
struct RollingWindow {
    /// Circular buffer of highs
    highs: [i64; MAX_LOOKBACK],
    /// Circular buffer of lows
    lows: [i64; MAX_LOOKBACK],
    /// Current write index
    index: AtomicUsize,
    /// Count of valid entries
    count: AtomicUsize,
    /// Cached midpoint sum for fast calculation
    _padding: [u8; 32], // Additional padding to avoid false sharing
}

impl RollingWindow {
    const fn new() -> Self {
        Self {
            highs: [0; MAX_LOOKBACK],
            lows: [0; MAX_LOOKBACK],
            index: AtomicUsize::new(0),
            count: AtomicUsize::new(0),
            _padding: [0; 32],
        }
    }

    #[inline]
    fn push(&mut self, high: i64, low: i64) {
        let idx = self.index.load(Ordering::Relaxed);
        self.highs[idx] = high;
        self.lows[idx] = low;
        
        self.index.store((idx + 1) % MAX_LOOKBACK, Ordering::Relaxed);
        
        let count = self.count.load(Ordering::Relaxed);
        if count < MAX_LOOKBACK {
            self.count.store(count + 1, Ordering::Relaxed);
        }
    }

    /// Find highest high in the last n periods
    #[inline]
    fn highest_high(&self, n: usize) -> i64 {
        let count = self.count.load(Ordering::Relaxed);
        let actual_n = n.min(count);
        if actual_n == 0 {
            return 0;
        }

        let idx = self.index.load(Ordering::Relaxed);
        let mut max_high = i64::MIN;

        for i in 0..actual_n {
            let read_idx = (idx.wrapping_sub(i + 1) + MAX_LOOKBACK) % MAX_LOOKBACK;
            let h = unsafe { *self.highs.get_unchecked(read_idx) };
            if h > max_high {
                max_high = h;
            }
        }

        max_high
    }

    /// Find lowest low in the last n periods
    #[inline]
    fn lowest_low(&self, n: usize) -> i64 {
        let count = self.count.load(Ordering::Relaxed);
        let actual_n = n.min(count);
        if actual_n == 0 {
            return i64::MAX;
        }

        let idx = self.index.load(Ordering::Relaxed);
        let mut min_low = i64::MAX;

        for i in 0..actual_n {
            let read_idx = (idx.wrapping_sub(i + 1) + MAX_LOOKBACK) % MAX_LOOKBACK;
            let l = unsafe { *self.lows.get_unchecked(read_idx) };
            if l < min_low {
                min_low = l;
            }
        }

        min_low
    }

    /// Get close price from n periods ago
    #[inline]
    fn close_ago(&self, closes: &[i64; MAX_LOOKBACK], n: usize) -> i64 {
        let count = self.count.load(Ordering::Relaxed);
        if n >= count {
            return 0;
        }

        let idx = self.index.load(Ordering::Relaxed);
        let read_idx = (idx.wrapping_sub(n + 1) + MAX_LOOKBACK) % MAX_LOOKBACK;
        unsafe { *closes.get_unchecked(read_idx) }
    }
}

/// Ichimoku Cloud components
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IchimokuComponents {
    /// Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    pub tenkan: i64,
    /// Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    pub kijun: i64,
    /// Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, projected 26 periods ahead
    pub senkou_a: i64,
    /// Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, projected 26 periods ahead
    pub senkou_b: i64,
    /// Chikou Span (Lagging Span): Current close plotted 26 periods back
    pub chikou: i64,
    /// Current close for reference
    pub close: i64,
    _padding: [u8; 8], // Cache-line alignment
}

impl Default for IchimokuComponents {
    fn default() -> Self {
        Self::new()
    }
}

impl IchimokuComponents {
    pub const fn new() -> Self {
        Self {
            tenkan: 0,
            kijun: 0,
            senkou_a: 0,
            senkou_b: 0,
            chikou: 0,
            close: 0,
            _padding: [0; 8],
        }
    }
}

/// Signal type from Ichimoku analysis
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum IchimokuSignal {
    None = 0,
    BullishTKCross = 1,      // Tenkan crosses above Kijun
    BearishTKCross = 2,     // Tenkan crosses below Kijun
    BullishCloudBreak = 3,  // Price breaks above cloud
    BearishCloudBreak = 4,  // Price breaks below cloud
    BullishChikou = 5,      // Chikou above price
    BearishChikou = 6,      // Chikou below price
    StrongBullish = 7,      // All bullish signals aligned
    StrongBearish = 8,      // All bearish signals aligned
}

/// Main Ichimoku calculator with optimized rolling windows
pub struct IchimokuCalculator {
    /// Primary rolling window for all calculations
    window: RollingWindow,
    /// Rolling closes for Chikou span
    closes: [i64; MAX_LOOKBACK],
    /// Standard periods: (tenkan, kijun, senkou)
    tenkan_period: usize,
    kijun_period: usize,
    senkou_period: usize,
    /// Leading offset (typically 26)
    leading_offset: usize,
    /// Previous values for signal detection
    prev_tenkan: i64,
    prev_kijun: i64,
    _padding: [u8; 48], // Ensure cache-line separation
}

impl Default for IchimokuCalculator {
    fn default() -> Self {
        Self::new(9, 26, 52)
    }
}

impl IchimokuCalculator {
    /// Create new calculator with custom periods
    pub const fn new(tenkan_period: usize, kijun_period: usize, senkou_period: usize) -> Self {
        Self {
            window: RollingWindow::new(),
            closes: [0; MAX_LOOKBACK],
            tenkan_period,
            kijun_period,
            senkou_period,
            leading_offset: 26,
            prev_tenkan: 0,
            prev_kijun: 0,
            _padding: [0; 48],
        }
    }

    /// Process a new candle and return updated Ichimoku components
    pub fn process_candle(
        &mut self,
        high: i64,
        low: i64,
        close: i64,
    ) -> Option<IchimokuComponents> {
        // Store close for Chikou calculation
        let idx = self.window.index.load(Ordering::Relaxed);
        self.closes[idx] = close;

        // Update rolling window
        self.window.push(high, low);

        let count = self.window.count.load(Ordering::Relaxed);
        
        // Need at least senkou_period candles for full calculation
        if count < self.senkou_period {
            return None;
        }

        // Calculate Tenkan-sen (Conversion Line)
        let tenkan_high = self.window.highest_high(self.tenkan_period);
        let tenkan_low = self.window.lowest_low(self.tenkan_period);
        let tenkan = (tenkan_high + tenkan_low) / 2;

        // Calculate Kijun-sen (Base Line)
        let kijun_high = self.window.highest_high(self.kijun_period);
        let kijun_low = self.window.lowest_low(self.kijun_period);
        let kijun = (kijun_high + kijun_low) / 2;

        // Calculate Senkou Span A (projected forward)
        let senkou_a = (tenkan + kijun) / 2;

        // Calculate Senkou Span B (projected forward)
        let senkou_b_high = self.window.highest_high(self.senkou_period);
        let senkou_b_low = self.window.lowest_low(self.senkou_period);
        let senkou_b = (senkou_b_high + senkou_b_low) / 2;

        // Calculate Chikou Span (lagging - current close plotted back)
        let chikou = close; // In real usage, this would be compared to price n periods ago

        let components = IchimokuComponents {
            tenkan,
            kijun,
            senkou_a,
            senkou_b,
            chikou,
            close,
            _padding: [0; 8],
        };

        // Store for signal detection
        self.prev_tenkan = tenkan;
        self.prev_kijun = kijun;

        Some(components)
    }

    /// Detect trading signals based on component relationships
    pub fn detect_signal(
        &self,
        components: &IchimokuComponents,
        price_26_ago: i64,
    ) -> IchimokuSignal {
        let mut bullish_signals = 0u8;
        let mut bearish_signals = 0u8;

        // TK Cross detection
        if components.tenkan > components.kijun {
            bullish_signals += 1;
            if self.prev_tenkan <= self.prev_kijun {
                return IchimokuSignal::BullishTKCross;
            }
        } else {
            bearish_signals += 1;
            if self.prev_tenkan >= self.prev_kijun {
                return IchimokuSignal::BearishTKCross;
            }
        }

        // Cloud position
        let cloud_top = components.senkou_a.max(components.senkou_b);
        let cloud_bottom = components.senkou_a.min(components.senkou_b);

        if components.close > cloud_top {
            bullish_signals += 1;
        } else if components.close < cloud_bottom {
            bearish_signals += 1;
        }

        // Chikou span vs historical price
        if components.chikou > price_26_ago {
            bullish_signals += 1;
        } else if components.chikou < price_26_ago {
            bearish_signals += 1;
        }

        // Determine overall signal
        if bullish_signals >= 3 {
            IchimokuSignal::StrongBullish
        } else if bearish_signals >= 3 {
            IchimokuSignal::StrongBearish
        } else if bullish_signals > bearish_signals {
            IchimokuSignal::BullishChikou
        } else if bearish_signals > bullish_signals {
            IchimokuSignal::BearishChikou
        } else {
            IchimokuSignal::None
        }
    }

    /// Check if price is in the cloud (neutral zone)
    #[inline]
    pub fn is_in_cloud(&self, components: &IchimokuComponents, current_price: i64) -> bool {
        let cloud_top = components.senkou_a.max(components.senkou_b);
        let cloud_bottom = components.senkou_a.min(components.senkou_b);
        current_price >= cloud_bottom && current_price <= cloud_top
    }

    /// Calculate cloud thickness (volatility indicator)
    #[inline]
    pub fn cloud_thickness(&self, components: &IchimokuComponents) -> i64 {
        (components.senkou_a - components.senkou_b).abs()
    }

    /// Get future cloud projection (Senkou spans are projected forward)
    pub fn get_future_cloud(&self, periods_ahead: usize) -> (i64, i64) {
        // This returns the cloud that will be visible 'periods_ahead' bars from now
        // Based on current Tenkan/Kijun values
        let senkou_a = (self.prev_tenkan + self.prev_kijun) / 2;
        
        // Senkou B requires looking back further - simplified here
        let senkou_b = self.prev_kijun; // Approximation
        
        (senkou_a, senkou_b)
    }

    /// Reset the calculator state
    pub fn reset(&mut self) {
        self.window = RollingWindow::new();
        self.closes = [0; MAX_LOOKBACK];
        self.prev_tenkan = 0;
        self.prev_kijun = 0;
    }
}

/// Kumo (Cloud) Twist detector - when Senkou A crosses Senkou B
pub struct KumoTwistDetector {
    prev_senkou_a: i64,
    prev_senkou_b: i64,
    initialized: bool,
}

impl KumoTwistDetector {
    pub const fn new() -> Self {
        Self {
            prev_senkou_a: 0,
            prev_senkou_b: 0,
            initialized: false,
        }
    }

    /// Check for cloud twist (trend change signal)
    pub fn check_twist(&mut self, senkou_a: i64, senkou_b: i64) -> Option<i8> {
        if !self.initialized {
            self.prev_senkou_a = senkou_a;
            self.prev_senkou_b = senkou_b;
            self.initialized = true;
            return None;
        }

        let prev_diff = self.prev_senkou_a - self.prev_senkou_b;
        let curr_diff = senkou_a - senkou_b;

        self.prev_senkou_a = senkou_a;
        self.prev_senkou_b = senkou_b;

        // Detect crossing
        if prev_diff <= 0 && curr_diff > 0 {
            Some(1) // Bullish twist
        } else if prev_diff >= 0 && curr_diff < 0 {
            Some(-1) // Bearish twist
        } else {
            None
        }
    }
}

impl Default for KumoTwistDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ichimoku_calculation() {
        let mut calc = IchimokuCalculator::default();

        // Feed 52 candles to initialize
        for i in 0..60 {
            let base = 100_0000_0000i64;
            let high = base + (i as i64) * 1000_0000i64 + 500_0000i64;
            let low = base + (i as i64) * 1000_0000i64 - 500_0000i64;
            let close = base + (i as i64) * 1000_0000i64;

            if let Some(components) = calc.process_candle(high, low, close) {
                if i >= 55 {
                    assert!(components.tenkan > 0);
                    assert!(components.kijun > 0);
                    assert!(components.senkou_a > 0);
                    assert!(components.senkou_b > 0);
                }
            }
        }
    }

    #[test]
    fn test_tk_cross_detection() {
        let mut calc = IchimokuCalculator::default();

        // Create scenario where Tenkan crosses above Kijun
        for i in 0..60 {
            let high = 100_0000_0000i64 + (i as i64) * 2000_0000i64;
            let low = 100_0000_0000i64 + (i as i64) * 1000_0000i64;
            let close = 100_0000_0000i64 + (i as i64) * 1500_0000i64;

            if let Some(components) = calc.process_candle(high, low, close) {
                let signal = calc.detect_signal(&components, close);
                // Signal should eventually trigger
                let _ = signal;
            }
        }
    }
}
