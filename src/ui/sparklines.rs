//! Ultra-Fast Sparkline and Equity Curve Renderer
//!
//! Zero-allocation rendering using Unicode braille characters
//! for maximum data density in the TUI dashboard.

/// Unicode braille characters for sparklines (8 dots per character)
/// Each character represents 8 vertical levels
const BRAILLE_CHARS: [[char; 8]; 256] = generate_braille_table();

/// Generate braille character table at compile time
const fn generate_braille_table() -> [[char; 8]; 256] {
    // Simplified: use actual braille unicode range
    // In practice, we compute these dynamically
    let mut table = [[' '; 8]; 256];
    let mut i = 0;
    while i < 256 {
        // Braille pattern: U+2800 + byte_value gives the character
        // We store the offset instead to avoid const fn limitations
        table[i][0] = '\u{2800}'; // Placeholder
        i += 1;
    }
    table
}

/// Actual braille base character
const BRAILLE_BASE: u32 = 0x2800;

/// Bit patterns for braille dots
const BRAILLE_DOTS: [u32; 8] = [
    0x01, // dot 1 (top left)
    0x08, // dot 2 (middle left)
    0x10, // dot 3 (bottom left)
    0x02, // dot 4 (top right)
    0x20, // dot 5 (middle right)
    0x40, // dot 6 (bottom right)
    0x80, // dot 7 (bottom-left extended)
    0x04, // dot 8 (bottom-right extended)
];

/// Pre-allocated buffer for sparkline rendering
pub struct SparklineBuffer {
    /// Pre-allocated string buffer
    buffer: String,
    /// Maximum width
    max_width: usize,
    /// Current data point count
    data_count: usize,
}

impl SparklineBuffer {
    pub fn new(max_width: usize) -> Self {
        // Each braille char = 4 bytes UTF-8 + some margin
        let capacity = max_width * 6;
        
        SparklineBuffer {
            buffer: String::with_capacity(capacity),
            max_width,
            data_count: 0,
        }
    }

    /// Clear for next render (zero allocation)
    #[inline]
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.data_count = 0;
    }

    /// Get rendered content
    #[inline]
    pub fn content(&self) -> &str {
        &self.buffer
    }

    /// Append a braille character
    #[inline]
    pub fn append_braille(&mut self, pattern: u8) {
        let code_point = BRAILLE_BASE + pattern as u32;
        if let Some(ch) = char::from_u32(code_point) {
            self.buffer.push(ch);
        }
        self.data_count += 1;
    }

    /// Append raw text (for labels)
    #[inline]
    pub fn append_text(&mut self, text: &str) {
        self.buffer.push_str(text);
    }

    /// Check capacity
    #[inline]
    pub fn is_full(&self) -> bool {
        self.data_count >= self.max_width
    }
}

/// Sparkline rendering style
#[derive(Debug, Clone, Copy)]
pub enum SparklineStyle {
    /// Simple line using braille
    Braille,
    /// Block-based bar chart
    Blocks,
    /// ASCII fallback
    Ascii,
}

/// Data series for sparkline rendering
pub struct DataSeries {
    /// Fixed-size circular buffer for values
    values: [f64; 512],
    /// Write index
    write_idx: usize,
    /// Count of valid values
    count: usize,
    /// Cached min value
    cached_min: f64,
    /// Cached max value
    cached_max: f64,
    /// Whether cache is valid
    cache_valid: bool,
}

impl DataSeries {
    pub fn new() -> Self {
        DataSeries {
            values: [0.0; 512],
            write_idx: 0,
            count: 0,
            cached_min: 0.0,
            cached_max: 0.0,
            cache_valid: false,
        }
    }

    /// Add a new value (circular buffer)
    #[inline]
    pub fn push(&mut self, value: f64) {
        self.values[self.write_idx] = value;
        self.write_idx = (self.write_idx + 1) % 512;
        if self.count < 512 {
            self.count += 1;
        }
        self.cache_valid = false;
    }

    /// Get min/max efficiently
    #[inline]
    pub fn get_range(&mut self) -> (f64, f64) {
        if !self.cache_valid || self.count == 0 {
            self.update_cache();
        }
        (self.cached_min, self.cached_max)
    }

    fn update_cache(&mut self) {
        if self.count == 0 {
            self.cached_min = 0.0;
            self.cached_max = 0.0;
            self.cache_valid = true;
            return;
        }

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for i in 0..self.count {
            let val = self.values[i];
            if val < min { min = val; }
            if val > max { max = val; }
        }

        self.cached_min = min;
        self.cached_max = max;
        self.cache_valid = true;
    }

    /// Get values as slice (in chronological order)
    pub fn values_slice(&self) -> &[f64] {
        if self.count == 0 {
            return &[];
        }

        // Return values in chronological order
        let start = if self.count < 512 {
            0
        } else {
            self.write_idx
        };

        // For simplicity, just return from start to write_idx
        // A more complete implementation would handle wrap-around
        if self.write_idx >= start {
            &self.values[start..self.write_idx]
        } else {
            &self.values[0..]
        }
    }

    /// Get number of data points
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Clear all data
    #[inline]
    pub fn clear(&mut self) {
        self.count = 0;
        self.write_idx = 0;
        self.cache_valid = false;
    }
}

impl Default for DataSeries {
    fn default() -> Self {
        Self::new()
    }
}

/// High-performance sparkline renderer
pub struct SparklineRenderer {
    /// Pre-allocated output buffer
    buffer: SparklineBuffer,
    /// Rendering style
    style: SparklineStyle,
    /// Block characters for block style
    block_chars: [char; 8],
    /// ASCII characters for ASCII style
    ascii_chars: [char; 8],
    /// Width of sparkline
    width: usize,
    /// Height (number of levels)
    height: usize,
}

impl SparklineRenderer {
    pub fn new(width: usize, height: usize, style: SparklineStyle) -> Self {
        let block_chars = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        let ascii_chars = [' ', '.', ':', '-', '=', '+', '*', '#'];

        SparklineRenderer {
            buffer: SparklineBuffer::new(width),
            style,
            block_chars: [' ', '▁', '▂', '▃', '▄', '▅', '▆', '█'],
            ascii_chars,
            width,
            height: height.min(8), // Max 8 levels for braille
        }
    }

    /// Render sparkline from data series
    pub fn render(&mut self, data: &mut DataSeries) -> &str {
        self.buffer.clear();

        if data.len() < 2 {
            self.buffer.append_text("─");
            return self.buffer.content();
        }

        let (min, max) = data.get_range();
        let range = max - min;
        
        if range < 1e-10 {
            // Flat line
            for _ in 0..self.width.min(data.len()) {
                match self.style {
                    SparklineStyle::Braille => self.buffer.append_braille(0x01), // Single dot
                    SparklineStyle::Blocks => self.buffer.append_text("▁"),
                    SparklineStyle::Ascii => self.buffer.append_text("-"),
                }
            }
            return self.buffer.content();
        }

        let values = data.values_slice();
        let step = (values.len() as f64 / self.width as f64).max(1.0) as usize;

        match self.style {
            SparklineStyle::Braille => self.render_braille(values, min, range, step),
            SparklineStyle::Blocks => self.render_blocks(values, min, range, step),
            SparklineStyle::Ascii => self.render_ascii(values, min, range, step),
        }

        self.buffer.content()
    }

    /// Render using braille characters (4 bits = 8 vertical levels)
    fn render_braille(&mut self, values: &[f64], min: f64, range: f64, step: usize) {
        let mut i = 0;
        while i < values.len() && !self.buffer.is_full() {
            // Sample 4 consecutive points for one braille character
            let mut pattern: u8 = 0;
            
            for j in 0..4 {
                let idx = i + j;
                if idx >= values.len() {
                    break;
                }
                
                let normalized = (values[idx] - min) / range;
                let level = (normalized * 3.0) as usize; // 0-3 for each dot pair
                
                // Map to braille dot pattern
                match j {
                    0 => {
                        if level >= 1 { pattern |= BRAILLE_DOTS[0] as u8; }
                        if level >= 2 { pattern |= BRAILLE_DOTS[1] as u8; }
                        if level >= 3 { pattern |= BRAILLE_DOTS[2] as u8; }
                    }
                    1 => {
                        if level >= 1 { pattern |= BRAILLE_DOTS[3] as u8; }
                        if level >= 2 { pattern |= BRAILLE_DOTS[4] as u8; }
                        if level >= 3 { pattern |= BRAILLE_DOTS[5] as u8; }
                    }
                    2 => {
                        if level >= 1 { pattern |= BRAILLE_DOTS[6] as u8; }
                    }
                    3 => {
                        if level >= 1 { pattern |= BRAILLE_DOTS[7] as u8; }
                    }
                    _ => {}
                }
            }
            
            self.buffer.append_braille(pattern);
            i += 4;
        }
    }

    /// Render using block characters
    fn render_blocks(&mut self, values: &[f64], min: f64, range: f64, step: usize) {
        let mut i = 0;
        while i < values.len() && !self.buffer.is_full() {
            let idx = i;
            let normalized = (values[idx] - min) / range;
            let level = (normalized * 7.0) as usize;
            
            let ch = self.block_chars[level.min(7)];
            self.buffer.append_text(&ch.to_string());
            
            i += step;
        }
    }

    /// Render using ASCII characters
    fn render_ascii(&mut self, values: &[f64], min: f64, range: f64, step: usize) {
        let mut i = 0;
        while i < values.len() && !self.buffer.is_full() {
            let idx = i;
            let normalized = (values[idx] - min) / range;
            let level = (normalized * 7.0) as usize;
            
            let ch = self.ascii_chars[level.min(7)];
            self.buffer.append_text(&ch.to_string());
            
            i += step;
        }
    }

    /// Render equity curve with drawdown visualization
    pub fn render_equity_curve(&mut self, equity: &mut DataSeries, initial: f64) -> &str {
        self.buffer.clear();

        if equity.len() < 2 {
            self.buffer.append_text("No data");
            return self.buffer.content();
        }

        let (min, max) = equity.get_range();
        
        // Calculate max drawdown
        let mut peak = initial;
        let mut max_dd = 0.0;
        for &val in equity.values_slice() {
            if val > peak {
                peak = val;
            }
            let dd = (peak - val) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        // Render main sparkline
        self.render(equity);
        
        // Append stats
        let pnl = max - initial;
        let pnl_pct = (pnl / initial) * 100.0;
        
        self.buffer.append_text(" | ");
        
        if pnl >= 0.0 {
            self.buffer.append_text("+");
        }
        self.buffer.append_text(&format!("{:.2}% DD:{:.1}%", pnl_pct, max_dd * 100.0));

        self.buffer.content()
    }

    /// Render CVD (Cumulative Volume Delta) sparkline
    pub fn render_cvd(&mut self, cvd: &mut DataSeries) -> &str {
        self.buffer.clear();

        if cvd.len() < 2 {
            self.buffer.append_text("CVD: --");
            return self.buffer.content();
        }

        let (min, max) = cvd.get_range();
        let current = cvd.values_slice().last().copied().unwrap_or(0.0);
        
        // Color based on trend
        let first = cvd.values_slice().first().copied().unwrap_or(0.0);
        let trend = if current >= first { "▲" } else { "▼" };
        
        self.render(cvd);
        self.buffer.append_text(&format!(" {}", trend));

        self.buffer.content()
    }

    /// Set rendering style
    pub fn set_style(&mut self, style: SparklineStyle) {
        self.style = style;
    }

    /// Get buffer dimensions
    pub fn dimensions(&self) -> usize {
        self.width
    }
}

impl Default for SparklineRenderer {
    fn default() -> Self {
        Self::new(50, 4, SparklineStyle::Braille)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_series() {
        let mut series = DataSeries::new();
        
        for i in 0..100 {
            series.push(i as f64);
        }
        
        assert_eq!(series.len(), 100);
        let (min, max) = series.get_range();
        assert!((min - 0.0).abs() < 1e-10);
        assert!((max - 99.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparkline_render() {
        let mut renderer = SparklineRenderer::new(20, 4, SparklineStyle::Blocks);
        let mut data = DataSeries::new();
        
        for i in 0..50 {
            data.push((i % 10) as f64);
        }
        
        let output = renderer.render(&mut data);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_equity_curve() {
        let mut renderer = SparklineRenderer::new(30, 4, SparklineStyle::Braille);
        let mut equity = DataSeries::new();
        
        // Simulate equity curve with some drawdown
        let mut eq = 10000.0;
        for _ in 0..100 {
            equity.push(eq);
            eq *= 1.0 + (rand_f64() - 0.5) * 0.02;
        }
        
        let output = renderer.render_equity_curve(&mut equity, 10000.0);
        assert!(output.contains("DD:"));
    }

    fn rand_f64() -> f64 {
        static mut SEED: u64 = 54321;
        unsafe {
            SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (SEED as f64) / (u64::MAX as f64)
        }
    }
}
