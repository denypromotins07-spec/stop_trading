//! High-Performance L2 Order Book Heatmap Renderer
//!
//! Uses custom color interpolation and block characters to visualize
//! liquidity walls and spoofing clusters in real-time.
//! All rendering uses pre-allocated string buffers to avoid GC pauses.

/// Color gradient presets for heatmap visualization
#[derive(Debug, Clone, Copy)]
pub enum HeatmapPalette {
    /// Blue (low) to Red (high) - classic financial heatmap
    BlueRed,
    /// Green (low) to Red (high) - intensity based
    GreenRed,
    /// Black to Green - matrix style
    BlackGreen,
    /// Purple to Orange - modern terminal
    PurpleOrange,
    /// Custom 3-color gradient
    Custom([u8; 3], [u8; 3], [u8; 3]),
}

/// Block characters for density visualization (increasing density)
const BLOCK_CHARS: [char; 5] = [' ', '░', '▒', '▓', '█'];

/// ANSI color codes for terminal rendering
const ANSI_RESET: &str = "\x1b[0m";
const ANSI_BOLD: &str = "\x1b[1m";

/// Pre-computed 256-color palette indices for gradients
pub struct ColorPalette {
    /// 32 pre-computed color codes for fast lookup
    colors: [String; 32],
}

impl ColorPalette {
    pub fn new(palette: HeatmapPalette) -> Self {
        let mut colors = [String::new(); 32];
        
        match palette {
            HeatmapPalette::BlueRed => {
                // Blue (21) to Red (196) through purple
                for i in 0..32 {
                    let r = (i * 7) as u8;
                    let b = (255 - i * 7) as u8;
                    colors[i] = format!("\x1b[38;2;{};0;{}m", r.min(255), b);
                }
            }
            HeatmapPalette::GreenRed => {
                // Green (green=low, red=high)
                for i in 0..32 {
                    let r = (i * 8) as u8;
                    let g = (255 - i * 8) as u8;
                    colors[i] = format!("\x1b[38;2;{};{};0m", r.min(255), g);
                }
            }
            HeatmapPalette::BlackGreen => {
                // Black to bright green
                for i in 0..32 {
                    let g = (i * 8) as u8;
                    colors[i] = format!("\x1b[38;2;0;{};0m", g.min(255));
                }
            }
            HeatmapPalette::PurpleOrange => {
                // Purple to orange
                for i in 0..32 {
                    let r = (128 + i * 4) as u8;
                    let b = (255 - i * 8) as u8;
                    let g = (i * 2) as u8;
                    colors[i] = format!("\x1b[38;2;{};{};{}m", r.min(255), g, b);
                }
            }
            HeatmapPalette::Custom(c1, c2, c3) => {
                // Interpolate through three colors
                for i in 0..32 {
                    let (r, g, b) = if i < 16 {
                        // c1 to c2
                        let t = i as f64 / 16.0;
                        (
                            ((c1[0] as f64) * (1.0 - t) + (c2[0] as f64) * t) as u8,
                            ((c1[1] as f64) * (1.0 - t) + (c2[1] as f64) * t) as u8,
                            ((c1[2] as f64) * (1.0 - t) + (c2[2] as f64) * t) as u8,
                        )
                    } else {
                        // c2 to c3
                        let t = (i - 16) as f64 / 16.0;
                        (
                            ((c2[0] as f64) * (1.0 - t) + (c3[0] as f64) * t) as u8,
                            ((c2[1] as f64) * (1.0 - t) + (c3[1] as f64) * t) as u8,
                            ((c2[2] as f64) * (1.0 - t) + (c3[2] as f64) * t) as u8,
                        )
                    };
                    colors[i] = format!("\x1b[38;2;{};{};{}m", r, g, b);
                }
            }
        }
        
        ColorPalette { colors }
    }

    /// Get color code for normalized value (0.0 to 1.0)
    #[inline]
    pub fn get_color(&self, value: f64) -> &str {
        let idx = ((value * 31.0).round() as usize).min(31);
        &self.colors[idx]
    }
}

/// Single price level in the order book
#[derive(Debug, Clone, Copy)]
pub struct PriceLevel {
    pub price: u64,
    pub volume: u64,
    pub order_count: u32,
}

/// Rendered frame buffer for zero-allocation rendering
pub struct HeatmapFrameBuffer {
    /// Pre-allocated output string (reused every frame)
    buffer: String,
    /// Width in characters
    width: usize,
    /// Height in characters
    height: usize,
    /// Maximum capacity to avoid reallocations
    max_capacity: usize,
}

impl HeatmapFrameBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        // Estimate max capacity: width * height * (color_code_len + char + reset)
        // ~ 20 bytes per cell worst case
        let max_capacity = width * height * 25;
        
        let mut buffer = String::with_capacity(max_capacity);
        buffer.fill(' ', max_capacity);
        buffer.clear();
        
        HeatmapFrameBuffer {
            buffer,
            width,
            height,
            max_capacity,
        }
    }

    /// Clear buffer for next frame (zero allocation)
    #[inline]
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Get current buffer content
    #[inline]
    pub fn content(&self) -> &str {
        &self.buffer
    }

    /// Write a colored character to the buffer
    #[inline]
    pub fn write_cell(&mut self, color: &str, ch: char) {
        self.buffer.push_str(color);
        self.buffer.push(ch);
        self.buffer.push_str(ANSI_RESET);
    }

    /// Write raw text (for labels, headers)
    #[inline]
    pub fn write_text(&mut self, text: &str) {
        self.buffer.push_str(text);
    }

    /// Write formatted number with fixed width
    #[inline]
    pub fn write_number(&mut self, num: u64, width: usize) {
        let s = format!("{:>width$}", num, width = width);
        self.buffer.push_str(&s);
    }

    /// Add newline
    #[inline]
    pub fn newline(&mut self) {
        self.buffer.push('\n');
    }

    /// Check if buffer needs resize (should rarely happen)
    #[inline]
    pub fn needs_resize(&self) -> bool {
        self.buffer.len() > self.max_capacity / 2
    }
}

/// Order Book Heatmap Renderer
pub struct OrderBookHeatmap {
    /// Pre-allocated frame buffer
    frame_buffer: HeatmapFrameBuffer,
    /// Color palette
    palette: ColorPalette,
    /// Cached max volume for normalization
    max_volume: u64,
    /// Volume decay factor for EMA of max volume
    volume_decay: f64,
    /// Number of price levels to display
    num_levels: usize,
    /// Width of volume bars
    bar_width: usize,
    /// Whether to show order counts
    show_order_counts: bool,
    /// Spoofing detection threshold (ratio)
    spoof_threshold: f64,
}

impl OrderBookHeatmap {
    pub fn new(width: usize, height: usize, palette: HeatmapPalette) -> Self {
        let num_levels = height / 2; // Bid and ask take half each
        
        OrderBookHeatmap {
            frame_buffer: HeatmapFrameBuffer::new(width, height),
            palette: ColorPalette::new(palette),
            max_volume: 1,
            volume_decay: 0.95,
            num_levels,
            bar_width: 20,
            show_order_counts: true,
            spoof_threshold: 5.0,
        }
    }

    /// Update max volume estimate using EMA
    #[inline]
    fn update_max_volume(&mut self, current_max: u64) {
        self.max_volume = ((self.max_volume as f64 * self.volume_decay) 
            + (current_max as f64 * (1.0 - self.volume_decay))) as u64;
        self.max_volume = self.max_volume.max(1);
    }

    /// Normalize volume to 0-1 range
    #[inline]
    fn normalize_volume(&self, volume: u64) -> f64 {
        (volume as f64 / self.max_volume as f64).min(1.0)
    }

    /// Detect potential spoofing (large order count relative to volume)
    #[inline]
    fn detect_spoofing(&self, volume: u64, order_count: u32) -> bool {
        if order_count == 0 || volume == 0 {
            return false;
        }
        
        let avg_size = volume as f64 / order_count as f64;
        let typical_size = self.max_volume as f64 / 100.0; // Assume 100 orders at max
        
        // Many small orders = potential spoofing
        avg_size < typical_size / self.spoof_threshold
    }

    /// Get block character for volume density
    #[inline]
    fn density_char(&self, normalized: f64) -> char {
        let idx = (normalized * 4.0) as usize;
        BLOCK_CHARS[idx.min(4)]
    }

    /// Render complete order book heatmap
    /// Returns reference to internal buffer (zero copy)
    pub fn render(&mut self, bids: &[PriceLevel], asks: &[PriceLevel]) -> &str {
        self.frame_buffer.clear();
        
        // Find max volume for normalization
        let max_vol = bids.iter()
            .chain(asks.iter())
            .map(|l| l.volume)
            .max()
            .unwrap_or(1);
        self.update_max_volume(max_vol);
        
        // Header
        self.frame_buffer.write_text("┌");
        for _ in 0..(self.bar_width + 15) {
            self.frame_buffer.write_text("─");
        }
        self.frame_buffer.write_text(" ORDER BOOK HEATMAP ");
        for _ in 0..(self.bar_width + 15) {
            self.frame_buffer.write_text("─");
        }
        self.frame_buffer.write_text("┐\n");
        
        // Column headers
        self.frame_buffer.write_text("│ ");
        self.frame_buffer.write_text("ASKS");
        for _ in 0..self.bar_width {
            self.frame_buffer.write_text(" ");
        }
        self.frame_buffer.write_text("│ BIDS ");
        for _ in 0..self.bar_width {
            self.frame_buffer.write_text(" ");
        }
        self.frame_buffer.write_text("│\n");
        
        self.frame_buffer.write_text("├");
        for _ in 0..(self.bar_width * 2 + 35) {
            self.frame_buffer.write_text("─");
        }
        self.frame_buffer.write_text("┤\n");
        
        // Render asks (top to bottom, highest price first)
        let ask_levels = asks.iter().take(self.num_levels);
        for level in ask_levels.rev() {
            self.render_ask_level(level);
        }
        
        // Mid price separator
        self.frame_buffer.write_text("├");
        for _ in 0..(self.bar_width * 2 + 35) {
            self.frame_buffer.write_text("─");
        }
        self.frame_buffer.write_text("┤\n");
        
        // Render bids (top to bottom, highest price first)
        let bid_levels = bids.iter().take(self.num_levels);
        for level in bid_levels {
            self.render_bid_level(level);
        }
        
        self.frame_buffer.write_text("└");
        for _ in 0..(self.bar_width * 2 + 35) {
            self.frame_buffer.write_text("─");
        }
        self.frame_buffer.write_text("┘\n");
        
        self.frame_buffer.content()
    }

    /// Render single ask level
    #[inline]
    fn render_ask_level(&mut self, level: &PriceLevel) {
        let norm_vol = self.normalize_volume(level.volume);
        let color = self.palette.get_color(norm_vol);
        let is_spoof = self.detect_spoofing(level.volume, level.order_count);
        
        self.frame_buffer.write_text("│ ");
        
        // Ask volume bar (right-aligned in left column)
        let bar_len = (norm_vol * self.bar_width as f64) as usize;
        for _ in 0..(self.bar_width - bar_len) {
            self.frame_buffer.write_text(" ");
        }
        
        for _ in 0..bar_len {
            let ch = if is_spoof { '⚠' } else { self.density_char(norm_vol) };
            self.frame_buffer.write_cell(color, ch);
        }
        
        // Price
        self.frame_buffer.write_text(" │");
        self.frame_buffer.write_number(level.price, 12);
        self.frame_buffer.write_text("│");
        
        if self.show_order_counts {
            self.frame_buffer.write_text(" ");
            self.frame_buffer.write_number(level.order_count as u64, 4);
            self.frame_buffer.write_text(" │");
        }
        
        self.frame_buffer.newline();
    }

    /// Render single bid level
    #[inline]
    fn render_bid_level(&mut self, level: &PriceLevel) {
        let norm_vol = self.normalize_volume(level.volume);
        let color = self.palette.get_color(norm_vol);
        let is_spoof = self.detect_spoofing(level.volume, level.order_count);
        
        self.frame_buffer.write_text("│");
        self.frame_buffer.write_number(level.price, 12);
        self.frame_buffer.write_text("│ ");
        
        // Bid volume bar (left-aligned in right column)
        let bar_len = (norm_vol * self.bar_width as f64) as usize;
        
        for _ in 0..bar_len {
            let ch = if is_spoof { '⚠' } else { self.density_char(norm_vol) };
            self.frame_buffer.write_cell(color, ch);
        }
        
        for _ in 0..(self.bar_width - bar_len) {
            self.frame_buffer.write_text(" ");
        }
        
        self.frame_buffer.write_text(" │");
        
        if self.show_order_counts {
            self.frame_buffer.write_text(" ");
            self.frame_buffer.write_number(level.order_count as u64, 4);
            self.frame_buffer.write_text(" │");
        }
        
        self.frame_buffer.newline();
    }

    /// Set bar width
    pub fn set_bar_width(&mut self, width: usize) {
        self.bar_width = width;
    }

    /// Toggle order count display
    pub fn toggle_order_counts(&mut self) {
        self.show_order_counts = !self.show_order_counts;
    }

    /// Set spoofing detection threshold
    pub fn set_spoof_threshold(&mut self, threshold: f64) {
        self.spoof_threshold = threshold;
    }

    /// Get frame buffer dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.frame_buffer.width, self.frame_buffer.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heatmap_creation() {
        let heatmap = OrderBookHeatmap::new(80, 24, HeatmapPalette::BlueRed);
        assert_eq!(heatmap.num_levels, 12);
    }

    #[test]
    fn test_volume_normalization() {
        let mut heatmap = OrderBookHeatmap::new(80, 24, HeatmapPalette::BlueRed);
        heatmap.max_volume = 1000;
        
        assert!((heatmap.normalize_volume(500) - 0.5).abs() < 0.01);
        assert!((heatmap.normalize_volume(1000) - 1.0).abs() < 0.01);
        assert!((heatmap.normalize_volume(2000) - 1.0).abs() < 0.01); // Capped at 1.0
    }

    #[test]
    fn test_render_output() {
        let mut heatmap = OrderBookHeatmap::new(80, 24, HeatmapPalette::BlueRed);
        
        let bids = vec![
            PriceLevel { price: 50000, volume: 100, order_count: 10 },
            PriceLevel { price: 49999, volume: 200, order_count: 20 },
        ];
        
        let asks = vec![
            PriceLevel { price: 50001, volume: 150, order_count: 15 },
            PriceLevel { price: 50002, volume: 50, order_count: 5 },
        ];
        
        let output = heatmap.render(&bids, &asks);
        assert!(output.contains("ORDER BOOK HEATMAP"));
        assert!(output.contains("BIDS"));
        assert!(output.contains("ASKS"));
    }

    #[test]
    fn test_spoof_detection() {
        let heatmap = OrderBookHeatmap::new(80, 24, HeatmapPalette::BlueRed);
        
        // Many small orders = spoofing
        assert!(heatmap.detect_spoofing(100, 50));
        
        // Few large orders = legitimate
        assert!(!heatmap.detect_spoofing(10000, 5));
    }
}
