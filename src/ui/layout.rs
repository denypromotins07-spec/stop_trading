//! Terminal UI Layout System
//! 
//! Multi-pane rendering layout with diff-optimized 60FPS rendering.
//! Displays live PnL, L2 order book heatmaps, active orders, RAM usage, and AI confidence.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Sparkline, Wrap},
    Frame,
};

use crate::ui::app::App;

/// Cache-line aligned layout configuration to prevent false sharing
#[repr(align(64))]
pub struct LayoutConfig {
    pub header_height: u16,
    pub metrics_height: u16,
    pub heatmap_min_height: u16,
    pub orders_height: u16,
    pub status_height: u16,
    pub margin: u16,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            header_height: 3,
            metrics_height: 10,
            heatmap_min_height: 5,
            orders_height: 10,
            status_height: 3,
            margin: 1,
        }
    }
}

/// Render region tracker for diff-based rendering optimization
#[derive(Default, Clone)]
pub struct RenderRegions {
    pub header_changed: bool,
    pub metrics_changed: bool,
    pub heatmap_changed: bool,
    pub orders_changed: bool,
    pub status_changed: bool,
}

impl RenderRegions {
    pub fn new() -> Self {
        Self {
            header_changed: true,
            metrics_changed: true,
            heatmap_changed: true,
            orders_changed: true,
            status_changed: true,
        }
    }

    /// Check if any region needs redrawing
    pub fn needs_redraw(&self) -> bool {
        self.header_changed
            || self.metrics_changed
            || self.heatmap_changed
            || self.orders_changed
            || self.status_changed
    }

    /// Mark all regions as unchanged (after render)
    pub fn mark_rendered(&mut self) {
        self.header_changed = false;
        self.metrics_changed = false;
        self.heatmap_changed = false;
        self.orders_changed = false;
        self.status_changed = false;
    }

    /// Invalidate specific regions based on app state changes
    pub fn invalidate_from_app(&mut self, app: &App, prev_app: &PrevAppState) {
        self.header_changed = app.trading_active != prev_app.trading_active;
        self.metrics_changed = (app.pnl - prev_app.pnl).abs() > 0.01
            || app.ram_usage != prev_app.ram_usage
            || (app.ai_confidence - prev_app.ai_confidence).abs() > 0.01
            || app.latency_us != prev_app.latency_us;
        self.heatmap_changed = app.order_book_heatmap.len() != prev_app.order_book_heatmap_len;
        self.orders_changed = app.active_orders.len() != prev_app.active_orders_len;
        self.status_changed = app.show_kill_confirm != prev_app.show_kill_confirm
            || app.trading_active != prev_app.trading_active;
    }
}

/// Previous application state for diff comparison
#[derive(Default)]
pub struct PrevAppState {
    pub trading_active: bool,
    pub pnl: f64,
    pub ram_usage: u64,
    pub ai_confidence: f32,
    pub latency_us: u64,
    pub order_book_heatmap_len: usize,
    pub active_orders_len: usize,
    pub show_kill_confirm: bool,
}

impl PrevAppState {
    pub fn from_app(app: &App) -> Self {
        Self {
            trading_active: app.trading_active,
            pnl: app.pnl,
            ram_usage: app.ram_usage,
            ai_confidence: app.ai_confidence,
            latency_us: app.latency_us,
            order_book_heatmap_len: app.order_book_heatmap.len(),
            active_orders_len: app.active_orders.len(),
            show_kill_confirm: app.show_kill_confirm,
        }
    }
}

/// Main layout splitter dividing the terminal into functional panes
pub fn create_main_layout(area: Rect, config: &LayoutConfig) -> Vec<Rect> {
    Layout::default()
        .direction(Direction::Vertical)
        .margin(config.margin)
        .constraints([
            Constraint::Length(config.header_height),
            Constraint::Length(config.metrics_height),
            Constraint::Min(config.heatmap_min_height),
            Constraint::Length(config.orders_height),
            Constraint::Length(config.status_height),
        ])
        .split(area)
}

/// Create horizontal sub-layout for metrics pane
pub fn create_metrics_layout(area: Rect) -> Vec<Rect> {
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area)
}

/// Render the header pane with trading status and active pairs
pub fn render_header(f: &mut Frame, app: &App, area: Rect) {
    let title = if app.trading_active {
        "🚀 HFT TRADING ENGINE [ACTIVE]"
    } else {
        "⏸️  HFT TRADING ENGINE [STANDBY]"
    };

    let pairs = vec![
        Span::styled("BTC/USDT", Style::default().fg(Color::Cyan)),
        Span::raw("  •  "),
        Span::styled("ETH/USDT", Style::default().fg(Color::Magenta)),
        Span::raw("  •  "),
        Span::styled("SOL/USDT", Style::default().fg(Color::Yellow)),
    ];

    let header = Paragraph::new(Line::from(vec![
        Span::styled(title, Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        Span::raw(" | "),
    ]))
    .block(Block::default().borders(Borders::ALL).title("SYSTEM").title_bottom(Line::from(pairs)));

    f.render_widget(header, area);
}

/// Render the metrics pane with PnL, RAM, AI confidence, and latency
pub fn render_metrics(f: &mut Frame, app: &App, areas: &[Rect]) {
    // PnL Display
    let pnl_color = if app.pnl >= 0.0 { Color::Green } else { Color::Red };
    let pnl_indicator = if app.pnl >= 0.0 { "▲" } else { "▼" };
    let pnl_text = format!("{} ${:+.2}", pnl_indicator, app.pnl);
    
    let pnl_widget = Paragraph::new(pnl_text)
        .style(Style::default().fg(pnl_color).add_modifier(Modifier::BOLD))
        .block(Block::default().borders(Borders::ALL).title("REAL-TIME PnL"));
    f.render_widget(pnl_widget, areas[0]);

    // RAM Usage with mini sparkline simulation
    let ram_mb = app.ram_usage / (1024 * 1024);
    let ram_bar_width = ((ram_mb as f32) / 100.0).min(20.0) as usize;
    let ram_bar = "▓".repeat(ram_bar_width);
    let ram_text = format!("{} {} MB", ram_bar, ram_mb);
    
    let ram_widget = Paragraph::new(ram_text)
        .style(Style::default().fg(Color::Blue))
        .block(Block::default().borders(Borders::ALL).title("MEMORY"));
    f.render_widget(ram_widget, areas[1]);

    // AI Confidence Gauge
    let confidence_normalized = app.ai_confidence.min(1.0).max(0.0);
    let confidence_gauge = Gauge::default()
        .gauge_style(Style::default().fg(Color::Cyan).bg(Color::DarkGray))
        .label(format!("{:.0}%", confidence_normalized * 100.0))
        .ratio(confidence_normalized as f64)
        .block(Block::default().title("AI CONFIDENCE"));
    f.render_widget(confidence_gauge, areas[2]);

    // Latency indicator with color coding
    let latency_text = format!("{} µs", app.latency_us);
    let (latency_color, latency_icon) = if app.latency_us < 50 {
        (Color::Green, "⚡")
    } else if app.latency_us < 200 {
        (Color::Yellow, "◐")
    } else if app.latency_us < 500 {
        (Color::Rgb(255, 140, 0), "◑")
    } else {
        (Color::Red, "⚠️")
    };
    
    let latency_widget = Paragraph::new(format!("{} {}", latency_icon, latency_text))
        .style(Style::default().fg(latency_color).add_modifier(Modifier::BOLD))
        .block(Block::default().borders(Borders::ALL).title("TICK-TO-TRADE"));
    f.render_widget(latency_widget, areas[3]);
}

/// Render the L2 order book heatmap visualization
pub fn render_heatmap(f: &mut Frame, app: &App, area: Rect) {
    let mut lines = Vec::with_capacity(12);
    
    // Header
    lines.push(Line::from(vec![
        Span::styled("Price     ", Style::default().fg(Color::White).add_modifier(Modifier::UNDERLINED)),
        Span::raw(" | "),
        Span::styled("Volume Profile", Style::default().fg(Color::White).add_modifier(Modifier::UNDERLINED)),
    ]));

    if app.order_book_heatmap.is_empty() {
        // Generate simulated heatmap data
        let base_price = 45000.0;
        for i in 0..10 {
            let is_ask = i < 5;
            let offset = (i % 5) as f64 * 5.0;
            let price = if is_ask {
                base_price + 25.0 - offset
            } else {
                base_price - 25.0 + offset
            };
            
            let volume = (1000.0 + (i as f64 * 150.0)) as u64;
            let bar_len = ((volume as f32) / 200.0).min(40.0) as usize;
            
            let (price_style, bar_char) = if is_ask {
                (Style::default().fg(Color::Red), "▒")
            } else {
                (Style::default().fg(Color::Green), "█")
            };
            
            let bar = bar_char.repeat(bar_len.max(1));
            
            lines.push(Line::from(vec![
                Span::styled(format!("{:<9.2}", price), price_style),
                Span::raw(" | "),
                Span::styled(bar, Style::default().fg(if is_ask { Color::Red } else { Color::Green })),
            ]));
        }
    } else {
        // Render actual heatmap data
        for (price, volume) in app.order_book_heatmap.iter().take(10) {
            let bar_len = ((*volume / 100.0) as usize).min(40);
            let bar = "█".repeat(bar_len.max(1));
            
            let color = if *price > 45000.0 { Color::Red } else { Color::Green };
            
            lines.push(Line::from(vec![
                Span::styled(format!("{:<9.2}", price), Style::default().fg(color)),
                Span::raw(" | "),
                Span::styled(bar, Style::default().fg(color)),
            ]));
        }
    }

    let heatmap = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title("L2 ORDER BOOK HEATMAP"))
        .wrap(Wrap { trim: false });
    
    f.render_widget(heatmap, area);
}

/// Render the active orders list
pub fn render_orders(f: &mut Frame, app: &App, area: Rect) {
    let items: Vec<ListItem> = if app.active_orders.is_empty() {
        vec![ListItem::new(Line::from(Span::styled(
            "No active orders",
            Style::default().fg(Color::DarkGray),
        )))]
    } else {
        app.active_orders
            .iter()
            .map(|order| {
                let style = if order.contains("BUY") {
                    Style::default().fg(Color::Green)
                } else if order.contains("SELL") {
                    Style::default().fg(Color::Red)
                } else {
                    Style::default()
                };
                ListItem::new(Line::from(Span::styled(order.clone(), style)))
            })
            .collect()
    };

    let orders_list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("ACTIVE ORDERS"))
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED));

    f.render_widget(orders_list, area);
}

/// Render the status bar
pub fn render_status(f: &mut Frame, app: &App, area: Rect) {
    let (status_text, status_color) = if app.show_kill_confirm {
        (
            "⚠️  CONFIRM KILL SWITCH? Press Y to confirm, N to cancel",
            Color::Red,
        )
    } else if app.trading_active {
        (
            "● LIVE TRADING | Press ESC or /KILL to stop | Ctrl+C for emergency halt",
            Color::Green,
        )
    } else {
        (
            "○ STANDBY | Press /START to initialize trading engine",
            Color::Yellow,
        )
    };

    let status = Paragraph::new(status_text)
        .style(Style::default().fg(status_color).add_modifier(Modifier::BOLD))
        .block(Block::default().borders(Borders::ALL).title("STATUS"));

    f.render_widget(status, area);
}

/// Render kill confirmation popup overlay
pub fn render_kill_confirm_overlay(f: &mut Frame, area: Rect) {
    let popup_area = centered_rect(50, 35, area);
    
    let confirm_lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            "╔══════════════════════════════════════╗",
            Style::default().fg(Color::Red),
        )),
        Line::from(Span::styled(
            "║  ⚠️  GLOBAL KILL SWITCH ACTIVATION  ⚠️  ║",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            "╚══════════════════════════════════════╝",
            Style::default().fg(Color::Red),
        )),
        Line::from(""),
        Line::from("This action will immediately:"),
        Line::from("  ❖ Cancel ALL open orders across all venues"),
        Line::from("  ❖ Halt all new order generation"),
        Line::from("  ❖ Park capital in safe assets (USDT)"),
        Line::from("  ❖ Disconnect from market data feeds"),
        Line::from(""),
        Line::from(Span::styled(
            "Are you ABSOLUTELY sure? (Y/N)",
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )),
    ];

    let popup = Paragraph::new(confirm_lines)
        .style(Style::default().bg(Color::Black).fg(Color::White))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))
                .title("SAFETY CRITICAL CONFIRMATION")
                .title_alignment(ratatui::layout::Alignment::Center),
        );

    // Clear the background first
    f.render_widget(ratatui::widgets::Clear, popup_area);
    f.render_widget(popup, popup_area);
}

/// Helper function to create a centered rectangle
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let vertical_split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical_split[1])[1]
}

/// Performance metrics tracker for rendering optimization
#[repr(align(64))]
pub struct RenderMetrics {
    pub frames_rendered: u64,
    pub last_frame_time_us: u64,
    pub avg_fps: f32,
    pub regions_invalidated: [u64; 5], // header, metrics, heatmap, orders, status
}

impl Default for RenderMetrics {
    fn default() -> Self {
        Self {
            frames_rendered: 0,
            last_frame_time_us: 0,
            avg_fps: 60.0,
            regions_invalidated: [0; 5],
        }
    }
}

impl RenderMetrics {
    pub fn record_frame(&mut self, frame_time_us: u64, regions: &RenderRegions) {
        self.frames_rendered += 1;
        self.last_frame_time_us = frame_time_us;
        
        // Calculate rolling average FPS
        let current_fps = 1_000_000.0 / (frame_time_us as f32).max(1.0);
        self.avg_fps = (self.avg_fps * 0.95) + (current_fps * 0.05);
        
        // Track which regions are being invalidated
        if regions.header_changed { self.regions_invalidated[0] += 1; }
        if regions.metrics_changed { self.regions_invalidated[1] += 1; }
        if regions.heatmap_changed { self.regions_invalidated[2] += 1; }
        if regions.orders_changed { self.regions_invalidated[3] += 1; }
        if regions.status_changed { self.regions_invalidated[4] += 1; }
    }
    
    pub fn get_target_frame_interval_us(&self) -> u64 {
        (1_000_000.0 / 60.0) as u64 // Target 60 FPS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_config_cache_alignment() {
        let config = LayoutConfig::default();
        let addr = &config as *const _ as usize;
        assert_eq!(addr % 64, 0, "LayoutConfig should be cache-line aligned");
    }

    #[test]
    fn test_render_regions_diff() {
        let mut regions = RenderRegions::new();
        assert!(regions.needs_redraw());
        
        regions.mark_rendered();
        assert!(!regions.needs_redraw());
        
        regions.header_changed = true;
        assert!(regions.needs_redraw());
    }

    #[test]
    fn test_prev_app_state_from_app() {
        let app = App::default();
        let prev = PrevAppState::from_app(&app);
        
        assert_eq!(prev.trading_active, app.trading_active);
        assert_eq!(prev.pnl, app.pnl);
        assert_eq!(prev.active_orders_len, app.active_orders.len());
    }

    #[test]
    fn test_render_metrics_tracking() {
        let mut metrics = RenderMetrics::default();
        let regions = RenderRegions::default();
        
        metrics.record_frame(16000, &regions); // ~60 FPS frame time
        
        assert!(metrics.frames_rendered > 0);
        assert!(metrics.avg_fps > 50.0 && metrics.avg_fps < 70.0);
    }
}
