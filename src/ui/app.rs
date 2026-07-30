//! Terminal UI Application Core
//! 
//! Implements the main TUI event loop using `ratatui` and `crossterm`.
//! Handles global keybindings for `/START`, `/KILL`, and safety shutdown prompts.

use std::{
    io,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Wrap},
    Frame, Terminal,
};

use crate::safety::kill_switch::KillSwitch;
use crate::portfolio::state::PortfolioState;
use crate::clock::heartbeat::HeartbeatMonitor;

/// Main application state
pub struct App {
    pub running: bool,
    pub trading_active: bool,
    pub show_kill_confirm: bool,
    pub pnl: f64,
    pub ram_usage: u64,
    pub ai_confidence: f32,
    pub active_orders: Vec<String>,
    pub order_book_heatmap: Vec<(f64, f64)>, // (price, volume)
    pub last_heartbeat: Instant,
    pub latency_us: u64,
    pub kill_switch: Arc<KillSwitch>,
    pub portfolio_state: Arc<PortfolioState>,
    pub heartbeat_monitor: Arc<HeartbeatMonitor>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            running: true,
            trading_active: false,
            show_kill_confirm: false,
            pnl: 0.0,
            ram_usage: 0,
            ai_confidence: 0.5,
            active_orders: Vec::new(),
            order_book_heatmap: Vec::new(),
            last_heartbeat: Instant::now(),
            latency_us: 0,
            kill_switch: Arc::new(KillSwitch::default()),
            portfolio_state: Arc::new(PortfolioState::default()),
            heartbeat_monitor: Arc::new(HeartbeatMonitor::default()),
        }
    }
}

impl App {
    pub fn new(
        kill_switch: Arc<KillSwitch>,
        portfolio_state: Arc<PortfolioState>,
        heartbeat_monitor: Arc<HeartbeatMonitor>,
    ) -> Self {
        Self {
            kill_switch,
            portfolio_state,
            heartbeat_monitor,
            ..Default::default()
        }
    }

    /// Handle keyboard input with strict global keybindings
    pub fn handle_key(&mut self, key: event::KeyEvent) {
        if key.kind != KeyEventKind::Press {
            return;
        }

        // If kill confirmation is shown, only accept Y/N
        if self.show_kill_confirm {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    self.kill_switch.trigger_manual("UI_CONFIRMATION");
                    self.running = false;
                    self.trading_active = false;
                }
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                    self.show_kill_confirm = false;
                }
                _ => {}
            }
            return;
        }

        // Global command handling
        match key.code {
            KeyCode::Char('/') => {
                // Command mode entry - in real impl would buffer command string
                // For now, we simulate immediate command execution
            }
            KeyCode::Char('s') if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                // Ctrl+S as alternative start
                if !self.trading_active {
                    self.trading_active = true;
                    self.kill_switch.reset();
                }
            }
            KeyCode::Char('c') if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                // Ctrl+C triggers kill confirmation
                self.show_kill_confirm = true;
            }
            KeyCode::Esc => {
                // Escape also triggers kill confirmation when trading
                if self.trading_active {
                    self.show_kill_confirm = true;
                } else {
                    self.running = false;
                }
            }
            KeyCode::Char('q') => {
                if !self.trading_active {
                    self.running = false;
                }
            }
            _ => {}
        }
    }

    /// Simulate command parsing for /START, /KILL, etc.
    pub fn execute_command(&mut self, cmd: &str) {
        let cmd = cmd.trim().to_uppercase();
        
        match cmd.as_str() {
            "/START" => {
                if !self.trading_active {
                    self.trading_active = true;
                    self.kill_switch.reset();
                    log_info("Trading engine initialized");
                }
            }
            "/KILL" => {
                self.show_kill_confirm = true;
            }
            "/STATUS" => {
                log_info(&format!(
                    "Status: Trading={}, PnL={:.2}, Latency={}µs",
                    self.trading_active, self.pnl, self.latency_us
                ));
            }
            _ => {}
        }
    }

    /// Update application state from shared data structures
    pub fn update(&mut self) {
        // Check heartbeat
        self.latency_us = self.heartbeat_monitor.get_current_latency_us();
        self.last_heartbeat = self.heartbeat_monitor.last_tick();

        // Update portfolio metrics
        self.pnl = self.portfolio_state.get_total_pnl();
        self.ram_usage = get_memory_usage();

        // Update AI confidence from strategy ensemble
        // In real impl, this would read from shared memory or atomic
        
        // Check if kill switch was triggered externally
        if self.kill_switch.is_triggered() && self.trading_active {
            self.trading_active = false;
        }
    }
}

/// Initialize the terminal UI
pub fn init_terminal() -> io::Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    Terminal::new(backend)
}

/// Shutdown the terminal UI gracefully
pub fn shutdown_terminal(mut terminal: Terminal<CrosstermBackend<io::Stdout>>) -> io::Result<()> {
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    Ok(())
}

/// Run the main TUI event loop
pub fn run_ui_loop(
    mut app: App,
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
) -> io::Result<()> {
    let mut last_render = Instant::now();
    let render_interval = Duration::from_millis(16); // ~60 FPS

    while app.running {
        // Non-blocking event polling
        if crossterm::event::poll(Duration::from_millis(1))? {
            if let Event::Key(key) = event::read()? {
                app.handle_key(key);
            }
        }

        // Update application state
        app.update();

        // Render at controlled frame rate
        if last_render.elapsed() >= render_interval {
            terminal.draw(|f| ui(f, &app))?;
            last_render = Instant::now();
        }

        // Small sleep to prevent CPU spinning
        std::thread::sleep(Duration::from_micros(100));
    }

    Ok(())
}

/// Main UI rendering function with diff-optimized layout
fn ui(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Length(10), // PnL & Metrics
            Constraint::Min(5),     // Order Book Heatmap
            Constraint::Length(10), // Active Orders
            Constraint::Length(3),  // Status Bar
        ])
        .split(f.area());

    render_header(f, app, chunks[0]);
    render_metrics(f, app, chunks[1]);
    render_heatmap(f, app, chunks[2]);
    render_orders(f, app, chunks[3]);
    render_status(f, app, chunks[4]);

    // Render kill confirmation popup if needed
    if app.show_kill_confirm {
        render_kill_confirm(f, app);
    }
}

fn render_header(f: &mut Frame, app: &App, area: Rect) {
    let title = if app.trading_active {
        "🚀 HFT TRADING ENGINE [ACTIVE]"
    } else {
        "⏸️  HFT TRADING ENGINE [STANDBY]"
    };

    let header = Paragraph::new(Line::from(vec![
        Span::styled(title, Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        Span::raw(" | "),
        Span::raw("BTC/USDT • ETH/USDT • SOL/USDT"),
    ]))
    .block(Block::default().borders(Borders::ALL).title("SYSTEM"));

    f.render_widget(header, area);
}

fn render_metrics(f: &mut Frame, app: &App, area: Rect) {
    let metrics_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);

    // PnL Display
    let pnl_color = if app.pnl >= 0.0 { Color::Green } else { Color::Red };
    let pnl_text = format!("PnL: ${:+.2}", app.pnl);
    let pnl_widget = Paragraph::new(pnl_text)
        .style(Style::default().fg(pnl_color).add_modifier(Modifier::BOLD))
        .block(Block::default().borders(Borders::ALL).title("REAL-TIME PnL"));
    f.render_widget(pnl_widget, metrics_layout[0]);

    // RAM Usage
    let ram_mb = app.ram_usage / (1024 * 1024);
    let ram_text = format!("RAM: {} MB", ram_mb);
    let ram_widget = Paragraph::new(ram_text)
        .block(Block::default().borders(Borders::ALL).title("MEMORY"));
    f.render_widget(ram_widget, metrics_layout[1]);

    // AI Confidence Gauge
    let confidence_pct = (app.ai_confidence * 100.0) as u16;
    let confidence_gauge = Gauge::default()
        .gauge_style(Style::default().fg(Color::Cyan))
        .label(format!("{}%", confidence_pct))
        .ratio(app.ai_confidence as f64 / 100.0)
        .block(Block::default().title("AI CONFIDENCE"));
    f.render_widget(confidence_gauge, metrics_layout[2]);

    // Latency
    let latency_text = format!("Latency: {} µs", app.latency_us);
    let latency_color = if app.latency_us < 100 {
        Color::Green
    } else if app.latency_us < 500 {
        Color::Yellow
    } else {
        Color::Red
    };
    let latency_widget = Paragraph::new(latency_text)
        .style(Style::default().fg(latency_color))
        .block(Block::default().borders(Borders::ALL).title("TICK-TO-TRADE"));
    f.render_widget(latency_widget, metrics_layout[3]);
}

fn render_heatmap(f: &mut Frame, app: &App, area: Rect) {
    // Generate simulated heatmap visualization
    let mut lines = Vec::new();
    
    if app.order_book_heatmap.is_empty() {
        // Simulated data for display
        for i in 0..8 {
            let price = 45000.0 + (i as f64 * 10.0);
            let volume = (i as f64 * 100.0) as u64;
            let bar_len = (volume / 1000).min(50) as usize;
            let bar = "█".repeat(bar_len);
            lines.push(Line::from(format!("{:<10.2} | {}", price, bar)));
        }
    } else {
        for (price, volume) in &app.order_book_heatmap {
            let bar_len = (*volume / 100.0) as usize;
            let bar = "█".repeat(bar_len.min(50));
            lines.push(Line::from(format!("{:<10.2} | {}", price, bar)));
        }
    }

    let heatmap = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title("L2 ORDER BOOK HEATMAP"))
        .wrap(Wrap { trim: true });
    
    f.render_widget(heatmap, area);
}

fn render_orders(f: &mut Frame, app: &App, area: Rect) {
    let items: Vec<ListItem> = app
        .active_orders
        .iter()
        .map(|order| ListItem::new(order.clone()))
        .collect();

    let orders_list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("ACTIVE ORDERS"))
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED));

    f.render_widget(orders_list, area);
}

fn render_status(f: &mut Frame, app: &App, area: Rect) {
    let status_text = if app.trading_active {
        "● LIVE | Press ESC or /KILL to stop"
    } else if app.show_kill_confirm {
        "⚠️  CONFIRM KILL? (Y/N)"
    } else {
        "○ STANDBY | Press /START to begin"
    };

    let status = Paragraph::new(status_text)
        .style(Style::default().fg(if app.trading_active {
            Color::Green
        } else {
            Color::Yellow
        }))
        .block(Block::default().borders(Borders::ALL).title("STATUS"));

    f.render_widget(status, area);
}

fn render_kill_confirm(f: &mut Frame, app: &App) {
    let area = centered_rect(40, 30, f.area());
    
    let confirm_text = vec![
        Line::from(""),
        Line::from(Span::styled(
            "⚠️  GLOBAL KILL SWITCH ACTIVATION ⚠️",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("This will:"),
        Line::from("  • Cancel all open orders"),
        Line::from("  • Halt all trading activity"),
        Line::from("  • Park capital in safe assets"),
        Line::from(""),
        Line::from("Confirm? (Y/N)"),
    ];

    let popup = Paragraph::new(confirm_text)
        .style(Style::default().bg(Color::Black).fg(Color::White))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Red))
                .title("SAFETY CONFIRMATION"),
        );

    f.render_widget(ratatui::widgets::Clear, area);
    f.render_widget(popup, area);
}

/// Helper to create a centered rectangle
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
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
        .split(popup_layout[1])[1]
}

/// Get current process memory usage (platform-specific)
fn get_memory_usage() -> u64 {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        return parts[1].parse::<u64>().unwrap_or(0) * 1024; // Convert KB to bytes
                    }
                }
            }
        }
    }
    
    // Fallback estimate
    50 * 1024 * 1024 // 50 MB default
}

/// Simple logging helper
fn log_info(msg: &str) {
    println!("[INFO] {}", msg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_default() {
        let app = App::default();
        assert!(!app.trading_active);
        assert!(app.running);
        assert!(!app.show_kill_confirm);
    }

    #[test]
    fn test_handle_key_kill_confirm() {
        let mut app = App::default();
        app.show_kill_confirm = true;

        // Test Y confirmation
        let key_y = event::KeyEvent::new(KeyCode::Char('y'), crossterm::event::KeyModifiers::NONE);
        app.handle_key(key_y);
        assert!(!app.running);
        assert!(!app.trading_active);

        // Reset and test N cancellation
        app.running = true;
        app.show_kill_confirm = true;
        let key_n = event::KeyEvent::new(KeyCode::Char('n'), crossterm::event::KeyModifiers::NONE);
        app.handle_key(key_n);
        assert!(!app.show_kill_confirm);
    }
}
