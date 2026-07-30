//! UI Module Root
//! 
//! Initializes the alternate screen buffer, mouse support, and raw terminal mode.
//! Re-exports all UI components for the application.

pub mod app;
pub mod layout;

use std::io;
use std::sync::Arc;

use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

use crate::safety::kill_switch::KillSwitch;
use crate::portfolio::state::PortfolioState;
use crate::clock::heartbeat::HeartbeatMonitor;

use self::app::{App, init_terminal, run_ui_loop, shutdown_terminal};

/// UI Manager handling terminal lifecycle and state coordination
pub struct UIManager {
    terminal: Option<Terminal<CrosstermBackend<io::Stdout>>>,
    app: Option<App>,
    kill_switch: Arc<KillSwitch>,
    portfolio_state: Arc<PortfolioState>,
    heartbeat_monitor: Arc<HeartbeatMonitor>,
}

impl UIManager {
    /// Create a new UI manager with shared system components
    pub fn new(
        kill_switch: Arc<KillSwitch>,
        portfolio_state: Arc<PortfolioState>,
        heartbeat_monitor: Arc<HeartbeatMonitor>,
    ) -> Self {
        Self {
            terminal: None,
            app: None,
            kill_switch,
            portfolio_state,
            heartbeat_monitor,
        }
    }

    /// Initialize the terminal and enter alternate screen mode
    pub fn initialize(&mut self) -> io::Result<()> {
        // Enable raw mode for direct terminal control
        enable_raw_mode()?;
        
        let mut stdout = io::stdout();
        
        // Enter alternate screen buffer
        execute!(stdout, EnterAlternateScreen)?;
        
        // Enable mouse capture for interactive elements
        execute!(stdout, EnableMouseCapture)?;
        
        // Create the terminal backend
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;
        
        self.terminal = Some(terminal);
        
        // Initialize the application state
        self.app = Some(App::new(
            self.kill_switch.clone(),
            self.portfolio_state.clone(),
            self.heartbeat_monitor.clone(),
        ));
        
        Ok(())
    }

    /// Run the main UI event loop
    pub fn run(&mut self) -> io::Result<()> {
        if self.terminal.is_none() || self.app.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "UI not initialized. Call initialize() first.",
            ));
        }

        let terminal = self.terminal.take().unwrap();
        let app = self.app.take().unwrap();

        // Run the main UI loop
        run_ui_loop(app, terminal)?;

        Ok(())
    }

    /// Shutdown the UI gracefully, restoring terminal state
    pub fn shutdown(&mut self) -> io::Result<()> {
        if let Some(mut terminal) = self.terminal.take() {
            shutdown_terminal(terminal)?;
        }
        
        self.app = None;
        Ok(())
    }

    /// Get a reference to the current application state
    pub fn get_app(&self) -> Option<&App> {
        self.app.as_ref()
    }

    /// Get a mutable reference to the current application state
    pub fn get_app_mut(&mut self) -> Option<&mut App> {
        self.app.as_mut()
    }

    /// Check if the UI is currently active
    pub fn is_active(&self) -> bool {
        self.terminal.is_some() && self.app.is_some()
    }

    /// Send a command to the application (e.g., "/START", "/KILL")
    pub fn send_command(&mut self, cmd: &str) {
        if let Some(app) = &mut self.app {
            app.execute_command(cmd);
        }
    }

    /// Trigger an emergency shutdown
    pub fn emergency_shutdown(&mut self) -> io::Result<()> {
        if let Some(ref mut app) = self.app {
            app.running = false;
            app.trading_active = false;
            self.kill_switch.trigger_manual("EMERGENCY_UI_SHUTDOWN");
        }
        self.shutdown()
    }
}

impl Drop for UIManager {
    fn drop(&mut self) {
        // Ensure terminal is restored on drop
        let _ = self.shutdown();
    }
}

/// Convenience function to run the UI with default configuration
pub fn run_default_ui(
    kill_switch: Arc<KillSwitch>,
    portfolio_state: Arc<PortfolioState>,
    heartbeat_monitor: Arc<HeartbeatMonitor>,
) -> io::Result<()> {
    let mut ui_manager = UIManager::new(kill_switch, portfolio_state, heartbeat_monitor);
    ui_manager.initialize()?;
    ui_manager.run()?;
    ui_manager.shutdown()
}

/// Quick startup helper for testing
#[cfg(test)]
pub fn create_test_ui() -> UIManager {
    UIManager::new(
        Arc::new(KillSwitch::default()),
        Arc::new(PortfolioState::default()),
        Arc::new(HeartbeatMonitor::default()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ui_manager_creation() {
        let ui_manager = UIManager::new(
            Arc::new(KillSwitch::default()),
            Arc::new(PortfolioState::default()),
            Arc::new(HeartbeatMonitor::default()),
        );
        
        assert!(!ui_manager.is_active());
        assert!(ui_manager.get_app().is_none());
    }

    #[test]
    fn test_ui_manager_send_command() {
        let mut ui_manager = create_test_ui();
        
        // Should not panic even without initialization
        ui_manager.send_command("/STATUS");
    }

    #[test]
    fn test_emergency_shutdown_without_init() {
        let mut ui_manager = create_test_ui();
        
        // Should handle gracefully without prior initialization
        let result = ui_manager.emergency_shutdown();
        assert!(result.is_ok());
    }
}
