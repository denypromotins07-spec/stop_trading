//! CLI Module for HFT Crypto Bot
//! 
//! This module provides the interactive command-line interface infrastructure:
//! - REPL shell for manual operator overrides
//! - Zero-allocation command parsing
//! - Lock-free channel integration with orchestrator and TUI

pub mod commands;
pub mod shell;

pub use commands::{Command, CommandParser, CommandResult, FaultType, ParseError};
pub use shell::{InteractiveShell, ShellConfig};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

/// CLI module configuration
#[derive(Debug, Clone)]
pub struct CliConfig {
    /// Enable interactive shell
    pub enable_shell: bool,
    /// Shell prompt string
    pub prompt: String,
    /// Enable command history
    pub history_enabled: bool,
    /// Maximum history entries
    pub max_history: usize,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            enable_shell: true,
            prompt: "hft> ".to_string(),
            history_enabled: true,
            max_history: 1000,
        }
    }
}

/// CLI Manager - coordinates shell, command routing, and responses
pub struct CliManager {
    config: CliConfig,
    running: Arc<AtomicBool>,
    shell: Option<InteractiveShell>,
    command_tx: mpsc::Sender<Command>,
    response_tx: mpsc::Sender<String>,
}

impl CliManager {
    /// Create a new CLI manager
    pub fn new(config: CliConfig) -> Self {
        let (cmd_tx, _cmd_rx) = mpsc::channel(100);
        let (resp_tx, _resp_rx) = mpsc::channel(100);
        
        Self {
            config,
            running: Arc::new(AtomicBool::new(false)),
            shell: None,
            command_tx: cmd_tx,
            response_tx: resp_tx,
        }
    }

    /// Initialize the CLI manager
    pub async fn init(&mut self) -> crate::Result<()> {
        if !self.config.enable_shell {
            return Ok(());
        }

        let shell_config = ShellConfig {
            prompt: self.config.prompt.clone(),
            history_enabled: self.config.history_enabled,
            max_history: self.config.max_history,
            autocomplete_enabled: true,
        };

        let (shell, cmd_rx, resp_rx) = InteractiveShell::new(shell_config);
        self.shell = Some(shell);

        // Note: cmd_rx and resp_rx should be wired into the orchestrator
        // This is done by the caller via take_command_receiver() pattern
        
        info!("CLI manager initialized");
        Ok(())
    }

    /// Start the interactive shell (non-blocking, runs in background)
    pub async fn start(&mut self) -> crate::Result<()> {
        if let Some(ref mut shell) = self.shell {
            self.running.store(true, Ordering::SeqCst);
            
            let running = self.running.clone();
            
            tokio::spawn(async move {
                if let Err(e) = shell.run().await {
                    error!("Shell error: {}", e);
                    running.store(false, Ordering::SeqCst);
                }
            });
            
            info!("Interactive shell started");
        }
        
        Ok(())
    }

    /// Stop the CLI manager
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        
        if let Some(ref shell) = self.shell {
            shell.stop();
        }
        
        info!("CLI manager stopped");
    }

    /// Check if CLI is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Send a command through the CLI channel
    pub async fn send_command(&self, command: Command) -> crate::Result<()> {
        self.command_tx.send(command).await
            .map_err(|e| crate::Error::Internal(e.to_string()))?;
        Ok(())
    }

    /// Get command receiver for wiring into orchestrator
    /// Returns None if shell wasn't initialized or already taken
    pub fn get_command_receiver(&mut self) -> Option<mpsc::Receiver<Command>> {
        self.shell.as_mut().and_then(|s| s.take_command_receiver())
    }

    /// Get response receiver for wiring into TUI event loop
    /// Returns None if shell wasn't initialized or already taken
    pub fn get_response_receiver(&mut self) -> Option<mpsc::Receiver<String>> {
        self.shell.as_mut().and_then(|s| s.take_response_receiver())
    }

    /// Broadcast a status message to the shell
    pub async fn broadcast_status(&self, message: &str) -> crate::Result<()> {
        self.response_tx.send(message.to_string()).await
            .map_err(|e| crate::Error::Internal(e.to_string()))?;
        Ok(())
    }
}

/// Builder for CliManager
pub struct CliManagerBuilder {
    config: CliConfig,
}

impl CliManagerBuilder {
    pub fn new() -> Self {
        Self {
            config: CliConfig::default(),
        }
    }

    pub fn enable_shell(mut self, enable: bool) -> Self {
        self.config.enable_shell = enable;
        self
    }

    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.prompt = prompt.into();
        self
    }

    pub fn history_enabled(mut self, enabled: bool) -> Self {
        self.config.history_enabled = enabled;
        self
    }

    pub fn max_history(mut self, max: usize) -> Self {
        self.config.max_history = max;
        self
    }

    pub fn build(self) -> CliManager {
        CliManager::new(self.config)
    }
}

impl Default for CliManagerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Re-export commonly used items at module root
pub use commands::CommandParser;
pub use shell::InteractiveShell;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_config_default() {
        let config = CliConfig::default();
        assert!(config.enable_shell);
        assert_eq!(config.prompt, "hft> ");
    }

    #[tokio::test]
    async fn test_cli_manager_creation() {
        let config = CliConfig {
            enable_shell: false,
            ..Default::default()
        };
        
        let mut manager = CliManager::new(config);
        assert!(manager.init().await.is_ok());
        assert!(!manager.is_running());
    }

    #[tokio::test]
    async fn test_cli_manager_builder() {
        let manager = CliManagerBuilder::new()
            .enable_shell(true)
            .prompt("trading> ")
            .max_history(500)
            .build();
        
        assert!(manager.config.enable_shell);
        assert_eq!(manager.config.prompt, "trading> ");
        assert_eq!(manager.config.max_history, 500);
    }
}
