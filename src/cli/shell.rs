//! Interactive REPL Shell for HFT Crypto Bot
//! 
//! Provides a non-blocking interactive shell for:
//! - Inspecting internal actor states
//! - Tweaking risk limits
//! - Injecting faults for chaos testing
//! - Manual trading overrides

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::{Duration, timeout};

use crate::cli::commands::{Command, CommandParser, CommandResult};

/// Shell configuration
#[derive(Debug, Clone)]
pub struct ShellConfig {
    /// Prompt string
    pub prompt: String,
    /// Enable command history
    pub history_enabled: bool,
    /// Maximum history size
    pub max_history: usize,
    /// Auto-complete enabled
    pub autocomplete_enabled: bool,
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            prompt: "hft> ".to_string(),
            history_enabled: true,
            max_history: 1000,
            autocomplete_enabled: true,
        }
    }
}

/// Interactive REPL shell
pub struct InteractiveShell {
    config: ShellConfig,
    running: Arc<AtomicBool>,
    command_tx: mpsc::Sender<Command>,
    command_rx: Option<mpsc::Receiver<Command>>,
    response_tx: mpsc::Sender<String>,
    response_rx: Option<mpsc::Receiver<String>>,
    history: Vec<String>,
    parser: CommandParser,
}

impl InteractiveShell {
    /// Create a new interactive shell
    pub fn new(config: ShellConfig) -> (Self, mpsc::Receiver<Command>, mpsc::Receiver<String>) {
        let (cmd_tx, cmd_rx) = mpsc::channel(100);
        let (resp_tx, resp_rx) = mpsc::channel(100);
        
        let shell = Self {
            config,
            running: Arc::new(AtomicBool::new(false)),
            command_tx: cmd_tx,
            command_rx: Some(cmd_rx),
            response_tx: resp_tx,
            response_rx: Some(resp_rx),
            history: Vec::new(),
            parser: CommandParser::new(),
        };
        
        (shell, cmd_rx, resp_rx)
    }

    /// Get the command receiver for wiring into the orchestrator
    pub fn take_command_receiver(&mut self) -> Option<mpsc::Receiver<Command>> {
        self.command_rx.take()
    }

    /// Get the response receiver for wiring into the TUI
    pub fn take_response_receiver(&mut self) -> Option<mpsc::Receiver<String>> {
        self.response_rx.take()
    }

    /// Start the interactive shell (non-blocking)
    pub async fn run(&mut self) -> io::Result<()> {
        self.running.store(true, Ordering::SeqCst);
        
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║     HFT Crypto Bot - Interactive Control Shell          ║");
        println!("║     Type 'help' for available commands                  ║");
        println!("║     Type 'exit' or Ctrl+D to quit                       ║");
        println!("╚══════════════════════════════════════════════════════════╝\n");
        
        print!("{}", self.config.prompt);
        io::stdout().flush()?;

        let mut input = String::new();
        
        loop {
            if !self.running.load(Ordering::Relaxed) {
                break;
            }

            // Read input with timeout for non-blocking behavior
            input.clear();
            
            match self.read_line_with_timeout(Duration::from_millis(100)).await {
                Ok(line) => {
                    let trimmed = line.trim();
                    
                    if trimmed.is_empty() {
                        print!("{}", self.config.prompt);
                        io::stdout().flush()?;
                        continue;
                    }

                    // Add to history
                    if self.config.history_enabled {
                        self.history.push(trimmed.to_string());
                        if self.history.len() > self.config.max_history {
                            self.history.remove(0);
                        }
                    }

                    // Check for exit commands
                    if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
                        break;
                    }

                    // Parse and execute command
                    match self.parser.parse(trimmed) {
                        Ok(command) => {
                            if let Err(e) = self.execute_command(command).await {
                                eprintln!("Error: {}", e);
                            }
                        }
                        Err(e) => {
                            eprintln!("Parse error: {}", e);
                        }
                    }
                }
                Err(_) => {
                    // Timeout, continue loop (non-blocking)
                }
            }

            print!("{}", self.config.prompt);
            io::stdout().flush()?;
        }

        self.running.store(false, Ordering::SeqCst);
        println!("\nShell exited.");
        
        Ok(())
    }

    /// Read a line with timeout (allows non-blocking checks)
    async fn read_line_with_timeout(&self, timeout_dur: Duration) -> Result<String, ()> {
        tokio::task::spawn_blocking(move || {
            let mut input = String::new();
            match timeout(timeout_dur, tokio::task::spawn_blocking(move || {
                io::stdin().read_line(&mut input)
            })).await {
                Ok(Ok(Ok(_))) => Ok(input),
                _ => Err(()),
            }
        }).await
        .map_err(|_| ())?
    }

    /// Execute a parsed command
    async fn execute_command(&self, command: Command) -> Result<(), String> {
        match &command {
            Command::Help => {
                self.print_help();
                Ok(())
            }
            Command::Status => {
                // Send status request through channel
                let _ = self.response_tx.send("STATUS_REQUEST".to_string()).await;
                Ok(())
            }
            Command::ForceKill => {
                println!("⚠️  WARNING: Force kill initiated...");
                let _ = self.command_tx.send(command).await;
                Ok(())
            }
            Command::DumpState { path } => {
                println!("Dumping state to: {}", path);
                let _ = self.command_tx.send(command).await;
                Ok(())
            }
            Command::ShadowMode { enable } => {
                if *enable {
                    println!("🎭 Entering shadow mode (orders simulated, not sent)");
                } else {
                    println!("📈 Exiting shadow mode (live trading active)");
                }
                let _ = self.command_tx.send(command).await;
                Ok(())
            }
            Command::SetRiskLimit { parameter, value } => {
                println!("Setting risk limit: {} = {}", parameter, value);
                let _ = self.command_tx.send(command).await;
                Ok(())
            }
            Command::InjectFault { fault_type, target } => {
                println!("⚡ Injecting fault: {:?} on {}", fault_type, target);
                let _ = self.command_tx.send(command).await;
                Ok(())
            }
            Command::ShowActors => {
                println!("Querying actor states...");
                let _ = self.command_tx.send(command).await;
                Ok(())
            }
            Command::Metrics => {
                println!("Fetching metrics...");
                let _ = self.command_tx.send(command).await;
                Ok(())
            }
            Command::Unknown(cmd) => {
                Err(format!("Unknown command: {}. Type 'help' for available commands.", cmd))
            }
        }
    }

    /// Print help message
    fn print_help(&self) {
        println!(r#"
╔══════════════════════════════════════════════════════════╗
║                    AVAILABLE COMMANDS                     ║
╠══════════════════════════════════════════════════════════╣
║  SYSTEM CONTROL                                          ║
║  ────────────────                                        ║
║  help              Show this help message                ║
║  status            Show system status                    ║
║  exit, quit        Exit the shell                        ║
║                                                          ║
║  TRADING CONTROL                                         ║
║  ────────────────                                        ║
║  shadow_mode [on|off]  Toggle shadow mode (simulated)    ║
║  force_kill          Emergency stop all trading          ║
║  dump_state <path>   Dump current state to file          ║
║                                                          ║
║  RISK MANAGEMENT                                         ║
║  ─────────────────                                       ║
║  set_risk <param> <value>  Set risk limit parameter      ║
║    Parameters: max_position, max_order_size,             ║
║                daily_loss_limit, var_limit               ║
║                                                          ║
║  DIAGNOSTICS                                             ║
║  ───────────                                             ║
║  show_actors       Display internal actor states         ║
║  metrics           Show performance metrics              ║
║                                                          ║
║  CHAOS TESTING                                           ║
║  ─────────────                                           ║
║  inject_fault <type> <target>  Inject fault for testing  ║
║    Types: latency, disconnect, order_reject,             ║
║           sequence_gap, timeout                          ║
║    Targets: gateway, exchange, disruptor, engine         ║
╚══════════════════════════════════════════════════════════╝
"#);
    }

    /// Stop the shell
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Check if shell is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shell_creation() {
        let config = ShellConfig::default();
        let (mut shell, _cmd_rx, _resp_rx) = InteractiveShell::new(config);
        
        assert!(shell.config.prompt == "hft> ");
        assert!(shell.history.is_empty());
        assert!(!shell.is_running());
    }

    #[tokio::test]
    async fn test_command_parsing() {
        let parser = CommandParser::new();
        
        assert!(parser.parse("help").is_ok());
        assert!(parser.parse("force_kill").is_ok());
        assert!(parser.parse("shadow_mode on").is_ok());
        assert!(parser.parse("unknown_cmd").is_ok()); // Returns Unknown variant
    }
}
