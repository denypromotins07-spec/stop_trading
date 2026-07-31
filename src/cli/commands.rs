//! Zero-Allocation Command Parser for CLI Directives
//! 
//! Parses operator commands without heap allocations where possible.
//! Supports: force_kill, dump_state, shadow_mode, risk limits, fault injection

use std::fmt;

/// Supported CLI commands
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    /// Show help message
    Help,
    
    /// Show system status
    Status,
    
    /// Emergency stop - kill all trading activity immediately
    ForceKill,
    
    /// Dump current state to file
    DumpState { path: String },
    
    /// Toggle shadow mode (simulate orders without sending)
    ShadowMode { enable: bool },
    
    /// Set a risk limit parameter
    SetRiskLimit { parameter: String, value: f64 },
    
    /// Inject a fault for chaos testing
    InjectFault { fault_type: FaultType, target: String },
    
    /// Display internal actor states
    ShowActors,
    
    /// Show performance metrics
    Metrics,
    
    /// Unknown command
    Unknown(String),
}

/// Types of faults that can be injected
#[derive(Debug, Clone, PartialEq)]
pub enum FaultType {
    Latency,
    Disconnect,
    OrderReject,
    SequenceGap,
    Timeout,
    PacketLoss,
    CorruptedData,
}

impl std::str::FromStr for FaultType {
    type Err = ();
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "latency" => Ok(FaultType::Latency),
            "disconnect" => Ok(FaultType::Disconnect),
            "order_reject" | "orderreject" => Ok(FaultType::OrderReject),
            "sequence_gap" | "sequencegap" => Ok(FaultType::SequenceGap),
            "timeout" => Ok(FaultType::Timeout),
            "packet_loss" | "packetloss" => Ok(FaultType::PacketLoss),
            "corrupted_data" | "corrupteddata" => Ok(FaultType::CorruptedData),
            _ => Err(()),
        }
    }
}

/// Result of command parsing
pub type CommandResult = Result<Command, ParseError>;

/// Command parsing errors
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    InvalidSyntax(String),
    UnknownCommand(String),
    InvalidValue { param: String, value: String },
    MissingArgument(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::InvalidSyntax(cmd) => write!(f, "Invalid syntax for command: {}", cmd),
            ParseError::UnknownCommand(cmd) => write!(f, "Unknown command: {}", cmd),
            ParseError::InvalidValue { param, value } => {
                write!(f, "Invalid value '{}' for parameter '{}'", value, param)
            }
            ParseError::MissingArgument(param) => write!(f, "Missing required argument: {}", param),
        }
    }
}

impl std::error::Error for ParseError {}

/// Zero-allocation command parser
pub struct CommandParser {
    // Pre-allocated buffer could be added here for true zero-allocation
    // For now, we minimize allocations through careful string handling
}

impl CommandParser {
    /// Create a new command parser
    pub const fn new() -> Self {
        Self
    }

    /// Parse a command line into a Command enum
    pub fn parse(&self, input: &str) -> CommandResult {
        let trimmed = input.trim();
        
        if trimmed.is_empty() {
            return Ok(Command::Help);
        }

        // Split into command and arguments without allocating Vec
        let mut parts = trimmed.splitn(2, char::is_whitespace);
        let cmd = parts.next().unwrap_or("").to_lowercase();
        let args = parts.next().unwrap_or("");

        match cmd.as_str() {
            "help" | "?" => Ok(Command::Help),
            "status" => Ok(Command::Status),
            "force_kill" | "kill" | "emergency_stop" => Ok(Command::ForceKill),
            
            "dump_state" | "save_state" => {
                if args.is_empty() {
                    return Err(ParseError::MissingArgument("path".to_string()));
                }
                Ok(Command::DumpState { path: args.trim().to_string() })
            }
            
            "shadow_mode" | "shadow" => {
                let enable = parse_bool(args)?;
                Ok(Command::ShadowMode { enable })
            }
            
            "set_risk" | "risk_limit" => {
                let (param, value) = parse_key_value(args)?;
                Ok(Command::SetRiskLimit { 
                    parameter: param.to_string(),
                    value,
                })
            }
            
            "inject_fault" | "fault" => {
                let (fault_type_str, target) = parse_key_value(args)?;
                let fault_type = fault_type_str
                    .parse::<FaultType>()
                    .map_err(|_| ParseError::InvalidValue {
                        param: "fault_type".to_string(),
                        value: fault_type_str.to_string(),
                    })?;
                Ok(Command::InjectFault {
                    fault_type,
                    target: target.to_string(),
                })
            }
            
            "show_actors" | "actors" | "state" => Ok(Command::ShowActors),
            "metrics" | "perf" | "performance" => Ok(Command::Metrics),
            
            // Exit commands (handled specially in shell)
            "exit" | "quit" => Ok(Command::Unknown(cmd)),
            
            _ => Ok(Command::Unknown(cmd)),
        }
    }

    /// Get list of available commands (for autocomplete)
    pub fn get_commands(&self) -> &'static [&'static str] {
        &[
            "help",
            "status",
            "force_kill",
            "dump_state",
            "shadow_mode",
            "set_risk",
            "inject_fault",
            "show_actors",
            "metrics",
        ]
    }

    /// Get risk parameter names (for autocomplete)
    pub fn get_risk_parameters(&self) -> &'static [&'static str] {
        &[
            "max_position",
            "max_order_size",
            "daily_loss_limit",
            "var_limit",
            "max_drawdown",
            "leverage_limit",
            "concentration_limit",
        ]
    }

    /// Get fault types (for autocomplete)
    pub fn get_fault_types(&self) -> &'static [&'static str] {
        &[
            "latency",
            "disconnect",
            "order_reject",
            "sequence_gap",
            "timeout",
            "packet_loss",
            "corrupted_data",
        ]
    }
}

/// Parse a boolean value from string
fn parse_bool(s: &str) -> Result<bool, ParseError> {
    let s = s.trim().to_lowercase();
    match s.as_str() {
        "on" | "true" | "1" | "yes" | "enable" => Ok(true),
        "off" | "false" | "0" | "no" | "disable" => Ok(false),
        "" => Err(ParseError::MissingArgument("value".to_string())),
        _ => Err(ParseError::InvalidValue {
            param: "boolean".to_string(),
            value: s,
        }),
    }
}

/// Parse a key-value pair from arguments
fn parse_key_value(s: &str) -> Result<(&str, f64), ParseError> {
    let s = s.trim();
    
    // Find the split point (first whitespace or '=')
    let mut key_end = 0;
    let mut value_start = 0;
    
    let chars: Vec<char> = s.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c == '=' {
            key_end = i;
            value_start = i + 1;
            break;
        } else if c.is_whitespace() {
            key_end = i;
            // Skip whitespace to find value start
            for (j, &c2) in chars.iter().enumerate().skip(i) {
                if !c2.is_whitespace() {
                    value_start = j;
                    break;
                }
            }
            break;
        }
    }
    
    if key_end == 0 || value_start == 0 {
        return Err(ParseError::MissingArgument("value".to_string()));
    }
    
    let key = &s[..key_end];
    let value_str = &s[value_start..];
    
    let value = value_str
        .trim()
        .parse::<f64>()
        .map_err(|_| ParseError::InvalidValue {
            param: key.to_string(),
            value: value_str.to_string(),
        })?;
    
    Ok((key, value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_help() {
        let parser = CommandParser::new();
        assert_eq!(parser.parse("help").unwrap(), Command::Help);
        assert_eq!(parser.parse("?").unwrap(), Command::Help);
    }

    #[test]
    fn test_parse_force_kill() {
        let parser = CommandParser::new();
        assert_eq!(parser.parse("force_kill").unwrap(), Command::ForceKill);
        assert_eq!(parser.parse("kill").unwrap(), Command::ForceKill);
        assert_eq!(parser.parse("emergency_stop").unwrap(), Command::ForceKill);
    }

    #[test]
    fn test_parse_shadow_mode() {
        let parser = CommandParser::new();
        assert_eq!(
            parser.parse("shadow_mode on").unwrap(),
            Command::ShadowMode { enable: true }
        );
        assert_eq!(
            parser.parse("shadow_mode off").unwrap(),
            Command::ShadowMode { enable: false }
        );
    }

    #[test]
    fn test_parse_dump_state() {
        let parser = CommandParser::new();
        assert_eq!(
            parser.parse("dump_state /tmp/state.bin").unwrap(),
            Command::DumpState { path: "/tmp/state.bin".to_string() }
        );
    }

    #[test]
    fn test_parse_set_risk() {
        let parser = CommandParser::new();
        let result = parser.parse("set_risk max_position 1000000").unwrap();
        match result {
            Command::SetRiskLimit { parameter, value } => {
                assert_eq!(parameter, "max_position");
                assert!((value - 1000000.0).abs() < f64::EPSILON);
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_parse_inject_fault() {
        let parser = CommandParser::new();
        let result = parser.parse("inject_fault latency gateway").unwrap();
        match result {
            Command::InjectFault { fault_type, target } => {
                assert_eq!(fault_type, FaultType::Latency);
                assert_eq!(target, "gateway");
            }
            _ => panic!("Wrong command variant"),
        }
    }

    #[test]
    fn test_parse_unknown() {
        let parser = CommandParser::new();
        match parser.parse("unknown_cmd").unwrap() {
            Command::Unknown(cmd) => assert_eq!(cmd, "unknown_cmd"),
            _ => panic!("Should be Unknown variant"),
        }
    }

    #[test]
    fn test_parse_error_missing_arg() {
        let parser = CommandParser::new();
        assert!(matches!(
            parser.parse("dump_state"),
            Err(ParseError::MissingArgument(_))
        ));
    }
}
