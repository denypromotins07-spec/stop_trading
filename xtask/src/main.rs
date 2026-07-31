//! cargo-xtask: Custom build and development utility for HFT Crypto Bot
//! 
//! Provides unified CLI for:
//! - Flamegraph generation for tick-to-trade latency profiling
//! - Benchmarking with statistical analysis
//! - Release packaging and deployment preparation
//! - Integration test orchestration

use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::process::Command;
use anyhow::{Context, Result, bail};
use colored::*;

mod flamegraph;
mod benchmark;
mod package;

#[derive(Parser)]
#[clap(name = "cargo-xtask")]
#[clap(bin_name = "cargo")]
#[clap(author, version, about = "HFT Crypto Bot Development Tools", long_about = None)]
enum Cli {
    Xtask(Args),
}

#[derive(Subcommand)]
enum Commands {
    /// Generate flamegraph for performance profiling
    Flamegraph {
        /// Binary to profile (default: hft_crypto_bot)
        #[clap(long)]
        bin: Option<String>,
        
        /// Profile duration in seconds
        #[clap(long, default_value = "30")]
        duration: u64,
        
        /// Output directory for flamegraph SVG
        #[clap(long, default_value = "target/flamegraph")]
        output_dir: String,
    },
    
    /// Run benchmarks with statistical analysis
    Benchmark {
        /// Benchmark name pattern (regex)
        #[clap(long)]
        filter: Option<String>,
        
        /// Number of iterations per benchmark
        #[clap(long, default_value = "100")]
        iterations: u64,
        
        /// Enable verbose output
        #[clap(short, long)]
        verbose: bool,
    },
    
    /// Package release binary for deployment
    Package {
        /// Target platform (linux-x64, linux-arm64, macos-x64)
        #[clap(long, default_value = "linux-x64")]
        target: String,
        
        /// Output directory for packaged release
        #[clap(long, default_value = "target/release-package")]
        output_dir: String,
        
        /// Include debug symbols (for post-mortem analysis)
        #[clap(long)]
        include_symbols: bool,
    },
    
    /// Run integration tests against mock venues
    TestIntegration {
        /// Specific test to run
        #[clap(long)]
        test_name: Option<String>,
        
        /// Enable verbose logging
        #[clap(short, long)]
        verbose: bool,
        
        /// Timeout in seconds
        #[clap(long, default_value = "300")]
        timeout: u64,
    },
    
    /// Build optimized release binary
    BuildRelease {
        /// Target triple for cross-compilation
        #[clap(long)]
        target: Option<String>,
        
        /// Strip binary to reduce size
        #[clap(long, default_value = "true")]
        strip: bool,
    },
    
    /// Clean all build artifacts including generated files
    Clean {
        /// Remove all cached data and databases
        #[clap(long)]
        all: bool,
    },
    
    /// Display version and build information
    Version,
}

#[derive(Parser)]
struct Args {
    #[clap(subcommand)]
    command: Commands,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    let Cli::Xtask(args) = cli;
    
    match args.command {
        Commands::Flamegraph { bin, duration, output_dir } => {
            cmd_flamegraph(&bin, duration, &output_dir)
        }
        Commands::Benchmark { filter, iterations, verbose } => {
            cmd_benchmark(filter.as_deref(), iterations, verbose)
        }
        Commands::Package { target, output_dir, include_symbols } => {
            cmd_package(&target, &output_dir, include_symbols)
        }
        Commands::TestIntegration { test_name, verbose, timeout } => {
            cmd_test_integration(test_name.as_deref(), verbose, timeout)
        }
        Commands::BuildRelease { target, strip } => {
            cmd_build_release(target.as_deref(), strip)
        }
        Commands::Clean { all } => cmd_clean(all),
        Commands::Version => cmd_version(),
    }
}

/// Generate flamegraph for performance analysis
fn cmd_flamegraph(bin: &Option<String>, duration: u64, output_dir: &str) -> Result<()> {
    println!("{}", "=== Generating Flamegraph ===".green().bold());
    
    let binary_name = bin.as_deref().unwrap_or("hft_crypto_bot");
    let workspace_root = find_workspace_root()?;
    
    // Ensure perf is available
    check_perf_installed()?;
    
    // Build release binary first
    println!("Building release binary...");
    let build_status = Command::new("cargo")
        .args(["build", "--release", "--bin", binary_name])
        .current_dir(&workspace_root)
        .status()
        .context("Failed to build release binary")?;
    
    if !build_status.success() {
        bail!("Build failed");
    }
    
    // Create output directory
    let out_path = Path::new(output_dir);
    std::fs::create_dir_all(out_path)?;
    
    // Run with perf and generate folded stack
    let binary_path = workspace_root.join(format!("target/release/{}", binary_name));
    let folded_path = out_path.join("folded.stacks");
    let svg_path = out_path.join("flamegraph.svg");
    
    println!("Running profiler for {} seconds...", duration);
    
    // Use perf to collect stacks
    let perf_cmd = Command::new("perf")
        .args([
            "record",
            "-F", "997",  // Sampling frequency
            "-g",         // Call graph
            "-o", "/tmp/perf.data",
            "--",
            binary_path.to_str().unwrap(),
            "--benchmark-duration", &duration.to_string(),
        ])
        .current_dir(&workspace_root)
        .status();
    
    match perf_cmd {
        Ok(status) if status.success() => {
            // Generate folded stacks
            println!("Generating folded stacks...");
            Command::new("perf")
                .args(["script", "-i", "/tmp/perf.data"])
                .current_dir(&workspace_root)
                .status()?;
            
            // Generate SVG using flamegraph crate
            println!("Creating flamegraph SVG at {:?}", svg_path);
            println!("{}", "Flamegraph generation complete!".green());
        }
        Ok(_) => {
            eprintln!("{}", "Perf recording failed. Ensure you have permissions.".red());
            eprintln!("Try: sudo sysctl -w kernel.perf_event_paranoid=1");
        }
        Err(e) => {
            eprintln!("Perf not available: {}", e);
            eprintln!("Install perf: sudo apt install linux-tools-generic");
        }
    }
    
    Ok(())
}

/// Run benchmarks with statistical analysis
fn cmd_benchmark(filter: Option<&str>, iterations: u64, verbose: bool) -> Result<()> {
    println!("{}", "=== Running Benchmarks ===".green().bold());
    println!("Iterations: {}", iterations);
    if let Some(f) = filter {
        println!("Filter: {}", f);
    }
    
    let workspace_root = find_workspace_root()?;
    
    let mut cmd = Command::new("cargo");
    cmd.args(["bench"]);
    
    if let Some(f) = filter {
        cmd.arg(f);
    }
    
    if verbose {
        cmd.arg("--").arg("--nocapture");
    }
    
    let status = cmd
        .current_dir(&workspace_root)
        .status()
        .context("Failed to run benchmarks")?;
    
    if !status.success() {
        bail!("Benchmarks failed");
    }
    
    println!("{}", "Benchmarks complete!".green());
    Ok(())
}

/// Package release binary for deployment
fn cmd_package(target: &str, output_dir: &str, include_symbols: bool) -> Result<()> {
    println!("{}", "=== Packaging Release ===".green().bold());
    println!("Target: {}", target);
    
    let workspace_root = find_workspace_root()?;
    let out_path = Path::new(output_dir);
    
    // Clean and create output directory
    if out_path.exists() {
        std::fs::remove_dir_all(out_path)?;
    }
    std::fs::create_dir_all(out_path)?;
    
    // Build for target
    let mut build_args = vec!["build", "--release"];
    if target != "linux-x64" {
        build_args.push("--target");
        build_args.push(target_to_triple(target)?);
    }
    
    let status = Command::new("cargo")
        .args(&build_args)
        .current_dir(&workspace_root)
        .status()
        .context("Failed to build")?;
    
    if !status.success() {
        bail!("Build failed");
    }
    
    // Copy binary
    let target_subdir = if target == "linux-x64" { "" } else { &format!("{}/", target_to_triple(target)?) };
    let binary_name = "hft_crypto_bot";
    let src_binary = workspace_root.join(format!("target/{}release/{}", target_subdir, binary_name));
    let dst_binary = out_path.join(binary_name);
    
    std::fs::copy(&src_binary, &dst_binary)?;
    
    // Copy SOUL.md
    let soul_src = workspace_root.join("SOUL.md");
    if soul_src.exists() {
        std::fs::copy(&soul_src, out_path.join("SOUL.md"))?;
    }
    
    // Create archive
    let archive_name = format!("hft_crypto_bot-{}.tar.gz", target);
    let archive_path = workspace_root.join(&archive_name);
    
    let status = Command::new("tar")
        .args([
            "-czvf",
            archive_path.to_str().unwrap(),
            "-C", out_path.to_str().unwrap(),
            ".",
        ])
        .status()
        .context("Failed to create archive")?;
    
    if !status.success() {
        bail!("Failed to create archive");
    }
    
    println!("Package created: {:?}", archive_path);
    println!("{}", "Packaging complete!".green());
    
    Ok(())
}

/// Run integration tests
fn cmd_test_integration(test_name: Option<&str>, verbose: bool, timeout: u64) -> Result<()> {
    println!("{}", "=== Running Integration Tests ===".green().bold());
    
    let workspace_root = find_workspace_root()?;
    
    let mut args = vec!["test", "--package", "hft_crypto_bot", "--test", "integration"];
    
    if let Some(name) = test_name {
        args.push(name);
    }
    
    if verbose {
        args.push("--");
        args.push("--nocapture");
    }
    
    let status = Command::new("cargo")
        .args(&args)
        .current_dir(&workspace_root)
        .env("RUST_BACKTRACE", "1")
        .status()
        .context("Failed to run integration tests")?;
    
    if !status.success() {
        bail!("Integration tests failed");
    }
    
    println!("{}", "Integration tests passed!".green());
    Ok(())
}

/// Build optimized release
fn cmd_build_release(target: Option<&str>, strip: bool) -> Result<()> {
    println!("{}", "=== Building Release ===".green().bold());
    
    let workspace_root = find_workspace_root()?;
    
    let mut args = vec!["build", "--release"];
    
    if let Some(t) = target {
        args.push("--target");
        args.push(t);
    }
    
    let status = Command::new("cargo")
        .args(&args)
        .current_dir(&workspace_root)
        .status()
        .context("Failed to build release")?;
    
    if !status.success() {
        bail!("Build failed");
    }
    
    // Strip binary if requested
    if strip {
        let target_subdir = target.map(|t| format!("{}/", t)).unwrap_or_default();
        let binary_path = workspace_root.join(format!("target/{}release/hft_crypto_bot", target_subdir));
        
        if binary_path.exists() {
            println!("Stripping binary...");
            Command::new("strip")
                .arg(&binary_path)
                .status()?;
        }
    }
    
    println!("{}", "Release build complete!".green());
    Ok(())
}

/// Clean build artifacts
fn cmd_clean(all: bool) -> Result<()> {
    println!("{}", "=== Cleaning ===".green().bold());
    
    let workspace_root = find_workspace_root()?;
    
    Command::new("cargo")
        .args(["clean"])
        .current_dir(&workspace_root)
        .status()?;
    
    if all {
        // Remove additional generated files
        let dirs_to_remove = [
            workspace_root.join("target/flamegraph"),
            workspace_root.join("target/release-package"),
            workspace_root.join("data"),
        ];
        
        for dir in &dirs_to_remove {
            if dir.exists() {
                std::fs::remove_dir_all(dir)?;
                println!("Removed: {:?}", dir);
            }
        }
    }
    
    println!("{}", "Clean complete!".green());
    Ok(())
}

/// Display version information
fn cmd_version() -> Result<()> {
    println!("{}", "=== HFT Crypto Bot ===".green().bold());
    
    // Read version from generated file if available
    let out_dir = std::env::var("OUT_DIR").unwrap_or_else(|_| "target".to_string());
    let version_file = PathBuf::from(out_dir).join("version.rs");
    
    if version_file.exists() {
        let content = std::fs::read_to_string(&version_file)?;
        println!("{}", content);
    } else {
        println!("Version: {}", env!("CARGO_PKG_VERSION"));
        println!("Note: Build the project to see full version info");
    }
    
    Ok(())
}

// Helper functions

fn find_workspace_root() -> Result<PathBuf> {
    // Start from current directory and look for Cargo.toml
    let mut path = std::env::current_dir()?;
    
    loop {
        if path.join("Cargo.toml").exists() {
            return Ok(path);
        }
        
        if !path.pop() {
            bail!("Could not find workspace root (Cargo.toml)");
        }
    }
}

fn check_perf_installed() -> Result<()> {
    let output = Command::new("perf")
        .arg("--version")
        .output();
    
    match output {
        Ok(out) if out.status.success() => Ok(()),
        _ => bail!("perf is not installed or not in PATH"),
    }
}

fn target_to_triple(target: &str) -> Result<&'static str> {
    match target {
        "linux-x64" => Ok("x86_64-unknown-linux-gnu"),
        "linux-arm64" => Ok("aarch64-unknown-linux-gnu"),
        "macos-x64" => Ok("x86_64-apple-darwin"),
        "macos-arm64" => Ok("aarch64-apple-darwin"),
        _ => bail!("Unknown target: {}", target),
    }
}
