#!/usr/bin/env python3
"""
Interactive Prompt - Stage 50
Sexy terminal prompt using rich and prompt_toolkit with Ctrl+C interception.
Routes "yes" to Rust Global Kill Switch.
"""

import os
import sys
import signal
import time
import logging
from datetime import datetime
from typing import Optional, Callable
from pathlib import Path
import threading
import zmq

# Try to import rich and prompt_toolkit, fallback gracefully if not available
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from prompt_toolkit import prompt
    from prompt_toolkit.styles import Style
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.key_binding import KeyBindings
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger('InteractivePrompt')

# Constants
ZMQ_KILL_URL = "tcp://localhost:5557"
HISTORY_FILE = str(Path.home() / '.crypto_bot_history')


class RichDisplay:
    """Rich terminal display manager."""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.live: Optional[Live] = None
    
    def print_welcome(self):
        """Print welcome panel."""
        if not self.console:
            print("=" * 60)
            print("CRYPTO MEDIUM FREQUENCY TRADING BOT - STAGE 50")
            print("=" * 60)
            return
        
        welcome_text = Text()
        welcome_text.append("🤖 CRYPTO MEDIUM FREQUENCY TRADING BOT\n", style="bold cyan")
        welcome_text.append("Stage 50 - Ultimate Handoff Infrastructure\n", style="dim")
        welcome_text.append("\nType /HELP for commands or press Ctrl+C for shutdown prompt", style="italic")
        
        self.console.print(Panel(
            welcome_text,
            title="[bold green]System Ready[/bold green]",
            border_style="green"
        ))
    
    def print_status(self, status_data: dict):
        """Print system status table."""
        if not self.console:
            for key, value in status_data.items():
                print(f"  {key}: {value}")
            return
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Status", style="yellow")
        
        for key, value in status_data.items():
            status_icon = "✅" if value else "❌" if isinstance(value, bool) else "•"
            table.add_row(str(key), str(value), status_icon)
        
        self.console.print(table)
    
    def print_shutdown_prompt(self) -> Optional[bool]:
        """Display shutdown confirmation prompt."""
        if not self.console:
            response = input("⚠️  Do you want to stop the bot? (yes/no): ").strip().lower()
            return response in ['yes', 'y']
        
        self.console.print(Panel(
            "[bold yellow]Do you want to stop the bot?[/bold yellow]\n\n"
            "Type [green]yes[/green] to activate Global Kill Switch\n"
            "Type [red]no[/red] to continue trading",
            title="⚠️  SHUTDOWN CONFIRMATION",
            border_style="yellow"
        ))
        
        try:
            response = input("> ").strip().lower()
            return response in ['yes', 'y']
        except:
            return None
    
    def print_killing_switch_activated(self, reason: str):
        """Display kill switch activation message."""
        if not self.console:
            print(f"🔴 GLOBAL KILL SWITCH ACTIVATED")
            print(f"   Reason: {reason}")
            return
        
        self.console.print(Panel(
            f"[bold red]GLOBAL KILL SWITCH ACTIVATED[/bold red]\n\n"
            f"Reason: {reason}\n\n"
            "Waiting for all processes to terminate...",
            title="🔴 EMERGENCY SHUTDOWN",
            border_style="red"
        ))
    
    def print_error(self, message: str):
        """Display error message."""
        if not self.console:
            print(f"❌ ERROR: {message}")
            return
        
        self.console.print(f"[bold red]❌ ERROR:[/bold red] {message}")
    
    def print_success(self, message: str):
        """Display success message."""
        if not self.console:
            print(f"✅ {message}")
            return
        
        self.console.print(f"[bold green]✅[/bold green] {message}")


class PromptToolkitShell:
    """Advanced shell using prompt_toolkit."""
    
    def __init__(self):
        self.style = Style.from_dict({
            'prompt': 'ansicyan bold',
            'input': 'ansigreen',
            'error': 'ansired bold',
            'success': 'ansigreen bold',
        })
        
        self.kb = KeyBindings()
        
        @self.kb.add('c-c')
        def _(event):
            """Handle Ctrl+C."""
            event.app.current_buffer.text = '/KILL'
            event.app.run()
        
        self.history = None
        if PROMPT_TOOLKIT_AVAILABLE:
            try:
                self.history = FileHistory(HISTORY_FILE)
            except:
                pass
    
    def get_input(self, prompt_text: str = "crypto-bot> ") -> Optional[str]:
        """Get user input with advanced features."""
        if not PROMPT_TOOLKIT_AVAILABLE:
            return input(prompt_text).strip()
        
        try:
            return prompt(
                [('class:prompt', f"🔹 {prompt_text}")],
                style=self.style,
                history=self.history,
                auto_suggest=AutoSuggestFromHistory(),
                key_bindings=self.kb,
                mouse_support=False,
            ).strip()
        except EOFError:
            return None
        except KeyboardInterrupt:
            return '/KILL'


class KillSwitchClient:
    """Client for communicating with Rust Global Kill Switch."""
    
    def __init__(self):
        self.zmq_context = zmq.Context()
        self.kill_socket = self.zmq_context.socket(zmq.PUSH)
        self.kill_socket.setsockopt(zmq.LINGER, 100)
        self.connected = False
    
    def connect(self):
        """Connect to kill switch socket."""
        try:
            self.kill_socket.connect(ZMQ_KILL_URL)
            self.connected = True
            logger.info(f"Connected to kill switch: {ZMQ_KILL_URL}")
        except Exception as e:
            logger.warning(f"Could not connect to kill switch: {e}")
            self.connected = False
    
    def trigger_kill(self, reason: str) -> bool:
        """Trigger the global kill switch."""
        if not self.connected:
            logger.warning("Not connected to kill switch, attempting fallback")
            return self._fallback_kill(reason)
        
        message = {
            'type': 'MANUAL_KILL',
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'source': 'interactive_prompt'
        }
        
        try:
            self.kill_socket.send_json(message, flags=zmq.NOBLOCK)
            logger.critical(f"Kill switch triggered: {reason}")
            return True
        except Exception as e:
            logger.error(f"Failed to send kill signal: {e}")
            return self._fallback_kill(reason)
    
    def _fallback_kill(self, reason: str) -> bool:
        """Fallback kill mechanism via file signal."""
        try:
            kill_file = Path('/tmp/crypto_bot_kill_signal.txt')
            kill_file.write_text(f"{reason}\n{datetime.now().isoformat()}")
            logger.critical(f"Fallback kill signal written: {reason}")
            return True
        except Exception as e:
            logger.critical(f"All kill mechanisms failed: {e}")
            return False
    
    def close(self):
        """Close ZMQ socket."""
        self.kill_socket.close()
        self.zmq_context.term()


class InteractivePromptManager:
    """Main interactive prompt manager."""
    
    def __init__(self):
        self.display = RichDisplay()
        self.shell = PromptToolkitShell()
        self.kill_client = KillSwitchClient()
        self.running = False
        self.ctrl_c_timestamps = []
        self.max_ctrl_c_window = 2.0  # seconds
    
    def run(self):
        """Main interactive prompt loop."""
        self.running = True
        self.kill_client.connect()
        
        self.display.print_welcome()
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self._handle_ctrl_c)
        
        while self.running:
            try:
                # Get user input
                user_input = self.shell.get_input()
                
                if user_input is None:
                    # EOF received
                    break
                
                # Process command
                self._process_command(user_input)
            
            except Exception as e:
                logger.error(f"Error in prompt loop: {e}")
                self.display.print_error(str(e))
        
        self.shutdown()
    
    def _handle_ctrl_c(self, signum, frame):
        """Handle Ctrl+C with double-tap detection and confirmation."""
        now = time.time()
        self.ctrl_c_timestamps.append(now)
        
        # Remove old timestamps
        self.ctrl_c_timestamps = [
            ts for ts in self.ctrl_c_timestamps
            if now - ts < self.max_ctrl_c_window
        ]
        
        # Double tap within window = immediate kill
        if len(self.ctrl_c_timestamps) >= 2:
            logger.warning("Double Ctrl+C detected - immediate shutdown")
            self._confirm_and_kill("Double Ctrl+C emergency shutdown")
            return
        
        # Single tap = confirmation prompt
        should_kill = self.display.print_shutdown_prompt()
        
        if should_kill:
            self._confirm_and_kill("User confirmed via prompt")
        elif should_kill is False:
            self.display.print_success("Continuing operation")
    
    def _confirm_and_kill(self, reason: str):
        """Confirm and execute kill switch."""
        self.display.print_killing_switch_activated(reason)
        
        if self.kill_client.trigger_kill(reason):
            self.display.print_success("Kill signal sent successfully")
        else:
            self.display.print_error("Failed to send kill signal")
        
        # Give processes time to shut down
        time.sleep(2)
        self.running = False
    
    def _process_command(self, command: str):
        """Process user command."""
        cmd = command.strip().upper()
        
        if cmd == '/START':
            self.display.print_success("Use the master CLI or launch script to start the system")
        
        elif cmd == '/KILL':
            self._confirm_and_kill("User requested via /KILL command")
        
        elif cmd == '/STATUS':
            self.display.print_status({
                'CLI Connected': True,
                'Kill Switch': self.kill_client.connected,
                'Rich Display': RICH_AVAILABLE,
                'Prompt Toolkit': PROMPT_TOOLKIT_AVAILABLE
            })
        
        elif cmd in ['/HELP', '?']:
            self._show_help()
        
        elif cmd in ['/QUIT', '/EXIT']:
            self.display.print_success("Exiting interactive prompt (system continues running)")
            self.running = False
        
        elif cmd.startswith('/'):
            self.display.print_error(f"Unknown command: {cmd}")
            self._show_help()
        
        else:
            # Echo unknown input
            print(f"  '{command}' - use /HELP for available commands")
    
    def _show_help(self):
        """Display help information."""
        help_text = """
Available Commands:
  /START          Start the trading system (via master orchestrator)
  /KILL           Emergency shutdown with confirmation
  /STATUS         Show connection status
  /HELP           Show this help message
  /QUIT, /EXIT    Exit interactive prompt

Keyboard Shortcuts:
  Ctrl+C          Shutdown confirmation prompt
  Ctrl+C (x2)     Immediate emergency shutdown
  Tab             Auto-completion (if available)
"""
        if self.display.console and RICH_AVAILABLE:
            self.display.console.print(Panel(help_text, title="Help"))
        else:
            print(help_text)
    
    def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down interactive prompt...")
        self.kill_client.close()
        self.display.print_success("Interactive prompt closed")


def main():
    """Entry point for interactive prompt."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Interactive Trading Bot Prompt')
    parser.add_argument('--no-rich', action='store_true', help='Disable rich display')
    parser.add_argument('--no-prompt-toolkit', action='store_true', help='Disable prompt_toolkit')
    args = parser.parse_args()
    
    if args.no_rich:
        global RICH_AVAILABLE
        RICH_AVAILABLE = False
    
    if args.no_prompt_toolkit:
        global PROMPT_TOOLKIT_AVAILABLE
        PROMPT_TOOLKIT_AVAILABLE = False
    
    manager = InteractivePromptManager()
    
    try:
        manager.run()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
