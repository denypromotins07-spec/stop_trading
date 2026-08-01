"""
Chapter 5: Python CLI, Fuzzing & Final Integration Testing
File: python/cli/interactive_shell.py

Python-side REPL using prompt_toolkit for inspecting Ray actor states,
ML model weights, and Nautilus portfolios. Allows operators to manually
inject faults or force model hot-swaps without stopping the main trading daemon.
Runs on separate non-blocking asyncio thread.
"""

import asyncio
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import json

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("prompt_toolkit not available, using basic input")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Style for the interactive shell
shell_style = Style.from_dict({
    'prompt': 'ansigreen bold',
    'command': 'ansiblue',
    'output': 'ansiwhite',
    'error': 'ansired bold',
    'warning': 'ansiyellow',
})

# Key bindings
bindings = KeyBindings()


@bindings.add('c-c')
def _(event):
    """Handle Ctrl+C."""
    event.app.exit(result=None)


class CommandCompleter(Completer):
    """Auto-completion for CLI commands."""
    
    def __init__(self, commands: List[str]):
        self.commands = commands
    
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        for cmd in self.commands:
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text))


class InteractiveShell:
    """
    Interactive REPL shell for system inspection and control.
    Runs on a separate non-blocking asyncio thread.
    """
    
    def __init__(self):
        self.is_running = False
        self.shell_thread: Optional[threading.Thread] = None
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Registered commands
        self.commands: Dict[str, Callable] = {}
        self.command_help: Dict[str, str] = {}
        
        # System state references (set by external code)
        self.ray_state_provider: Optional[Callable] = None
        self.model_weights_provider: Optional[Callable] = None
        self.portfolio_provider: Optional[Callable] = None
        self.fault_injector: Optional[Callable] = None
        self.model_swapper: Optional[Callable] = None
        
        # Register default commands
        self._register_default_commands()
    
    def _register_default_commands(self):
        """Register built-in CLI commands."""
        
        def cmd_help(args):
            """Show available commands."""
            output = "Available commands:\n"
            for cmd_name, help_text in self.command_help.items():
                output += f"  {cmd_name}: {help_text}\n"
            return output
        
        def cmd_status(args):
            """Show system status."""
            status = {
                "timestamp": datetime.utcnow().isoformat(),
                "shell_running": self.is_running,
                "ray_connected": self.ray_state_provider is not None,
                "model_loaded": self.model_weights_provider is not None
            }
            return json.dumps(status, indent=2)
        
        def cmd_ray(args):
            """Show Ray cluster state."""
            if self.ray_state_provider:
                return json.dumps(self.ray_state_provider(), indent=2)
            return "Ray state provider not configured"
        
        def cmd_weights(args):
            """Show ML model weights summary."""
            if self.model_weights_provider:
                weights = self.model_weights_provider()
                return f"Model weights: {len(weights)} parameters"
            return "Model weights provider not configured"
        
        def cmd_portfolio(args):
            """Show Nautilus portfolio state."""
            if self.portfolio_provider:
                return json.dumps(self.portfolio_provider(), indent=2)
            return "Portfolio provider not configured"
        
        def cmd_inject(args):
            """Inject a fault for testing. Usage: /inject <fault_type>"""
            if not self.fault_injector:
                return "Fault injector not configured"
            
            if len(args) < 1:
                return "Usage: /inject <fault_type> [params]"
            
            fault_type = args[0]
            params = args[1:] if len(args) > 1 else []
            
            result = self.fault_injector(fault_type, params)
            return f"Fault injected: {result}"
        
        def cmd_swap(args):
            """Force model hot-swap. Usage: /swap <model_path>"""
            if not self.model_swapper:
                return "Model swapper not configured"
            
            if len(args) < 1:
                return "Usage: /swap <model_path>"
            
            model_path = args[0]
            result = self.model_swapper(model_path)
            return f"Model swapped: {result}"
        
        def cmd_exit(args):
            """Exit the interactive shell."""
            self.stop()
            return "Shutting down..."
        
        # Register all commands
        self.register_command("help", cmd_help, "Show available commands")
        self.register_command("status", cmd_status, "Show system status")
        self.register_command("ray", cmd_ray, "Show Ray cluster state")
        self.register_command("weights", cmd_weights, "Show ML model weights")
        self.register_command("portfolio", cmd_portfolio, "Show portfolio state")
        self.register_command("inject", cmd_inject, "Inject fault for testing")
        self.register_command("swap", cmd_swap, "Force model hot-swap")
        self.register_command("exit", cmd_exit, "Exit the shell")
        self.register_command("quit", cmd_exit, "Exit the shell")
    
    def register_command(
        self, 
        name: str, 
        handler: Callable, 
        help_text: str = ""
    ):
        """Register a custom command."""
        self.commands[name] = handler
        self.command_help[name] = help_text
        logger.debug(f"Registered command: {name}")
    
    def _process_command(self, line: str) -> str:
        """Process a command line and return output."""
        line = line.strip()
        
        if not line:
            return ""
        
        # Parse command and arguments
        parts = line.split()
        cmd_name = parts[0].lstrip('/')
        args = parts[1:]
        
        # Look up command
        if cmd_name in self.commands:
            try:
                result = self.commands[cmd_name](args)
                return str(result) if result else ""
            except Exception as e:
                return f"Error: {str(e)}"
        else:
            return f"Unknown command: {cmd_name}. Type /help for available commands."
    
    async def _run_async_shell(self):
        """Async shell runner."""
        if not PROMPT_TOOLKIT_AVAILABLE:
            # Fallback to basic input
            print("prompt_toolkit not available, using basic input mode")
            print("Type /help for commands, /exit to quit")
            
            while self.is_running:
                try:
                    line = input("> ")
                    if line.strip() == '/exit':
                        break
                    output = self._process_command(line)
                    if output:
                        print(output)
                except EOFError:
                    break
                except KeyboardInterrupt:
                    continue
            return
        
        # Use prompt_toolkit
        session = PromptSession(
            history=FileHistory('/tmp/hft_shell_history.txt'),
            auto_suggest=AutoSuggestFromHistory(),
            completer=CommandCompleter([f"/{cmd}" for cmd in self.commands.keys()]),
            style=shell_style,
            key_bindings=bindings
        )
        
        print("\n=== HFT Interactive Shell ===")
        print("Type /help for commands, /exit to quit\n")
        
        while self.is_running:
            try:
                line = await session.prompt_async(
                    [('class:prompt', '> ')]
                )
                output = self._process_command(line)
                
                if output:
                    if "Error" in output:
                        print([('class:error', output)])
                    elif "Fault" in output or "swapped" in output:
                        print([('class:warning', output)])
                    else:
                        print([('class:output', output)])
                        
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
    
    def _run_shell_thread(self):
        """Run shell in dedicated thread with its own event loop."""
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)
        
        self.is_running = True
        self.event_loop.run_until_complete(self._run_async_shell())
        
        self.event_loop.close()
        self.is_running = False
        logger.info("Interactive shell stopped")
    
    def start(self):
        """Start the interactive shell on a background thread."""
        if self.is_running:
            logger.warning("Shell already running")
            return
        
        self.shell_thread = threading.Thread(
            target=self._run_shell_thread,
            daemon=True,
            name="InteractiveShell"
        )
        self.shell_thread.start()
        logger.info("Interactive shell started on background thread")
    
    def stop(self):
        """Stop the interactive shell."""
        self.is_running = False
        
        if self.shell_thread and self.shell_thread.is_alive():
            self.shell_thread.join(timeout=5)
        
        logger.info("Interactive shell stopped")
    
    def is_active(self) -> bool:
        """Check if shell is running."""
        return self.is_running and self.shell_thread and self.shell_thread.is_alive()


# Module-level singleton
_shell: Optional[InteractiveShell] = None


def get_shell() -> InteractiveShell:
    """Get or create the module-level shell singleton."""
    global _shell
    if _shell is None:
        _shell = InteractiveShell()
    return _shell


def start_cli():
    """Start the interactive CLI."""
    shell = get_shell()
    shell.start()
    return shell


def stop_cli():
    """Stop the interactive CLI."""
    global _shell
    if _shell:
        _shell.stop()


# Export for module use
__all__ = [
    "InteractiveShell",
    "get_shell",
    "start_cli",
    "stop_cli",
    "PROMPT_TOOLKIT_AVAILABLE"
]
