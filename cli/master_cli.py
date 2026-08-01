#!/usr/bin/env python3
"""
Master CLI - Stage 50
User-facing CLI wrapper capturing /START, /KILL commands and Ctrl+C.
Routes shutdown requests to Rust Global Kill Switch via ZMQ.
"""

import os
import sys
import signal
import time
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path
import zmq

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger('MasterCLI')

# Constants
ZMQ_CTRL_URL = "tcp://localhost:5558"
ZMQ_STATUS_URL = "tcp://localhost:5559"
COMMAND_TIMEOUT_SEC = 30


class CommandProcessor:
    """Processes user commands and routes to appropriate handlers."""
    
    def __init__(self):
        self.zmq_context = zmq.Context()
        self.ctrl_socket = self.zmq_context.socket(zmq.REQ)
        self.ctrl_socket.setsockopt(zmq.LINGER, 1000)
        self.status_socket = self.zmq_context.socket(zmq.SUB)
        self.status_socket.setsockopt(zmq.SUBSCRIBE, b"")
        
        self.running = False
        self.command_handlers = {
            '/START': self._handle_start,
            '/KILL': self._handle_kill,
            '/STATUS': self._handle_status,
            '/HELP': self._handle_help,
            '/QUIT': self._handle_quit,
            '/EXIT': self._handle_quit,
        }
    
    def connect(self):
        """Connect to control sockets."""
        try:
            self.ctrl_socket.connect(ZMQ_CTRL_URL)
            logger.info(f"Connected to control socket: {ZMQ_CTRL_URL}")
        except Exception as e:
            logger.warning(f"Could not connect to control socket: {e}")
        
        try:
            self.status_socket.connect(ZMQ_STATUS_URL)
            logger.info(f"Connected to status socket: {ZMQ_STATUS_URL}")
        except Exception as e:
            logger.warning(f"Could not connect to status socket: {e}")
    
    def process_command(self, command: str) -> bool:
        """Process a single command. Returns True if should continue running."""
        command = command.strip().upper()
        
        if not command:
            return True
        
        # Check for exact match first, then prefix match
        handler = self.command_handlers.get(command)
        if not handler:
            for cmd, hdlr in self.command_handlers.items():
                if command.startswith(cmd):
                    handler = hdlr
                    break
        
        if handler:
            return handler(command)
        else:
            print(f"❌ Unknown command: {command}")
            self._handle_help("")
            return True
    
    def _send_ctrl_message(self, msg_type: str, payload: dict = None) -> dict:
        """Send message to control socket and wait for response."""
        message = {
            'type': msg_type,
            'timestamp': datetime.now().isoformat(),
            'payload': payload or {}
        }
        
        try:
            self.ctrl_socket.send_json(message, flags=zmq.NOBLOCK)
            
            # Wait for response with timeout
            poller = zmq.Poller()
            poller.register(self.ctrl_socket, zmq.POLLIN)
            
            socks = dict(poller.poll(timeout=COMMAND_TIMEOUT_SEC * 1000))
            
            if self.ctrl_socket in socks:
                response = self.ctrl_socket.recv_json()
                return response
            else:
                return {'status': 'error', 'message': 'Command timeout'}
        
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _handle_start(self, args: str) -> bool:
        """Handle /START command."""
        print("🚀 Initiating system startup...")
        
        response = self._send_ctrl_message('START')
        
        if response.get('status') == 'ok':
            print("✅ System startup initiated successfully")
            if 'pid' in response:
                print(f"   Master Orchestrator PID: {response['pid']}")
            if 'trading_window' in response:
                print(f"   Trading window: {response['trading_window']} hours")
        else:
            print(f"❌ Startup failed: {response.get('message', 'Unknown error')}")
        
        return True
    
    def _handle_kill(self, args: str) -> bool:
        """Handle /KILL command - triggers Rust Global Kill Switch."""
        print("⚠️  INITIATING EMERGENCY SHUTDOWN...")
        
        reason = args.replace('/KILL', '').strip() or "User requested via CLI"
        
        response = self._send_ctrl_message('KILL', {'reason': reason})
        
        if response.get('status') == 'ok':
            print("✅ Global Kill Switch activated")
            print(f"   Reason: {reason}")
            print("   Waiting for processes to terminate...")
            return False  # Exit CLI after kill
        else:
            print(f"❌ Kill command failed: {response.get('message', 'Unknown error')}")
            print("   Attempting direct signal fallback...")
            os.kill(os.getpid(), signal.SIGTERM)
            return False
    
    def _handle_status(self, args: str) -> bool:
        """Handle /STATUS command."""
        print("📊 System Status:")
        
        response = self._send_ctrl_message('STATUS')
        
        if response.get('status') == 'ok':
            data = response.get('data', {})
            
            print(f"   State: {data.get('state', 'UNKNOWN')}")
            print(f"   Uptime: {data.get('uptime', 'N/A')}")
            print(f"   Trading Window: {data.get('window_remaining', 'N/A')}")
            print(f"   Rust Core: {'✅ Running' if data.get('rust_alive') else '❌ Stopped'}")
            print(f"   Python Daemon: {'✅ Running' if data.get('python_alive') else '❌ Stopped'}")
            
            if 'pnl' in data:
                pnl = data['pnl']
                print(f"   P&L: {pnl.get('value', 0):.2f} INR ({pnl.get('percent', 0):.2f}%)")
        else:
            print(f"   Could not retrieve status: {response.get('message', 'Unknown')}")
        
        return True
    
    def _handle_help(self, args: str) -> bool:
        """Handle /HELP command."""
        print("\n📖 Available Commands:")
        print("   /START          - Start the trading system")
        print("   /KILL [reason]  - Emergency shutdown (routes to Rust Kill Switch)")
        print("   /STATUS         - Show current system status")
        print("   /HELP           - Show this help message")
        print("   /QUIT, /EXIT    - Exit CLI (does not stop trading)")
        print("   Ctrl+C          - Interrupt with shutdown prompt")
        print("\n💡 Tips:")
        print("   - Use /KILL for immediate emergency shutdown")
        print("   - The system auto-shuts after 4 hour trading window")
        print("   - Press Ctrl+C twice for force quit")
        return True
    
    def _handle_quit(self, args: str) -> bool:
        """Handle /QUIT command."""
        print("👋 Exiting CLI (trading system continues running)")
        print("   Use /KILL to stop the trading system")
        return False
    
    def close(self):
        """Close ZMQ sockets."""
        self.ctrl_socket.close()
        self.status_socket.close()
        self.zmq_context.term()


class InteractiveCLIShell:
    """Interactive shell with rich prompts."""
    
    def __init__(self):
        self.processor = CommandProcessor()
        self.running = False
        self.ctrl_c_count = 0
        self.last_ctrl_c_time = 0
    
    def run(self):
        """Main CLI loop."""
        self.running = True
        self.processor.connect()
        
        print("=" * 60)
        print("🤖 CRYPTO MEDIUM FREQUENCY TRADING BOT - STAGE 50")
        print("   Master CLI Interface")
        print("=" * 60)
        print()
        self.processor._handle_help("")
        print()
        print("Type /HELP for available commands")
        print("-" * 60)
        
        # Setup signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._handle_ctrl_c)
        
        try:
            while self.running:
                try:
                    # Get user input
                    user_input = input("\n🔹 crypto-bot> ").strip()
                    
                    if user_input:
                        self.running = self.processor.process_command(user_input)
                
                except EOFError:
                    # End of input (pipe closed)
                    self.running = False
                
                except KeyboardInterrupt:
                    # Handled by signal handler
                    pass
        
        finally:
            self.processor.close()
            print("\n👋 CLI session ended")
    
    def _handle_ctrl_c(self, signum, frame):
        """Handle Ctrl+C with confirmation prompt."""
        import time
        now = time.time()
        
        # Double tap detection (within 2 seconds)
        if now - self.last_ctrl_c_time < 2:
            print("\n⚡ Force quit detected")
            self.running = False
            return
        
        self.last_ctrl_c_time = now
        self.ctrl_c_count += 1
        
        print("\n⚠️  Do you want to stop the bot? (yes/no): ", end="", flush=True)
        
        try:
            # Try to get input with timeout
            import select
            if select.select([sys.stdin], [], [], 5.0)[0]:
                response = sys.stdin.readline().strip().lower()
                if response in ['yes', 'y']:
                    print("🔄 Routing shutdown to Rust Global Kill Switch...")
                    self.processor._handle_kill("/KILL User confirmed via Ctrl+C prompt")
                elif response in ['no', 'n']:
                    print("✅ Continuing operation")
                else:
                    print(f"Unrecognized response: {response}")
            else:
                print("\n(No response received, continuing operation)")
        except Exception as e:
            print(f"\n(Input error: {e})")
        
        self.ctrl_c_count = 0


def main():
    """Entry point for Master CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Crypto Bot Master CLI')
    parser.add_argument('--command', '-c', type=str, help='Execute single command and exit')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress welcome message')
    args = parser.parse_args()
    
    cli = InteractiveCLIShell()
    
    if args.command:
        # Single command mode
        cli.processor.connect()
        cli.processor.process_command(args.command)
        cli.processor.close()
    else:
        # Interactive mode
        try:
            cli.run()
        except KeyboardInterrupt:
            print("\n\n👋 CLI interrupted")
            sys.exit(0)


if __name__ == '__main__':
    main()
