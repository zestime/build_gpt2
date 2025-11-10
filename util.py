import code
import sys
import readline # The module needed for completion and history
import rlcompleter # Provides the default Python-specific completer logic
from types import SimpleNamespace

def start_interactive_shell(local_vars=None):
    # 1. Check if readline is available (primarily for Windows compatibility)
    if 'libedit' in readline.__doc__:
        # On macOS, readline is often libedit, which sometimes needs a different setup
        # This is a common workaround for macOS systems
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        # Standard setup for Linux/most other systems
        readline.parse_and_bind("tab: complete")

    # 2. Set the completer function to Python's default
    # rlcompleter.Completer provides completion for module names, variables, etc.
    readline.set_completer(rlcompleter.Completer(local_vars).complete)

    # 3. Define a nice banner message
    banner = (
        "--- Entering Interactive Debug Shell --- \n"
        "Local variables 'data' and 'result' are available.\n"
        "Press Ctrl+D or Ctrl+Z (Windows) to exit."
    )
    
    local_vars = {
        'data': SimpleNamespace(**local_vars),
        'sys': sys, # Useful to include common modules
    }
    # 4. Start the interactive session
    code.interact(banner=banner, local=local_vars)
