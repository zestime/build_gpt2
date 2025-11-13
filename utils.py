import code
import sys
import readline # The module needed for completion and history
import rlcompleter # Provides the default Python-specific completer logic
from types import SimpleNamespace
import uuid
import time
from datetime import datetime
import random

BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BASE_SIZE = len(BASE62_ALPHABET)

def base62_encode(number: int, limit: int = 4) -> str:
    if number == 0:
        return "0"
    encoded = ""
    while number > 0 and len(encoded) < limit:
        number, remainder = divmod(number, BASE_SIZE)
        encoded = BASE62_ALPHABET[remainder] + encoded
    return encoded

def make_execution_id():
    now = datetime.now()
    date_component = now.strftime('%m%d%H%M')
    time_component = base62_encode(int(now.timestamp()), limit=2)
    random_component = ''.join(random.choices(BASE62_ALPHABET, k=3))
    return f"{date_component}-{time_component}{random_component}"

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


def create_timer(is_laps=False):
    start = time.time()
    def print_laps(msg,is_reset=False ):
        nonlocal start
        print(f"{msg}: {(time.time() - start):.4f}")
        if is_laps:
            start = time.time()

    return print_laps
