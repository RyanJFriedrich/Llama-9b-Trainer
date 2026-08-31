# File: log.py
import datetime
import os
import sys
from typing import Any


def log(
    *args: Any,
    sep: str = ' ',
    end: str = '\n',
    flush: bool = False,
    level: str = 'INFO',
    filename: str = 'common.log',
    print_console: bool = False,
    add_header: bool = True
) -> None:
    """
    Drop-in replacement for print() that writes to a log file and optionally console.
    
    Usage (from the repo root, where `train` imports as a package):
        from train.utils.log import log
        log("Just like print", variable, "but logs to file")
    
    Args:
        *args: Values to log (works exactly like print).
        sep: String inserted between values (default: ' ').
        end: String appended after the last value (default: '\\n').
        flush: Whether to forcibly flush the file (default: False).
        level: Logging level (default: 'INFO').
        filename: Log file name (default: 'common.log').
        print_console: Also print to console (default: False).
        add_header: Add a header if the log file is new (default: True).
    """
    # Add timestamp and log level
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    prefix = f"[{timestamp}] [{level}] "
    
    # Convert all arguments to strings and join them with sep
    output = sep.join(str(arg) for arg in args)
    message = prefix + output
    
    # Check if file exists to add header if needed
    file_exists = os.path.exists(filename)
    
    # Append to the log file. Windows AV/indexer can briefly hold the file
    # between rapid open/close cycles, so retry a few times before giving up.
    import time
    last_err = None
    for _ in range(10):
        try:
            with open(filename, 'a', encoding='utf-8') as log_file:
                # Add header if it's a new file
                if add_header and not file_exists:
                    log_file.write(f"=== Log started at {timestamp} ===\n")

                # Write the actual log message
                log_file.write(message + end)
                if flush:
                    log_file.flush()
            last_err = None
            break
        except PermissionError as err:
            last_err = err
            time.sleep(0.1)
    if last_err is not None:
        raise last_err
    
    # Also print to console if requested. Use stdout.buffer with utf-8 so
    # Windows consoles do not crash on non-ASCII generated text.
    if print_console:
        try:
            encoded = (message + end).encode("utf-8")
            sys.stdout.buffer.write(encoded)
            if flush:
                sys.stdout.buffer.flush()
        except Exception:
            print(message, end=end)
