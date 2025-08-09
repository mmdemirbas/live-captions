#!/usr/bin/env python3
import sys


# ——— Color + logging helpers ———
class Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


def _use_color():
    # Color only when stderr is a TTY (not when redirected). You can add a CLI flag to override.
    return sys.stderr.isatty()


def _c(text, color):
    if _use_color():
        return f"{color}{text}{Ansi.RESET}"
    return text


def log_info(msg):
    print(_c("[info ] " + msg, Ansi.CYAN), file=sys.stderr, flush=True)


def log_warn(msg):
    print(_c("[warn ] " + msg, Ansi.YELLOW), file=sys.stderr, flush=True)


def log_error(msg):
    print(_c("[error] " + msg, Ansi.RED), file=sys.stderr, flush=True)


def log_debug(msg):
    # Dim debug to keep it subtle
    print(_c("[debug] " + msg, Ansi.GRAY), file=sys.stderr, flush=True)
