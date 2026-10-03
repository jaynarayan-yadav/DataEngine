"""
Entry point for python -m data_engine.
"""

from __future__ import annotations

import sys
from data_engine.cli import main

if __name__ == "__main__":
    sys.exit(main())
