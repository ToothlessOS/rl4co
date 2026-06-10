#!/usr/bin/env python
"""Thin wrapper: `python scripts/sweep.py ...` -> `python -m nrp sweep ...`."""
from nrp.cli import main

if __name__ == "__main__":
    import sys

    sys.exit(main(["sweep", *sys.argv[1:]]))
