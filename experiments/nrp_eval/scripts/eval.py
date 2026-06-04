#!/usr/bin/env python
"""Thin wrapper: `python -m nrp eval ...` or `python scripts/eval.py ...`."""
from nrp.cli import main

if __name__ == "__main__":
    import sys

    sys.exit(main(["eval", *sys.argv[1:]]))
