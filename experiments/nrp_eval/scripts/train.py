#!/usr/bin/env python
"""Thin wrapper: `python scripts/train.py ...` -> `python -m nrp train ...`."""
from nrp.cli import main

if __name__ == "__main__":
    import sys

    sys.exit(main(["train", *sys.argv[1:]]))
