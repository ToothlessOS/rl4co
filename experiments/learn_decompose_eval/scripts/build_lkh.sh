#!/usr/bin/env bash
# Rebuild the patched LKH-3 binary.
set -euo pipefail

cd "$(dirname "$0")/../LKH-3.0.14"
make clean
make
echo "Built: $(pwd)/LKH"
