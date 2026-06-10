#!/usr/bin/env bash
# NRP Eval Pipeline — Classical solver installation guide.
#
# Stage 1 does NOT auto-install these. The user provides binaries and points
# the pipeline at them via:
#   - environment variables: NRP_LKH_BINARY, NRP_CONCORDE_BINARY, NRP_GUROBI_BINARY
#   - or per-solver config: solver.binary_path: /path/to/binary
#
# --- OR-Tools (recommended; pre-built wheels available) ---
# pip install ortools
# This is handled by the Python env, not this script.
#
# --- LKH-3 ---
# http://akira.ruc.dk/~keld/research/LKH-3/
# tar xvfz LKH-3.tgz
# cd LKH-3
# make
# cp LKH /usr/local/bin/lkh
# export NRP_LKH_BINARY=/usr/local/bin/lkh
#
# --- Concorde (requires a C compiler; free for academic use) ---
# http://www.math.uwaterloo.ca/tsp/concorde.html
# ./configure
# make
# cp concorde /usr/local/bin/concorde
# cp LINKERN/linkern /usr/local/bin/linkern
# export NRP_CONCORDE_BINARY=/usr/local/bin/concorde
#
# --- Gurobi (commercial; free academic license) ---
# https://www.gurobi.com/downloads/
# Follow the installer, then `pip install gurobipy`
# export NRP_GUROBI_BINARY=/opt/gurobi*/bin/gurobi_cl
# export GRB_LICENSE_FILE=/path/to/gurobi.lic
echo "This is a documentation script. See comments above for installation instructions."
