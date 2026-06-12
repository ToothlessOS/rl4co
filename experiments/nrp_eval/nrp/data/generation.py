"""Thin re-exports of RL4CO's data generation utilities.

Wave 2 (sub-agent C) wires the harness's caching layer on top of these.
"""
from rl4co.data.generate_data import generate_dataset, generate_env_data

__all__ = ["generate_env_data", "generate_dataset"]
