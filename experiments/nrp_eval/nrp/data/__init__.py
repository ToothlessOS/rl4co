"""Data module: dataset specs, generation re-exports, benchmark loaders."""
from nrp.data.dataset import DatasetSpec, build_eval_dataset
from nrp.data.generation import generate_dataset, generate_env_data

__all__ = [
    "DatasetSpec",
    "build_eval_dataset",
    "generate_env_data",
    "generate_dataset",
]
