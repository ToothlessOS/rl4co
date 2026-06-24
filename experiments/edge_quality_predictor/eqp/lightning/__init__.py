"""Lightning module + data module for the edge quality predictor."""
from .module import EdgeQualityModule
from .data import EdgeQualityDataModule

__all__ = ["EdgeQualityModule", "EdgeQualityDataModule"]