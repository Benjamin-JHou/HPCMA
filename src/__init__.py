"""Hypertension Pan-Comorbidity Multi-Modal Atlas (HPCMA)"""

__version__ = "1.0.0"
__author__ = "Benjamin-JHou"
__license__ = "MIT"
__description__ = "Integrated genomic-clinical resource for hypertension-mediated end-organ risk prediction"

from .inference.api_server import app, predict, predict_batch

__all__ = [
    "app",
    "predict", 
    "predict_batch",
    "__version__",
    "__description__"
]
