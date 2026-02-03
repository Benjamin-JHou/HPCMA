"""MMRP Clinical AI Package"""

__version__ = "1.0.0"
__author__ = "Benjamin-JHou"
__license__ = "MIT"

from .inference.api_server import app, predict, predict_batch

__all__ = [
    "app",
    "predict", 
    "predict_batch",
    "__version__"
]
