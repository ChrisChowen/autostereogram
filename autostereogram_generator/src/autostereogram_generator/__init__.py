"""
Autostereogram Generator

A Python package for generating autostereograms from input images using 
neural depth estimation with support for Apple Silicon MPS acceleration.
"""

__version__ = "0.1.0"
__author__ = "Autostereogram Generator"

from .depth_estimator import generate_depth_map
from .stereogram import generate_autostereogram
from .utils import get_device, load_image, save_image

__all__ = [
    "generate_depth_map",
    "generate_autostereogram", 
    "get_device",
    "load_image",
    "save_image",
]
