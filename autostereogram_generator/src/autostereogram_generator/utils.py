"""
Utility functions for the autostereogram generator.

This module provides helper functions for device detection, image loading/saving,
and other common operations.
"""

import os
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch operations.
    
    Prioritizes MPS (Apple Silicon) > CUDA > CPU.
    
    Returns:
        torch.device: The best available device
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        np.ndarray: Image as numpy array in RGB format
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image cannot be loaded
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Use PIL for better format support
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")
            return np.array(img)
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")


def save_image(image: np.ndarray, output_path: Union[str, Path]) -> None:
    """
    Save a numpy array as an image file.
    
    Args:
        image: Image as numpy array (RGB format)
        output_path: Path where to save the image
        
    Raises:
        ValueError: If the image array is invalid
        OSError: If the file cannot be saved
    """
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate image array
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    
    if image.ndim not in [2, 3]:
        raise ValueError("Image must be 2D (grayscale) or 3D (RGB/RGBA)")
    
    # Ensure image is in correct range
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    
    try:
        # Use PIL for saving
        img = Image.fromarray(image)
        img.save(output_path)
    except Exception as e:
        raise OSError(f"Failed to save image to {output_path}: {e}")


def validate_image_path(path: Union[str, Path], must_exist: bool = True) -> Path:
    """
    Validate and convert image path to Path object.
    
    Args:
        path: Input path as string or Path
        must_exist: Whether the file must already exist
        
    Returns:
        Path: Validated path object
        
    Raises:
        FileNotFoundError: If must_exist=True and file doesn't exist
        ValueError: If path is invalid
    """
    path = Path(path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Check if parent directory is writable (for output files)
    if not must_exist:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Cannot create directory {path.parent}: {e}")
    
    return path


def normalize_depth_map(depth_map: np.ndarray) -> np.ndarray:
    """
    Normalize a depth map to 0-255 uint8 range.
    
    Args:
        depth_map: Input depth map as numpy array
        
    Returns:
        np.ndarray: Normalized depth map as uint8
    """
    if depth_map.size == 0:
        raise ValueError("Empty depth map")
    
    # Handle potential NaN values
    if np.any(np.isnan(depth_map)):
        depth_map = np.nan_to_num(depth_map, nan=0.0)
    
    # Normalize to 0-1 range
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    
    if depth_max == depth_min:
        # Constant depth map
        normalized = np.zeros_like(depth_map)
    else:
        normalized = (depth_map - depth_min) / (depth_max - depth_min)
    
    # Convert to 0-255 uint8
    return (normalized * 255).astype(np.uint8)


def get_image_info(image_path: Union[str, Path]) -> dict:
    """
    Get basic information about an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: Image information including size, mode, format
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        with Image.open(image_path) as img:
            return {
                "size": img.size,
                "mode": img.mode,
                "format": img.format,
                "width": img.width,
                "height": img.height,
            }
    except Exception as e:
        raise ValueError(f"Failed to read image info from {image_path}: {e}")


def print_device_info() -> None:
    """Print information about available compute devices."""
    device = get_device()
    print(f"Using device: {device}")
    
    if device.type == "mps":
        print("✓ Apple Silicon MPS acceleration enabled")
    elif device.type == "cuda":
        print(f"✓ CUDA acceleration enabled (GPU: {torch.cuda.get_device_name()})")
    else:
        print("⚠ Using CPU (consider using a device with MPS or CUDA for faster processing)")
        
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()} (version: {torch.version.cuda})")
