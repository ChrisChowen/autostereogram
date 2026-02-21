"""
Depth estimation module using pre-trained neural networks.

This module provides functionality to generate depth maps from input images
using MiDaS or DPT models via torch.hub, with MPS acceleration support.
"""

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from .utils import get_device, load_image, normalize_depth_map, validate_image_path


class DepthEstimator:
    """
    Neural depth estimator using pre-trained models.
    
    This class loads and manages a pre-trained depth estimation model,
    providing methods to generate depth maps from input images.
    """
    
    def __init__(self, model_name: str = "DPT_Large", device: torch.device = None):
        """
        Initialize the depth estimator.
        
        Args:
            model_name: Name of the model to use ("DPT_Large", "DPT_Hybrid", "MiDaS")
            device: Device to run the model on (auto-detected if None)
        """
        self.device = device or get_device()
        self.model_name = model_name
        self.model = None
        self.transform = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the pre-trained depth estimation model."""
        print(f"Loading {self.model_name} model...")
        
        try:
            # Load model from torch hub
            if self.model_name.startswith("DPT"):
                self.model = torch.hub.load("intel-isl/MiDaS", self.model_name)
            else:
                # Default to MiDaS
                self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
                
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            # Load the appropriate transform
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            
            if self.model_name == "DPT_Large":
                self.transform = midas_transforms.dpt_transform
            elif self.model_name == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.default_transform
                
            print(f"✓ Model loaded successfully on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_name} model: {e}")
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from an RGB image.
        
        Args:
            image: Input image as numpy array (RGB format)
            
        Returns:
            np.ndarray: Depth map as numpy array
        """
        if self.model is None or self.transform is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Ensure image is numpy array in correct format
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array")
        
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Apply transforms - MiDaS transforms expect numpy arrays, not PIL Images
        input_tensor = self.transform(image).to(self.device)
        
        # Run inference
        with torch.no_grad():
            depth_map = self.model(input_tensor)
            
            # Convert to numpy array
            depth_map = depth_map.squeeze().cpu().numpy()
            
        return depth_map


# Global estimator instance (lazy loaded)
_estimator = None


def get_estimator(model_name: str = "DPT_Large", force_reload: bool = False) -> DepthEstimator:
    """
    Get a global depth estimator instance.
    
    Args:
        model_name: Name of the model to use
        force_reload: Whether to force reload the model
        
    Returns:
        DepthEstimator: The estimator instance
    """
    global _estimator
    
    if _estimator is None or force_reload or _estimator.model_name != model_name:
        _estimator = DepthEstimator(model_name=model_name)
    
    return _estimator


def generate_depth_map(
    input_image_path: Union[str, Path], 
    model_name: str = "DPT_Large"
) -> np.ndarray:
    """
    Generate a normalized depth map from an input image.
    
    This is the main function for depth map generation. It loads the image,
    runs depth estimation, and returns a normalized depth map.
    
    Args:
        input_image_path: Path to the input image file
        model_name: Name of the depth estimation model to use
        
    Returns:
        np.ndarray: Normalized depth map as uint8 (0-255 range)
        
    Raises:
        FileNotFoundError: If the input image file doesn't exist
        ValueError: If the image cannot be processed
        RuntimeError: If the model fails to load or run
    """
    # Validate input path
    image_path = validate_image_path(input_image_path, must_exist=True)
    
    print(f"Processing image: {image_path}")
    
    # Load image
    try:
        image = load_image(image_path)
        print(f"✓ Image loaded: {image.shape}")
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")
    
    # Get estimator
    try:
        estimator = get_estimator(model_name=model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize depth estimator: {e}")
    
    # Generate depth map
    try:
        print("Estimating depth...")
        depth_map = estimator.estimate_depth(image)
        print(f"✓ Depth estimation complete: {depth_map.shape}")
    except Exception as e:
        raise RuntimeError(f"Depth estimation failed: {e}")
    
    # Normalize depth map
    try:
        normalized_depth = normalize_depth_map(depth_map)
        print(f"✓ Depth map normalized to range [0, 255]")
        return normalized_depth
    except Exception as e:
        raise ValueError(f"Failed to normalize depth map: {e}")


def generate_and_save_depth_map(
    input_image_path: Union[str, Path],
    output_depth_path: Union[str, Path],
    model_name: str = "DPT_Large"
) -> np.ndarray:
    """
    Generate and save a depth map from an input image.
    
    Args:
        input_image_path: Path to the input image file
        output_depth_path: Path where to save the depth map
        model_name: Name of the depth estimation model to use
        
    Returns:
        np.ndarray: The generated depth map
    """
    from .utils import save_image
    
    # Generate depth map
    depth_map = generate_depth_map(input_image_path, model_name)
    
    # Save depth map
    save_image(depth_map, output_depth_path)
    print(f"✓ Depth map saved to: {output_depth_path}")
    
    return depth_map


def list_available_models() -> list:
    """
    List available depth estimation models.
    
    Returns:
        list: List of available model names
    """
    return [
        "DPT_Large",
        "DPT_Hybrid", 
        "MiDaS",
        "MiDaS_small"
    ]


def get_model_info(model_name: str) -> dict:
    """
    Get information about a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        dict: Model information
    """
    model_info = {
        "DPT_Large": {
            "description": "Dense Prediction Transformer - Large variant",
            "input_size": (384, 384),
            "accuracy": "High",
            "speed": "Slow"
        },
        "DPT_Hybrid": {
            "description": "Dense Prediction Transformer - Hybrid variant", 
            "input_size": (384, 384),
            "accuracy": "High",
            "speed": "Medium"
        },
        "MiDaS": {
            "description": "MiDaS depth estimation model",
            "input_size": (384, 384),
            "accuracy": "Medium",
            "speed": "Fast"
        },
        "MiDaS_small": {
            "description": "MiDaS small model for faster inference",
            "input_size": (256, 256),
            "accuracy": "Medium",
            "speed": "Very Fast"
        }
    }
    
    return model_info.get(model_name, {"description": "Unknown model"})
