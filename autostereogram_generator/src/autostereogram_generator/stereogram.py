"""
Autostereogram generation module.

This module provides functionality to generate autostereograms (Magic Eye images)
from depth maps and pattern tiles.
"""

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from .utils import load_image, save_image, validate_image_path


def generate_autostereogram(
    depth_map: np.ndarray,
    pattern_path: Union[str, Path],
    output_path: Union[str, Path],
    eye_separation: int = 60,
    pixel_size: int = 1
) -> None:
    """
    Generate an autostereogram from a depth map and pattern.
    
    Args:
        depth_map: Depth map as numpy array (grayscale, 0-255)
        pattern_path: Path to the pattern tile image
        output_path: Path where to save the autostereogram
        eye_separation: Distance between eyes in pixels (default: 60)
        pixel_size: Pixel size multiplier (default: 1)
        
    Raises:
        FileNotFoundError: If pattern file doesn't exist
        ValueError: If inputs are invalid
        OSError: If output cannot be saved
    """
    # Validate inputs
    pattern_path = validate_image_path(pattern_path, must_exist=True)
    output_path = validate_image_path(output_path, must_exist=False)
    
    if not isinstance(depth_map, np.ndarray):
        raise ValueError("depth_map must be a numpy array")
    
    if depth_map.ndim != 2:
        raise ValueError("depth_map must be a 2D grayscale array")
    
    if eye_separation <= 0:
        raise ValueError("eye_separation must be positive")
    
    if pixel_size <= 0:
        raise ValueError("pixel_size must be positive")
    
    print(f"Generating autostereogram...")
    print(f"Depth map shape: {depth_map.shape}")
    print(f"Eye separation: {eye_separation}px")
    print(f"Pixel size: {pixel_size}")
    
    # Load and prepare pattern
    pattern = load_image(pattern_path)
    print(f"✓ Pattern loaded: {pattern.shape}")
    
    # Generate the stereogram
    stereogram = _create_stereogram(
        depth_map=depth_map,
        pattern=pattern,
        eye_separation=eye_separation,
        pixel_size=pixel_size
    )
    
    # Save the result
    save_image(stereogram, output_path)
    print(f"✓ Autostereogram saved to: {output_path}")


def _create_stereogram(
    depth_map: np.ndarray,
    pattern: np.ndarray,
    eye_separation: int,
    pixel_size: int
) -> np.ndarray:
    """
    Create the actual stereogram from depth map and pattern.
    
    Args:
        depth_map: Normalized depth map (0-255)
        pattern: Pattern image as RGB numpy array
        eye_separation: Eye separation in pixels
        pixel_size: Pixel size multiplier
        
    Returns:
        np.ndarray: Generated stereogram as RGB array
    """
    height, width = depth_map.shape
    pattern_height, pattern_width = pattern.shape[:2]
    
    # Scale dimensions by pixel size
    output_width = width * pixel_size
    output_height = height * pixel_size
    
    # Create output array
    if pattern.ndim == 3:
        stereogram = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    else:
        stereogram = np.zeros((output_height, output_width), dtype=np.uint8)
    
    # Tile the pattern horizontally to cover the output width
    tiled_pattern = _tile_pattern_horizontally(pattern, output_width)
    
    # Apply depth-based shifting for each row
    for y in range(height):
        output_y_start = y * pixel_size
        output_y_end = output_y_start + pixel_size
        
        # Get the row from tiled pattern
        pattern_y = y % pattern_height
        base_row = tiled_pattern[pattern_y]
        
        # Create shifted row based on depth
        shifted_row = _apply_depth_shift(
            base_row, 
            depth_map[y], 
            eye_separation, 
            pixel_size
        )
        
        # Fill all pixel_size rows with the same shifted pattern
        for dy in range(pixel_size):
            if output_y_start + dy < output_height:
                stereogram[output_y_start + dy] = shifted_row
    
    return stereogram


def _tile_pattern_horizontally(pattern: np.ndarray, target_width: int) -> np.ndarray:
    """
    Tile a pattern horizontally to reach the target width.
    
    Args:
        pattern: Input pattern as numpy array
        target_width: Target width for tiling
        
    Returns:
        np.ndarray: Horizontally tiled pattern
    """
    pattern_height, pattern_width = pattern.shape[:2]
    
    # Calculate how many full tiles we need
    num_tiles = (target_width + pattern_width - 1) // pattern_width
    
    # Tile the pattern
    if pattern.ndim == 3:
        tiled = np.tile(pattern, (1, num_tiles, 1))
    else:
        tiled = np.tile(pattern, (1, num_tiles))
    
    # Crop to exact target width
    tiled = tiled[:, :target_width]
    
    return tiled


def _apply_depth_shift(
    base_row: np.ndarray, 
    depth_row: np.ndarray, 
    eye_separation: int,
    pixel_size: int
) -> np.ndarray:
    """
    Apply depth-based shifting to a pattern row.
    
    Args:
        base_row: Base pattern row
        depth_row: Depth values for this row (0-255)
        eye_separation: Eye separation in pixels
        pixel_size: Pixel size multiplier
        
    Returns:
        np.ndarray: Shifted pattern row
    """
    width = len(depth_row) * pixel_size
    
    if base_row.ndim == 2:
        shifted_row = np.zeros((width, base_row.shape[1]), dtype=np.uint8)
    else:
        shifted_row = np.zeros(width, dtype=np.uint8)
    
    # Process each original pixel
    for x in range(len(depth_row)):
        depth_value = depth_row[x]
        
        # Calculate shift based on depth (normalize depth to fraction of eye separation)
        # Closer objects (higher depth values) get more shift
        shift = int((depth_value / 255.0) * eye_separation * 0.5)
        
        # Apply shift to all pixel_size columns for this original pixel
        for dx in range(pixel_size):
            src_x = x * pixel_size + dx
            dst_x = (src_x + shift) % width
            
            if base_row.ndim == 2:
                shifted_row[dst_x] = base_row[src_x % base_row.shape[0]]
            else:
                shifted_row[dst_x] = base_row[src_x % len(base_row)]
    
    return shifted_row


def create_simple_pattern(
    width: int = 64, 
    height: int = 64, 
    pattern_type: str = "random_dots"
) -> np.ndarray:
    """
    Create a simple pattern for stereogram generation.
    
    Args:
        width: Pattern width in pixels
        height: Pattern height in pixels
        pattern_type: Type of pattern ("random_dots", "stripes", "checkerboard")
        
    Returns:
        np.ndarray: Generated pattern as RGB array
    """
    if pattern_type == "random_dots":
        # Random colored dots
        pattern = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
    elif pattern_type == "stripes":
        # Vertical stripes
        pattern = np.zeros((height, width, 3), dtype=np.uint8)
        stripe_width = max(1, width // 8)
        for x in range(width):
            if (x // stripe_width) % 2 == 0:
                pattern[:, x] = [255, 255, 255]  # White
            else:
                pattern[:, x] = [0, 0, 0]        # Black
                
    elif pattern_type == "checkerboard":
        # Checkerboard pattern
        pattern = np.zeros((height, width, 3), dtype=np.uint8)
        check_size = max(1, min(width, height) // 8)
        for y in range(height):
            for x in range(width):
                if ((x // check_size) + (y // check_size)) % 2 == 0:
                    pattern[y, x] = [255, 255, 255]  # White
                else:
                    pattern[y, x] = [128, 128, 128]  # Gray
                    
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    return pattern


def save_pattern(
    pattern: np.ndarray, 
    output_path: Union[str, Path]
) -> None:
    """
    Save a pattern to file.
    
    Args:
        pattern: Pattern as numpy array
        output_path: Where to save the pattern
    """
    save_image(pattern, output_path)


def validate_stereogram_inputs(
    depth_map: np.ndarray,
    pattern_path: Union[str, Path],
    eye_separation: int,
    pixel_size: int
) -> tuple:
    """
    Validate inputs for stereogram generation.
    
    Args:
        depth_map: Input depth map
        pattern_path: Path to pattern file
        eye_separation: Eye separation value
        pixel_size: Pixel size value
        
    Returns:
        tuple: (validated_depth_map, validated_pattern_path)
        
    Raises:
        ValueError: If any input is invalid
    """
    # Validate depth map
    if not isinstance(depth_map, np.ndarray):
        raise ValueError("depth_map must be a numpy array")
    
    if depth_map.ndim != 2:
        raise ValueError("depth_map must be 2D")
    
    if depth_map.size == 0:
        raise ValueError("depth_map cannot be empty")
    
    # Validate pattern path
    pattern_path = validate_image_path(pattern_path, must_exist=True)
    
    # Validate numeric parameters
    if not isinstance(eye_separation, int) or eye_separation <= 0:
        raise ValueError("eye_separation must be a positive integer")
    
    if not isinstance(pixel_size, int) or pixel_size <= 0:
        raise ValueError("pixel_size must be a positive integer")
    
    if eye_separation > min(depth_map.shape) // 2:
        raise ValueError("eye_separation is too large for the given depth map size")
    
    return depth_map, pattern_path


def get_stereogram_info(
    depth_map: np.ndarray,
    eye_separation: int = 60,
    pixel_size: int = 1
) -> dict:
    """
    Get information about the stereogram that would be generated.
    
    Args:
        depth_map: Input depth map
        eye_separation: Eye separation in pixels
        pixel_size: Pixel size multiplier
        
    Returns:
        dict: Information about the output stereogram
    """
    height, width = depth_map.shape
    output_width = width * pixel_size
    output_height = height * pixel_size
    
    return {
        "input_size": (width, height),
        "output_size": (output_width, output_height),
        "eye_separation": eye_separation,
        "pixel_size": pixel_size,
        "max_shift": int(eye_separation * 0.5),
        "estimated_memory_mb": (output_width * output_height * 3) / (1024 * 1024)
    }
