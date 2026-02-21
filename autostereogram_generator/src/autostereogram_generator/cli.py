"""
Command-line interface for the autostereogram generator.

This module provides the main CLI entry point for generating autostereograms
from input images using neural depth estimation.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .depth_estimator import generate_and_save_depth_map, list_available_models
from .stereogram import generate_autostereogram, create_simple_pattern, save_pattern
from .utils import get_device, print_device_info, get_image_info


def create_parser() -> argparse.ArgumentParser:
    """
    Create the command-line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="ag-cli",
        description="Generate autostereograms from images using neural depth estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  ag-cli --input photo.jpg --pattern tile.png \\
         --out-depth depth.png --out-stereo stereogram.png

  # With custom parameters  
  ag-cli --input landscape.jpg --pattern texture.png \\
         --out-depth depth_map.png --out-stereo result.png \\
         --eye-sep 80 --pixel-size 2

  # Create a simple pattern
  ag-cli --create-pattern dots.png --pattern-type random_dots

  # List available models
  ag-cli --list-models

  # Show device information
  ag-cli --device-info
        """
    )
    
    # Main operation arguments
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input image (JPEG/PNG)"
    )
    
    parser.add_argument(
        "--pattern", 
        type=str,
        help="Path to pattern tile image (PNG)"
    )
    
    parser.add_argument(
        "--out-depth",
        type=str,
        help="Path to save intermediate depth map (PNG)"
    )
    
    parser.add_argument(
        "--out-stereo",
        type=str, 
        help="Path to save final autostereogram (PNG)"
    )
    
    # Stereogram parameters
    parser.add_argument(
        "--eye-sep",
        type=int,
        default=60,
        help="Eye separation distance in pixels (default: 60)"
    )
    
    parser.add_argument(
        "--pixel-size",
        type=int,
        default=1,
        help="Pixel size multiplier (default: 1)"
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="DPT_Large",
        choices=list_available_models(),
        help="Depth estimation model to use (default: DPT_Large)"
    )
    
    # Utility commands
    parser.add_argument(
        "--create-pattern",
        type=str,
        help="Create a simple pattern and save to specified path"
    )
    
    parser.add_argument(
        "--pattern-type",
        type=str,
        default="random_dots",
        choices=["random_dots", "stripes", "checkerboard"],
        help="Type of pattern to create (default: random_dots)"
    )
    
    parser.add_argument(
        "--pattern-size",
        type=int,
        nargs=2,
        default=[64, 64],
        metavar=("WIDTH", "HEIGHT"),
        help="Size of pattern to create (default: 64 64)"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available depth estimation models"
    )
    
    parser.add_argument(
        "--device-info",
        action="store_true",
        help="Show information about available compute devices"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    return parser


def validate_args(args: argparse.Namespace) -> bool:
    """
    Validate command-line arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        bool: True if arguments are valid
    """
    # Check for utility commands first
    if args.list_models or args.device_info or args.create_pattern:
        return True
    
    # For main operation, require input, pattern, and output paths
    if not args.input:
        print("Error: --input is required for stereogram generation", file=sys.stderr)
        return False
    
    if not args.pattern:
        print("Error: --pattern is required for stereogram generation", file=sys.stderr)
        return False
    
    if not args.out_depth:
        print("Error: --out-depth is required for stereogram generation", file=sys.stderr)
        return False
    
    if not args.out_stereo:
        print("Error: --out-stereo is required for stereogram generation", file=sys.stderr)
        return False
    
    # Validate file paths
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return False
    
    if not args.create_pattern:  # Only check pattern if we're not creating one
        pattern_path = Path(args.pattern)
        if not pattern_path.exists():
            print(f"Error: Pattern file not found: {pattern_path}", file=sys.stderr)
            return False
    
    # Validate numeric parameters
    if args.eye_sep <= 0:
        print("Error: --eye-sep must be positive", file=sys.stderr)
        return False
    
    if args.pixel_size <= 0:
        print("Error: --pixel-size must be positive", file=sys.stderr)
        return False
    
    return True


def handle_utility_commands(args: argparse.Namespace) -> bool:
    """
    Handle utility commands that don't require main processing.
    
    Args:
        args: Parsed arguments
        
    Returns:
        bool: True if a utility command was handled
    """
    if args.device_info:
        print("=== Device Information ===")
        print_device_info()
        return True
    
    if args.list_models:
        from .depth_estimator import get_model_info
        
        print("=== Available Depth Estimation Models ===")
        for model_name in list_available_models():
            info = get_model_info(model_name)
            print(f"\n{model_name}:")
            print(f"  Description: {info['description']}")
            if 'accuracy' in info:
                print(f"  Accuracy: {info['accuracy']}")
                print(f"  Speed: {info['speed']}")
        return True
    
    if args.create_pattern:
        print(f"Creating {args.pattern_type} pattern...")
        width, height = args.pattern_size
        pattern = create_simple_pattern(width, height, args.pattern_type)
        save_pattern(pattern, args.create_pattern)
        print(f"✓ Pattern saved to: {args.create_pattern}")
        return True
    
    return False


def run_main_pipeline(args: argparse.Namespace) -> None:
    """
    Run the main autostereogram generation pipeline.
    
    Args:
        args: Parsed command-line arguments
    """
    print("=== Autostereogram Generator ===")
    
    if args.verbose:
        print(f"Input image: {args.input}")
        print(f"Pattern: {args.pattern}")
        print(f"Depth output: {args.out_depth}")
        print(f"Stereogram output: {args.out_stereo}")
        print(f"Model: {args.model}")
        print(f"Eye separation: {args.eye_sep}px")
        print(f"Pixel size: {args.pixel_size}")
        print()
        
        # Show device info
        print_device_info()
        print()
        
        # Show input image info
        try:
            info = get_image_info(args.input)
            print(f"Input image info: {info['width']}x{info['height']} {info['mode']}")
            print()
        except Exception as e:
            print(f"Warning: Could not read image info: {e}")
    
    try:
        # Step 1: Generate depth map
        print("Step 1: Generating depth map...")
        depth_map = generate_and_save_depth_map(
            input_image_path=args.input,
            output_depth_path=args.out_depth,
            model_name=args.model
        )
        
        # Step 2: Generate autostereogram
        print("\nStep 2: Generating autostereogram...")
        generate_autostereogram(
            depth_map=depth_map,
            pattern_path=args.pattern,
            output_path=args.out_stereo,
            eye_separation=args.eye_sep,
            pixel_size=args.pixel_size
        )
        
        print(f"\n✓ Complete! Autostereogram saved to: {args.out_stereo}")
        
    except KeyboardInterrupt:
        print("\n⚠ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main() -> None:
    """
    Main entry point for the CLI.
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle utility commands first
    if handle_utility_commands(args):
        return
    
    # Validate arguments for main operation
    if not validate_args(args):
        parser.print_help()
        sys.exit(1)
    
    # Run main pipeline
    run_main_pipeline(args)


if __name__ == "__main__":
    main()
