# Autostereogram Generator

A Python tool for generating autostereograms (Magic Eye images) from input images using neural depth estimation. This project leverages MiDaS/DPT models for depth estimation and supports Apple Silicon M-series processors via PyTorch MPS backend.

## Features

- Automatic depth map generation using pre-trained neural networks
- Apple Silicon M1/M2/M4 acceleration via PyTorch MPS backend
- Customizable pattern tiles and stereogram parameters
- Command-line interface for easy usage
- Clean, modular code architecture

## Installation

### Prerequisites

- Python ≥3.9
- macOS with Apple Silicon (M1/M2/M4) for MPS acceleration (optional)

### Install from source

```bash
# Clone or download the project
cd autostereogram_generator

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### MPS Backend Note

This project automatically detects and uses Apple's Metal Performance Shaders (MPS) backend when available on Apple Silicon Macs. This provides significant acceleration for the neural depth estimation. If MPS is not available, the code will fallback to CPU processing.

## Usage

### Command Line Interface

The main entry point is the `ag-cli` command:

```bash
ag-cli --input scene.jpg --pattern tile.png \
       --out-depth scene_depth.png --out-stereo scene_stereo.png
```

#### Parameters

- `--input`: Path to input image (JPEG/PNG)
- `--pattern`: Path to pattern tile image (PNG)
- `--out-depth`: Path to save intermediate depth map (PNG)
- `--out-stereo`: Path to save final autostereogram (PNG)
- `--eye-sep`: Eye separation distance in pixels (default: 60)
- `--pixel-size`: Pixel size multiplier (default: 1)

#### Examples

Basic usage:
```bash
ag-cli --input photo.jpg --pattern dots.png \
       --out-depth depth.png --out-stereo stereogram.png
```

With custom parameters:
```bash
ag-cli --input landscape.jpg --pattern texture.png \
       --out-depth depth_map.png --out-stereo result.png \
       --eye-sep 80 --pixel-size 2
```

### Python API

You can also use the modules directly in your Python code:

```python
from autostereogram_generator.depth_estimator import generate_depth_map
from autostereogram_generator.stereogram import generate_autostereogram

# Generate depth map
depth_map = generate_depth_map("input_image.jpg")

# Create autostereogram
generate_autostereogram(
    depth_map=depth_map,
    pattern_path="pattern.png",
    output_path="stereogram.png",
    eye_separation=60,
    pixel_size=1
)
```

## How It Works

1. **Depth Estimation**: The input image is processed by a pre-trained MiDaS or DPT model to generate a depth map
2. **Pattern Tiling**: A small pattern image is tiled horizontally across the output canvas
3. **Depth-based Shifting**: Each pixel column is shifted based on the corresponding depth value to create the stereoscopic effect
4. **Output Generation**: The final autostereogram is saved as a PNG image

## Project Structure

```
autostereogram_generator/
├── pyproject.toml              # Project configuration and dependencies
├── README.md                   # This file
├── src/
│   └── autostereogram_generator/
│       ├── __init__.py         # Package initialization
│       ├── cli.py              # Command-line interface
│       ├── depth_estimator.py  # Neural depth estimation
│       ├── stereogram.py       # Autostereogram generation
│       └── utils.py            # Utility functions
└── tests/
    └── test_integration.py     # Integration tests
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

## Requirements

- torch>=2.0.0
- torchvision>=0.15.0
- opencv-python>=4.8.0
- numpy>=1.24.0
- pillow>=10.0.0

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
