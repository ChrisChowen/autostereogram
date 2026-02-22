# Autostereogram Generator

A Python tool for generating autostereograms (Magic Eye images) from input images using neural depth estimation. Leverages MiDaS/DPT models for depth estimation with Apple Silicon acceleration via PyTorch MPS backend.

## Features

- Automatic depth map generation using pre-trained neural networks
- Apple Silicon M1/M2/M4 acceleration via PyTorch MPS backend
- Customizable pattern tiles and stereogram parameters
- Command-line interface for easy usage
- Clean, modular code architecture

## Installation

```bash
cd autostereogram_generator
pip install -e .
```

## Usage

### CLI

```bash
ag-cli --input photo.jpg --pattern dots.png \
       --out-depth depth.png --out-stereo stereogram.png
```

**Parameters:**
- `--input` - Path to input image (JPEG/PNG)
- `--pattern` - Path to pattern tile image (PNG)
- `--out-depth` - Path to save intermediate depth map
- `--out-stereo` - Path to save final autostereogram
- `--eye-sep` - Eye separation distance in pixels (default: 60)
- `--pixel-size` - Pixel size multiplier (default: 1)

### Python API

```python
from autostereogram_generator.depth_estimator import generate_depth_map
from autostereogram_generator.stereogram import generate_autostereogram

depth_map = generate_depth_map("input_image.jpg")
generate_autostereogram(
    depth_map=depth_map,
    pattern_path="pattern.png",
    output_path="stereogram.png",
    eye_separation=60,
    pixel_size=1
)
```

## How It Works

1. **Depth Estimation** - Input image processed by MiDaS/DPT model to generate a depth map
2. **Pattern Tiling** - A pattern image is tiled horizontally across the output canvas
3. **Depth-based Shifting** - Each pixel column is shifted based on depth values to create the stereoscopic effect
4. **Output** - Final autostereogram saved as PNG

## Tech Stack

- Python 3.9+
- PyTorch (MiDaS depth estimation)
- OpenCV
- NumPy / Pillow

## Project Structure

```
autostereogram_generator/
├── src/autostereogram_generator/
│   ├── cli.py              # Command-line interface
│   ├── depth_estimator.py  # Neural depth estimation
│   ├── stereogram.py       # Autostereogram generation
│   └── utils.py            # Utility functions
└── tests/
    └── test_integration.py
```
