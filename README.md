# Wasserstein Barycenter Texture Mixing

A Python implementation of texture synthesis and mixing using Wasserstein barycenters and steerable pyramid decomposition. This library supports both first-order and higher-order statistical texture mixing with GPU acceleration.

## Overview

This project implements texture synthesis algorithms based on optimal transport theory and the sliced Wasserstein distance. The approach decomposes textures into multi-scale oriented representations using steerable pyramids, then computes Wasserstein barycenters to blend textures while preserving their statistical properties.

## Features

- First-order statistical texture mixing using sliced Wasserstein barycenters
- Higher-order statistical mixing with spatial block structures
- Multiple color space support (RGB, LAB, YUV)
- GPU acceleration via PyTorch
- Steerable pyramid decomposition for multi-scale texture analysis
- Configurable mixing weights for interpolation between multiple textures

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- pyrtools

## Usage

### First-Order Texture Mixing

```python
from src.first_order_statistical_mixing import first_order_texture_mixing
from src.utils import ColorSpace
import cv2

# Load textures
texture1 = cv2.imread('texture1.jpg')
texture2 = cv2.imread('texture2.jpg')

# Mix textures with equal weights
textures = [texture1, texture2]
weights = np.array([0.5, 0.5])

result = first_order_texture_mixing(
    textures=textures,
    rhos=weights,
    height=4,
    order=3,
    n_iter=100,
    K=128,
    device='cuda',
    color_space=ColorSpace.RGB
)
```

### Higher-Order Texture Mixing

```python
from src.higher_order_statistical_mixing import higher_order_texture_mixing

result = higher_order_texture_mixing(
    textures=textures,
    rhos=weights,
    height=4,
    order=3,
    block_size=4,
    n_iter=100,
    K=128,
    device='cuda',
    color_space=ColorSpace.LAB
)
```

## Project Structure

```
src/
├── first_order_statistical_mixing.py  # First-order texture mixing
├── higher_order_statistical_mixing.py # Higher-order texture mixing
├── sliced_wasserstein.py              # Sliced Wasserstein distance computations
└── utils.py                            # Utility functions and color space handling

notebook/
└── texture_mixing.ipynb                # Example usage and experiments

output/                                  # Experimental results
```

## Algorithm Details

The implementation uses steerable pyramid decomposition to analyze textures at multiple scales and orientations. Texture mixing is performed by computing Wasserstein barycenters in the coefficient space, either at the global level (first-order) or within local spatial blocks (higher-order) to preserve spatial correlations.

The sliced Wasserstein distance is computed using random projections for computational efficiency, enabling GPU-accelerated optimization via gradient descent.

## License

This project is part of academic work at MVA (Master Vision Apprentissage).

## References

This implementation is based on optimal transport methods for texture synthesis and the theory of Wasserstein barycenters applied to image processing.