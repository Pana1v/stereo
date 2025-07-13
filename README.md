# Stereo Depth Estimation

A comprehensive implementation of stereo depth estimation techniques comparing classical computer vision methods with modern deep learning approaches.

## Overview

This repository contains implementations of various depth estimation methods:
- **Classical Methods**: Gradient-based, Laplacian-based, Structure tensor, and Blur-based approaches
- **Deep Learning Methods**: MiDaS (Monocular Depth Estimation) integration
- **Evaluation Framework**: Comprehensive comparison and visualization tools

## Features

- 🔍 **Multiple Depth Estimation Algorithms**
  - Gradient-based edge detection
  - Laplacian edge detection
  - Structure tensor analysis
  - Blur-based depth estimation
  - MiDaS deep learning model

- 📊 **Comprehensive Evaluation**
  - Statistical analysis of depth distributions
  - Visual comparison of results
  - Performance metrics and timing

- 🎨 **Visualization Tools**
  - Side-by-side depth map comparisons
  - Statistical distribution plots
  - Color-coded depth visualizations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Depth Estimation
Run the basic depth estimation methods:

```bash
python run_depth_estimation.py
```

This will process the test images and generate `depth_estimation_results.png` with comparisons of gradient, Laplacian, and blur-based methods.

### Advanced Depth Estimation
Run the advanced depth estimation including MiDaS:

```bash
python advanced_depth_estimation.py
```

This will generate `advanced_depth_estimation_results.png` with comprehensive comparisons including the MiDaS deep learning model.

### Jupyter Notebook
For interactive experimentation:

```bash
jupyter notebook stereo_depth_estimation.ipynb
```

## Results

### Classical vs Deep Learning Comparison

The repository demonstrates the significant differences between classical computer vision approaches and modern deep learning methods:

- **Classical Methods** (Gradient, Laplacian, Structure Tensor): Focus on edge detection and local features
- **Deep Learning Methods** (MiDaS): Provide realistic depth perception with better understanding of object relationships

### Sample Results

![Basic Depth Estimation](results/depth_estimation_results.png)
*Basic depth estimation methods comparison*

![Advanced Depth Estimation](results/advanced_depth_estimation_results.png)
*Advanced depth estimation including MiDaS deep learning model*

## Test Images

The repository includes test images of mechanical/robotic devices:
- `images/download.jpeg` - Primary test image
- `images/images (1).jpeg` - Secondary test image

## Technical Details

### Classical Methods
- **Gradient-based**: Uses Sobel operators to detect edges as depth cues
- **Laplacian-based**: Applies Laplacian of Gaussian for edge detection
- **Structure Tensor**: Analyzes local image structure for depth estimation
- **Blur-based**: Uses image blur as a depth indicator

### Deep Learning Methods
- **MiDaS**: Intel's Monocular Depth Estimation model
  - Model: MiDaS_small (81.8MB)
  - Provides realistic depth maps with object-aware depth perception
  - Significantly outperforms classical methods in depth quality

## Performance Metrics

Typical results on test images:
- **Gradient Method**: Mean depth ~0.04
- **Laplacian Method**: Mean depth ~0.03
- **Structure Tensor**: Mean depth ~0.005
- **MiDaS**: Mean depth ~0.48 (normalized depth values)

## Dependencies

- OpenCV (cv2)
- NumPy
- Matplotlib
- PIL (Pillow)
- PyTorch
- timm (for MiDaS)

## Repository Structure

```
stereo-depth-estimation/
├── images/                     # Test images
├── results/                    # Generated depth estimation results
├── advanced_depth_estimation.py # Advanced methods including MiDaS
├── run_depth_estimation.py     # Basic depth estimation methods
├── stereo_depth_estimation.ipynb # Jupyter notebook for experimentation
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Contributing

Feel free to contribute by:
- Adding new depth estimation methods
- Improving visualization tools
- Adding more test images
- Enhancing evaluation metrics

## License

This project is open source and available under the MIT License. 