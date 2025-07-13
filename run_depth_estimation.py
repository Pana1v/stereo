#!/usr/bin/env python3
"""
Run depth estimation on test images
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def run_depth_estimation():
    """Run depth estimation using various methods"""
    
    # Check if images directory exists
    if not os.path.exists('images'):
        print("Images directory not found. Creating test images...")
        os.makedirs('images', exist_ok=True)
        
    # List available images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend([f for f in os.listdir('.') if f.lower().endswith(ext.replace('*', ''))])
        if os.path.exists('images'):
            image_files.extend([f'images/{f}' for f in os.listdir('images') if f.lower().endswith(ext.replace('*', ''))])
    
    print(f"Found {len(image_files)} images: {image_files}")
    
    if not image_files:
        print("No images found. Please add some test images.")
        return
    
    # Use the first available image
    test_image = image_files[0]
    print(f"Using test image: {test_image}")
    
    # Load image
    img = cv2.imread(test_image)
    if img is None:
        print(f"Could not load image: {test_image}")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Simple gradient-based depth estimation
    print("Running gradient-based depth estimation...")
    
    # Calculate gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine gradients to estimate depth
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize and invert (assuming closer objects have higher gradients)
    depth_map = 255 - cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Method 2: Laplacian-based depth estimation
    print("Running Laplacian-based depth estimation...")
    
    # Apply Laplacian filter
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = np.absolute(laplacian)
    
    # Normalize and invert
    depth_map_laplacian = 255 - cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Method 3: Gaussian blur difference (simulating depth of field)
    print("Running blur-based depth estimation...")
    
    # Create different blur levels
    blur1 = cv2.GaussianBlur(gray, (5, 5), 0)
    blur2 = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Difference indicates focus/depth
    blur_diff = cv2.absdiff(blur1, blur2)
    depth_map_blur = cv2.normalize(blur_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Gradient-based depth
    axes[0, 1].imshow(depth_map, cmap='plasma')
    axes[0, 1].set_title('Gradient-based Depth')
    axes[0, 1].axis('off')
    
    # Laplacian-based depth
    axes[1, 0].imshow(depth_map_laplacian, cmap='plasma')
    axes[1, 0].set_title('Laplacian-based Depth')
    axes[1, 0].axis('off')
    
    # Blur-based depth
    axes[1, 1].imshow(depth_map_blur, cmap='plasma')
    axes[1, 1].set_title('Blur-based Depth')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('depth_estimation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Depth estimation completed!")
    print("Results saved as 'depth_estimation_results.png'")
    
    # Print some statistics
    print(f"\nDepth map statistics:")
    print(f"Gradient-based - Min: {depth_map.min()}, Max: {depth_map.max()}, Mean: {depth_map.mean():.2f}")
    print(f"Laplacian-based - Min: {depth_map_laplacian.min()}, Max: {depth_map_laplacian.max()}, Mean: {depth_map_laplacian.mean():.2f}")
    print(f"Blur-based - Min: {depth_map_blur.min()}, Max: {depth_map_blur.max()}, Mean: {depth_map_blur.mean():.2f}")

if __name__ == "__main__":
    run_depth_estimation() 