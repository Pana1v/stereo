#!/usr/bin/env python3
"""
Advanced depth estimation using MiDaS and other deep learning models
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import timm
import torch.nn.functional as F
from torchvision import transforms

class MiDaSDepthEstimator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load MiDaS model
        try:
            self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
            self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            self.transform = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform
            print("MiDaS model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading MiDaS: {e}")
            self.model = None
            self.transform = None

    def estimate_depth(self, image_path):
        """Estimate depth using MiDaS"""
        if self.model is None:
            return None
            
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return None
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Transform image
        input_tensor = self.transform(img_rgb).to(self.device)
        
        # Predict depth
        with torch.no_grad():
            depth_map = self.model(input_tensor)
            
            # Convert to numpy
            depth_map = depth_map.squeeze().cpu().numpy()
            
            # Normalize depth map
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            
        return depth_map, img_rgb

def run_advanced_depth_estimation():
    """Run advanced depth estimation with multiple methods"""
    
    # Find test images
    image_files = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_files.extend([f for f in os.listdir('.') if f.lower().endswith(ext)])
        if os.path.exists('images'):
            image_files.extend([f'images/{f}' for f in os.listdir('images') if f.lower().endswith(ext)])
    
    print(f"Found {len(image_files)} images: {image_files}")
    
    if not image_files:
        print("No images found. Please add some test images.")
        return
    
    # Use the first available image
    test_image = image_files[0]
    print(f"Using test image: {test_image}")
    
    # Initialize MiDaS estimator
    midas_estimator = MiDaSDepthEstimator()
    
    # Classical methods
    img = cv2.imread(test_image)
    if img is None:
        print(f"Could not load image: {test_image}")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Method 1: Gradient-based depth estimation
    print("Running gradient-based depth estimation...")
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    depth_gradient = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
    
    # Method 2: Laplacian-based depth estimation
    print("Running Laplacian-based depth estimation...")
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    depth_laplacian = cv2.normalize(np.abs(laplacian), None, 0, 1, cv2.NORM_MINMAX)
    
    # Method 3: MiDaS depth estimation
    print("Running MiDaS depth estimation...")
    depth_midas = None
    if midas_estimator.model is not None:
        try:
            depth_midas, _ = midas_estimator.estimate_depth(test_image)
            print("MiDaS depth estimation completed!")
        except Exception as e:
            print(f"MiDaS estimation failed: {e}")
    
    # Method 4: Structure Tensor based depth
    print("Running structure tensor depth estimation...")
    
    # Compute structure tensor
    Ixx = cv2.GaussianBlur(grad_x * grad_x, (5, 5), 0)
    Iyy = cv2.GaussianBlur(grad_y * grad_y, (5, 5), 0)
    Ixy = cv2.GaussianBlur(grad_x * grad_y, (5, 5), 0)
    
    # Compute eigenvalues
    det = Ixx * Iyy - Ixy * Ixy
    trace = Ixx + Iyy
    
    # Harris corner response (can indicate depth features)
    harris_response = det - 0.04 * (trace ** 2)
    depth_structure = cv2.normalize(np.abs(harris_response), None, 0, 1, cv2.NORM_MINMAX)
    
    # Create comprehensive visualization
    n_methods = 4 if depth_midas is not None else 3
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Gradient-based depth
    axes[0, 1].imshow(depth_gradient, cmap='plasma')
    axes[0, 1].set_title('Gradient-based Depth')
    axes[0, 1].axis('off')
    
    # Laplacian-based depth
    axes[0, 2].imshow(depth_laplacian, cmap='plasma')
    axes[0, 2].set_title('Laplacian-based Depth')
    axes[0, 2].axis('off')
    
    # Structure tensor depth
    axes[1, 0].imshow(depth_structure, cmap='plasma')
    axes[1, 0].set_title('Structure Tensor Depth')
    axes[1, 0].axis('off')
    
    # MiDaS depth (if available)
    if depth_midas is not None:
        axes[1, 1].imshow(depth_midas, cmap='plasma')
        axes[1, 1].set_title('MiDaS Depth Estimation')
        axes[1, 1].axis('off')
        
        # Comparison plot
        axes[1, 2].plot(depth_gradient.flatten()[::1000], label='Gradient', alpha=0.7)
        axes[1, 2].plot(depth_laplacian.flatten()[::1000], label='Laplacian', alpha=0.7)
        axes[1, 2].plot(depth_midas.flatten()[::1000], label='MiDaS', alpha=0.7)
        axes[1, 2].set_title('Depth Profile Comparison')
        axes[1, 2].legend()
        axes[1, 2].set_xlabel('Pixel Index (sampled)')
        axes[1, 2].set_ylabel('Normalized Depth')
    else:
        # Without MiDaS, show comparison of classical methods
        axes[1, 1].plot(depth_gradient.flatten()[::1000], label='Gradient', alpha=0.7)
        axes[1, 1].plot(depth_laplacian.flatten()[::1000], label='Laplacian', alpha=0.7)
        axes[1, 1].plot(depth_structure.flatten()[::1000], label='Structure', alpha=0.7)
        axes[1, 1].set_title('Classical Methods Comparison')
        axes[1, 1].legend()
        axes[1, 1].set_xlabel('Pixel Index (sampled)')
        axes[1, 1].set_ylabel('Normalized Depth')
        
        # Empty the last subplot
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('advanced_depth_estimation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Advanced depth estimation completed!")
    print("Results saved as 'advanced_depth_estimation_results.png'")
    
    # Print statistics
    print(f"\nDepth estimation statistics:")
    print(f"Gradient-based - Min: {depth_gradient.min():.4f}, Max: {depth_gradient.max():.4f}, Mean: {depth_gradient.mean():.4f}")
    print(f"Laplacian-based - Min: {depth_laplacian.min():.4f}, Max: {depth_laplacian.max():.4f}, Mean: {depth_laplacian.mean():.4f}")
    print(f"Structure tensor - Min: {depth_structure.min():.4f}, Max: {depth_structure.max():.4f}, Mean: {depth_structure.mean():.4f}")
    
    if depth_midas is not None:
        print(f"MiDaS - Min: {depth_midas.min():.4f}, Max: {depth_midas.max():.4f}, Mean: {depth_midas.mean():.4f}")

if __name__ == "__main__":
    run_advanced_depth_estimation() 