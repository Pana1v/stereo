import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
import time
from transformers import DPTImageProcessor, DPTForDepthEstimation
import warnings
warnings.filterwarnings("ignore")

class AdvancedDepthModels:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.models = {}
        self.processors = {}
        
    def load_dpt_large(self):
        """Load DPT-Large model (Intel's follow-up to MiDaS)"""
        try:
            print("Loading DPT-Large model...")
            processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
            model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(self.device)
            self.models['DPT-Large'] = model
            self.processors['DPT-Large'] = processor
            print("✓ DPT-Large loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to load DPT-Large: {e}")
            return False
    
    def load_midas_large(self):
        """Load MiDaS Large as baseline"""
        try:
            print("Loading MiDaS Large model...")
            processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
            model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(self.device)
            self.models['MiDaS-Large'] = model
            self.processors['MiDaS-Large'] = processor
            print("✓ MiDaS Large loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to load MiDaS Large: {e}")
            return False
    
    def estimate_depth(self, image_path, model_name):
        """Estimate depth using specified model"""
        if model_name not in self.models:
            print(f"Model {model_name} not loaded")
            return None
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            processor = self.processors[model_name]
            model = self.models[model_name]
            
            # Process image
            inputs = processor(images=image, return_tensors="pt").to(self.device)
            
            # Get depth prediction
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            depth_map = prediction.cpu().numpy()
            
            # Normalize for visualization
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            depth_map_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            
            return depth_map_normalized
            
        except Exception as e:
            print(f"Error estimating depth with {model_name}: {e}")
            return None
    
    def create_comparison_window(self, image_path):
        """Create a window showing depth estimation results from all models"""
        print(f"\nProcessing image: {image_path}")
        
        # Load original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Could not load image: {image_path}")
            return
        
        # Resize for display
        height, width = original_image.shape[:2]
        max_size = 400
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            original_image = cv2.resize(original_image, (new_width, new_height))
        
        # Get depth maps from all models
        depth_maps = {}
        model_names = list(self.models.keys())
        
        print(f"\nEstimating depth with {len(model_names)} models...")
        for model_name in model_names:
            print(f"Processing {model_name}...")
            start_time = time.time()
            depth_map = self.estimate_depth(image_path, model_name)
            if depth_map is not None:
                # Resize depth map to match original image
                depth_map_resized = cv2.resize(depth_map, (original_image.shape[1], original_image.shape[0]))
                depth_maps[model_name] = depth_map_resized
                elapsed_time = time.time() - start_time
                print(f"✓ {model_name} completed in {elapsed_time:.2f}s")
            else:
                print(f"✗ {model_name} failed")
        
        if not depth_maps:
            print("No depth maps generated successfully")
            return
        
        # Create comparison visualization
        num_models = len(depth_maps)
        cols = 2
        rows = (num_models + 1) // cols  # +1 for original image
        
        # Create window
        window_width = original_image.shape[1] * cols
        window_height = original_image.shape[0] * rows
        comparison_image = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        
        # Add original image
        comparison_image[0:original_image.shape[0], 0:original_image.shape[1]] = original_image
        
        # Add depth maps
        idx = 1
        for model_name, depth_map in depth_maps.items():
            row = idx // cols
            col = idx % cols
            
            # Convert depth map to BGR for OpenCV
            depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
            
            # Add text label
            cv2.putText(depth_colored, model_name, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Calculate position
            y_start = row * original_image.shape[0]
            y_end = y_start + original_image.shape[0]
            x_start = col * original_image.shape[1]
            x_end = x_start + original_image.shape[1]
            
            # Ensure we don't exceed bounds
            if y_end <= comparison_image.shape[0] and x_end <= comparison_image.shape[1]:
                comparison_image[y_start:y_end, x_start:x_end] = depth_colored
            idx += 1
        
        # Add title to original image
        cv2.putText(comparison_image, "Original Image", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display results
        cv2.imshow("Advanced Depth Estimation Comparison", comparison_image)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save comparison image
        output_path = f"results/advanced_depth_comparison_{image_path.split('/')[-1].split('.')[0]}.png"
        cv2.imwrite(output_path, comparison_image)
        print(f"Comparison saved to: {output_path}")
        
        return depth_maps
    
    def run_comparison(self, image_path):
        """Run complete comparison on the given image"""
        print("=" * 60)
        print("ADVANCED DEPTH ESTIMATION MODEL COMPARISON")
        print("=" * 60)
        
        # Load models
        print("\nLoading depth estimation models...")
        models_loaded = 0
        models_loaded += self.load_dpt_large()
        models_loaded += self.load_midas_large()
        
        if models_loaded == 0:
            print("No models loaded successfully. Exiting.")
            return
        
        print(f"\nSuccessfully loaded {models_loaded} models")
        
        # Run comparison
        depth_maps = self.create_comparison_window(image_path)
        
        if depth_maps:
            print(f"\nComparison completed with {len(depth_maps)} models")
            print("Models used:", list(depth_maps.keys()))
        else:
            print("Comparison failed")

def main():
    # Initialize the comparison system
    depth_comparison = AdvancedDepthModels()
    
    # Test images
    test_images = [
        "images/download.jpeg",
        "images/images (1).jpeg"
    ]
    
    for image_path in test_images:
        try:
            depth_comparison.run_comparison(image_path)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 