import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from transformers import DPTImageProcessor, DPTForDepthEstimation
import warnings
warnings.filterwarnings("ignore")

class ComprehensiveDepthComparison:
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
    
    def load_dpt_beit_large(self):
        """Load DPT-BEiT-Large model (another variant)"""
        try:
            print("Loading DPT-BEiT-Large model...")
            processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
            model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(self.device)
            self.models['DPT-BEiT-Large'] = model
            self.processors['DPT-BEiT-Large'] = processor
            print("✓ DPT-BEiT-Large loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to load DPT-BEiT-Large: {e}")
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
    
    def load_midas_small(self):
        """Load MiDaS Small for comparison"""
        try:
            print("Loading MiDaS Small model...")
            processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
            model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(self.device)
            self.models['MiDaS-Small'] = model
            self.processors['MiDaS-Small'] = processor
            print("✓ MiDaS Small loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to load MiDaS Small: {e}")
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
    
    def create_comprehensive_comparison(self, image_path):
        """Create a comprehensive comparison window with 3-4 models"""
        print(f"\nProcessing image: {image_path}")
        
        # Load original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Could not load image: {image_path}")
            return
        
        # Resize for display
        height, width = original_image.shape[:2]
        max_size = 300  # Smaller for more models
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
        
        # Create comprehensive visualization (2x3 grid)
        num_models = len(depth_maps)
        cols = 3
        rows = 2
        
        # Create window
        window_width = original_image.shape[1] * cols
        window_height = original_image.shape[0] * rows
        comparison_image = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        
        # Add original image (top-left)
        comparison_image[0:original_image.shape[0], 0:original_image.shape[1]] = original_image
        
        # Add depth maps in a 2x3 grid
        model_list = list(depth_maps.keys())
        positions = [
            (0, 1),  # Top middle
            (0, 2),  # Top right
            (1, 0),  # Bottom left
            (1, 1),  # Bottom middle
            (1, 2)   # Bottom right
        ]
        
        for i, (model_name, depth_map) in enumerate(depth_maps.items()):
            if i < len(positions):
                row, col = positions[i]
                
                # Convert depth map to BGR for OpenCV
                depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
                
                # Add text label
                cv2.putText(depth_colored, model_name, (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Calculate position
                y_start = row * original_image.shape[0]
                y_end = y_start + original_image.shape[0]
                x_start = col * original_image.shape[1]
                x_end = x_start + original_image.shape[1]
                
                # Ensure we don't exceed bounds
                if y_end <= comparison_image.shape[0] and x_end <= comparison_image.shape[1]:
                    comparison_image[y_start:y_end, x_start:x_end] = depth_colored
        
        # Add title to original image
        cv2.putText(comparison_image, "Original Image", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display results
        cv2.imshow("Comprehensive Depth Estimation Comparison", comparison_image)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save comparison image
        output_path = f"results/comprehensive_depth_comparison_{image_path.split('/')[-1].split('.')[0]}.png"
        cv2.imwrite(output_path, comparison_image)
        print(f"Comprehensive comparison saved to: {output_path}")
        
        return depth_maps
    
    def run_comprehensive_comparison(self, image_path):
        """Run comprehensive comparison on the given image"""
        print("=" * 70)
        print("COMPREHENSIVE DEPTH ESTIMATION MODEL COMPARISON")
        print("=" * 70)
        
        # Load models
        print("\nLoading depth estimation models...")
        models_loaded = 0
        models_loaded += self.load_dpt_large()
        models_loaded += self.load_dpt_beit_large()
        models_loaded += self.load_midas_large()
        models_loaded += self.load_midas_small()
        
        if models_loaded == 0:
            print("No models loaded successfully. Exiting.")
            return
        
        print(f"\nSuccessfully loaded {models_loaded} models")
        
        # Run comparison
        depth_maps = self.create_comprehensive_comparison(image_path)
        
        if depth_maps:
            print(f"\nComprehensive comparison completed with {len(depth_maps)} models")
            print("Models used:", list(depth_maps.keys()))
            
            # Print performance summary
            print("\n" + "=" * 50)
            print("PERFORMANCE SUMMARY")
            print("=" * 50)
            for model_name in depth_maps.keys():
                print(f"✓ {model_name}: Successfully generated depth map")
        else:
            print("Comprehensive comparison failed")

def main():
    # Initialize the comparison system
    depth_comparison = ComprehensiveDepthComparison()
    
    # Test with the robot image
    test_image = "images/download.jpeg"
    
    try:
        depth_comparison.run_comprehensive_comparison(test_image)
    except Exception as e:
        print(f"Error processing {test_image}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 