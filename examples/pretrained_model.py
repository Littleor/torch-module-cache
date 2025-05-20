import torch
import torch.nn as nn
import time
import os
import sys
from PIL import Image
import urllib.request
from io import BytesIO

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch_module_cache import cache_module, CacheLevel, clear_cache

# Try to import torchvision, but don't fail if it's not available
try:
    import torchvision
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("Torchvision is not available. Please install it with: pip install torchvision")
    sys.exit(1)

@cache_module(cache_name="resnet_feature_extractor")
class FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        print("Initializing ResNet feature extractor")
        # This will only be executed when cache is missed
        self.model = torchvision.models.resnet18(pretrained=pretrained)
        # Remove the final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
    
    def forward(self, x, cache_key=None):
        print(f"Running feature extraction with cache_key: {cache_key}")
        with torch.no_grad():
            features = self.model(x)
            # Reshape features to vector
            return features.view(features.size(0), -1)

def load_image_from_url(url):
    """Download an image from URL and convert to tensor"""
    response = urllib.request.urlopen(url)
    img = Image.open(BytesIO(response.read())).convert('RGB')
    
    # Apply transformations similar to what the model expects
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return transform(img).unsqueeze(0)  # Add batch dimension

def main():
    # Clear any existing cache
    clear_cache(cache_name="resnet_feature_extractor")
    
    # Create the feature extractor (won't initialize ResNet yet)
    feature_extractor = FeatureExtractor()
    print("Feature extractor created but ResNet not yet initialized")
    
    # Example image URLs
    image_urls = [
        "https://raw.githubusercontent.com/pytorch/pytorch.github.io/master/assets/images/resnet.png",
        "https://raw.githubusercontent.com/pytorch/pytorch.github.io/master/assets/images/pytorch-logo.png",
    ]
    
    # Process the first image
    print("\nProcessing first image...")
    img1 = load_image_from_url(image_urls[0])
    
    # First run - should initialize the model and cache the result
    print("First run (should initialize model and cache):")
    start_time = time.time()
    features1 = feature_extractor(img1, cache_key="image1")
    first_time = time.time() - start_time
    print(f"Time: {first_time:.4f} seconds")
    print(f"Feature shape: {features1.shape}")
    
    # Second run with same image - should use cache
    print("\nSecond run with same image (should use cache):")
    start_time = time.time()
    features1_cached = feature_extractor(img1, cache_key="image1")
    second_time = time.time() - start_time
    print(f"Time: {second_time:.4f} seconds")
    print(f"Feature shape: {features1_cached.shape}")
    print(f"Cache speedup: {first_time / second_time:.2f}x")
    print(f"Same features: {torch.allclose(features1, features1_cached)}")
    
    # Process a second image
    print("\nProcessing second image...")
    img2 = load_image_from_url(image_urls[1])
    
    # Third run with different image - should use the initialized model and cache the result
    print("Run with different image (model already initialized):")
    start_time = time.time()
    features2 = feature_extractor(img2, cache_key="image2")
    print(f"Time: {time.time() - start_time:.4f} seconds")
    print(f"Feature shape: {features2.shape}")
    
    # Verify that the features are different
    print(f"\nFeatures are different: {not torch.allclose(features1, features2)}")

if __name__ == "__main__":
    main() 