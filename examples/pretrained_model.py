import os
import sys
import time

import timm
import torch
import torch.nn as nn

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch_module_cache import cache_module, clear_cache


@cache_module()
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained ViT
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.vit.eval()  # Set to eval mode

    def forward(self, x):
        # Extract features from ViT
        with torch.no_grad():
            features = self.vit.forward_features(x)
        return features


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # `feature_extractor` is frozen, so we can use cache to speed up
        self.feature_extractor = FeatureExtractor()
        self.classifier = nn.Linear(768, 10)  # ViT-Base features are 768-dim

    def forward(self, x, cache_key=None):
        # Features will be cached automatically
        features = self.feature_extractor(x, cache_key=cache_key)
        return self.classifier(features)


if __name__ == "__main__":
    # Clear cache first
    clear_cache(cache_name="FeatureExtractor")

    model = MyModel()
    x = torch.randn(10, 3, 224, 224)
    cache_keys = [f"vit_features_{x.shape}_{i}" for i in range(10)]

    # Test without cache
    start_time = time.time()
    _ = model(x, cache_key=cache_keys)
    no_cache_time = time.time() - start_time
    print(f"Time without cache: {no_cache_time:.4f} seconds")

    # Test with cache
    start_time = time.time()
    _ = model(x, cache_key=cache_keys)
    with_cache_time = time.time() - start_time
    print(f"Time with cache: {with_cache_time:.4f} seconds")

    print(f"Speedup: {no_cache_time/with_cache_time:.2f}x")
