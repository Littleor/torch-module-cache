# Torch Module Cache

🚀 **One-line code to implement PyTorch feature caching, accelerate training by 30x+!**

Torch Module Cache is a simple yet powerful PyTorch tool that enables model feature caching with just one line of code, significantly boosting training and inference speed. Whether it's dataset preprocessing or pretrained model feature caching, it's all made easy.

## ✨ Key Features

- 🚀 **Minimal Code**: Enable caching with just one decorator
- 📈 **Significant Speedup**: Real-world tests show 30x+ acceleration per epoch
- 💻 **VRAM Friendly**: Model will not be loaded until not hit cache, save your VRAM
- 🔄 **Flexible Caching**: Support for both dataset and model feature caching
- 🎯 **Smart Inference**: Support for inference mode with global cache disabling
- 💾 **Memory Optimized**: Automatic cache memory management to prevent leaks

## 🚀 Quick Start

### 1. Installation

```bash
pip install torch-module-cache
```

### 2. Basic Usage

Simply add the `@cache_module()` decorator to enable feature caching, this will be extremely effective when extracting features within the model using pre-trained models:

```python
from torch_module_cache import cache_module

# Only need to add one line of code to enable caching
@cache_module()
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 3)
    
    def forward(self, x):
        return self.linear(x)

# Using cache
model = MyModel()
# First run will compute and cache the result
output1 = model(x, cache_key="key1")
# Second run will use the cached result
output2 = model(x, cache_key="key1")

# For batch processing, you can use a list of cache keys:
cache_keys = [f"key_{i}" for i in range(10)]
outputs = model(torch.randn(10, 10), cache_key=cache_keys)
```

### 3 Pretrained Model Feature Caching

Accelerate your model by caching features from pretrained models like ViT, ResNet, etc.:

```python
# Only need to add one line of code to enable caching
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
```

### 4. Dataset Feature Caching

Still manually extracting features and saving them to `.pt` files? Use caching in your dataset to accelerate data loading with **only one-line code**:

```python
@cache_module(cache_name="feature_processor")
class FeatureProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 256)
    
    def forward(self, x):
        return self.linear(x)

class CachedDataset(Dataset):
    def __init__(self):
        self.processor = FeatureProcessor()
    
    def __getitem__(self, idx):
        raw_data = self.data[idx]
        # Use sample index as cache key, the second epoch will start using cache to speed up.
        processed_data = self.processor(raw_data, cache_key=f"sample_{idx}")
        return processed_data, self.labels[idx]
```

### 5. Inference Mode

Disable caching during inference:

```python
from torch_module_cache import enable_inference_mode

# Enable inference mode (disable caching and model will be init when instance is created)
enable_inference_mode()

# Model will compute directly without using cache
model = MyModel()
output = model(x)
```

## 📊 Performance Comparison

| Scenario | Without Cache | With Cache | Speedup |
|----------|--------------|------------|---------|
| Dataset Preprocessing | 100s | 3.2s | 31.25x |
| ViT Feature Extraction | 2.10s | 0.024s | 86.82x |

## 📚 More Examples

Check out the [examples](./examples) directory for more usage examples:
- [Basic Usage](./examples/basic_usage.py)
- [Dataset Feature Caching](./examples/dataset_feature_cache.py)
- [Inference Mode](./examples/infer_usage.py)
- [Batch Processing](./examples/batch_usage.py)
- [Custom Cache Options](./examples/custom_cache_options.py)

## ⚙️ Configuration Options

The `@cache_module()` decorator accepts several configuration parameters:

```python
from torch_module_cache import cache_module, CacheLevel

@cache_module(
    # Path to store cache files (default: ~/.cache/torch-module-cache)
    cache_path="/path/to/cache",
    
    # Subfolder name for this specific model (default: class name)
    cache_name="my_model_cache",
    
    # Cache level: CacheLevel.DISK or CacheLevel.MEMORY
    cache_level=CacheLevel.MEMORY,
    
    # Whether to use safer loading options (recommended for untrusted data)
    safe_load=False,
    
    # Maximum memory usage in MB (default: None)
    max_memory_cache_size_mb=None,
)
class MyModel(nn.Module):
    # ... your model implementation
```

## 🔧 Cache Management

### Memory Management

```python
from torch_module_cache import clear_memory_caches, clear_disk_caches

# Clear all in-memory caches
clear_memory_caches()

# Clear all disk caches
clear_disk_caches()

# Clear caches for a specific model
clear_memory_caches(cache_name="my_model_cache")
clear_disk_caches(cache_name="my_model_cache")
```

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

MIT License 