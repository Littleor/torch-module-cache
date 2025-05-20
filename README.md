# Torch Module Cache

A PyTorch module caching decorator that enables efficient caching of module outputs, with support for both single inference and batch processing.

## Features

- Cache PyTorch module outputs to disk or memory
- Support for both single inputs and batched inputs
- Automatic smart batching for performance optimization
- Safe loading options for improved security
- Memory cache for ultra-fast repeated access
- Configurable cache paths and naming

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/torch-module-cache.git
cd torch-module-cache

# Install the package
pip install -e .
```

## Basic Usage

### Simple Example

```python
import torch
import torch.nn as nn
from torch_module_cache import cache_module

@cache_module()
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize your model here
        self.linear = nn.Linear(10, 5)
        
    def forward(self, x):
        # The cache_key parameter is injected by the decorator
        # When provided, results will be cached
        print("Not hit the cache, forwarding", x.shape)
        return self.linear(x)

# Create model instance
model = MyModel()

print("Normal forward pass (no caching)")

# Normal forward pass (no caching)
input_tensor = torch.randn(1, 10)
output = model(input_tensor)

print("Cached forward pass (first time will compute and cache)")
# Cached forward pass (first time will compute and cache)
output_cached = model(input_tensor, cache_key="my_unique_key")

print("Subsequent calls with the same key will load from cache")
# Subsequent calls with the same key will load from cache
output_from_cache = model(input_tensor, cache_key="my_unique_key")
```

## Batch Processing

The decorator supports batched inference, which can significantly improve performance when processing multiple inputs:

```python
import torch
import torch.nn as nn
from torch_module_cache import cache_module

@cache_module()
class BatchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    def forward(self, x):
        print("Not hit the cache, forwarding", x.shape)
        return self.encoder(x)

# Create model instance
model = BatchModel()

print("Begin test")

# Create a batch of inputs
batch_size = 4
batch_input = torch.randn(batch_size, 10)

# Create a list of cache keys (one for each item in the batch)
batch_keys = ["item1", "item2", "item3", "item4"]

print("Process the entire batch with unique keys for each item")
# Process the entire batch with unique keys for each item
# The decorator will handle caching each result individually
batch_output = model(batch_input, cache_key=batch_keys)

print("The next time you use the same keys, results will be loaded from cache")
# The next time you use the same keys, results will be loaded from cache
cached_batch_output = model(batch_input, cache_key=batch_keys)
```

### Partial Cache Hits

One of the key features is the ability to handle partial cache hits efficiently:

```python
# Some keys are already cached, some are new
mixed_keys = ["item1", "item2", "new_item1", "new_item2"]

# Only the new items will be processed, cached items will be loaded from cache
mixed_output = model(batch_input, cache_key=mixed_keys)
```

## Configuration Options

The `@cache_module()` decorator accepts several configuration parameters:

```python
@cache_module(
    # Path to store cache files (default: ~/.cache/torch-module-cache)
    cache_path="/path/to/cache",
    
    # Subfolder name for this specific model (default: class name)
    cache_name="my_model_cache",
    
    # Cache level: CacheLevel.DISK or CacheLevel.MEMORY
    cache_level=CacheLevel.MEMORY,
    
    # Whether to use safer loading options (recommended for untrusted data)
    safe_load=True
)
```

## Cache Management

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

## Performance Considerations

- **Memory vs. Disk Caching**: Memory caching is much faster but limited by available RAM
- **Batch Processing**: Processing inputs in batches is typically much faster than individual processing
- **Cache Keys**: Choose unique and meaningful cache keys that represent your inputs
- **Cache Path**: For large models, ensure the cache path has sufficient disk space

## Advanced Usage

### Custom Cache Path

```python
# Custom cache path
@cache_module(cache_path="/tmp/my_model_cache")
class CustomPathModel(nn.Module):
    # ...
```

### Batch Processing with Mixed Types

```python
# The decorator handles various input types
# Cache keys can be strings, numbers, or any hashable types
model_output = model(inputs, cache_key=[1, 2, 3, 4])
```

## Example Scripts

The package includes several example scripts in the `examples/` directory:

- `basic_usage.py`: Simple example showing basic caching functionality
- `batch_usage.py`: Demonstrates batch processing and performance comparison

## Notes

- The first forward pass with a specific cache key will always execute the model
- For best performance with batches, try to reuse the same batch size and structure
- Non-tensor inputs and outputs are supported but may have serialization limitations

## License

[MIT License](LICENSE) 