# torch-module-cache

A Python package for caching PyTorch module outputs. This package provides a decorator that adds caching capabilities to any PyTorch module, which helps reduce computational overhead by reusing previously computed results.

## Features

- Simple decorator-based API
- Lazy model initialization to save GPU memory
- Configurable cache location
- Support for both disk and memory-based caching
- Utilities for cache management

## Installation

```bash
pip install torch-module-cache
```

## Usage

Basic usage example:

```python
import torch
import torch.nn as nn
from torch_module_cache import cache_module

@cache_module()
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        
    def forward(self, x, cache_key=None):
        return self.linear(x)

# Create model (will not initialize immediately)
model = MyModel()

# First time will initialize the model and compute the result
input_tensor = torch.randn(1, 10)
output = model(input_tensor, cache_key="test_input")

# Second time with the same cache_key will load from cache
output_cached = model(input_tensor, cache_key="test_input")
```

## Configuration

The `cache_module` decorator accepts several configuration options:

```python
@cache_module(
    cache_path="./my_cache",  # Custom cache directory
    cache_name="my_model",    # Custom name for the cache subfolder
    cache_level=CacheLevel.MEMORY  # Use memory caching
)
class MyModel(nn.Module):
    # ...
```

## Cache Management

The package includes utilities for managing the cache:

```python
from torch_module_cache import clear_cache, get_cache_size, list_cache_entries

# Clear all caches
clear_cache()

# Clear specific module cache
clear_cache(cache_name="MyModel")

# Get cache size
size_bytes = get_cache_size()

# List all cache entries
entries = list_cache_entries()
```

## License

MIT 