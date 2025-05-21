import os
import sys
import time

import torch
import torch.nn as nn

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch_module_cache import cache_module, clear_cache
from torch_module_cache.decorator import CacheLevel


@cache_module(
    cache_level=CacheLevel.MEMORY,
    max_memory_cache_size_mb=100,
    cache_name="CustomCacheOptionsModel",
    safe_load=False
)
class CustomCacheOptionsModel(nn.Module):
    def __init__(self):
        super().__init__()
        print("Initializing model (this should only happen once)")
        time.sleep(0.2)  # Add delay to simulate a heavy model initialization
        self.linear1 = nn.Linear(10, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.linear3 = nn.Linear(2048, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        print(
            f"Running forward with x: {x.shape} (Only be called when cache is not fully hit)"
        )
        # Simulate a more expensive computation when not using cache
        time.sleep(1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return self.linear3(x)


def main():
    # Clear any existing cache first
    clear_cache(cache_name="CustomCacheOptionsModel")

    input_batch = [
        torch.randn(1, 10),
        torch.randn(1, 10),
        torch.randn(1, 10),
        torch.randn(1, 10),
    ]
    input_batch_keys = ["key1", "key2", "key3", "new_key"]

    # Create the model (this doesn't initialize it yet)
    model = CustomCacheOptionsModel()

    # 1. Run the model without cache

    print("1. Running model without cache")
    result = model(torch.randn(3, 10))
    print(result)
    print()

    # 2. Run the model with cache, but at the first time, it will not hit the cache
    print("2. Running model with cache (not hit the cache)")
    result = model(torch.cat(input_batch[:3], dim=0), cache_key=input_batch_keys[:3])
    print(result)
    print()

    # 3. Run the model with cache, but at the second time, it will hit the cache
    print("3. Running model with cache (fully hit the cache)")
    result = model(torch.cat(input_batch[:3], dim=0), cache_key=input_batch_keys[:3])
    print(result)
    print()

    # 4. Run the model with cache, but at the third time, it will partially hit the cache
    print(
        "4. Running model with cache (partially hit the cache, the last one is not cached)"
    )
    result = model(torch.cat(input_batch, dim=0), cache_key=input_batch_keys)
    print(result)
    print()

    # 5.After the cache is fully hit, the cache will be used
    print("5. After the cache is fully hit, the cache will be used")
    result = model(torch.cat(input_batch, dim=0), cache_key=input_batch_keys)
    print(result)
    print()


if __name__ == "__main__":
    main()
