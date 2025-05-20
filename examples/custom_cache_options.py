import torch
import torch.nn as nn
import time
import os
import sys
import tempfile
import shutil

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch_module_cache import cache_module, CacheLevel, clear_cache, get_cache_size, list_cache_entries

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        print(f"Initializing {self.__class__.__name__}")
        self.linear1 = nn.Linear(10, 64)
        self.linear2 = nn.Linear(64, 5)
        self.relu = nn.ReLU()
        
    def forward(self, x, cache_key=None):
        print(f"Running forward pass on {self.__class__.__name__} with cache_key: {cache_key}")
        x = self.relu(self.linear1(x))
        return self.linear2(x)


# Model with default caching (disk-based, default location)
@cache_module()
class DefaultCacheModel(BaseModel):
    pass


# Model with custom cache path
@cache_module(cache_path="./temp_cache")
class CustomPathModel(BaseModel):
    pass


# Model with custom cache name
@cache_module(cache_name="custom_name_model")
class CustomNameModel(BaseModel):
    pass


# Model with memory-level caching
@cache_module(cache_level=CacheLevel.MEMORY)
class MemoryCacheModel(BaseModel):
    pass


# Model with disabled caching
@cache_module(cache_path=None)
class NoCacheModel(BaseModel):
    pass


def run_model_test(model, name):
    print(f"\n{'='*20} Testing {name} {'='*20}")
    
    # Create input tensor
    input_tensor = torch.randn(1, 10)
    
    # First run (should initialize and cache)
    print(f"\nFirst run with {name}")
    start_time = time.time()
    output1 = model(input_tensor, cache_key="test_key")
    first_time = time.time() - start_time
    print(f"Time taken: {first_time:.4f} seconds")
    
    # Second run (should use cache)
    print(f"\nSecond run with {name}")
    start_time = time.time()
    output2 = model(input_tensor, cache_key="test_key")
    second_time = time.time() - start_time
    print(f"Time taken: {second_time:.4f} seconds")
    print(f"Cache speedup: {first_time / second_time:.2f}x")
    print(f"Outputs match: {torch.allclose(output1, output2)}")
    
    return first_time, second_time


def main():
    # Create temporary directory for the custom path model
    os.makedirs("./temp_cache", exist_ok=True)
    
    try:
        # Clear any existing cache
        clear_cache()
        
        # Create all models
        default_model = DefaultCacheModel()
        custom_path_model = CustomPathModel()
        custom_name_model = CustomNameModel()
        memory_model = MemoryCacheModel()
        no_cache_model = NoCacheModel()
        
        # Test each model
        results = {}
        results["Default Cache"] = run_model_test(default_model, "Default Cache Model")
        results["Custom Path"] = run_model_test(custom_path_model, "Custom Path Model")
        results["Custom Name"] = run_model_test(custom_name_model, "Custom Name Model")
        results["Memory Cache"] = run_model_test(memory_model, "Memory Cache Model")
        results["No Cache"] = run_model_test(no_cache_model, "No Cache Model")
        
        # Show cache information
        print("\n" + "="*50)
        print("Cache Information:")
        print(f"Default cache size: {get_cache_size()} bytes")
        print(f"Custom path cache size: {get_cache_size('./temp_cache')} bytes")
        print("\nCache entries:")
        print(list_cache_entries())
        print(f"Custom path cache entries: {list_cache_entries('./temp_cache')}")
        
        # Print speedup comparison
        print("\n" + "="*50)
        print("Speedup Comparison (second run vs first run):")
        for name, (first, second) in results.items():
            if second > 0:
                speedup = first / second
                print(f"{name}: {speedup:.2f}x speedup")
    
    finally:
        # Clean up the temporary directory
        if os.path.exists("./temp_cache"):
            shutil.rmtree("./temp_cache")


if __name__ == "__main__":
    main() 