import torch
import torch.nn as nn
import time
import os
import sys

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch_module_cache import cache_module, clear_cache

@cache_module()
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        print("Initializing model (this should only happen once)")
        time.sleep(0.2)  # Add delay to simulate a heavy model initialization
        self.linear1 = nn.Linear(10, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.linear3 = nn.Linear(2048, 5)
        self.relu = nn.ReLU()
        
    def forward(self, x, cache_key=None):
        print(f"Cache not work, Running forward pass with cache_key: {cache_key}")
        if cache_key is None:
            # Simulate a more expensive computation when not using cache
            time.sleep(1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return self.linear3(x)

def main():
    # Clear any existing cache first
    clear_cache()

    # Create the model (this doesn't initialize it yet)
    model = SimpleModel()
    print("Model created but not initialized")
    
    # Create input tensor
    input_tensor = torch.randn(1, 10)
    
    # First forward pass with no cache key (should initialize and run the model)
    print("\nRunning first forward pass without cache key")
    start_time = time.time()
    output1 = model(input_tensor)
    print(f"Time taken: {time.time() - start_time:.4f} seconds")
    print(f"Output shape: {output1.shape}")
    
    # Second forward pass with a cache key (should already have initialized model and cache the result)
    print("\nRunning second forward pass with cache key 'test1'")
    start_time = time.time()
    output2 = model(input_tensor, cache_key="test1")
    time_normal = time.time() - start_time
    print(f"Time taken: {time_normal:.4f} seconds")
    print(f"Output shape: {output2.shape}")
    
    # Third forward pass with the same cache key (should load from cache)
    print("\nRunning third forward pass with same cache key 'test1'")
    start_time = time.time()
    output3 = model(input_tensor, cache_key="test1")
    time_cached = time.time() - start_time
    print(f"Time taken: {time_cached:.4f} seconds")
    print(f"Output shape: {output3.shape}")
    print(f"Cache speedup: {time_normal/time_cached:.2f}x faster")
    
    # Check if the outputs are the same
    print("\nChecking if outputs are the same:")
    print(f"Output2 and Output3 are identical: {torch.allclose(output2, output3)}")
    
    # Fourth forward pass with a different cache key (should use initialized model and cache)
    print("\nRunning fourth forward pass with different cache key 'test2'")
    start_time = time.time()
    output4 = model(input_tensor, cache_key="test2")
    time_normal = time.time() - start_time
    print(f"Time taken: {time_normal:.4f} seconds")
    print(f"Output shape: {output4.shape}")
    
    # Fifth forward pass with the second cache key (should load from cache)
    print("\nRunning fifth forward pass with cache key 'test2'")
    start_time = time.time()
    output5 = model(input_tensor, cache_key="test2")
    time_cached = time.time() - start_time
    print(f"Time taken: {time_cached:.4f} seconds")
    print(f"Output shape: {output5.shape}")
    print(f"Cache speedup: {time_normal/time_cached:.2f}x faster")

if __name__ == "__main__":
    main() 