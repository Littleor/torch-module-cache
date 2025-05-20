import torch
import torch.nn as nn
import time
import os
import sys

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch_module_cache import cache_module, clear_memory_caches

@cache_module()
class SimpleBatchModel(nn.Module):
    def __init__(self):
        super().__init__()
        print("Initializing model (this should only happen once)")
        time.sleep(0.2)  # Add delay to simulate a heavy model initialization
        self.linear1 = nn.Linear(10, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.linear3 = nn.Linear(2048, 5)
        self.relu = nn.ReLU()
        
    def forward(self, x, cache_key=None):
        # For demonstration, we'll print information on batched processing
        if cache_key is not None:
            if isinstance(cache_key, (list, tuple)):
                print(f"Batch processing with {len(cache_key)} items")
            else:
                print(f"Single item processing with key: {cache_key}")
        else:
            print("Running without cache key")
            
        # Simulate computation that takes longer for larger batches
        if isinstance(x, torch.Tensor) and x.dim() > 0:
            # Add slight delay based on batch size to simulate real workload
            time.sleep(0.1 * x.size(0))
            
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return self.linear3(x)

def main():
    # Clear any existing caches first
    clear_memory_caches()
    
    # Create the model
    model = SimpleBatchModel()
    print("Model created but not initialized")
    
    # ===== SINGLE ITEM PROCESSING (for comparison) =====
    print("\n===== SINGLE ITEM PROCESSING =====")
    
    # Create single input tensor
    single_input = torch.randn(1, 10)
    
    # Process single input without caching
    print("\nProcessing single input without cache key")
    start_time = time.time()
    single_output = model(single_input)
    print(f"Time taken: {time.time() - start_time:.4f} seconds")
    print(f"Output shape: {single_output.shape}")
    
    # Process single input with string cache key
    print("\nProcessing single input with string cache key 'item1'")
    start_time = time.time()
    single_cached_output = model(single_input, cache_key="item1")
    time_first = time.time() - start_time
    print(f"Time taken (first run): {time_first:.4f} seconds")
    
    # Access cached result
    print("\nAccessing cached result for 'item1'")
    start_time = time.time()
    single_cached_output_again = model(single_input, cache_key="item1")
    time_cached = time.time() - start_time
    print(f"Time taken (cached): {time_cached:.4f} seconds")
    print(f"Speedup: {time_first / time_cached:.2f}x faster")
    print(f"Results identical: {torch.allclose(single_cached_output, single_cached_output_again)}")
    
    # ===== BATCH PROCESSING =====
    print("\n===== BATCH PROCESSING =====")
    
    # Create a batch of inputs
    batch_size = 4
    batch_input = torch.randn(batch_size, 10)
    batch_keys = ["batch_item1", "batch_item2", "batch_item3", "batch_item4"]
    
    # Process batch with list of cache keys - first run
    print("\nProcessing batch with list of cache keys (first run)")
    start_time = time.time()
    batch_output = model(batch_input, cache_key=batch_keys)
    batch_time_first = time.time() - start_time
    print(f"Time taken (first batch): {batch_time_first:.4f} seconds")
    print(f"Output shape: {batch_output.shape}")
    
    # Process batch with cached results
    print("\nProcessing batch with cached results")
    start_time = time.time()
    batch_output_cached = model(batch_input, cache_key=batch_keys)
    batch_time_cached = time.time() - start_time
    print(f"Time taken (cached batch): {batch_time_cached:.4f} seconds")
    print(f"Batch speedup: {batch_time_first / batch_time_cached:.2f}x faster")
    print(f"Results identical: {torch.allclose(batch_output, batch_output_cached)}")
    
    # ===== PARTIAL CACHE HITS =====
    print("\n===== PARTIAL CACHE HITS =====")
    
    # Create new keys, some of which were already cached
    mixed_keys = ["batch_item1", "batch_item2", "new_item1", "new_item2"]
    
    # Process with mix of cached and new items
    print("\nProcessing batch with mix of cached and new keys")
    start_time = time.time()
    mixed_output = model(batch_input, cache_key=mixed_keys)
    mixed_time = time.time() - start_time
    print(f"Time for mixed batch: {mixed_time:.4f} seconds")
    print(f"Output shape: {mixed_output.shape}")
    
    # Now all should be cached
    print("\nProcessing again with all keys now cached")
    start_time = time.time()
    mixed_output_cached = model(batch_input, cache_key=mixed_keys)
    mixed_time_cached = time.time() - start_time
    print(f"Time (all cached): {mixed_time_cached:.4f} seconds")
    print(f"Speedup: {mixed_time / mixed_time_cached:.2f}x faster")
    
    # ===== INDIVIDUAL VS BATCH COMPARISON =====
    print("\n===== INDIVIDUAL VS BATCH COMPARISON =====")
    
    # Create new batch for testing
    new_batch = torch.randn(batch_size, 10)
    new_keys = [f"comparison_item{i}" for i in range(batch_size)]
    
    # Process individually (one by one)
    print("\nProcessing items individually (one by one)")
    start_time = time.time()
    individual_results = []
    for i in range(batch_size):
        result = model(new_batch[i:i+1], cache_key=new_keys[i])
        individual_results.append(result)
    individual_time = time.time() - start_time
    print(f"Time for individual processing: {individual_time:.4f} seconds")
    
    # Clear cache and process as batch
    clear_memory_caches()
    print("\nProcessing as a single batch")
    start_time = time.time()
    batch_results = model(new_batch, cache_key=new_keys)
    batch_time = time.time() - start_time
    print(f"Time for batch processing: {batch_time:.4f} seconds")
    print(f"Batch vs Individual speedup: {individual_time / batch_time:.2f}x faster")
    
    # Compare results
    individual_combined = torch.cat(individual_results, dim=0)
    print(f"Results identical: {torch.allclose(individual_combined, batch_results)}")

if __name__ == "__main__":
    main() 