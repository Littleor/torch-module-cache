import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch_module_cache import CacheLevel, cache_module

torch.manual_seed(42)

@cache_module(cache_name="DictReturnModel")
class DictReturnModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim * 2)
        
    def forward(self, x, cache_key=None):
        """
        Returns a dictionary containing multiple results
        
        Args:
            x: Input tensor
            cache_key: Cache key, if None then cache is not used
            
        Returns:
            dict: Dictionary containing different outputs
        """
        # Main computation path
        hidden = F.relu(self.fc1(x))
        primary = self.fc2(hidden)
        secondary = self.fc3(hidden)
        
        # Return dictionary containing multiple values
        return {
            "primary_output": primary,
            "secondary_output": secondary,
            "hidden_features": hidden,
            "metadata": {
                "input_shape": x.shape,
                "primary_shape": primary.shape,
                "secondary_shape": secondary.shape
            }
        }

# Model that returns tuple type
@cache_module(cache_name="TupleReturnModel")
class TupleReturnModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.decoder1 = nn.Linear(hidden_dim // 2, output_dim)
        self.decoder2 = nn.Linear(hidden_dim // 2, output_dim * 2)
        
    def forward(self, x, cache_key=None):
        """
        Returns a tuple containing multiple results
        
        Args:
            x: Input tensor
            cache_key: Cache key, if None then cache is not used
            
        Returns:
            tuple: Tuple containing encoder features and two different decoder outputs
        """
        # Encode input
        encoded = self.encoder(x)
        
        # Two different decodings
        output1 = self.decoder1(encoded)
        output2 = self.decoder2(encoded)
        
        # Return triple: (encoded features, output1, output2)
        return (encoded, output1, output2)

def main():
    print("Demonstrating cache_module decorator for models returning dict and tuple types\n")
    
    # Create input data
    input_data = torch.randn(1, 10)
    
    # Test model that returns dictionary
    print("=== Testing model that returns dict ===")
    dict_model = DictReturnModel()
    
    # Run once without cache
    print("First run (without cache)...")
    result1 = dict_model(input_data)
    print(f"Return type: {(result1)}")
    print(f"Dictionary keys: {list(result1.keys())}")
    
    # Run with cache
    cache_key = "test_input"
    print("\nRunning with cache...")
    result2 = dict_model(input_data, cache_key=cache_key)
    print(f"Return type: {(result2)}")
    
    # Run again with the same cache key (will load from cache)
    print("\nRunning again with the same cache key (loading from cache)...")
    result3 = dict_model(input_data, cache_key=cache_key)
    
    # Check if results are the same
    all_tensors_equal = all(
        torch.allclose(result2[k], result3[k]) 
        for k in result2.keys() 
        if isinstance(result2[k], torch.Tensor)
    )
    print(f"Cached results match original results: {all_tensors_equal}")
    
    # Test model that returns tuple
    print("\n\n=== Testing model that returns tuple ===")
    tuple_model = TupleReturnModel()
    
    # Run once without cache
    print("First run (without cache)...")
    result1 = tuple_model(input_data)
    print(f"Return type: {type(result1)}")
    print(f"Tuple length: {len(result1)}")
    print(f"Types of elements in tuple: {[(x) for x in result1]}")
    
    # Run with cache
    print("\nRunning with cache...")
    result2 = tuple_model(input_data, cache_key=cache_key)
    
    # Run again with the same cache key (will load from cache)
    print("\nRunning again with the same cache key (loading from cache)...")
    result3 = tuple_model(input_data, cache_key=cache_key)
    
    # Check if results are the same
    all_equal = all(
        torch.allclose(result2[i], result3[i]) 
        for i in range(len(result2))
    )
    print(f"Cached results match original results: {all_equal}")
    
    print("\nDemonstration complete!")

if __name__ == "__main__":
    main() 