import torch
import torch.nn as nn
import time
import os
import sys
from torch.utils.data import Dataset

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch_module_cache import cache_module, clear_cache


@cache_module(cache_name="dataset_feature_processor")
class FeatureProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        print("Initializing feature processor (this should only happen once)")
        time.sleep(0.2)  # Simulate initialization time
        self.linear1 = nn.Linear(10, 256)
        self.relu = nn.ReLU()

    def forward(self, x):
        print(f"Processing features for shape: {x.shape}")
        # Simulate expensive feature processing
        time.sleep(0.1)
        x = self.relu(self.linear1(x))
        return x


class CachedDataset(Dataset):
    """A dataset that caches processed features"""
    def __init__(self, num_samples=100, feature_dim=10):
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        # Generate some random data
        self.data = torch.randn(num_samples, feature_dim)
        self.labels = torch.randint(0, 3, (num_samples,))
        # Initialize the feature processor
        self.processor = FeatureProcessor()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get the raw data
        raw_data = self.data[idx]
        label = self.labels[idx]
        
        # Process the data with caching
        # Use the index as cache key
        processed_data = self.processor(raw_data.unsqueeze(0), cache_key=f"sample_{idx}")
        
        # Remove the batch dimension
        return processed_data.squeeze(0), label


def main():
    # Clear any existing cache
    clear_cache(cache_name="dataset_feature_processor")
    
    # Create the dataset
    dataset = CachedDataset(num_samples=10)
    
    print("\nFirst run - should process all samples:")
    # First run - process all samples
    start_time = time.time()
    for i in range(len(dataset)):
        data, label = dataset[i]
        print(f"Sample {i}: Processed data shape: {data.shape}, Label: {label}")
    first_run_time = time.time() - start_time
    print(f"\nFirst run total time: {first_run_time:.2f} seconds")
    
    print("\nSecond run - should use cache:")
    # Second run - should use cache
    start_time = time.time()
    for i in range(len(dataset)):
        data, label = dataset[i]
        print(f"Sample {i}: Processed data shape: {data.shape}, Label: {label}")
    second_run_time = time.time() - start_time
    print(f"\nSecond run total time: {second_run_time:.2f} seconds")
    
    # Calculate speedup
    speedup = first_run_time / second_run_time
    print(f"\nCache speedup: {speedup:.2f}x faster")


if __name__ == "__main__":
    main()
