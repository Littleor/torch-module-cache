import os
import sys
import unittest
import tempfile
import shutil
import torch
import torch.nn as nn

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch_module_cache import cache_module, CacheLevel, clear_cache, list_cache_entries

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x, cache_key=None):
        return self.linear(x)

class TestCacheModule(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for testing
        self.test_cache_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.test_cache_dir)
    
    def test_default_cache(self):
        """Test the default caching behavior"""
        @cache_module()
        class TestDefaultCache(TestModel):
            pass
        
        model = TestDefaultCache()
        self.assertFalse(model._model_initialized)
        
        # Run with a cache key
        x = torch.randn(1, 10)
        out1 = model(x, cache_key="test1")
        self.assertTrue(model._model_initialized)
        
        # Run again with same cache key
        out2 = model(x, cache_key="test1")
        self.assertTrue(torch.allclose(out1, out2))
    
    def test_custom_cache_path(self):
        """Test using a custom cache path"""
        @cache_module(cache_path=self.test_cache_dir)
        class TestCustomPath(TestModel):
            pass
        
        model = TestCustomPath()
        
        # Run with a cache key
        x = torch.randn(1, 10)
        model(x, cache_key="test_custom_path")
        
        # Check that the cache file was created
        cache_entries = list_cache_entries(self.test_cache_dir)
        self.assertIn("TestCustomPath", cache_entries)
        self.assertEqual(len(cache_entries["TestCustomPath"]), 1)
    
    def test_disable_cache(self):
        """Test disabling the cache"""
        @cache_module(cache_path=None)
        class TestNoCache(TestModel):
            pass
        
        model = TestNoCache()
        
        # Run with a cache key
        x = torch.randn(1, 10)
        out1 = model(x, cache_key="test_no_cache")
        
        # Run again with same cache key (should use same model, not reinitialize)
        out2 = model(x, cache_key="test_no_cache")
        
        # Results should be identical because the model is only initialized once
        # but with caching disabled, it should recompute each time
        self.assertTrue(torch.allclose(out1, out2))
        
        # But running with a different input should produce different results
        y = torch.randn(1, 10)
        out3 = model(y, cache_key="test_no_cache")
        
        # Results should be different with different inputs
        self.assertFalse(torch.allclose(out1, out3))
    
    def test_memory_cache(self):
        """Test memory-level caching"""
        @cache_module(cache_level=CacheLevel.MEMORY)
        class TestMemoryCache(TestModel):
            pass
        
        model = TestMemoryCache()
        
        # Run with a cache key
        x = torch.randn(1, 10)
        out1 = model(x, cache_key="test_memory")
        
        # Run again with same cache key
        out2 = model(x, cache_key="test_memory")
        
        # Should be identical
        self.assertTrue(torch.all(out1 == out2))

if __name__ == "__main__":
    unittest.main() 