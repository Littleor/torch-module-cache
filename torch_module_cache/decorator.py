import functools
import os
import pickle
import hashlib
import torch
import torch.nn as nn
import warnings
from pathlib import Path
from enum import Enum, auto
from typing import Optional, Callable, Any, Dict, Union, List, Tuple, Type

class CacheLevel(Enum):
    DISK = auto()
    MEMORY = auto()

class SaveMode(Enum):
    """Cache serialization mode for controlling security vs. functionality."""
    WEIGHTS_ONLY = auto()  # More secure, limited to tensors and basic types
    FULL = auto()          # Less secure, supports all types but potential security risk

# Class-specific memory caches - accessible at module level
_CLASS_MEMORY_CACHES = {}

def clear_memory_caches(cache_name=None):
    """Clear in-memory caches.
    
    Args:
        cache_name: If provided, attempt to clear caches specifically for this name
                   (note: this is a best-effort match, as cache_name doesn't directly map to class ID)
    """
    # Currently we just clear all memory caches regardless of cache_name
    # In a more sophisticated implementation, we could track which cache_name maps to which class ID
    for cache_dict in _CLASS_MEMORY_CACHES.values():
        cache_dict.clear()

def cache_module(
    cache_path: Optional[str] = os.path.join(os.path.expanduser("~"), ".cache", "torch-module-cache"),
    cache_name: Optional[str] = None,
    cache_level: CacheLevel = CacheLevel.DISK,
    safe_load: bool = True
):
    """
    Decorator for PyTorch modules to add caching functionality.
    
    Args:
        cache_path: Path to store cache files. If None, caching is disabled.
                    If not specified, defaults to ~/.cache/torch-module-cache
        cache_name: Name for the cache subfolder. If not specified, uses the module class name
        cache_level: Level of caching (DISK or MEMORY)
        safe_load: If True, uses safer loading options for torch.load to mitigate security risks
        
    Returns:
        Cached module results which can be of various types:
        - torch.Tensor: Single tensor results
        - List: Lists containing tensors or other serializable objects
        - Tuple: Tuples containing tensors or other serializable objects
        - Dict: Dictionaries with string keys and tensor/serializable values
        - Any other pickle-serializable Python object
    """
    def decorator(cls):
        if not issubclass(cls, nn.Module):
            raise TypeError(f"cache_module can only be applied to torch.nn.Module subclasses, got {cls}")
        
        # Create a class-specific memory cache
        cache_id = id(cls)
        if cache_id not in _CLASS_MEMORY_CACHES:
            _CLASS_MEMORY_CACHES[cache_id] = {}
        
        original_init = cls.__init__
        original_forward = cls.forward
        
        @functools.wraps(original_init)
        def init_wrapper(self, *args, **kwargs):
            # Call nn.Module.__init__ to set up basic module attributes
            nn.Module.__init__(self)
            
            # Set initialization flag
            self._model_initialized = False
            self._init_args = args
            self._init_kwargs = kwargs
            
            # Store device and dtype information
            self._cached_device = kwargs.get('device', torch.device('cpu'))
            self._cached_dtype = kwargs.get('dtype', torch.float32)
            
            # Store reference to class memory cache
            self._memory_cache = _CLASS_MEMORY_CACHES[cache_id]
            
            # Set up cache paths and settings
            self._cache_level = cache_level
            self._safe_load = safe_load
            
            if cache_path is None:
                self._cache_enabled = False
                self._cache_dir = None
            else:
                self._cache_enabled = True
                if cache_path == "":  # Use default
                    self._cache_dir = Path.home() / ".cache" / "torch-module-cache"
                else:
                    self._cache_dir = Path(cache_path)
                
                # Set up cache name subdirectory
                if cache_name is None:
                    self._cache_subdir = self._cache_dir / cls.__name__
                else:
                    self._cache_subdir = self._cache_dir / cache_name
                
                # Create cache directory if it doesn't exist
                if not self._cache_subdir.exists():
                    self._cache_subdir.mkdir(parents=True, exist_ok=True)
        
        @functools.wraps(original_forward)
        def forward_wrapper(self, *args, **kwargs):
            # Get the cache_key directly from kwargs
            cache_key = kwargs.get('cache_key', None)
            
            if not self._cache_enabled or cache_key is None:
                # If caching is disabled or no cache key provided, initialize if needed and forward
                if not self._model_initialized:
                    self._initialize_model()
                # Pass through to original forward method
                return original_forward(self, *args, **kwargs)
            
            # Generate a hash from the cache key for filename
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
            cache_file = self._cache_subdir / f"{cache_hash}.pt"
            
            # Check memory cache first if memory caching is enabled
            if self._cache_level == CacheLevel.MEMORY and cache_key in self._memory_cache:
                return self._memory_cache[cache_key]
            
            # Check disk cache if applicable
            if self._cache_enabled and cache_file.exists():
                try:
                    # Load the cached result with appropriate security settings
                    if self._safe_load:
                        try:
                            # Use weights_only=True if this version of PyTorch supports it
                            result = torch.load(
                                cache_file, 
                                map_location=self._cached_device,
                                weights_only=True  # Safer loading option
                            )
                        except TypeError:
                            # Fallback for older PyTorch versions that don't support weights_only
                            warnings.warn("Your PyTorch version doesn't support 'weights_only' parameter. "
                                          "Using default loading method, which could have security implications.")
                            result = torch.load(cache_file, map_location=self._cached_device)
                    else:
                        # Use standard loading method if safe_load is False
                        result = torch.load(cache_file, map_location=self._cached_device)
                    
                    # Validate the loaded result
                    _validate_cache_result(result)
                    
                    # Also store in memory if memory caching is enabled
                    if self._cache_level == CacheLevel.MEMORY:
                        self._memory_cache[cache_key] = result
                    
                    return result
                except Exception as e:
                    warnings.warn(f"Failed to load cache from {cache_file}: {e}")
            
            # Cache miss - initialize model if needed
            if not self._model_initialized:
                self._initialize_model()
            
            # Run the forward pass (pass kwargs unchanged, it includes cache_key)
            result = original_forward(self, *args, **kwargs)
            
            # Validate the result
            _validate_cache_result(result)
            
            # Cache the result
            if self._cache_enabled:
                try:
                    # For dictionaries, ensure all keys are strings for better compatibility
                    if isinstance(result, dict):
                        for k in result.keys():
                            if not isinstance(k, str):
                                warnings.warn(f"Non-string dict key detected in cached result: {type(k)}. " 
                                             f"Consider using string keys for better compatibility.")
                    
                    torch.save(result, cache_file)
                    
                    # Also store in memory if memory caching is enabled
                    if self._cache_level == CacheLevel.MEMORY:
                        self._memory_cache[cache_key] = result
                        
                except Exception as e:
                    warnings.warn(f"Failed to save cache to {cache_file}: {e}")
            
            return result
        
        def _initialize_model(self):
            """Initialize the model with stored args and kwargs"""
            # Save the currently registered hooks and buffers before re-initialization
            # to prevent them from being lost when original_init is called
            old_state_dict = {
                k: v for k, v in self.__dict__.items() 
                if k.startswith('_') and not k in ['_model_initialized', '_init_args', '_init_kwargs',
                                                 '_cached_device', '_cached_dtype', '_cache_level',
                                                 '_cache_enabled', '_cache_dir', '_cache_subdir',
                                                 '_memory_cache', '_safe_load']
            }
            
            # Call the original init
            original_init(self, *self._init_args, **self._init_kwargs)
            
            # Restore any PyTorch Module internal state
            for k, v in old_state_dict.items():
                if not hasattr(self, k):
                    setattr(self, k, v)
                    
            self._model_initialized = True
            
            # Move model to the correct device and dtype
            self.to(device=self._cached_device, dtype=self._cached_dtype)
        
        cls.__init__ = init_wrapper
        cls.forward = forward_wrapper
        cls._initialize_model = _initialize_model
        
        return cls
    
    return decorator 

def _validate_cache_result(result):
    """
    Validate that the result is of a supported type for caching.
    Raises a warning if there might be serialization issues.
    
    Args:
        result: The result to validate
    """
    if result is None:
        return
        
    if isinstance(result, (torch.Tensor, list, tuple, dict, str, int, float, bool)):
        # These types are directly supported
        pass
    elif hasattr(result, "__dict__"):
        # Custom objects with __dict__ attribute can be pickled
        # but might have issues with torch.save/load
        warnings.warn(f"Caching custom object of type {type(result)}. "
                      f"Ensure it can be properly serialized with torch.save.")
    else:
        # For other types, issue a warning
        warnings.warn(f"Unsupported cache result type: {type(result)}. "
                      f"This may cause serialization issues.")
    
    # Check for nested containers
    if isinstance(result, (list, tuple)):
        for item in result:
            _validate_cache_result(item)
    elif isinstance(result, dict):
        for k, v in result.items():
            # Check dictionary keys
            if not isinstance(k, str):
                warnings.warn(f"Non-string dictionary key {k} of type {type(k)} may cause "
                             f"serialization issues. Consider using string keys.")
            # Check dictionary values
            _validate_cache_result(v) 