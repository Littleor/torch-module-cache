import functools
import hashlib
import os
import pickle
import shutil
import warnings
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn


class CacheLevel(Enum):
    DISK = auto()
    MEMORY = auto()


class SaveMode(Enum):
    """Cache serialization mode for controlling security vs. functionality."""

    WEIGHTS_ONLY = auto()  # More secure, limited to tensors and basic types
    FULL = auto()  # Less secure, supports all types but potential security risk


# Class-specific memory caches - accessible at module level
_CLASS_MEMORY_CACHES = {}


def clear_disk_caches(cache_name=None):
    """Clear disk caches.

    Args:
        cache_name: If provided, attempt to clear caches specifically for this name
    """
    cache_dir = Path.home() / ".cache" / "torch-module-cache"
    if cache_name is not None:
        cache_dir = cache_dir / cache_name
    if cache_dir.exists():
        # delete the directory
        shutil.rmtree(cache_dir)


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
    cache_path: Optional[str] = os.path.join(
        os.path.expanduser("~"), ".cache", "torch-module-cache"
    ),
    cache_name: Optional[str] = None,
    cache_level: CacheLevel = CacheLevel.DISK,
    safe_load: bool = True,
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
            raise TypeError(
                f"cache_module can only be applied to torch.nn.Module subclasses, got {cls}"
            )

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
            self._cached_device = kwargs.get("device", torch.device("cpu"))
            self._cached_dtype = kwargs.get("dtype", torch.float32)

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
            
            # Overrides for nn.Module methods that would affect device/dtype
            original_to = self.to
            @functools.wraps(original_to)
            def to_wrapper(*args, **kwargs):
                # Update cached device and dtype if present in args/kwargs
                if not self._model_initialized:
                    # Extract device from args/kwargs
                    if args and isinstance(args[0], (torch.device, str, int)):
                        self._cached_device = torch.device(args[0])
                    elif "device" in kwargs:
                        self._cached_device = torch.device(kwargs["device"])
                        
                    # Extract dtype from args/kwargs
                    dtype_arg_index = 1 if args and isinstance(args[0], (torch.device, str, int)) else 0
                    if len(args) > dtype_arg_index and isinstance(args[dtype_arg_index], torch.dtype):
                        self._cached_dtype = args[dtype_arg_index]
                    elif "dtype" in kwargs:
                        self._cached_dtype = kwargs["dtype"]
                    
                    # Just return self since the model isn't initialized yet
                    return self
                # If initialized, call the original to() method
                return original_to(*args, **kwargs)
            
            # Override type() method
            original_type = self.type
            @functools.wraps(original_type)
            def type_wrapper(dtype=None, *args, **kwargs):
                if not self._model_initialized and dtype is not None:
                    # Track the requested dtype
                    if isinstance(dtype, str):
                        # Convert string dtype to torch.dtype
                        dtype_map = {
                            'torch.FloatTensor': torch.float32,
                            'torch.DoubleTensor': torch.float64,
                            'torch.HalfTensor': torch.float16,
                            'torch.BFloat16Tensor': torch.bfloat16,
                            'torch.ByteTensor': torch.uint8,
                            'torch.CharTensor': torch.int8,
                            'torch.ShortTensor': torch.int16,
                            'torch.IntTensor': torch.int32,
                            'torch.LongTensor': torch.int64,
                            'torch.BoolTensor': torch.bool,
                        }
                        if dtype in dtype_map:
                            self._cached_dtype = dtype_map[dtype]
                        else:
                            # Try to interpret the string directly
                            try:
                                self._cached_dtype = getattr(torch, dtype.split('.')[-1])
                            except (AttributeError, IndexError):
                                warnings.warn(f"Could not interpret dtype string: {dtype}")
                    else:
                        self._cached_dtype = dtype
                    return self
                # If initialized, call original type() method
                return original_type(dtype, *args, **kwargs)
            
            # Override cuda() method
            original_cuda = self.cuda
            @functools.wraps(original_cuda)
            def cuda_wrapper(device=None, *args, **kwargs):
                if not self._model_initialized:
                    # Track the device change
                    if device is None:
                        self._cached_device = torch.device('cuda')
                    else:
                        self._cached_device = torch.device(f'cuda:{device}')
                    return self
                # If initialized, call original cuda() method
                return original_cuda(device, *args, **kwargs)
            
            # Override cpu() method
            original_cpu = self.cpu
            @functools.wraps(original_cpu)
            def cpu_wrapper(*args, **kwargs):
                if not self._model_initialized:
                    self._cached_device = torch.device('cpu')
                    return self
                # If initialized, call original cpu() method
                return original_cpu(*args, **kwargs)
            
            # Assign overridden methods
            self.to = to_wrapper
            self.type = type_wrapper
            self.cuda = cuda_wrapper
            self.cpu = cpu_wrapper
            
            # Save original methods to restore after initialization
            self._original_methods = {
                'to': original_to,
                'type': original_type,
                'cuda': original_cuda,
                'cpu': original_cpu,
            }

        @functools.wraps(original_forward)
        def forward_wrapper(self, *args, **kwargs):
            # Get the cache_key from kwargs
            cache_key = kwargs.get("cache_key", None)

            # If no cache_key or caching is disabled
            if not self._cache_enabled or cache_key is None:
                # Initialize if needed and forward
                if not self._model_initialized:
                    self._initialize_model()
                # Pass through to original forward method
                return original_forward(self, *args, **kwargs)

            # Initialize model if not already initialized
            if not self._model_initialized:
                self._initialize_model()

            # Convert single cache_key to list for unified handling
            is_single_key = not isinstance(cache_key, (list, tuple))
            cache_keys = [cache_key] if is_single_key else cache_key
            batch_size = len(cache_keys)

            # Create a copy of kwargs without the cache_key
            forward_kwargs = {k: v for k, v in kwargs.items() if k != "cache_key"}

            # Check if we need batch processing
            if batch_size == 1:
                # Single key processing
                result = self._check_and_process_single(
                    cache_keys[0], *args, **forward_kwargs
                )
                return result
            else:
                # Batch processing
                results, cache_info = self._check_cache_batch(cache_keys)

                # If all results are in cache, return them
                if len(cache_info["missed_indices"]) == 0:
                    # Return as single item or list based on input type
                    if is_single_key:
                        return results[0]
                    else:
                        return self._combine_results(results)

                # Run the forward pass once for the entire batch
                all_results = original_forward(self, *args, **forward_kwargs)

                # Split the results and cache them individually
                if (
                    batch_size > 1
                    and isinstance(all_results, torch.Tensor)
                    and all_results.size(0) == batch_size
                ):
                    # Batch tensor result case - cache each item individually
                    for i, key in enumerate(cache_keys):
                        item_result = all_results[i : i + 1]  # Preserve the dimension
                        if i in cache_info["missed_indices"]:
                            # Cache this result and update our results list
                            self._cache_result(key, item_result)
                            results[i] = item_result
                else:
                    # Non-batchable result case
                    warnings.warn(
                        "Model returned a result that doesn't appear to be batched. "
                        "Caching the entire result for each key, which may not be intended."
                    )
                    for i, key in enumerate(cache_keys):
                        if i in cache_info["missed_indices"]:
                            self._cache_result(key, all_results)
                            results[i] = all_results

                # Return the combined results based on the input type
                if is_single_key:
                    return results[0]
                else:
                    return self._combine_results(results)

        def _check_and_process_single(self, cache_key, *args, **kwargs):
            """Check cache for a single key and process if needed"""
            # Generate a hash from the cache key for filename
            cache_hash = hashlib.md5(str(cache_key).encode()).hexdigest()
            cache_file = self._cache_subdir / f"{cache_hash}.pt"

            # Check memory cache first if memory caching is enabled
            if (
                self._cache_level == CacheLevel.MEMORY
                and cache_key in self._memory_cache
            ):
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
                                weights_only=True,  # Safer loading option
                            )
                        except TypeError:
                            # Fallback for older PyTorch versions that don't support weights_only
                            warnings.warn(
                                "Your PyTorch version doesn't support 'weights_only' parameter. "
                                "Using default loading method, which could have security implications."
                            )
                            result = torch.load(
                                cache_file, map_location=self._cached_device
                            )
                    else:
                        # Use standard loading method if safe_load is False
                        result = torch.load(
                            cache_file, map_location=self._cached_device
                        )

                    # Validate the loaded result
                    _validate_cache_result(result)

                    # Also store in memory if memory caching is enabled
                    if self._cache_level == CacheLevel.MEMORY:
                        self._memory_cache[cache_key] = result

                    return result
                except Exception as e:
                    warnings.warn(f"Failed to load cache from {cache_file}: {e}")

            # Cache miss - run the forward pass
            # Create new kwargs without the cache_key to avoid passing it to the model
            # (forward_kwargs should already be clean, but we'll double-check)
            forward_kwargs = {k: v for k, v in kwargs.items() if k != "cache_key"}
            result = original_forward(self, *args, **forward_kwargs)

            # Validate and cache the result
            self._cache_result(cache_key, result)

            return result

        def _check_cache_batch(self, cache_keys):
            """Check cache for a batch of keys and return results and cache info"""
            batch_size = len(cache_keys)
            results = [None] * batch_size
            missed_indices = []

            # Check cache for each item in the batch
            for i, key in enumerate(cache_keys):
                cache_hit = False

                # Check memory cache first
                if self._cache_level == CacheLevel.MEMORY and key in self._memory_cache:
                    results[i] = self._memory_cache[key]
                    cache_hit = True

                # If not in memory, check disk cache
                if not cache_hit and self._cache_enabled:
                    cache_hash = hashlib.md5(str(key).encode()).hexdigest()
                    cache_file = self._cache_subdir / f"{cache_hash}.pt"

                    if cache_file.exists():
                        try:
                            # Load from disk with appropriate security settings
                            if self._safe_load:
                                try:
                                    result = torch.load(
                                        cache_file,
                                        map_location=self._cached_device,
                                        weights_only=True,
                                    )
                                except TypeError:
                                    result = torch.load(
                                        cache_file, map_location=self._cached_device
                                    )
                            else:
                                result = torch.load(
                                    cache_file, map_location=self._cached_device
                                )

                            # Validate the loaded result
                            _validate_cache_result(result)

                            # Store result
                            results[i] = result

                            # Also store in memory if memory caching is enabled
                            if self._cache_level == CacheLevel.MEMORY:
                                self._memory_cache[key] = result

                            cache_hit = True
                        except Exception as e:
                            warnings.warn(
                                f"Failed to load cache for key {key} from {cache_file}: {e}"
                            )

                # If cache miss, add to list of indices to process
                if not cache_hit:
                    missed_indices.append(i)

            return results, {"missed_indices": missed_indices}

        def _cache_result(self, cache_key, result):
            """Validate and cache a result for a given key"""
            # Validate the result
            _validate_cache_result(result)

            # Cache the result if enabled
            if self._cache_enabled:
                try:
                    # For dictionaries, ensure all keys are strings for better compatibility
                    if isinstance(result, dict):
                        for k in result.keys():
                            if not isinstance(k, str):
                                warnings.warn(
                                    f"Non-string dict key detected in cached result: {type(k)}. "
                                    f"Consider using string keys for better compatibility."
                                )

                    cache_hash = hashlib.md5(str(cache_key).encode()).hexdigest()
                    cache_file = self._cache_subdir / f"{cache_hash}.pt"
                    torch.save(result, cache_file)

                    # Also store in memory if memory caching is enabled
                    if self._cache_level == CacheLevel.MEMORY:
                        self._memory_cache[cache_key] = result

                except Exception as e:
                    warnings.warn(
                        f"Failed to save cache for key {cache_key} to {cache_file}: {e}"
                    )

        def _combine_results(self, results):
            """Combine a list of results into a batch if possible"""
            # Check if all results are tensors that can be combined
            if all(isinstance(r, torch.Tensor) for r in results):
                # If all results are tensors with the same shape (except batch dim)
                # and all have a batch dimension
                first_shape = results[0].shape
                if all(len(r.shape) > 0 for r in results) and all(
                    r.shape[1:] == first_shape[1:] for r in results
                ):
                    return torch.cat(results, dim=0)

            # For mixed result types or non-batchable tensors, return the list
            return results

        def _initialize_model(self):
            """Initialize the model with stored args and kwargs"""
            # Save the currently registered hooks and buffers before re-initialization
            # to prevent them from being lost when original_init is called
            old_state_dict = {
                k: v
                for k, v in self.__dict__.items()
                if k.startswith("_")
                and not k
                in [
                    "_model_initialized",
                    "_init_args",
                    "_init_kwargs",
                    "_cached_device",
                    "_cached_dtype",
                    "_cache_level",
                    "_cache_enabled",
                    "_cache_dir",
                    "_cache_subdir",
                    "_memory_cache",
                    "_safe_load",
                    "_original_methods",
                ]
            }

            # Print initialization information
            print(f"Initializing model: {self.__class__.__name__}")

            # Call the original init
            original_init(self, *self._init_args, **self._init_kwargs)

            # Restore any PyTorch Module internal state
            for k, v in old_state_dict.items():
                if not hasattr(self, k):
                    setattr(self, k, v)

            # Restore original methods
            for method_name, original_method in getattr(self, "_original_methods", {}).items():
                setattr(self, method_name, original_method)

            self._model_initialized = True

            # Move model to the correct device and dtype
            self.to(device=self._cached_device, dtype=self._cached_dtype)

        cls.__init__ = init_wrapper
        cls.forward = forward_wrapper
        cls._check_and_process_single = _check_and_process_single
        cls._check_cache_batch = _check_cache_batch
        cls._cache_result = _cache_result
        cls._combine_results = _combine_results
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
        warnings.warn(
            f"Caching custom object of type {type(result)}. "
            f"Ensure it can be properly serialized with torch.save."
        )
    else:
        # For other types, issue a warning
        warnings.warn(
            f"Unsupported cache result type: {type(result)}. "
            f"This may cause serialization issues."
        )

    # Check for nested containers
    if isinstance(result, (list, tuple)):
        for item in result:
            _validate_cache_result(item)
    elif isinstance(result, dict):
        for k, v in result.items():
            # Check dictionary keys
            if not isinstance(k, str):
                warnings.warn(
                    f"Non-string dictionary key {k} of type {type(k)} may cause "
                    f"serialization issues. Consider using string keys."
                )
            # Check dictionary values
            _validate_cache_result(v)
