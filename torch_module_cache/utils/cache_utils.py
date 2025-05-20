import os
import shutil
from pathlib import Path
from typing import Optional, Union

from torch_module_cache.decorator import clear_memory_caches

def clear_cache(cache_path: Optional[str] = None, cache_name: Optional[str] = None, clear_memory: bool = True):
    """
    Clear cached results for a specific module or all modules.
    
    Args:
        cache_path: Path to the cache directory. If None, uses default (~/.cache/torch-module-cache)
        cache_name: Name of the module cache to clear. If None, clears all caches.
        clear_memory: Whether to also clear in-memory caches
    """
    # Clear disk cache
    if cache_path is None:
        cache_dir = Path.home() / ".cache" / "torch-module-cache"
    else:
        cache_dir = Path(cache_path)
    
    if not cache_dir.exists():
        pass  # Nothing to clear on disk
    elif cache_name is None:
        # Clear all caches
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Clear specific module cache
        module_cache_dir = cache_dir / cache_name
        if module_cache_dir.exists():
            shutil.rmtree(module_cache_dir)
            module_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear memory cache if requested
    if clear_memory:
        clear_memory_caches(cache_name)

def get_cache_size(cache_path: Optional[str] = None, cache_name: Optional[str] = None) -> int:
    """
    Get the size of the cache in bytes.
    
    Args:
        cache_path: Path to the cache directory. If None, uses default (~/.cache/torch-module-cache)
        cache_name: Name of the module cache to measure. If None, measures all caches.
    
    Returns:
        Total size of the cache in bytes
    """
    if cache_path is None:
        cache_dir = Path.home() / ".cache" / "torch-module-cache"
    else:
        cache_dir = Path(cache_path)
    
    if not cache_dir.exists():
        return 0
    
    if cache_name is None:
        # Measure all caches
        target_dir = cache_dir
    else:
        # Measure specific module cache
        target_dir = cache_dir / cache_name
        if not target_dir.exists():
            return 0
    
    total_size = 0
    for dirpath, _, filenames in os.walk(target_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    
    return total_size

def list_cache_entries(cache_path: Optional[str] = None, cache_name: Optional[str] = None):
    """
    List all cache entries.
    
    Args:
        cache_path: Path to the cache directory. If None, uses default (~/.cache/torch-module-cache)
        cache_name: Name of the module cache to list. If None, lists all caches.
    
    Returns:
        A dictionary mapping module names to lists of cache file hashes
    """
    if cache_path is None:
        cache_dir = Path.home() / ".cache" / "torch-module-cache"
    else:
        cache_dir = Path(cache_path)
    
    if not cache_dir.exists():
        return {}
    
    result = {}
    
    if cache_name is None:
        # List all caches
        for module_dir in cache_dir.iterdir():
            if module_dir.is_dir():
                module_name = module_dir.name
                result[module_name] = []
                for cache_file in module_dir.glob("*.pt"):
                    result[module_name].append(cache_file.stem)
    else:
        # List specific module cache
        module_dir = cache_dir / cache_name
        if module_dir.exists():
            result[cache_name] = []
            for cache_file in module_dir.glob("*.pt"):
                result[cache_name].append(cache_file.stem)
    
    return result 