from torch_module_cache.decorator import (
    cache_module,
    CacheLevel,
    clear_memory_caches,
    disable_global_cache,
    enable_global_cache,
    is_global_cache_enabled,
    enable_inference_mode,
    disable_inference_mode,
    is_inference_mode
)
from torch_module_cache.utils.cache_utils import (
    clear_cache,
    get_cache_size,
    list_cache_entries,
)

__all__ = [
    "cache_module",
    "CacheLevel",
    "clear_cache",
    "get_cache_size",
    "list_cache_entries",
    "clear_memory_caches",
    "disable_global_cache",
    "enable_global_cache",
    "is_global_cache_enabled",
    "enable_inference_mode",
    "disable_inference_mode",
    "is_inference_mode"
]
