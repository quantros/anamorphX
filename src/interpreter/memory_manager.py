"""Memory Manager for AnamorphX Stage 3.2"""
"""
Memory Manager for Anamorph Language.

This module implements memory management, garbage collection, and resource
monitoring for the Anamorph interpreter.
"""

import gc
import psutil
import time
import threading
import weakref
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Generic
from enum import Enum, auto
import logging
from collections import defaultdict, deque

T = TypeVar('T')


class MemoryState(Enum):
    """Memory manager state."""
    NORMAL = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_allocated: int = 0
    total_freed: int = 0
    current_usage: int = 0
    peak_usage: int = 0
    gc_collections: int = 0
    gc_time: float = 0.0
    object_count: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    
    @property
    def allocation_rate(self) -> float:
        """Get allocation rate in bytes per second."""
        return self.total_allocated / max(time.time(), 1)
    
    @property
    def pool_hit_rate(self) -> float:
        """Get object pool hit rate."""
        total = self.pool_hits + self.pool_misses
        return self.pool_hits / max(total, 1)


@dataclass
class ResourceLimits:
    """Resource usage limits."""
    max_memory: int = 512 * 1024 * 1024  # 512MB
    max_objects: int = 1000000
    max_execution_time: float = 300.0  # 5 minutes
    gc_threshold: float = 0.8  # Trigger GC at 80% memory usage
    emergency_threshold: float = 0.95  # Emergency cleanup at 95%


class ObjectPool(Generic[T]):
    """
    Object pool for reusing objects to reduce allocation overhead.
    """
    
    def __init__(self, factory: callable, max_size: int = 1000):
        """Initialize object pool."""
        self.factory = factory
        self.max_size = max_size
        self.pool: deque = deque()
        self.created_count = 0
        self.reused_count = 0
        self._lock = threading.Lock()
    
    def acquire(self) -> T:
        """Acquire an object from the pool."""
        with self._lock:
            if self.pool:
                self.reused_count += 1
                return self.pool.popleft()
            else:
                self.created_count += 1
                return self.factory()
    
    def release(self, obj: T):
        """Release an object back to the pool."""
        with self._lock:
            if len(self.pool) < self.max_size:
                # Reset object state if it has a reset method
                if hasattr(obj, 'reset'):
                    obj.reset()
                self.pool.append(obj)
    
    def clear(self):
        """Clear the pool."""
        with self._lock:
            self.pool.clear()
    
    @property
    def size(self) -> int:
        """Get current pool size."""
        return len(self.pool)
    
    @property
    def hit_rate(self) -> float:
        """Get pool hit rate."""
        total = self.created_count + self.reused_count
        return self.reused_count / max(total, 1)


class GarbageCollector:
    """
    Enhanced garbage collector with monitoring and tuning.
    """
    
    def __init__(self):
        """Initialize garbage collector."""
        self.enabled = True
        self.collections = 0
        self.total_time = 0.0
        self.last_collection = time.time()
        self.collection_interval = 10.0  # seconds
        self._lock = threading.Lock()
        
        # Configure Python GC
        gc.set_threshold(700, 10, 10)
        
    def collect(self, generation: Optional[int] = None) -> int:
        """Perform garbage collection."""
        if not self.enabled:
            return 0
        
        with self._lock:
            start_time = time.time()
            
            if generation is not None:
                collected = gc.collect(generation)
            else:
                collected = gc.collect()
            
            collection_time = time.time() - start_time
            self.collections += 1
            self.total_time += collection_time
            self.last_collection = time.time()
            
            return collected
    
    def should_collect(self) -> bool:
        """Check if garbage collection should be triggered."""
        if not self.enabled:
            return False
        
        # Time-based collection
        if time.time() - self.last_collection > self.collection_interval:
            return True
        
        # Memory pressure-based collection
        stats = gc.get_stats()
        if stats and stats[0]['collections'] > 100:
            return True
        
        return False
    
    def auto_collect(self):
        """Automatically collect if needed."""
        if self.should_collect():
            self.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics."""
        return {
            'collections': self.collections,
            'total_time': self.total_time,
            'average_time': self.total_time / max(self.collections, 1),
            'last_collection': self.last_collection,
            'gc_stats': gc.get_stats(),
            'gc_counts': gc.get_count()
        }


class MemoryProfiler:
    """
    Memory profiler for tracking allocation patterns and hotspots.
    """
    
    def __init__(self):
        """Initialize memory profiler."""
        self.enabled = False
        self.allocations: Dict[str, int] = defaultdict(int)
        self.allocation_sizes: Dict[str, int] = defaultdict(int)
        self.allocation_times: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def record_allocation(self, obj_type: str, size: int):
        """Record an allocation."""
        if not self.enabled:
            return
        
        with self._lock:
            self.allocations[obj_type] += 1
            self.allocation_sizes[obj_type] += size
            self.allocation_times[obj_type].append(time.time())
    
    def get_hotspots(self, top_n: int = 10) -> List[tuple]:
        """Get top allocation hotspots."""
        with self._lock:
            sorted_allocs = sorted(
                self.allocations.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_allocs[:top_n]
    
    def get_memory_usage_by_type(self) -> Dict[str, int]:
        """Get memory usage by object type."""
        with self._lock:
            return dict(self.allocation_sizes)
    
    def reset(self):
        """Reset profiler statistics."""
        with self._lock:
            self.allocations.clear()
            self.allocation_sizes.clear()
            self.allocation_times.clear()


class MemoryManager:
    """
    Comprehensive memory manager for the Anamorph interpreter.
    
    Provides memory allocation tracking, garbage collection, object pooling,
    and resource limit enforcement.
    """
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        """Initialize memory manager."""
        self.limits = limits or ResourceLimits()
        self.stats = MemoryStats()
        self.state = MemoryState.NORMAL
        
        # Components
        self.gc = GarbageCollector()
        self.profiler = MemoryProfiler()
        self.pools: Dict[str, ObjectPool] = {}
        
        # Tracking
        self.tracked_objects: Set[weakref.ref] = set()
        self.allocation_history: deque = deque(maxlen=1000)
        
        # Threading
        self._lock = threading.Lock()
        self._monitor_thread = None
        self._monitoring = False
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Start monitoring
        self.start_monitoring()
    
    def allocate(self, obj_type: Type[T], *args, **kwargs) -> T:
        """Allocate an object with tracking."""
        # Try object pool first
        pool_key = obj_type.__name__
        if pool_key in self.pools:
            obj = self.pools[pool_key].acquire()
            self.stats.pool_hits += 1
            return obj
        
        # Create new object
        obj = obj_type(*args, **kwargs)
        self.stats.pool_misses += 1
        
        # Track allocation
        self._track_allocation(obj)
        
        return obj
    
    def deallocate(self, obj: Any):
        """Deallocate an object."""
        obj_type = type(obj).__name__
        
        # Try to return to pool
        if obj_type in self.pools:
            self.pools[obj_type].release(obj)
        
        # Track deallocation
        self._track_deallocation(obj)
    
    def create_pool(self, obj_type: Type[T], factory: callable, max_size: int = 1000):
        """Create an object pool for a specific type."""
        pool_key = obj_type.__name__
        self.pools[pool_key] = ObjectPool(factory, max_size)
        self.logger.info(f"Created object pool for {pool_key} with max size {max_size}")
    
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        process = psutil.Process()
        return process.memory_info().rss
    
    def check_limits(self):
        """Check if resource limits are exceeded."""
        current_memory = self.get_memory_usage()
        
        # Update stats
        self.stats.current_usage = current_memory
        if current_memory > self.stats.peak_usage:
            self.stats.peak_usage = current_memory
        
        # Check memory limit
        memory_ratio = current_memory / self.limits.max_memory
        
        if memory_ratio >= self.limits.emergency_threshold:
            self.state = MemoryState.EMERGENCY
            self._emergency_cleanup()
        elif memory_ratio >= self.limits.gc_threshold:
            self.state = MemoryState.CRITICAL
            self.gc.collect()
        elif memory_ratio >= 0.6:
            self.state = MemoryState.WARNING
        else:
            self.state = MemoryState.NORMAL
        
        # Check object count
        if self.stats.object_count > self.limits.max_objects:
            self.logger.warning(f"Object count limit exceeded: {self.stats.object_count}")
            self.gc.collect()
    
    def _track_allocation(self, obj: Any):
        """Track object allocation."""
        with self._lock:
            # Add weak reference
            ref = weakref.ref(obj, self._object_finalized)
            self.tracked_objects.add(ref)
            
            # Update stats
            obj_size = self._estimate_size(obj)
            self.stats.total_allocated += obj_size
            self.stats.current_usage += obj_size
            self.stats.object_count += 1
            
            # Record in profiler
            self.profiler.record_allocation(type(obj).__name__, obj_size)
            
            # Add to history
            self.allocation_history.append({
                'type': type(obj).__name__,
                'size': obj_size,
                'time': time.time()
            })
    
    def _track_deallocation(self, obj: Any):
        """Track object deallocation."""
        with self._lock:
            obj_size = self._estimate_size(obj)
            self.stats.total_freed += obj_size
            self.stats.current_usage -= obj_size
            self.stats.object_count -= 1
    
    def _object_finalized(self, ref: weakref.ref):
        """Called when a tracked object is finalized."""
        with self._lock:
            self.tracked_objects.discard(ref)
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            import sys
            return sys.getsizeof(obj)
        except:
            return 64  # Default estimate
    
    def _emergency_cleanup(self):
        """Perform emergency memory cleanup."""
        self.logger.warning("Emergency memory cleanup triggered")
        
        # Clear object pools
        for pool in self.pools.values():
            pool.clear()
        
        # Force garbage collection
        self.gc.collect()
        self.gc.collect(0)
        self.gc.collect(1)
        self.gc.collect(2)
        
        # Clear allocation history
        self.allocation_history.clear()
    
    def start_monitoring(self):
        """Start memory monitoring thread."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        self.logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Memory monitoring loop."""
        while self._monitoring:
            try:
                self.check_limits()
                self.gc.auto_collect()
                time.sleep(1.0)  # Check every second
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                time.sleep(5.0)  # Back off on error
    
    def get_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        # Update current usage
        self.stats.current_usage = self.get_memory_usage()
        return self.stats
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed memory statistics."""
        return {
            'memory_stats': self.get_stats(),
            'gc_stats': self.gc.get_stats(),
            'pool_stats': {
                name: {
                    'size': pool.size,
                    'created': pool.created_count,
                    'reused': pool.reused_count,
                    'hit_rate': pool.hit_rate
                }
                for name, pool in self.pools.items()
            },
            'profiler_hotspots': self.profiler.get_hotspots(),
            'state': self.state.name,
            'tracked_objects': len(self.tracked_objects)
        }
    
    def cleanup(self):
        """Cleanup memory manager resources."""
        self.stop_monitoring()
        
        # Clear pools
        for pool in self.pools.values():
            pool.clear()
        
        # Final garbage collection
        self.gc.collect()
        
        self.logger.info("Memory manager cleanup completed")
    
    def __del__(self):
        """Destructor."""
        self.cleanup() 