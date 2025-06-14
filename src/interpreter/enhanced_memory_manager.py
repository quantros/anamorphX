"""
Enhanced Memory Manager for AnamorphX Stage 3.2.

This module implements advanced memory management with full integration
into the AnamorphX ecosystem, including signal events, configurable limits,
trend analysis, and async monitoring.
"""

import gc
import psutil
import time
import threading
import weakref
import asyncio
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Generic, Callable, Union
from enum import Enum, auto
import logging
from collections import defaultdict, deque
from pathlib import Path
import yaml

T = TypeVar('T')


class MemoryState(Enum):
    """Memory manager state."""
    NORMAL = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


class MemoryEventType(Enum):
    """Types of memory events."""
    ALLOCATION = "allocation"
    DEALLOCATION = "deallocation"
    GC_TRIGGERED = "gc_triggered"
    LIMIT_EXCEEDED = "limit_exceeded"
    EMERGENCY_CLEANUP = "emergency_cleanup"
    POOL_CREATED = "pool_created"
    TREND_ALERT = "trend_alert"


@dataclass
class MemoryEvent:
    """Memory event for signal system."""
    event_type: MemoryEventType
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, error, critical
    
    def to_signal_data(self) -> Dict[str, Any]:
        """Convert to signal data format."""
        return {
            'type': 'memory_event',
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'severity': self.severity,
            'data': self.data
        }


@dataclass
class MemoryStats:
    """Enhanced memory usage statistics."""
    total_allocated: int = 0
    total_freed: int = 0
    current_usage: int = 0
    peak_usage: int = 0
    gc_collections: int = 0
    gc_time: float = 0.0
    object_count: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    
    # New trend analysis fields
    allocation_trend: List[float] = field(default_factory=list)
    usage_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_trend_analysis: float = 0.0
    
    @property
    def allocation_rate(self) -> float:
        """Get allocation rate in bytes per second."""
        return self.total_allocated / max(time.time(), 1)
    
    @property
    def pool_hit_rate(self) -> float:
        """Get object pool hit rate."""
        total = self.pool_hits + self.pool_misses
        return self.pool_hits / max(total, 1)
    
    @property
    def memory_efficiency(self) -> float:
        """Get memory efficiency ratio."""
        if self.total_allocated == 0:
            return 1.0
        return (self.total_allocated - self.current_usage) / self.total_allocated
    
    def add_usage_sample(self, usage: int):
        """Add usage sample for trend analysis."""
        self.usage_history.append({
            'timestamp': time.time(),
            'usage': usage,
            'objects': self.object_count
        })


@dataclass
class ResourceLimits:
    """Configurable resource usage limits."""
    max_memory: int = 512 * 1024 * 1024  # 512MB
    max_objects: int = 1000000
    max_execution_time: float = 300.0  # 5 minutes
    gc_threshold: float = 0.8  # Trigger GC at 80% memory usage
    emergency_threshold: float = 0.95  # Emergency cleanup at 95%
    
    # New configurable parameters
    monitoring_interval: float = 1.0  # seconds
    trend_analysis_interval: float = 30.0  # seconds
    pool_max_size_default: int = 1000
    gc_generation_thresholds: tuple = (700, 10, 10)
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> 'ResourceLimits':
        """Load limits from configuration file."""
        config_path = Path(config_path)
        
        if config_path.suffix.lower() == '.yaml':
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        return cls(**config.get('memory_limits', {}))
    
    def save_config(self, config_path: Union[str, Path]):
        """Save limits to configuration file."""
        config_path = Path(config_path)
        config = {'memory_limits': asdict(self)}
        
        if config_path.suffix.lower() == '.yaml':
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)


class EnhancedObjectPool(Generic[T]):
    """
    Enhanced object pool with error handling and monitoring.
    """
    
    def __init__(self, factory: Callable[[], T], max_size: int = 1000, 
                 name: str = "unnamed"):
        """Initialize enhanced object pool."""
        self.factory = factory
        self.max_size = max_size
        self.name = name
        self.pool: deque = deque()
        self.created_count = 0
        self.reused_count = 0
        self.error_count = 0
        self.last_error: Optional[Exception] = None
        self._lock = threading.Lock()
        
        # Monitoring
        self.creation_times: deque = deque(maxlen=100)
        self.usage_stats: Dict[str, int] = defaultdict(int)
    
    def acquire(self) -> Optional[T]:
        """Acquire an object from the pool with error handling."""
        with self._lock:
            if self.pool:
                obj = self.pool.popleft()
                self.reused_count += 1
                self.usage_stats['reused'] += 1
                return obj
            else:
                try:
                    start_time = time.time()
                    obj = self.factory()
                    creation_time = time.time() - start_time
                    
                    self.created_count += 1
                    self.usage_stats['created'] += 1
                    self.creation_times.append(creation_time)
                    
                    return obj
                except Exception as e:
                    self.error_count += 1
                    self.last_error = e
                    self.usage_stats['errors'] += 1
                    return None
    
    def release(self, obj: T) -> bool:
        """Release an object back to the pool."""
        if obj is None:
            return False
            
        with self._lock:
            if len(self.pool) < self.max_size:
                try:
                    # Reset object state if it has a reset method
                    if hasattr(obj, 'reset'):
                        obj.reset()
                    elif hasattr(obj, 'clear'):
                        obj.clear()
                    
                    self.pool.append(obj)
                    self.usage_stats['released'] += 1
                    return True
                except Exception as e:
                    self.error_count += 1
                    self.last_error = e
                    return False
            else:
                self.usage_stats['discarded'] += 1
                return False
    
    def clear(self):
        """Clear the pool."""
        with self._lock:
            self.pool.clear()
            self.usage_stats['cleared'] += 1
    
    @property
    def size(self) -> int:
        """Get current pool size."""
        return len(self.pool)
    
    @property
    def hit_rate(self) -> float:
        """Get pool hit rate."""
        total = self.created_count + self.reused_count
        return self.reused_count / max(total, 1)
    
    @property
    def average_creation_time(self) -> float:
        """Get average object creation time."""
        if not self.creation_times:
            return 0.0
        return sum(self.creation_times) / len(self.creation_times)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed pool statistics."""
        return {
            'name': self.name,
            'size': self.size,
            'max_size': self.max_size,
            'created': self.created_count,
            'reused': self.reused_count,
            'errors': self.error_count,
            'hit_rate': self.hit_rate,
            'average_creation_time': self.average_creation_time,
            'last_error': str(self.last_error) if self.last_error else None,
            'usage_stats': dict(self.usage_stats)
        }


class TrendAnalyzer:
    """
    Memory trend analyzer for predictive monitoring.
    """
    
    def __init__(self, window_size: int = 100):
        """Initialize trend analyzer."""
        self.window_size = window_size
        self.samples: deque = deque(maxlen=window_size)
        self.alerts: List[Dict[str, Any]] = []
        
    def add_sample(self, usage: int, timestamp: float = None):
        """Add a memory usage sample."""
        if timestamp is None:
            timestamp = time.time()
        
        self.samples.append({
            'timestamp': timestamp,
            'usage': usage
        })
    
    def analyze_trend(self) -> Dict[str, Any]:
        """Analyze memory usage trend."""
        if len(self.samples) < 10:
            return {'trend': 'insufficient_data', 'confidence': 0.0}
        
        # Calculate linear regression for trend
        x_values = list(range(len(self.samples)))
        y_values = [sample['usage'] for sample in self.samples]
        
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        # Linear regression: y = mx + b
        if n * sum_x2 - sum_x * sum_x == 0:
            slope = 0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Determine trend
        if abs(slope) < 1000:  # bytes per sample
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        # Calculate confidence based on variance
        mean_y = sum_y / n
        variance = sum((y - mean_y) ** 2 for y in y_values) / n
        confidence = max(0.0, min(1.0, 1.0 - (variance / (mean_y ** 2))))
        
        return {
            'trend': trend,
            'slope': slope,
            'confidence': confidence,
            'current_usage': y_values[-1],
            'samples_count': n,
            'variance': variance
        }
    
    def check_alerts(self, current_usage: int, limits: ResourceLimits) -> List[Dict[str, Any]]:
        """Check for trend-based alerts."""
        alerts = []
        
        if len(self.samples) < 5:
            return alerts
        
        trend_data = self.analyze_trend()
        
        # Rapid growth alert
        if (trend_data['trend'] == 'increasing' and 
            trend_data['slope'] > 10000 and  # 10KB per sample
            trend_data['confidence'] > 0.7):
            
            # Predict when limit will be reached
            remaining_capacity = limits.max_memory - current_usage
            if trend_data['slope'] > 0:
                samples_to_limit = remaining_capacity / trend_data['slope']
                time_to_limit = samples_to_limit * limits.monitoring_interval
                
                alerts.append({
                    'type': 'rapid_growth',
                    'severity': 'warning',
                    'message': f"Rapid memory growth detected. Limit may be reached in {time_to_limit:.1f}s",
                    'data': {
                        'slope': trend_data['slope'],
                        'confidence': trend_data['confidence'],
                        'time_to_limit': time_to_limit
                    }
                })
        
        # Memory leak detection
        if (len(self.samples) >= self.window_size and
            trend_data['trend'] == 'increasing' and
            trend_data['confidence'] > 0.8):
            
            # Check if usage has been consistently increasing
            recent_samples = list(self.samples)[-20:]
            if all(recent_samples[i]['usage'] <= recent_samples[i+1]['usage'] 
                   for i in range(len(recent_samples)-1)):
                
                alerts.append({
                    'type': 'potential_leak',
                    'severity': 'error',
                    'message': "Potential memory leak detected - consistent growth pattern",
                    'data': trend_data
                })
        
        return alerts


class EnhancedMemoryManager:
    """
    Enhanced memory manager with full AnamorphX integration.
    """
    
    def __init__(self, 
                 limits: Optional[ResourceLimits] = None,
                 signal_env = None,
                 config_path: Optional[Union[str, Path]] = None):
        """Initialize enhanced memory manager."""
        
        # Load configuration
        if config_path:
            self.limits = ResourceLimits.from_config(config_path)
        else:
            self.limits = limits or ResourceLimits()
        
        self.stats = MemoryStats()
        self.state = MemoryState.NORMAL
        self.signal_env = signal_env
        
        # Enhanced components
        self.gc = self._create_enhanced_gc()
        self.pools: Dict[str, EnhancedObjectPool] = {}
        self.trend_analyzer = TrendAnalyzer()
        
        # Tracking with weak references and dependencies
        self.tracked_objects: Set[weakref.ref] = set()
        self.object_dependencies: Dict[int, Set[int]] = defaultdict(set)
        self.allocation_history: deque = deque(maxlen=2000)
        
        # Event handlers
        self.event_handlers: Dict[MemoryEventType, List[Callable]] = defaultdict(list)
        
        # Threading and async
        self._lock = threading.RLock()
        self._monitor_thread = None
        self._monitoring = False
        self._async_loop = None
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize
        self._setup_default_handlers()
        self.start_monitoring()
    
    def _create_enhanced_gc(self):
        """Create enhanced garbage collector."""
        class EnhancedGC:
            def __init__(self, limits: ResourceLimits):
                self.enabled = True
                self.collections = 0
                self.total_time = 0.0
                self.last_collection = time.time()
                self.collection_interval = limits.monitoring_interval * 10
                self._lock = threading.Lock()
                
                # Configure Python GC with custom thresholds
                gc.set_threshold(*limits.gc_generation_thresholds)
            
            def collect(self, generation: Optional[int] = None) -> int:
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
                if not self.enabled:
                    return False
                
                # Time-based collection
                if time.time() - self.last_collection > self.collection_interval:
                    return True
                
                # Memory pressure-based collection
                stats = gc.get_stats()
                if stats and any(stat['collections'] > 50 for stat in stats):
                    return True
                
                return False
            
            def auto_collect(self):
                if self.should_collect():
                    return self.collect()
                return 0
            
            def get_stats(self) -> Dict[str, Any]:
                return {
                    'collections': self.collections,
                    'total_time': self.total_time,
                    'average_time': self.total_time / max(self.collections, 1),
                    'last_collection': self.last_collection,
                    'gc_stats': gc.get_stats(),
                    'gc_counts': gc.get_count()
                }
        
        return EnhancedGC(self.limits)
    
    def _setup_default_handlers(self):
        """Setup default event handlers."""
        # Memory limit exceeded handler
        self.add_event_handler(MemoryEventType.LIMIT_EXCEEDED, self._handle_limit_exceeded)
        
        # Emergency cleanup handler
        self.add_event_handler(MemoryEventType.EMERGENCY_CLEANUP, self._handle_emergency_cleanup)
        
        # Trend alert handler
        self.add_event_handler(MemoryEventType.TREND_ALERT, self._handle_trend_alert)
    
    def add_event_handler(self, event_type: MemoryEventType, handler: Callable):
        """Add event handler."""
        self.event_handlers[event_type].append(handler)
    
    def _emit_event(self, event: MemoryEvent):
        """Emit memory event."""
        # Send to signal environment if available
        if self.signal_env and hasattr(self.signal_env, 'send_signal'):
            try:
                self.signal_env.send_signal(
                    'memory_event',
                    event.to_signal_data(),
                    target='memory_monitor'
                )
            except Exception as e:
                self.logger.error(f"Failed to send memory event signal: {e}")
        
        # Call registered handlers
        for handler in self.event_handlers[event.event_type]:
            try:
                handler(event)
            except Exception as e:
                self.logger.error(f"Event handler error: {e}")
    
    def allocate(self, obj_type: Type[T], *args, **kwargs) -> Optional[T]:
        """Allocate an object with enhanced tracking."""
        # Try object pool first
        pool_key = obj_type.__name__
        if pool_key in self.pools:
            obj = self.pools[pool_key].acquire()
            if obj is not None:
                self.stats.pool_hits += 1
                self._track_allocation(obj, from_pool=True)
                return obj
        
        # Create new object
        try:
            obj = obj_type(*args, **kwargs)
            self.stats.pool_misses += 1
            self._track_allocation(obj, from_pool=False)
            
            # Emit allocation event
            self._emit_event(MemoryEvent(
                event_type=MemoryEventType.ALLOCATION,
                data={
                    'object_type': obj_type.__name__,
                    'size': self._estimate_size(obj),
                    'from_pool': False
                }
            ))
            
            return obj
            
        except Exception as e:
            self.logger.error(f"Failed to allocate {obj_type.__name__}: {e}")
            return None
    
    def deallocate(self, obj: Any):
        """Deallocate an object with enhanced tracking."""
        if obj is None:
            return
        
        obj_type = type(obj).__name__
        obj_size = self._estimate_size(obj)
        
        # Try to return to pool
        if obj_type in self.pools:
            if self.pools[obj_type].release(obj):
                self._track_deallocation(obj)
                
                # Emit deallocation event
                self._emit_event(MemoryEvent(
                    event_type=MemoryEventType.DEALLOCATION,
                    data={
                        'object_type': obj_type,
                        'size': obj_size,
                        'returned_to_pool': True
                    }
                ))
                return
        
        # Track deallocation
        self._track_deallocation(obj)
        
        # Emit deallocation event
        self._emit_event(MemoryEvent(
            event_type=MemoryEventType.DEALLOCATION,
            data={
                'object_type': obj_type,
                'size': obj_size,
                'returned_to_pool': False
            }
        ))
    
    def create_pool(self, obj_type: Type[T], factory: Callable[[], T], 
                   max_size: Optional[int] = None, name: Optional[str] = None):
        """Create an enhanced object pool."""
        pool_key = obj_type.__name__
        max_size = max_size or self.limits.pool_max_size_default
        name = name or f"{pool_key}_pool"
        
        self.pools[pool_key] = EnhancedObjectPool(factory, max_size, name)
        
        self.logger.info(f"Created enhanced object pool for {pool_key} with max size {max_size}")
        
        # Emit pool creation event
        self._emit_event(MemoryEvent(
            event_type=MemoryEventType.POOL_CREATED,
            data={
                'object_type': pool_key,
                'max_size': max_size,
                'name': name
            }
        ))
    
    def add_object_dependency(self, parent_obj: Any, child_obj: Any):
        """Add object dependency for group cleanup."""
        parent_id = id(parent_obj)
        child_id = id(child_obj)
        
        with self._lock:
            self.object_dependencies[parent_id].add(child_id)
    
    def cleanup_object_group(self, root_obj: Any):
        """Cleanup object and all its dependencies."""
        root_id = id(root_obj)
        
        with self._lock:
            # Get all dependent objects
            to_cleanup = {root_id}
            queue = [root_id]
            
            while queue:
                current_id = queue.pop(0)
                for dep_id in self.object_dependencies.get(current_id, set()):
                    if dep_id not in to_cleanup:
                        to_cleanup.add(dep_id)
                        queue.append(dep_id)
            
            # Clean up dependencies mapping
            for obj_id in to_cleanup:
                if obj_id in self.object_dependencies:
                    del self.object_dependencies[obj_id]
        
        self.logger.info(f"Cleaned up object group: {len(to_cleanup)} objects")
    
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except:
            return 0
    
    def check_limits(self):
        """Enhanced limit checking with trend analysis."""
        current_memory = self.get_memory_usage()
        
        # Update stats
        self.stats.current_usage = current_memory
        if current_memory > self.stats.peak_usage:
            self.stats.peak_usage = current_memory
        
        self.stats.add_usage_sample(current_memory)
        self.trend_analyzer.add_sample(current_memory)
        
        # Check memory limit
        memory_ratio = current_memory / self.limits.max_memory
        
        if memory_ratio >= self.limits.emergency_threshold:
            self.state = MemoryState.EMERGENCY
            self._emit_event(MemoryEvent(
                event_type=MemoryEventType.EMERGENCY_CLEANUP,
                severity="critical",
                data={'memory_ratio': memory_ratio, 'current_usage': current_memory}
            ))
        elif memory_ratio >= self.limits.gc_threshold:
            self.state = MemoryState.CRITICAL
            collected = self.gc.collect()
            self._emit_event(MemoryEvent(
                event_type=MemoryEventType.GC_TRIGGERED,
                severity="warning",
                data={'memory_ratio': memory_ratio, 'objects_collected': collected}
            ))
        elif memory_ratio >= 0.6:
            self.state = MemoryState.WARNING
        else:
            self.state = MemoryState.NORMAL
        
        # Check object count
        if self.stats.object_count > self.limits.max_objects:
            self._emit_event(MemoryEvent(
                event_type=MemoryEventType.LIMIT_EXCEEDED,
                severity="error",
                data={'limit_type': 'object_count', 'current': self.stats.object_count}
            ))
        
        # Trend analysis
        if time.time() - self.stats.last_trend_analysis > self.limits.trend_analysis_interval:
            self._analyze_trends()
            self.stats.last_trend_analysis = time.time()
    
    def _analyze_trends(self):
        """Analyze memory trends and emit alerts."""
        alerts = self.trend_analyzer.check_alerts(self.stats.current_usage, self.limits)
        
        for alert in alerts:
            self._emit_event(MemoryEvent(
                event_type=MemoryEventType.TREND_ALERT,
                severity=alert['severity'],
                data=alert
            ))
    
    def _track_allocation(self, obj: Any, from_pool: bool = False):
        """Enhanced allocation tracking."""
        with self._lock:
            # Add weak reference
            ref = weakref.ref(obj, self._object_finalized)
            self.tracked_objects.add(ref)
            
            # Update stats
            obj_size = self._estimate_size(obj)
            self.stats.total_allocated += obj_size
            if not from_pool:
                self.stats.current_usage += obj_size
            self.stats.object_count += 1
            
            # Add to history
            self.allocation_history.append({
                'type': type(obj).__name__,
                'size': obj_size,
                'time': time.time(),
                'from_pool': from_pool,
                'id': id(obj)
            })
    
    def _track_deallocation(self, obj: Any):
        """Enhanced deallocation tracking."""
        with self._lock:
            obj_size = self._estimate_size(obj)
            self.stats.total_freed += obj_size
            self.stats.current_usage -= obj_size
            self.stats.object_count -= 1
            
            # Clean up dependencies
            obj_id = id(obj)
            if obj_id in self.object_dependencies:
                del self.object_dependencies[obj_id]
    
    def _object_finalized(self, ref: weakref.ref):
        """Called when a tracked object is finalized."""
        with self._lock:
            self.tracked_objects.discard(ref)
    
    def _estimate_size(self, obj: Any) -> int:
        """Enhanced object size estimation."""
        try:
            import sys
            size = sys.getsizeof(obj)
            
            # Add size of contained objects for containers
            if isinstance(obj, (list, tuple)):
                size += sum(sys.getsizeof(item) for item in obj)
            elif isinstance(obj, dict):
                size += sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in obj.items())
            elif isinstance(obj, set):
                size += sum(sys.getsizeof(item) for item in obj)
            
            return size
        except:
            return 64  # Default estimate
    
    def _handle_limit_exceeded(self, event: MemoryEvent):
        """Handle limit exceeded event."""
        self.logger.warning(f"Memory limit exceeded: {event.data}")
        
        # Trigger garbage collection
        collected = self.gc.collect()
        self.logger.info(f"Emergency GC collected {collected} objects")
    
    def _handle_emergency_cleanup(self, event: MemoryEvent):
        """Handle emergency cleanup event."""
        self.logger.critical(f"Emergency memory cleanup triggered: {event.data}")
        
        # Clear object pools
        for pool in self.pools.values():
            pool.clear()
        
        # Force aggressive garbage collection
        for generation in [2, 1, 0]:
            collected = self.gc.collect(generation)
            self.logger.info(f"Emergency GC generation {generation}: {collected} objects")
        
        # Clear allocation history
        self.allocation_history.clear()
        
        # Clear trend data
        self.trend_analyzer.samples.clear()
    
    def _handle_trend_alert(self, event: MemoryEvent):
        """Handle trend alert event."""
        alert_data = event.data
        self.logger.warning(f"Memory trend alert: {alert_data.get('message', 'Unknown trend issue')}")
        
        # Could trigger preemptive actions based on trend type
        if alert_data.get('type') == 'rapid_growth':
            # Preemptive GC
            self.gc.collect()
        elif alert_data.get('type') == 'potential_leak':
            # More aggressive monitoring
            self.limits.monitoring_interval = min(0.5, self.limits.monitoring_interval)
    
    def start_monitoring(self):
        """Start enhanced memory monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Enhanced memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        self.logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Enhanced monitoring loop."""
        while self._monitoring:
            try:
                self.check_limits()
                self.gc.auto_collect()
                time.sleep(self.limits.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                time.sleep(5.0)  # Back off on error
    
    async def async_monitor(self):
        """Async monitoring for better integration."""
        while self._monitoring:
            try:
                self.check_limits()
                collected = self.gc.auto_collect()
                if collected > 0:
                    self.logger.debug(f"Async GC collected {collected} objects")
                
                await asyncio.sleep(self.limits.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Async memory monitoring error: {e}")
                await asyncio.sleep(5.0)
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        trend_data = self.trend_analyzer.analyze_trend()
        
        return {
            'memory_stats': asdict(self.stats),
            'gc_stats': self.gc.get_stats(),
            'pool_stats': {
                name: pool.get_stats()
                for name, pool in self.pools.items()
            },
            'trend_analysis': trend_data,
            'state': self.state.name,
            'tracked_objects': len(self.tracked_objects),
            'object_dependencies': len(self.object_dependencies),
            'limits': asdict(self.limits),
            'recent_alerts': self.trend_analyzer.alerts[-10:] if self.trend_analyzer.alerts else []
        }
    
    def export_stats(self, filepath: Union[str, Path], format: str = 'json'):
        """Export detailed statistics to file."""
        stats = self.get_detailed_stats()
        filepath = Path(filepath)
        
        if format.lower() == 'yaml':
            with open(filepath, 'w') as f:
                yaml.dump(stats, f, default_flow_style=False)
        else:
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
        
        self.logger.info(f"Memory stats exported to {filepath}")
    
    def cleanup(self):
        """Enhanced cleanup with dependency handling."""
        self.stop_monitoring()
        
        # Clear all pools
        for pool in self.pools.values():
            pool.clear()
        
        # Clear dependencies
        self.object_dependencies.clear()
        
        # Final garbage collection
        self.gc.collect()
        
        self.logger.info("Enhanced memory manager cleanup completed")
    
    def __del__(self):
        """Destructor."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during destruction 