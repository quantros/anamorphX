"""
🚦 Rate Limiter - Enterprise Security
Продвинутая система ограничения скорости запросов
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import redis.asyncio as redis
import json
import hashlib

@dataclass
class RateLimit:
    """Конфигурация ограничения скорости"""
    requests: int           # Количество запросов
    window_seconds: int     # Окно времени в секундах
    burst: Optional[int] = None  # Burst лимит

@dataclass
class RateLimitResult:
    """Результат проверки rate limit"""
    allowed: bool
    requests_remaining: int
    reset_time: float
    retry_after: Optional[int] = None

class RateLimiter:
    """
    🚦 Enterprise Rate Limiter
    Многоуровневая система ограничения скорости запросов
    """
    
    def __init__(self,
                 requests_per_minute: int = 60,
                 redis_client: Optional[redis.Redis] = None,
                 key_prefix: str = 'ratelimit',
                 enable_burst: bool = True,
                 burst_multiplier: float = 1.5):
        
        self.requests_per_minute = requests_per_minute
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        self.enable_burst = enable_burst
        self.burst_multiplier = burst_multiplier
        self.logger = logging.getLogger(__name__)
        
        # Local storage for when Redis is not available
        self.local_storage = {}
        self.local_cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        
        # Default rate limits
        self.default_limits = {
            'global': RateLimit(requests_per_minute, 60),
            'per_ip': RateLimit(requests_per_minute, 60),
            'per_user': RateLimit(requests_per_minute * 2, 60),  # Higher limit for authenticated users
            'burst': RateLimit(
                int(requests_per_minute * burst_multiplier), 
                60
            ) if enable_burst else None
        }
        
        # Custom limits for specific endpoints/users
        self.custom_limits = {}
        
        # Whitelist and blacklist
        self.whitelist = set()
        self.blacklist = set()
        
        self.logger.info(f"Rate limiter initialized: {requests_per_minute} req/min")
    
    async def is_allowed(self, 
                        key: str,
                        limit_type: str = 'per_ip',
                        custom_limit: Optional[RateLimit] = None,
                        user_id: Optional[str] = None) -> RateLimitResult:
        """
        Проверка разрешения запроса
        
        Args:
            key: Уникальный ключ (IP, user_id, etc.)
            limit_type: Тип лимита
            custom_limit: Кастомный лимит
            user_id: ID пользователя для персональных лимитов
        
        Returns:
            RateLimitResult с информацией о разрешении
        """
        
        # Check whitelist
        if key in self.whitelist:
            return RateLimitResult(
                allowed=True,
                requests_remaining=999999,
                reset_time=time.time() + 3600
            )
        
        # Check blacklist
        if key in self.blacklist:
            return RateLimitResult(
                allowed=False,
                requests_remaining=0,
                reset_time=time.time() + 3600,
                retry_after=3600
            )
        
        # Get rate limit configuration
        rate_limit = custom_limit or self._get_rate_limit(limit_type, user_id)
        
        # Check rate limit
        if self.redis_client:
            return await self._check_redis_rate_limit(key, rate_limit)
        else:
            return await self._check_local_rate_limit(key, rate_limit)
    
    def _get_rate_limit(self, limit_type: str, user_id: Optional[str] = None) -> RateLimit:
        """Получение конфигурации rate limit"""
        
        # Check custom limits for user
        if user_id and user_id in self.custom_limits:
            return self.custom_limits[user_id]
        
        # Check custom limits for limit type
        if limit_type in self.custom_limits:
            return self.custom_limits[limit_type]
        
        # Return default limit
        return self.default_limits.get(limit_type, self.default_limits['global'])
    
    async def _check_redis_rate_limit(self, key: str, rate_limit: RateLimit) -> RateLimitResult:
        """Проверка rate limit с помощью Redis"""
        
        try:
            redis_key = f"{self.key_prefix}:{key}"
            current_time = time.time()
            window_start = current_time - rate_limit.window_seconds
            
            # Use Redis sorted set for sliding window
            pipe = self.redis_client.pipeline()
            
            # Remove expired entries
            pipe.zremrangebyscore(redis_key, 0, window_start)
            
            # Count current requests
            pipe.zcard(redis_key)
            
            # Add current request
            pipe.zadd(redis_key, {str(current_time): current_time})
            
            # Set expiry
            pipe.expire(redis_key, rate_limit.window_seconds)
            
            results = await pipe.execute()
            current_requests = results[1]
            
            # Check if limit exceeded
            allowed = current_requests < rate_limit.requests
            requests_remaining = max(0, rate_limit.requests - current_requests - 1)
            reset_time = current_time + rate_limit.window_seconds
            
            # Remove the request we just added if not allowed
            if not allowed:
                await self.redis_client.zrem(redis_key, str(current_time))
                retry_after = int(rate_limit.window_seconds)
            else:
                retry_after = None
            
            return RateLimitResult(
                allowed=allowed,
                requests_remaining=requests_remaining,
                reset_time=reset_time,
                retry_after=retry_after
            )
            
        except Exception as e:
            self.logger.error(f"Redis rate limit check failed: {e}")
            # Fallback to local storage
            return await self._check_local_rate_limit(key, rate_limit)
    
    async def _check_local_rate_limit(self, key: str, rate_limit: RateLimit) -> RateLimitResult:
        """Проверка rate limit с локальным хранилищем"""
        
        current_time = time.time()
        window_start = current_time - rate_limit.window_seconds
        
        # Cleanup old entries periodically
        if current_time - self.last_cleanup > self.local_cleanup_interval:
            await self._cleanup_local_storage(current_time)
        
        # Get or create entry for key
        if key not in self.local_storage:
            self.local_storage[key] = []
        
        requests = self.local_storage[key]
        
        # Remove expired requests
        requests[:] = [req_time for req_time in requests if req_time > window_start]
        
        # Check if limit exceeded
        allowed = len(requests) < rate_limit.requests
        requests_remaining = max(0, rate_limit.requests - len(requests) - 1)
        reset_time = current_time + rate_limit.window_seconds
        
        if allowed:
            requests.append(current_time)
            retry_after = None
        else:
            retry_after = int(rate_limit.window_seconds)
        
        return RateLimitResult(
            allowed=allowed,
            requests_remaining=requests_remaining,
            reset_time=reset_time,
            retry_after=retry_after
        )
    
    async def _cleanup_local_storage(self, current_time: float):
        """Очистка локального хранилища от старых записей"""
        
        cutoff_time = current_time - max(
            limit.window_seconds for limit in self.default_limits.values()
        )
        
        for key in list(self.local_storage.keys()):
            requests = self.local_storage[key]
            requests[:] = [req_time for req_time in requests if req_time > cutoff_time]
            
            # Remove empty entries
            if not requests:
                del self.local_storage[key]
        
        self.last_cleanup = current_time
        self.logger.debug("Local rate limit storage cleaned up")
    
    def add_custom_limit(self, identifier: str, rate_limit: RateLimit):
        """
        Добавление кастомного лимита
        
        Args:
            identifier: Идентификатор (user_id, endpoint, etc.)
            rate_limit: Конфигурация лимита
        """
        
        self.custom_limits[identifier] = rate_limit
        self.logger.info(f"Added custom rate limit for {identifier}: {rate_limit}")
    
    def remove_custom_limit(self, identifier: str):
        """Удаление кастомного лимита"""
        
        if identifier in self.custom_limits:
            del self.custom_limits[identifier]
            self.logger.info(f"Removed custom rate limit for {identifier}")
    
    def add_to_whitelist(self, key: str):
        """Добавление в whitelist"""
        
        self.whitelist.add(key)
        self.logger.info(f"Added {key} to whitelist")
    
    def remove_from_whitelist(self, key: str):
        """Удаление из whitelist"""
        
        self.whitelist.discard(key)
        self.logger.info(f"Removed {key} from whitelist")
    
    def add_to_blacklist(self, key: str, duration: Optional[int] = None):
        """
        Добавление в blacklist
        
        Args:
            key: Ключ для блокировки
            duration: Длительность блокировки в секундах (None = навсегда)
        """
        
        self.blacklist.add(key)
        
        # If duration specified, remove after timeout
        if duration:
            async def remove_after_timeout():
                await asyncio.sleep(duration)
                self.blacklist.discard(key)
                self.logger.info(f"Removed {key} from blacklist after {duration}s")
            
            asyncio.create_task(remove_after_timeout())
        
        self.logger.warning(f"Added {key} to blacklist")
    
    def remove_from_blacklist(self, key: str):
        """Удаление из blacklist"""
        
        self.blacklist.discard(key)
        self.logger.info(f"Removed {key} from blacklist")
    
    async def get_rate_limit_status(self, key: str, limit_type: str = 'per_ip') -> Dict[str, Any]:
        """
        Получение статуса rate limit для ключа
        
        Args:
            key: Ключ для проверки
            limit_type: Тип лимита
        
        Returns:
            Статус rate limit
        """
        
        rate_limit = self._get_rate_limit(limit_type)
        current_time = time.time()
        window_start = current_time - rate_limit.window_seconds
        
        if self.redis_client:
            try:
                redis_key = f"{self.key_prefix}:{key}"
                
                # Count requests in window
                request_count = await self.redis_client.zcount(redis_key, window_start, current_time)
                
                # Get oldest request time
                oldest_requests = await self.redis_client.zrange(redis_key, 0, 0, withscores=True)
                oldest_time = oldest_requests[0][1] if oldest_requests else current_time
                
            except Exception as e:
                self.logger.error(f"Failed to get Redis rate limit status: {e}")
                request_count = 0
                oldest_time = current_time
        else:
            requests = self.local_storage.get(key, [])
            request_count = len([req for req in requests if req > window_start])
            oldest_time = min(requests) if requests else current_time
        
        return {
            'key': key,
            'limit_type': limit_type,
            'requests_in_window': request_count,
            'requests_limit': rate_limit.requests,
            'window_seconds': rate_limit.window_seconds,
            'requests_remaining': max(0, rate_limit.requests - request_count),
            'reset_time': oldest_time + rate_limit.window_seconds,
            'whitelisted': key in self.whitelist,
            'blacklisted': key in self.blacklist
        }
    
    async def get_global_stats(self) -> Dict[str, Any]:
        """Получение глобальной статистики rate limiter"""
        
        stats = {
            'default_limits': {k: v.__dict__ for k, v in self.default_limits.items()},
            'custom_limits_count': len(self.custom_limits),
            'whitelist_size': len(self.whitelist),
            'blacklist_size': len(self.blacklist),
            'using_redis': self.redis_client is not None,
            'local_storage_keys': len(self.local_storage) if not self.redis_client else 0
        }
        
        # Redis-specific stats
        if self.redis_client:
            try:
                # Count total rate limit keys
                pattern = f"{self.key_prefix}:*"
                keys = await self.redis_client.keys(pattern)
                stats['redis_keys_count'] = len(keys)
                
                # Sample some key info
                if keys:
                    sample_key = keys[0]
                    sample_count = await self.redis_client.zcard(sample_key)
                    stats['sample_key_requests'] = sample_count
                
            except Exception as e:
                self.logger.error(f"Failed to get Redis stats: {e}")
                stats['redis_error'] = str(e)
        
        return stats
    
    async def reset_rate_limit(self, key: str):
        """
        Сброс rate limit для ключа
        
        Args:
            key: Ключ для сброса
        """
        
        if self.redis_client:
            try:
                redis_key = f"{self.key_prefix}:{key}"
                await self.redis_client.delete(redis_key)
                self.logger.info(f"Reset Redis rate limit for {key}")
            except Exception as e:
                self.logger.error(f"Failed to reset Redis rate limit: {e}")
        
        # Also reset local storage
        if key in self.local_storage:
            del self.local_storage[key]
            self.logger.info(f"Reset local rate limit for {key}")
    
    def create_composite_key(self, *parts: str) -> str:
        """
        Создание составного ключа
        
        Args:
            *parts: Части ключа
        
        Returns:
            Хешированный составной ключ
        """
        
        combined = ":".join(str(part) for part in parts if part)
        
        # Hash long keys to keep them manageable
        if len(combined) > 100:
            return hashlib.sha256(combined.encode()).hexdigest()[:16]
        
        return combined
    
    async def apply_adaptive_limits(self, 
                                  key: str,
                                  error_rate: float,
                                  response_time: float):
        """
        Адаптивные лимиты на основе метрик
        
        Args:
            key: Ключ для применения лимитов
            error_rate: Процент ошибок (0.0-1.0)
            response_time: Среднее время ответа в секундах
        """
        
        base_limit = self.default_limits['per_ip']
        
        # Reduce limits if high error rate or slow response time
        if error_rate > 0.1 or response_time > 2.0:  # 10% errors or 2s+ response
            severity = max(error_rate, min(response_time / 5.0, 1.0))
            reduced_requests = int(base_limit.requests * (1.0 - severity * 0.5))
            
            adaptive_limit = RateLimit(
                requests=max(1, reduced_requests),
                window_seconds=base_limit.window_seconds
            )
            
            self.add_custom_limit(f"adaptive:{key}", adaptive_limit)
            
            self.logger.warning(
                f"Applied adaptive rate limit for {key}: "
                f"{reduced_requests} req/{base_limit.window_seconds}s "
                f"(error_rate={error_rate:.2f}, response_time={response_time:.2f}s)"
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья rate limiter"""
        
        health = {
            'status': 'healthy',
            'redis_connected': False,
            'local_storage_size': len(self.local_storage),
            'whitelist_size': len(self.whitelist),
            'blacklist_size': len(self.blacklist)
        }
        
        # Test Redis connection
        if self.redis_client:
            try:
                await self.redis_client.ping()
                health['redis_connected'] = True
            except Exception as e:
                health['status'] = 'degraded'
                health['redis_error'] = str(e)
        
        return health 