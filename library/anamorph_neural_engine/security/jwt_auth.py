"""
🔐 JWT Authentication - Enterprise Security
Профессиональная система JWT аутентификации
"""

import jwt
import secrets
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import redis
import json
import logging

@dataclass
class TokenClaims:
    """JWT токен claims"""
    user_id: str
    username: str
    email: Optional[str] = None
    role: str = 'user'
    permissions: List[str] = None
    session_id: Optional[str] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []

class JWTAuth:
    """
    🔐 Enterprise JWT Authentication
    Безопасная система аутентификации с поддержкой refresh токенов
    """
    
    def __init__(self,
                 secret: str,
                 algorithm: str = 'HS256',
                 access_token_expiry: int = 3600,    # 1 hour
                 refresh_token_expiry: int = 86400,  # 24 hours
                 redis_client: Optional[redis.Redis] = None):
        
        self.secret = secret
        self.algorithm = algorithm
        self.access_token_expiry = access_token_expiry
        self.refresh_token_expiry = refresh_token_expiry
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        
        # Ensure secret is secure
        if len(secret) < 32:
            self.logger.warning("JWT secret is less than 32 characters - not secure for production")
        
        # Token blacklist for revoked tokens
        self.blacklist = set()
    
    def create_token(self, claims: TokenClaims, token_type: str = 'access') -> str:
        """
        Создание JWT токена
        
        Args:
            claims: Данные пользователя
            token_type: Тип токена (access/refresh)
        
        Returns:
            JWT токен
        """
        
        now = datetime.utcnow()
        
        if token_type == 'access':
            expiry = now + timedelta(seconds=self.access_token_expiry)
        else:
            expiry = now + timedelta(seconds=self.refresh_token_expiry)
        
        # Generate unique token ID
        token_id = secrets.token_urlsafe(16)
        
        payload = {
            'jti': token_id,  # JWT ID
            'sub': claims.user_id,  # Subject
            'username': claims.username,
            'email': claims.email,
            'role': claims.role,
            'permissions': claims.permissions,
            'session_id': claims.session_id,
            'token_type': token_type,
            'iat': int(now.timestamp()),  # Issued at
            'exp': int(expiry.timestamp()),  # Expires
            'iss': 'anamorph-neural-engine',  # Issuer
            'aud': 'anamorph-api'  # Audience
        }
        
        token = jwt.encode(payload, self.secret, algorithm=self.algorithm)
        
        # Store token metadata in Redis if available
        if self.redis_client:
            self._store_token_metadata(token_id, claims, token_type, expiry)
        
        self.logger.debug(f"Created {token_type} token for user {claims.user_id}")
        
        return token
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """
        Декодирование JWT токена
        
        Args:
            token: JWT токен
        
        Returns:
            Декодированные данные
        
        Raises:
            jwt.InvalidTokenError: Неверный токен
        """
        
        try:
            # Decode token
            payload = jwt.decode(
                token, 
                self.secret, 
                algorithms=[self.algorithm],
                audience='anamorph-api',
                issuer='anamorph-neural-engine'
            )
            
            # Check if token is blacklisted
            token_id = payload.get('jti')
            if token_id in self.blacklist:
                raise jwt.InvalidTokenError("Token is revoked")
            
            # Check Redis blacklist if available
            if self.redis_client and token_id:
                if self.redis_client.sismember('jwt:blacklist', token_id):
                    raise jwt.InvalidTokenError("Token is revoked")
            
            # Validate token type
            if payload.get('token_type') not in ['access', 'refresh']:
                raise jwt.InvalidTokenError("Invalid token type")
            
            self.logger.debug(f"Decoded token for user {payload.get('sub')}")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token has expired")
            raise
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid token: {e}")
            raise
    
    def create_refresh_token(self, claims: TokenClaims) -> str:
        """Создание refresh токена"""
        return self.create_token(claims, token_type='refresh')
    
    def decode_refresh_token(self, token: str) -> Dict[str, Any]:
        """Декодирование refresh токена"""
        
        payload = self.decode_token(token)
        
        if payload.get('token_type') != 'refresh':
            raise jwt.InvalidTokenError("Not a refresh token")
        
        return payload
    
    def refresh_access_token(self, refresh_token: str) -> str:
        """
        Обновление access токена с помощью refresh токена
        
        Args:
            refresh_token: Refresh токен
        
        Returns:
            Новый access токен
        """
        
        try:
            # Decode refresh token
            payload = self.decode_refresh_token(refresh_token)
            
            # Create new claims
            claims = TokenClaims(
                user_id=payload['sub'],
                username=payload['username'],
                email=payload.get('email'),
                role=payload.get('role', 'user'),
                permissions=payload.get('permissions', []),
                session_id=payload.get('session_id')
            )
            
            # Create new access token
            new_token = self.create_token(claims, token_type='access')
            
            self.logger.info(f"Refreshed access token for user {claims.user_id}")
            
            return new_token
            
        except jwt.InvalidTokenError:
            self.logger.warning("Failed to refresh token - invalid refresh token")
            raise
    
    def revoke_token(self, token: str):
        """
        Отзыв токена (добавление в blacklist)
        
        Args:
            token: Токен для отзыва
        """
        
        try:
            payload = self.decode_token(token)
            token_id = payload.get('jti')
            
            if token_id:
                # Add to local blacklist
                self.blacklist.add(token_id)
                
                # Add to Redis blacklist if available
                if self.redis_client:
                    self.redis_client.sadd('jwt:blacklist', token_id)
                    
                    # Set TTL to token expiry
                    exp = payload.get('exp', 0)
                    ttl = max(0, exp - int(time.time()))
                    if ttl > 0:
                        self.redis_client.expire('jwt:blacklist', ttl)
                
                self.logger.info(f"Revoked token {token_id}")
        
        except jwt.InvalidTokenError:
            # Token is already invalid, no need to revoke
            pass
    
    def revoke_all_user_tokens(self, user_id: str):
        """
        Отзыв всех токенов пользователя
        
        Args:
            user_id: ID пользователя
        """
        
        if not self.redis_client:
            self.logger.warning("Cannot revoke all user tokens without Redis")
            return
        
        # Get all user tokens from Redis
        user_tokens_key = f'jwt:user:{user_id}:tokens'
        token_ids = self.redis_client.smembers(user_tokens_key)
        
        for token_id in token_ids:
            token_id_str = token_id.decode() if isinstance(token_id, bytes) else token_id
            
            # Add to blacklist
            self.blacklist.add(token_id_str)
            self.redis_client.sadd('jwt:blacklist', token_id_str)
        
        # Clear user tokens set
        self.redis_client.delete(user_tokens_key)
        
        self.logger.info(f"Revoked all tokens for user {user_id}")
    
    def _store_token_metadata(self, 
                            token_id: str, 
                            claims: TokenClaims, 
                            token_type: str, 
                            expiry: datetime):
        """Сохранение метаданных токена в Redis"""
        
        if not self.redis_client:
            return
        
        try:
            # Token metadata
            metadata = {
                'token_id': token_id,
                'user_id': claims.user_id,
                'username': claims.username,
                'token_type': token_type,
                'created_at': datetime.utcnow().isoformat(),
                'expires_at': expiry.isoformat(),
                'session_id': claims.session_id
            }
            
            # Store token metadata
            metadata_key = f'jwt:token:{token_id}'
            self.redis_client.set(
                metadata_key,
                json.dumps(metadata),
                ex=int((expiry - datetime.utcnow()).total_seconds())
            )
            
            # Add to user tokens set
            user_tokens_key = f'jwt:user:{claims.user_id}:tokens'
            self.redis_client.sadd(user_tokens_key, token_id)
            self.redis_client.expire(user_tokens_key, self.refresh_token_expiry)
            
        except Exception as e:
            self.logger.error(f"Failed to store token metadata: {e}")
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Получение активных сессий пользователя
        
        Args:
            user_id: ID пользователя
        
        Returns:
            Список активных сессий
        """
        
        if not self.redis_client:
            return []
        
        try:
            user_tokens_key = f'jwt:user:{user_id}:tokens'
            token_ids = self.redis_client.smembers(user_tokens_key)
            
            sessions = []
            for token_id in token_ids:
                token_id_str = token_id.decode() if isinstance(token_id, bytes) else token_id
                
                metadata_key = f'jwt:token:{token_id_str}'
                metadata_json = self.redis_client.get(metadata_key)
                
                if metadata_json:
                    metadata = json.loads(metadata_json)
                    sessions.append(metadata)
            
            return sessions
            
        except Exception as e:
            self.logger.error(f"Failed to get user sessions: {e}")
            return []
    
    def validate_permissions(self, token: str, required_permissions: List[str]) -> bool:
        """
        Проверка разрешений токена
        
        Args:
            token: JWT токен
            required_permissions: Требуемые разрешения
        
        Returns:
            True если есть все разрешения
        """
        
        try:
            payload = self.decode_token(token)
            user_permissions = payload.get('permissions', [])
            
            # Check if user has all required permissions
            return all(perm in user_permissions for perm in required_permissions)
            
        except jwt.InvalidTokenError:
            return False
    
    def validate_role(self, token: str, required_roles: List[str]) -> bool:
        """
        Проверка роли токена
        
        Args:
            token: JWT токен
            required_roles: Требуемые роли
        
        Returns:
            True если роль подходит
        """
        
        try:
            payload = self.decode_token(token)
            user_role = payload.get('role', 'user')
            
            return user_role in required_roles
            
        except jwt.InvalidTokenError:
            return False
    
    def get_token_info(self, token: str) -> Dict[str, Any]:
        """
        Получение информации о токене
        
        Args:
            token: JWT токен
        
        Returns:
            Информация о токене
        """
        
        try:
            payload = self.decode_token(token)
            
            return {
                'valid': True,
                'user_id': payload.get('sub'),
                'username': payload.get('username'),
                'role': payload.get('role'),
                'permissions': payload.get('permissions', []),
                'token_type': payload.get('token_type'),
                'issued_at': datetime.fromtimestamp(payload.get('iat', 0)).isoformat(),
                'expires_at': datetime.fromtimestamp(payload.get('exp', 0)).isoformat(),
                'session_id': payload.get('session_id')
            }
            
        except jwt.InvalidTokenError as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def cleanup_expired_tokens(self):
        """Очистка истекших токенов из blacklist"""
        
        if not self.redis_client:
            # Clean local blacklist (simplified)
            self.blacklist.clear()
            return
        
        try:
            # Redis automatically expires keys, but we can clean up blacklist
            current_time = int(time.time())
            
            # Remove expired tokens from blacklist
            blacklisted_tokens = self.redis_client.smembers('jwt:blacklist')
            
            for token_id in blacklisted_tokens:
                token_id_str = token_id.decode() if isinstance(token_id, bytes) else token_id
                metadata_key = f'jwt:token:{token_id_str}'
                
                # If token metadata doesn't exist, it's expired
                if not self.redis_client.exists(metadata_key):
                    self.redis_client.srem('jwt:blacklist', token_id_str)
                    self.blacklist.discard(token_id_str)
            
            self.logger.debug("Cleaned up expired tokens")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired tokens: {e}")
    
    def generate_secure_secret(self) -> str:
        """Генерация безопасного секрета"""
        return secrets.token_urlsafe(64)
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """
        Хеширование пароля
        
        Args:
            password: Пароль
            salt: Соль (генерируется если не указана)
        
        Returns:
            Tuple (hashed_password, salt)
        """
        
        if salt is None:
            salt = secrets.token_urlsafe(16)
        
        # Use PBKDF2 with SHA-256
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 100k iterations
        )
        
        return hashed.hex(), salt
    
    def verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """
        Проверка пароля
        
        Args:
            password: Пароль для проверки
            hashed_password: Хешированный пароль
            salt: Соль
        
        Returns:
            True если пароль верный
        """
        
        try:
            test_hash, _ = self.hash_password(password, salt)
            return test_hash == hashed_password
        except Exception:
            return False 