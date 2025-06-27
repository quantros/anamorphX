"""
⚙️ Configuration Manager - Enterprise Edition
Управление конфигурацией enterprise сервера
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class ServerConfig:
    """Конфигурация сервера"""
    host: str = 'localhost'
    port: int = 8080
    redis_url: str = 'redis://localhost:6379'
    
@dataclass 
class NeuralConfig:
    """Конфигурация нейронной сети"""
    device: str = 'auto'
    model_path: Optional[str] = None
    max_workers: int = 4
    model_config: Optional[Dict[str, Any]] = None
    
@dataclass
class AuthConfig:
    """Конфигурация аутентификации"""
    jwt_secret: str = 'your-secret-key-change-in-production'
    token_expiry: int = 3600  # seconds
    refresh_expiry: int = 86400  # seconds
    
@dataclass
class SecurityConfig:
    """Конфигурация безопасности"""
    cors_origins: list = None
    rate_limit: Dict[str, int] = None
    enable_https: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ['*']
        if self.rate_limit is None:
            self.rate_limit = {'requests_per_minute': 60}

@dataclass
class FrontendConfig:
    """Конфигурация frontend"""
    static_dir: str = 'frontend/dist'
    index_file: str = 'index.html'
    api_prefix: str = '/api'
    enable_caching: bool = True
    cache_max_age: int = 3600

@dataclass
class LoggingConfig:
    """Конфигурация логирования"""
    level: str = 'INFO'
    file: Optional[str] = None
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5

@dataclass
class MetricsConfig:
    """Конфигурация метрик"""
    redis_url: str = 'redis://localhost:6379'
    collection_interval: int = 10
    retention_days: int = 7
    enable_prometheus: bool = True

@dataclass
class EnterpriseConfig:
    """Полная конфигурация enterprise сервера"""
    server: ServerConfig = None
    neural: NeuralConfig = None
    auth: AuthConfig = None
    security: SecurityConfig = None
    frontend: FrontendConfig = None
    logging: LoggingConfig = None
    metrics: MetricsConfig = None
    
    def __post_init__(self):
        if self.server is None:
            self.server = ServerConfig()
        if self.neural is None:
            self.neural = NeuralConfig()
        if self.auth is None:
            self.auth = AuthConfig()
        if self.security is None:
            self.security = SecurityConfig()
        if self.frontend is None:
            self.frontend = FrontendConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.metrics is None:
            self.metrics = MetricsConfig()

class ConfigManager:
    """
    ⚙️ Enterprise Configuration Manager
    Управление конфигурацией с поддержкой множественных источников
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = None
        self.logger = logging.getLogger(__name__)
        
        # Default config paths to try
        self.default_paths = [
            'config.yaml',
            'config.yml', 
            'anamorph_config.yaml',
            'anamorph_config.yml',
            '/etc/anamorph/config.yaml',
            os.path.expanduser('~/.anamorph/config.yaml')
        ]
        
        self._load_config()
    
    def _load_config(self):
        """Загрузка конфигурации"""
        
        # Start with default config
        self.config = self._get_default_config()
        
        # Try to load from file
        config_file = self._find_config_file()
        if config_file:
            try:
                file_config = self._load_config_file(config_file)
                self.config = self._merge_configs(self.config, file_config)
                self.logger.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_file}: {e}")
        
        # Override with environment variables
        env_config = self._load_env_config()
        self.config = self._merge_configs(self.config, env_config)
        
        # Validate configuration
        self._validate_config()
        
    def _find_config_file(self) -> Optional[str]:
        """Поиск конфигурационного файла"""
        
        # Use specified path first
        if self.config_path:
            if os.path.exists(self.config_path):
                return self.config_path
            else:
                self.logger.warning(f"Specified config file not found: {self.config_path}")
        
        # Try default paths
        for path in self.default_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _load_config_file(self, file_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации из файла"""
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            else:
                return yaml.safe_load(f) or {}
    
    def _load_env_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации из переменных окружения"""
        
        env_config = {}
        
        # Server config
        if os.getenv('ANAMORPH_HOST'):
            env_config.setdefault('server', {})['host'] = os.getenv('ANAMORPH_HOST')
        if os.getenv('ANAMORPH_PORT'):
            env_config.setdefault('server', {})['port'] = int(os.getenv('ANAMORPH_PORT'))
        if os.getenv('REDIS_URL'):
            env_config.setdefault('server', {})['redis_url'] = os.getenv('REDIS_URL')
        
        # Neural config
        if os.getenv('NEURAL_DEVICE'):
            env_config.setdefault('neural', {})['device'] = os.getenv('NEURAL_DEVICE')
        if os.getenv('NEURAL_MODEL_PATH'):
            env_config.setdefault('neural', {})['model_path'] = os.getenv('NEURAL_MODEL_PATH')
        if os.getenv('NEURAL_MAX_WORKERS'):
            env_config.setdefault('neural', {})['max_workers'] = int(os.getenv('NEURAL_MAX_WORKERS'))
        
        # Auth config
        if os.getenv('JWT_SECRET'):
            env_config.setdefault('auth', {})['jwt_secret'] = os.getenv('JWT_SECRET')
        if os.getenv('TOKEN_EXPIRY'):
            env_config.setdefault('auth', {})['token_expiry'] = int(os.getenv('TOKEN_EXPIRY'))
        
        # Security config
        if os.getenv('CORS_ORIGINS'):
            origins = os.getenv('CORS_ORIGINS').split(',')
            env_config.setdefault('security', {})['cors_origins'] = origins
        if os.getenv('RATE_LIMIT_RPM'):
            env_config.setdefault('security', {}).setdefault('rate_limit', {})['requests_per_minute'] = int(os.getenv('RATE_LIMIT_RPM'))
        
        # Logging config
        if os.getenv('LOG_LEVEL'):
            env_config.setdefault('logging', {})['level'] = os.getenv('LOG_LEVEL')
        if os.getenv('LOG_FILE'):
            env_config.setdefault('logging', {})['file'] = os.getenv('LOG_FILE')
        
        return env_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Получение конфигурации по умолчанию"""
        
        default_config = EnterpriseConfig()
        return asdict(default_config)
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Слияние конфигураций"""
        
        def merge_dict(d1, d2):
            result = d1.copy()
            for key, value in d2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            return result
        
        return merge_dict(base, override)
    
    def _validate_config(self):
        """Валидация конфигурации"""
        
        try:
            # Check required fields
            server_config = self.config.get('server', {})
            if not server_config.get('host'):
                raise ValueError("Server host is required")
            if not isinstance(server_config.get('port'), int):
                raise ValueError("Server port must be an integer")
            
            # Check neural config
            neural_config = self.config.get('neural', {})
            device = neural_config.get('device', 'auto')
            if device not in ['auto', 'cpu', 'cuda']:
                raise ValueError(f"Invalid neural device: {device}")
            
            # Check auth config
            auth_config = self.config.get('auth', {})
            if not auth_config.get('jwt_secret'):
                self.logger.warning("JWT secret not set - using default (not secure for production)")
            
            # Check paths
            frontend_config = self.config.get('frontend', {})
            static_dir = frontend_config.get('static_dir')
            if static_dir and not os.path.exists(static_dir):
                self.logger.warning(f"Frontend static directory not found: {static_dir}")
            
            self.logger.info("Configuration validation passed")
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise
    
    def get_config(self) -> Dict[str, Any]:
        """Получить полную конфигурацию"""
        return self.config.copy()
    
    def get_server_config(self) -> ServerConfig:
        """Получить конфигурацию сервера"""
        server_dict = self.config.get('server', {})
        return ServerConfig(**server_dict)
    
    def get_neural_config(self) -> NeuralConfig:
        """Получить конфигурацию нейронной сети"""
        neural_dict = self.config.get('neural', {})
        return NeuralConfig(**neural_dict)
    
    def get_auth_config(self) -> AuthConfig:
        """Получить конфигурацию аутентификации"""
        auth_dict = self.config.get('auth', {})
        return AuthConfig(**auth_dict)
    
    def get_security_config(self) -> SecurityConfig:
        """Получить конфигурацию безопасности"""
        security_dict = self.config.get('security', {})
        return SecurityConfig(**security_dict)
    
    def get_frontend_config(self) -> FrontendConfig:
        """Получить конфигурацию frontend"""
        frontend_dict = self.config.get('frontend', {})
        return FrontendConfig(**frontend_dict)
    
    def get_logging_config(self) -> LoggingConfig:
        """Получить конфигурацию логирования"""
        logging_dict = self.config.get('logging', {})
        return LoggingConfig(**logging_dict)
    
    def get_metrics_config(self) -> MetricsConfig:
        """Получить конфигурацию метрик"""
        metrics_dict = self.config.get('metrics', {})
        return MetricsConfig(**metrics_dict)
    
    def update_config(self, updates: Dict[str, Any]):
        """Обновить конфигурацию"""
        self.config = self._merge_configs(self.config, updates)
        self._validate_config()
    
    def save_config(self, file_path: str):
        """Сохранить конфигурацию в файл"""
        
        try:
            # Add metadata
            config_with_meta = {
                'metadata': {
                    'generated': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'generator': 'AnamorphX Enterprise Config Manager'
                },
                **self.config
            }
            
            if file_path.endswith('.json'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_with_meta, f, indent=2, ensure_ascii=False)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_with_meta, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config to {file_path}: {e}")
            raise
    
    def generate_sample_config(self, file_path: str = 'anamorph_config.yaml'):
        """Генерация примера конфигурации"""
        
        sample_config = {
            'metadata': {
                'description': 'AnamorphX Enterprise Neural Server Configuration',
                'version': '1.0.0',
                'generated': datetime.now().isoformat()
            },
            
            'server': {
                'host': 'localhost',
                'port': 8080,
                'redis_url': 'redis://localhost:6379'
            },
            
            'neural': {
                'device': 'auto',  # auto, cpu, cuda
                'model_path': None,  # Path to pretrained model
                'max_workers': 4,
                'model_config': {
                    'vocab_size': 2000,
                    'embedding_dim': 128,
                    'hidden_dim': 256,
                    'num_layers': 3,
                    'num_classes': 10,
                    'dropout': 0.3,
                    'use_attention': True
                }
            },
            
            'auth': {
                'jwt_secret': 'CHANGE-THIS-SECRET-IN-PRODUCTION',
                'token_expiry': 3600,  # 1 hour
                'refresh_expiry': 86400  # 24 hours
            },
            
            'security': {
                'cors_origins': ['http://localhost:3000', 'http://localhost:8081'],
                'rate_limit': {
                    'requests_per_minute': 60
                },
                'enable_https': False,
                'ssl_cert_path': None,
                'ssl_key_path': None
            },
            
            'frontend': {
                'static_dir': 'frontend/dist',
                'index_file': 'index.html',
                'api_prefix': '/api',
                'enable_caching': True,
                'cache_max_age': 3600
            },
            
            'logging': {
                'level': 'INFO',
                'file': 'anamorph.log',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'max_file_size': 10485760,  # 10MB
                'backup_count': 5
            },
            
            'metrics': {
                'redis_url': 'redis://localhost:6379',
                'collection_interval': 10,
                'retention_days': 7,
                'enable_prometheus': True
            }
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(sample_config, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"Sample configuration generated: {file_path}")
            print(f"✅ Sample configuration file created: {file_path}")
            print(f"📝 Edit this file to customize your server settings")
            
        except Exception as e:
            self.logger.error(f"Failed to generate sample config: {e}")
            raise
    
    def print_config(self):
        """Вывод текущей конфигурации"""
        
        print("🏢 Current Enterprise Configuration:")
        print("=" * 50)
        
        # Server
        server = self.config.get('server', {})
        print(f"🌐 Server: {server.get('host')}:{server.get('port')}")
        
        # Neural
        neural = self.config.get('neural', {})
        print(f"🧠 Neural Device: {neural.get('device')}")
        print(f"👥 Neural Workers: {neural.get('max_workers')}")
        
        # Auth
        auth = self.config.get('auth', {})
        print(f"🔐 JWT Expiry: {auth.get('token_expiry')}s")
        
        # Frontend
        frontend = self.config.get('frontend', {})
        print(f"🌐 Static Dir: {frontend.get('static_dir')}")
        
        # Logging
        logging_cfg = self.config.get('logging', {})
        print(f"📝 Log Level: {logging_cfg.get('level')}")
        
        print("=" * 50) 