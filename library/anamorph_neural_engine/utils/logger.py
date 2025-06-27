"""
📝 Enterprise Logger - Advanced Logging System
Профессиональная система логирования для enterprise сервера
"""

import logging
import logging.handlers
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import traceback
import threading
from logging import LogRecord

class ColoredFormatter(logging.Formatter):
    """Цветной форматтер для консольного вывода"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green  
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record: LogRecord) -> str:
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        # Format the message
        formatted = super().format(record)
        
        return formatted

class JSONFormatter(logging.Formatter):
    """JSON форматтер для структурированного логирования"""
    
    def format(self, record: LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process': record.process,
            'thread': record.thread,
            'thread_name': record.threadName
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)

class EnterpriseLogger:
    """
    📝 Enterprise Logger
    Профессиональная система логирования
    """
    
    _instance = None
    _lock = threading.Lock()
    _loggers = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def setup_enterprise_logging(cls,
                                level: str = 'INFO',
                                log_file: Optional[str] = None,
                                log_format: Optional[str] = None,
                                max_file_size: int = 10 * 1024 * 1024,  # 10MB
                                backup_count: int = 5,
                                enable_json: bool = False,
                                enable_console: bool = True) -> logging.Logger:
        """
        Настройка enterprise логирования
        
        Args:
            level: Уровень логирования
            log_file: Путь к файлу логов
            log_format: Формат логов
            max_file_size: Максимальный размер файла
            backup_count: Количество backup файлов
            enable_json: Включить JSON форматирование
            enable_console: Включить консольный вывод
        
        Returns:
            Настроенный logger
        """
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Default format
        if not log_format:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            
            if enable_json:
                console_formatter = JSONFormatter()
            else:
                console_formatter = ColoredFormatter(log_format)
            
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, level.upper()))
            
            if enable_json:
                file_formatter = JSONFormatter()
            else:
                file_formatter = logging.Formatter(log_format)
            
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        
        # Add enterprise info
        enterprise_logger = cls.get_logger('enterprise')
        enterprise_logger.info("🏢 Enterprise Logging System initialized")
        enterprise_logger.info(f"📝 Log level: {level}")
        enterprise_logger.info(f"📄 Log file: {log_file or 'Console only'}")
        enterprise_logger.info(f"🎨 JSON format: {'Enabled' if enable_json else 'Disabled'}")
        
        return enterprise_logger
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Получить logger с заданным именем"""
        
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
        
        return cls._loggers[name]
    
    @classmethod
    def log_with_context(cls, 
                        logger_name: str,
                        level: str,
                        message: str,
                        **context):
        """Логирование с контекстом"""
        
        logger = cls.get_logger(logger_name)
        
        # Create record with extra context
        record = logger.makeRecord(
            logger.name,
            getattr(logging, level.upper()),
            __file__,
            0,
            message,
            (),
            None
        )
        
        # Add context as extra fields
        record.extra_fields = context
        
        logger.handle(record)
    
    @classmethod
    def log_neural_inference(cls,
                           request_id: str,
                           path: str,
                           method: str,
                           classification: str,
                           confidence: float,
                           processing_time: float):
        """Специальное логирование нейронных инференций"""
        
        cls.log_with_context(
            'neural',
            'INFO',
            f"Neural inference: {classification} ({confidence:.3f}) - {processing_time:.3f}s",
            request_id=request_id,
            path=path,
            method=method,
            classification=classification,
            confidence=confidence,
            processing_time=processing_time,
            event_type='neural_inference'
        )
    
    @classmethod
    def log_api_request(cls,
                       method: str,
                       path: str,
                       status_code: int,
                       processing_time: float,
                       client_ip: str,
                       user_agent: Optional[str] = None):
        """Логирование API запросов"""
        
        cls.log_with_context(
            'api',
            'INFO',
            f"{method} {path} - {status_code} - {processing_time:.3f}s",
            method=method,
            path=path,
            status_code=status_code,
            processing_time=processing_time,
            client_ip=client_ip,
            user_agent=user_agent,
            event_type='api_request'
        )
    
    @classmethod
    def log_security_event(cls,
                          event_type: str,
                          client_ip: str,
                          details: Dict[str, Any]):
        """Логирование событий безопасности"""
        
        cls.log_with_context(
            'security',
            'WARNING',
            f"Security event: {event_type}",
            event_type=event_type,
            client_ip=client_ip,
            details=details,
            security_event=True
        )
    
    @classmethod
    def log_performance_metrics(cls,
                              component: str,
                              metrics: Dict[str, Any]):
        """Логирование метрик производительности"""
        
        cls.log_with_context(
            'performance',
            'DEBUG',
            f"Performance metrics for {component}",
            component=component,
            metrics=metrics,
            event_type='performance_metrics'
        )
    
    @classmethod
    def log_system_event(cls,
                        event_type: str,
                        message: str,
                        **details):
        """Логирование системных событий"""
        
        cls.log_with_context(
            'system',
            'INFO',
            message,
            event_type=event_type,
            **details
        )
    
    @classmethod
    def log_error_with_context(cls,
                             logger_name: str,
                             error: Exception,
                             context: Dict[str, Any] = None):
        """Логирование ошибок с контекстом"""
        
        logger = cls.get_logger(logger_name)
        
        error_context = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'event_type': 'error'
        }
        
        if context:
            error_context.update(context)
        
        cls.log_with_context(
            logger_name,
            'ERROR',
            f"Error occurred: {error}",
            **error_context
        )
    
    @classmethod
    def create_request_logger(cls, request_id: str) -> logging.LoggerAdapter:
        """Создать logger для конкретного запроса"""
        
        base_logger = cls.get_logger('request')
        
        class RequestAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                return f"[{request_id}] {msg}", kwargs
        
        return RequestAdapter(base_logger, {'request_id': request_id})
    
    @classmethod
    def setup_audit_logging(cls, audit_file: str):
        """Настройка аудит логирования"""
        
        # Create audit logger
        audit_logger = logging.getLogger('audit')
        audit_logger.setLevel(logging.INFO)
        
        # Don't propagate to root logger
        audit_logger.propagate = False
        
        # Create audit file handler
        audit_handler = logging.handlers.RotatingFileHandler(
            audit_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        
        # JSON formatter for audit logs
        audit_formatter = JSONFormatter()
        audit_handler.setFormatter(audit_formatter)
        
        audit_logger.addHandler(audit_handler)
        
        cls._loggers['audit'] = audit_logger
        
        # Log audit setup
        cls.log_system_event(
            'audit_setup',
            'Audit logging initialized',
            audit_file=audit_file
        )
    
    @classmethod
    def log_audit_event(cls,
                       event_type: str,
                       user_id: Optional[str],
                       action: str,
                       resource: str,
                       result: str,
                       **details):
        """Логирование аудит событий"""
        
        if 'audit' not in cls._loggers:
            cls.setup_audit_logging('audit.log')
        
        audit_logger = cls._loggers['audit']
        
        audit_record = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'result': result,
            'details': details
        }
        
        audit_logger.info(json.dumps(audit_record, ensure_ascii=False))
    
    @classmethod
    def get_log_stats(cls) -> Dict[str, Any]:
        """Получить статистику логирования"""
        
        stats = {
            'active_loggers': len(cls._loggers),
            'logger_names': list(cls._loggers.keys()),
            'root_level': logging.getLogger().level,
            'handlers': []
        }
        
        # Get handler info
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler_info = {
                'type': type(handler).__name__,
                'level': handler.level,
                'formatter': type(handler.formatter).__name__ if handler.formatter else None
            }
            
            if isinstance(handler, logging.FileHandler):
                handler_info['file'] = handler.baseFilename
            
            stats['handlers'].append(handler_info)
        
        return stats
    
    @classmethod
    def setup_monitoring_logger(cls, 
                               redis_url: Optional[str] = None,
                               log_level: str = 'WARNING'):
        """Настройка логирования для мониторинга"""
        
        monitor_logger = cls.get_logger('monitoring')
        monitor_logger.setLevel(getattr(logging, log_level.upper()))
        
        # If Redis is available, send critical logs there
        if redis_url:
            try:
                import redis
                redis_client = redis.from_url(redis_url)
                
                class RedisHandler(logging.Handler):
                    def __init__(self, redis_client):
                        super().__init__()
                        self.redis_client = redis_client
                    
                    def emit(self, record):
                        try:
                            log_entry = {
                                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                                'level': record.levelname,
                                'message': record.getMessage(),
                                'logger': record.name
                            }
                            
                            self.redis_client.lpush(
                                'logs:critical',
                                json.dumps(log_entry)
                            )
                            
                            # Keep only last 1000 entries
                            self.redis_client.ltrim('logs:critical', 0, 999)
                            
                        except Exception:
                            pass  # Fail silently to avoid logging loops
                
                redis_handler = RedisHandler(redis_client)
                redis_handler.setLevel(logging.ERROR)
                monitor_logger.addHandler(redis_handler)
                
                cls.log_system_event(
                    'monitoring_setup',
                    'Redis monitoring logging enabled',
                    redis_url=redis_url
                )
                
            except ImportError:
                cls.log_system_event(
                    'monitoring_setup',
                    'Redis not available for monitoring logs'
                )

# Convenience functions
def get_logger(name: str) -> logging.Logger:
    """Получить enterprise logger"""
    return EnterpriseLogger.get_logger(name)

def setup_logging(**kwargs) -> logging.Logger:
    """Быстрая настройка логирования"""
    return EnterpriseLogger.setup_enterprise_logging(**kwargs) 