"""
🏢 Enterprise Neural Server - Main Application
Главный сервер который объединяет все компоненты
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json
from datetime import datetime

from .core.neural_engine import NeuralEngine
from .backend.api_server import NeuralAPIServer
from .frontend.spa_handler import SPAHandler
from .monitoring.metrics_collector import MetricsCollector
from .utils.config_manager import ConfigManager
from .utils.logger import EnterpriseLogger

class EnterpriseNeuralServer:
    """
    🏢 Enterprise Neural Server
    Главный класс enterprise нейронного сервера
    """
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        
        # Configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Override config with kwargs
        for key, value in kwargs.items():
            if '.' in key:
                # Nested key like 'neural.device'
                keys = key.split('.')
                current = self.config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                self.config[key] = value
        
        # Logger
        self.logger = EnterpriseLogger.setup_enterprise_logging(
            level=self.config.get('logging', {}).get('level', 'INFO'),
            log_file=self.config.get('logging', {}).get('file')
        )
        
        # Components
        self.neural_engine = None
        self.api_server = None
        self.spa_handler = None
        self.metrics_collector = None
        
        # Runtime state
        self.is_running = False
        self.startup_time = None
        
        self.logger.info("🏢 Enterprise Neural Server initialized")
    
    async def initialize(self):
        """Инициализация всех компонентов"""
        
        self.logger.info("🚀 Initializing Enterprise Neural Server...")
        
        try:
            # Initialize Neural Engine
            await self._initialize_neural_engine()
            
            # Initialize Metrics Collector
            await self._initialize_metrics()
            
            # Initialize API Server
            await self._initialize_api_server()
            
            # Initialize SPA Handler
            await self._initialize_spa_handler()
            
            self.logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            raise
    
    async def _initialize_neural_engine(self):
        """Инициализация нейронного движка"""
        
        neural_config = self.config.get('neural', {})
        
        self.neural_engine = NeuralEngine(
            model_config=neural_config.get('model_config'),
            device=neural_config.get('device', 'auto'),
            max_workers=neural_config.get('max_workers', 4),
            model_path=neural_config.get('model_path')
        )
        
        self.logger.info("🧠 Neural Engine initialized")
    
    async def _initialize_metrics(self):
        """Инициализация сбора метрик"""
        
        metrics_config = self.config.get('metrics', {})
        
        self.metrics_collector = MetricsCollector(
            redis_url=metrics_config.get('redis_url', 'redis://localhost:6379'),
            collection_interval=metrics_config.get('collection_interval', 10)
        )
        
        await self.metrics_collector.initialize()
        
        self.logger.info("📊 Metrics Collector initialized")
    
    async def _initialize_api_server(self):
        """Инициализация API сервера"""
        
        server_config = self.config.get('server', {})
        auth_config = self.config.get('auth', {})
        security_config = self.config.get('security', {})
        
        self.api_server = NeuralAPIServer(
            neural_engine=self.neural_engine,
            host=server_config.get('host', 'localhost'),
            port=server_config.get('port', 8080),
            redis_url=server_config.get('redis_url', 'redis://localhost:6379'),
            jwt_secret=auth_config.get('jwt_secret', 'your-secret-key'),
            cors_origins=security_config.get('cors_origins', ['*']),
            rate_limit=security_config.get('rate_limit', {'requests_per_minute': 60})
        )
        
        self.logger.info("🚀 API Server initialized")
    
    async def _initialize_spa_handler(self):
        """Инициализация SPA handler"""
        
        frontend_config = self.config.get('frontend', {})
        
        self.spa_handler = SPAHandler(
            static_dir=frontend_config.get('static_dir', 'frontend/dist'),
            index_file=frontend_config.get('index_file', 'index.html'),
            api_prefix=frontend_config.get('api_prefix', '/api'),
            enable_caching=frontend_config.get('enable_caching', True),
            cache_max_age=frontend_config.get('cache_max_age', 3600)
        )
        
        # Setup SPA routes in API server
        self.spa_handler.setup_routes(self.api_server.app)
        
        self.logger.info("🌐 SPA Handler initialized")
    
    async def start(self):
        """Запуск сервера"""
        
        if self.is_running:
            self.logger.warning("Server is already running")
            return
        
        try:
            # Initialize if not done
            if not self.neural_engine:
                await self.initialize()
            
            # Start metrics collection
            if self.metrics_collector:
                await self.metrics_collector.start()
            
            # Start API server
            await self.api_server.start()
            
            self.is_running = True
            self.startup_time = datetime.now()
            
            # Log startup information
            self._log_startup_info()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start server: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Остановка сервера"""
        
        if not self.is_running:
            return
        
        self.logger.info("🛑 Shutting down Enterprise Neural Server...")
        
        try:
            # Stop API server
            if self.api_server:
                await self.api_server.stop()
            
            # Stop metrics collection
            if self.metrics_collector:
                await self.metrics_collector.stop()
            
            self.is_running = False
            self.logger.info("✅ Server stopped gracefully")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")
    
    def _log_startup_info(self):
        """Вывод информации о запуске"""
        
        server_config = self.config.get('server', {})
        host = server_config.get('host', 'localhost')
        port = server_config.get('port', 8080)
        
        self.logger.info("=" * 80)
        self.logger.info("🏢 ENTERPRISE NEURAL SERVER STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"🌐 URL: http://{host}:{port}")
        self.logger.info(f"📡 API: http://{host}:{port}/api")
        self.logger.info(f"🔗 WebSocket: ws://{host}:{port}/api/ws/neural")
        self.logger.info(f"📊 Metrics: http://{host}:{port}/api/metrics")
        self.logger.info(f"❤️ Health: http://{host}:{port}/api/health")
        self.logger.info("=" * 80)
        self.logger.info(f"🧠 Neural Engine: {self.neural_engine.model.__class__.__name__}")
        self.logger.info(f"🔧 Device: {self.neural_engine.device}")
        self.logger.info(f"📊 Parameters: {self.neural_engine.stats['total_parameters']:,}")
        self.logger.info(f"🗂️ Classes: {len(self.neural_engine.classes)}")
        self.logger.info(f"📚 Vocabulary: {len(self.neural_engine.vocab)}")
        self.logger.info("=" * 80)
        self.logger.info("🛑 Press Ctrl+C to stop")
        self.logger.info("=" * 80)
    
    async def run_forever(self):
        """Запуск сервера и ожидание остановки"""
        
        # Setup signal handlers
        def signal_handler(sig, frame):
            self.logger.info(f"Received signal {sig}")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            await self.start()
            
            # Keep running until stopped
            while self.is_running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
        finally:
            await self.stop()
    
    def get_status(self) -> Dict[str, Any]:
        """Получить статус сервера"""
        
        if not self.is_running:
            return {
                'status': 'stopped',
                'is_running': False
            }
        
        return {
            'status': 'running',
            'is_running': True,
            'startup_time': self.startup_time.isoformat() if self.startup_time else None,
            'uptime_seconds': (
                (datetime.now() - self.startup_time).total_seconds() 
                if self.startup_time else 0
            ),
            'config': self.config,
            'neural_engine': self.neural_engine.get_stats() if self.neural_engine else None,
            'components': {
                'neural_engine': self.neural_engine is not None,
                'api_server': self.api_server is not None,
                'spa_handler': self.spa_handler is not None,
                'metrics_collector': self.metrics_collector is not None
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья всех компонентов"""
        
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        # Check neural engine
        if self.neural_engine:
            try:
                neural_health = self.neural_engine.health_check()
                health['components']['neural_engine'] = neural_health
            except Exception as e:
                health['components']['neural_engine'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health['status'] = 'degraded'
        
        # Check API server
        if self.api_server and self.is_running:
            health['components']['api_server'] = {
                'status': 'healthy',
                'host': self.api_server.host,
                'port': self.api_server.port,
                'websocket_connections': len(self.api_server.websocket_connections)
            }
        else:
            health['components']['api_server'] = {
                'status': 'unhealthy' if self.api_server else 'not_initialized'
            }
            health['status'] = 'unhealthy'
        
        # Check metrics collector
        if self.metrics_collector:
            try:
                metrics_health = await self.metrics_collector.health_check()
                health['components']['metrics_collector'] = metrics_health
            except Exception as e:
                health['components']['metrics_collector'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        return health

# Factory function для создания и запуска сервера
async def create_and_run_server(config_path: Optional[str] = None, **kwargs):
    """Создание и запуск enterprise сервера"""
    
    server = EnterpriseNeuralServer(config_path=config_path, **kwargs)
    await server.run_forever()
    return server

# Main entry point
async def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='AnamorphX Enterprise Neural Server')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--device', default='auto', help='Neural device (cpu/cuda/auto)')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Create and run server
    await create_and_run_server(
        config_path=args.config,
        host=args.host,
        port=args.port,
        neural={'device': args.device},
        logging={'level': args.log_level}
    )

if __name__ == '__main__':
    asyncio.run(main()) 