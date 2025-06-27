"""
🚀 Neural API Server - Enterprise Backend
Высокопроизводительный асинхронный API сервер
"""

import asyncio
import json
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import asdict
import aiohttp
from aiohttp import web, websocket
from aiohttp.web_middlewares import cors_handler
from aiohttp_cors import setup as cors_setup, ResourceOptions
import jwt
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from ..core.neural_engine import NeuralEngine, NeuralRequest, NeuralResponse
from ..monitoring.metrics_collector import MetricsCollector
from ..security.jwt_auth import JWTAuth
from ..security.rate_limiter import RateLimiter
from ..utils.logger import EnterpriseLogger

# Prometheus metrics
REQUEST_COUNT = Counter('neural_api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('neural_api_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('neural_api_active_connections', 'Active WebSocket connections')
NEURAL_INFERENCE_TIME = Histogram('neural_inference_duration_seconds', 'Neural inference time')

class NeuralAPIServer:
    """
    🚀 Enterprise Neural API Server
    Асинхронный сервер с WebSocket поддержкой
    """
    
    def __init__(self, 
                 neural_engine: NeuralEngine,
                 host: str = 'localhost',
                 port: int = 8080,
                 redis_url: str = 'redis://localhost:6379',
                 jwt_secret: str = None,
                 cors_origins: List[str] = None,
                 rate_limit: Dict[str, int] = None):
        
        self.neural_engine = neural_engine
        self.host = host
        self.port = port
        self.logger = EnterpriseLogger.get_logger(__name__)
        
        # Redis for session storage and caching
        self.redis_url = redis_url
        self.redis_client = None
        
        # Security components
        self.jwt_auth = JWTAuth(secret=jwt_secret)
        self.rate_limiter = RateLimiter(
            requests_per_minute=rate_limit.get('requests_per_minute', 60) if rate_limit else 60,
            redis_client=self.redis_client
        )
        
        # CORS configuration
        self.cors_origins = cors_origins or ['http://localhost:3000', 'http://localhost:8081']
        
        # Metrics collector
        self.metrics = MetricsCollector()
        
        # WebSocket connections
        self.websocket_connections = {}
        
        # App initialization
        self.app = None
        self.runner = None
        self.site = None
        
        # Startup timestamp
        self.startup_time = datetime.now()
        
    async def initialize(self):
        """Инициализация сервера"""
        try:
            # Redis connection
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            self.logger.info("✅ Redis connected")
            
            # Update rate limiter with redis client
            self.rate_limiter.redis_client = self.redis_client
            
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
        
        # Create aiohttp app
        self.app = web.Application(
            middlewares=[
                self.auth_middleware,
                self.rate_limit_middleware,
                self.metrics_middleware,
                self.cors_middleware,
                self.error_middleware
            ]
        )
        
        # Setup CORS
        cors = cors_setup(self.app, defaults={
            "*": ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # API Routes
        self.setup_routes()
        
        # Setup CORS for all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
        
        self.logger.info("🚀 Neural API Server initialized")
    
    def setup_routes(self):
        """Настройка маршрутов API"""
        
        # Health & Status
        self.app.router.add_get('/api/health', self.health_check)
        self.app.router.add_get('/api/status', self.status_check)
        self.app.router.add_get('/api/metrics', self.prometheus_metrics)
        
        # Neural processing
        self.app.router.add_post('/api/neural/predict', self.neural_predict)
        self.app.router.add_post('/api/neural/batch', self.neural_batch_predict)
        self.app.router.add_get('/api/neural/stats', self.neural_stats)
        self.app.router.add_get('/api/neural/model-info', self.model_info)
        
        # Authentication
        self.app.router.add_post('/api/auth/login', self.auth_login)
        self.app.router.add_post('/api/auth/refresh', self.auth_refresh)
        self.app.router.add_post('/api/auth/logout', self.auth_logout)
        self.app.router.add_get('/api/auth/profile', self.auth_profile)
        
        # Real-time WebSocket
        self.app.router.add_get('/api/ws/neural', self.websocket_handler)
        self.app.router.add_get('/api/ws/metrics', self.websocket_metrics)
        
        # Admin endpoints (protected)
        self.app.router.add_get('/api/admin/connections', self.admin_connections)
        self.app.router.add_post('/api/admin/broadcast', self.admin_broadcast)
        self.app.router.add_get('/api/admin/logs', self.admin_logs)
        
        # Development endpoints
        self.app.router.add_get('/api/dev/test', self.dev_test)
        self.app.router.add_post('/api/dev/simulate', self.dev_simulate)
    
    # Middleware
    @web.middleware
    async def auth_middleware(self, request, handler):
        """JWT Authentication middleware"""
        # Skip auth for certain endpoints
        skip_auth = ['/api/health', '/api/status', '/api/auth/login', '/api/metrics']
        
        if any(request.path.startswith(path) for path in skip_auth):
            return await handler(request)
        
        # Check JWT token
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return web.json_response({'error': 'Missing or invalid authorization header'}, status=401)
        
        token = auth_header[7:]  # Remove 'Bearer '
        
        try:
            payload = self.jwt_auth.decode_token(token)
            request['user'] = payload
            return await handler(request)
        except jwt.InvalidTokenError:
            return web.json_response({'error': 'Invalid token'}, status=401)
    
    @web.middleware
    async def rate_limit_middleware(self, request, handler):
        """Rate limiting middleware"""
        client_ip = request.remote
        
        # Check rate limit
        allowed = await self.rate_limiter.is_allowed(client_ip)
        if not allowed:
            return web.json_response(
                {'error': 'Rate limit exceeded'}, 
                status=429,
                headers={'Retry-After': '60'}
            )
        
        return await handler(request)
    
    @web.middleware
    async def metrics_middleware(self, request, handler):
        """Metrics collection middleware"""
        start_time = time.time()
        
        try:
            response = await handler(request)
            duration = time.time() - start_time
            
            # Record metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.path,
                status=response.status
            ).inc()
            
            REQUEST_DURATION.observe(duration)
            
            # Add timing header
            response.headers['X-Processing-Time'] = f"{duration:.3f}s"
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.path,
                status=500
            ).inc()
            raise
    
    @web.middleware
    async def cors_middleware(self, request, handler):
        """CORS middleware"""
        # Handle preflight requests
        if request.method == 'OPTIONS':
            response = web.Response()
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            return response
        
        response = await handler(request)
        
        # Add CORS headers
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        
        return response
    
    @web.middleware
    async def error_middleware(self, request, handler):
        """Error handling middleware"""
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"API Error: {e}", exc_info=True)
            return web.json_response(
                {
                    'error': 'Internal server error',
                    'type': type(e).__name__,
                    'timestamp': datetime.now().isoformat()
                },
                status=500
            )
    
    # API Handlers
    async def health_check(self, request):
        """Health check endpoint"""
        neural_health = self.neural_engine.health_check()
        
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime': str(datetime.now() - self.startup_time),
            'neural_engine': neural_health,
            'redis_connected': self.redis_client is not None,
            'websocket_connections': len(self.websocket_connections),
            'version': '1.0.0'
        })
    
    async def status_check(self, request):
        """Detailed status endpoint"""
        stats = self.neural_engine.get_stats()
        
        return web.json_response({
            'server': {
                'host': self.host,
                'port': self.port,
                'startup_time': self.startup_time.isoformat(),
                'uptime_seconds': (datetime.now() - self.startup_time).total_seconds()
            },
            'neural_engine': stats,
            'connections': {
                'websocket': len(self.websocket_connections),
                'redis': self.redis_client is not None
            },
            'metrics': await self.metrics.get_current_metrics()
        })
    
    async def neural_predict(self, request):
        """Single neural prediction"""
        try:
            data = await request.json()
            
            # Create neural request
            neural_request = NeuralRequest(
                request_id=str(uuid.uuid4()),
                path=data.get('path', '/'),
                method=data.get('method', 'GET'),
                headers=data.get('headers', {}),
                body=data.get('body', '').encode() if data.get('body') else None,
                timestamp=datetime.now(),
                client_ip=request.remote,
                user_id=request.get('user', {}).get('user_id')
            )
            
            # Process with neural engine
            start_time = time.time()
            response = await self.neural_engine.process_request_async(neural_request)
            inference_time = time.time() - start_time
            
            # Record neural metrics
            NEURAL_INFERENCE_TIME.observe(inference_time)
            
            # Broadcast to WebSocket clients
            await self.broadcast_neural_result(response)
            
            return web.json_response({
                'success': True,
                'request_id': response.request_id,
                'classification': response.classification,
                'confidence': response.confidence,
                'processing_time': response.processing_time,
                'features': response.features,
                'metadata': response.metadata
            })
            
        except Exception as e:
            self.logger.error(f"Neural prediction error: {e}")
            return web.json_response({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, status=500)
    
    async def neural_batch_predict(self, request):
        """Batch neural predictions"""
        try:
            data = await request.json()
            requests_data = data.get('requests', [])
            
            if len(requests_data) > 100:  # Limit batch size
                return web.json_response({
                    'success': False,
                    'error': 'Batch size too large (max 100)'
                }, status=400)
            
            # Create neural requests
            neural_requests = []
            for req_data in requests_data:
                neural_request = NeuralRequest(
                    request_id=str(uuid.uuid4()),
                    path=req_data.get('path', '/'),
                    method=req_data.get('method', 'GET'),
                    headers=req_data.get('headers', {}),
                    body=req_data.get('body', '').encode() if req_data.get('body') else None,
                    timestamp=datetime.now(),
                    client_ip=request.remote,
                    user_id=request.get('user', {}).get('user_id')
                )
                neural_requests.append(neural_request)
            
            # Process batch
            start_time = time.time()
            tasks = [
                self.neural_engine.process_request_async(req) 
                for req in neural_requests
            ]
            responses = await asyncio.gather(*tasks)
            batch_time = time.time() - start_time
            
            # Convert to JSON-serializable format
            results = []
            for response in responses:
                results.append({
                    'request_id': response.request_id,
                    'classification': response.classification,
                    'confidence': response.confidence,
                    'processing_time': response.processing_time,
                    'metadata': response.metadata
                })
            
            return web.json_response({
                'success': True,
                'batch_size': len(results),
                'total_processing_time': batch_time,
                'average_processing_time': batch_time / len(results),
                'results': results
            })
            
        except Exception as e:
            self.logger.error(f"Batch prediction error: {e}")
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def neural_stats(self, request):
        """Neural engine statistics"""
        stats = self.neural_engine.get_stats()
        return web.json_response(stats)
    
    async def model_info(self, request):
        """Model information"""
        return web.json_response({
            'model_name': self.neural_engine.model.__class__.__name__,
            'model_config': self.neural_engine.model_config,
            'vocab_size': len(self.neural_engine.vocab),
            'classes': self.neural_engine.classes,
            'device': str(self.neural_engine.device),
            'parameters': self.neural_engine.stats['total_parameters']
        })
    
    # Authentication endpoints
    async def auth_login(self, request):
        """User login"""
        try:
            data = await request.json()
            username = data.get('username')
            password = data.get('password')
            
            # Simple demo authentication
            if username == 'admin' and password == 'admin123':
                payload = {
                    'user_id': 'admin',
                    'username': 'admin',
                    'role': 'administrator',
                    'permissions': ['read', 'write', 'admin']
                }
                
                access_token = self.jwt_auth.create_token(payload)
                refresh_token = self.jwt_auth.create_refresh_token(payload)
                
                return web.json_response({
                    'success': True,
                    'access_token': access_token,
                    'refresh_token': refresh_token,
                    'user': payload
                })
            else:
                return web.json_response({
                    'success': False,
                    'error': 'Invalid credentials'
                }, status=401)
                
        except Exception as e:
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def auth_refresh(self, request):
        """Refresh JWT token"""
        try:
            data = await request.json()
            refresh_token = data.get('refresh_token')
            
            payload = self.jwt_auth.decode_refresh_token(refresh_token)
            new_access_token = self.jwt_auth.create_token(payload)
            
            return web.json_response({
                'success': True,
                'access_token': new_access_token
            })
            
        except Exception as e:
            return web.json_response({
                'success': False,
                'error': 'Invalid refresh token'
            }, status=401)
    
    async def auth_logout(self, request):
        """User logout"""
        # In a real implementation, you'd invalidate the token
        return web.json_response({
            'success': True,
            'message': 'Logged out successfully'
        })
    
    async def auth_profile(self, request):
        """User profile"""
        user = request.get('user', {})
        return web.json_response({
            'success': True,
            'user': user
        })
    
    # WebSocket handlers
    async def websocket_handler(self, request):
        """Main WebSocket handler for neural updates"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        connection_id = str(uuid.uuid4())
        self.websocket_connections[connection_id] = {
            'ws': ws,
            'type': 'neural',
            'connected_at': datetime.now(),
            'client_ip': request.remote
        }
        
        ACTIVE_CONNECTIONS.inc()
        self.logger.info(f"WebSocket connected: {connection_id}")
        
        try:
            # Send welcome message
            await ws.send_str(json.dumps({
                'type': 'connected',
                'connection_id': connection_id,
                'timestamp': datetime.now().isoformat()
            }))
            
            # Handle incoming messages
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self.handle_websocket_message(connection_id, data)
                    except json.JSONDecodeError:
                        await ws.send_str(json.dumps({
                            'type': 'error',
                            'message': 'Invalid JSON'
                        }))
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.logger.error(f"WebSocket error: {ws.exception()}")
                    break
        
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
        
        finally:
            # Cleanup
            if connection_id in self.websocket_connections:
                del self.websocket_connections[connection_id]
            ACTIVE_CONNECTIONS.dec()
            self.logger.info(f"WebSocket disconnected: {connection_id}")
        
        return ws
    
    async def websocket_metrics(self, request):
        """WebSocket for metrics streaming"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        connection_id = str(uuid.uuid4())
        self.websocket_connections[connection_id] = {
            'ws': ws,
            'type': 'metrics',
            'connected_at': datetime.now(),
            'client_ip': request.remote
        }
        
        try:
            # Send metrics every 5 seconds
            while not ws.closed:
                stats = self.neural_engine.get_stats()
                metrics = await self.metrics.get_current_metrics()
                
                await ws.send_str(json.dumps({
                    'type': 'metrics_update',
                    'neural_stats': stats,
                    'server_metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }))
                
                await asyncio.sleep(5)
        
        except Exception as e:
            self.logger.error(f"Metrics WebSocket error: {e}")
        
        finally:
            if connection_id in self.websocket_connections:
                del self.websocket_connections[connection_id]
        
        return ws
    
    async def handle_websocket_message(self, connection_id, data):
        """Handle incoming WebSocket messages"""
        msg_type = data.get('type')
        ws = self.websocket_connections[connection_id]['ws']
        
        if msg_type == 'ping':
            await ws.send_str(json.dumps({
                'type': 'pong',
                'timestamp': datetime.now().isoformat()
            }))
        
        elif msg_type == 'neural_request':
            # Process neural request via WebSocket
            try:
                neural_request = NeuralRequest(
                    request_id=str(uuid.uuid4()),
                    path=data.get('path', '/'),
                    method=data.get('method', 'GET'),
                    headers=data.get('headers', {}),
                    timestamp=datetime.now(),
                    client_ip=self.websocket_connections[connection_id]['client_ip']
                )
                
                response = await self.neural_engine.process_request_async(neural_request)
                
                await ws.send_str(json.dumps({
                    'type': 'neural_response',
                    'request_id': response.request_id,
                    'classification': response.classification,
                    'confidence': response.confidence,
                    'processing_time': response.processing_time,
                    'timestamp': datetime.now().isoformat()
                }))
                
            except Exception as e:
                await ws.send_str(json.dumps({
                    'type': 'error',
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }))
    
    async def broadcast_neural_result(self, response: NeuralResponse):
        """Broadcast neural result to WebSocket clients"""
        message = {
            'type': 'neural_broadcast',
            'request_id': response.request_id,
            'classification': response.classification,
            'confidence': response.confidence,
            'processing_time': response.processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Send to all neural WebSocket connections
        disconnected = []
        for conn_id, conn_info in self.websocket_connections.items():
            if conn_info['type'] == 'neural':
                try:
                    await conn_info['ws'].send_str(json.dumps(message))
                except Exception:
                    disconnected.append(conn_id)
        
        # Clean up disconnected connections
        for conn_id in disconnected:
            del self.websocket_connections[conn_id]
    
    # Admin endpoints
    async def admin_connections(self, request):
        """Get active connections info"""
        connections = []
        for conn_id, conn_info in self.websocket_connections.items():
            connections.append({
                'id': conn_id,
                'type': conn_info['type'],
                'connected_at': conn_info['connected_at'].isoformat(),
                'client_ip': conn_info['client_ip']
            })
        
        return web.json_response({
            'total_connections': len(connections),
            'connections': connections
        })
    
    async def admin_broadcast(self, request):
        """Admin broadcast message"""
        try:
            data = await request.json()
            message = data.get('message', {})
            target_type = data.get('target_type', 'all')
            
            broadcast_message = {
                'type': 'admin_broadcast',
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
            
            count = 0
            for conn_info in self.websocket_connections.values():
                if target_type == 'all' or conn_info['type'] == target_type:
                    try:
                        await conn_info['ws'].send_str(json.dumps(broadcast_message))
                        count += 1
                    except Exception:
                        pass
            
            return web.json_response({
                'success': True,
                'broadcasted_to': count
            })
            
        except Exception as e:
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def admin_logs(self, request):
        """Get recent logs"""
        # This would integrate with your logging system
        return web.json_response({
            'logs': [
                {
                    'timestamp': datetime.now().isoformat(),
                    'level': 'INFO',
                    'message': 'Server running normally',
                    'component': 'neural_api_server'
                }
            ]
        })
    
    # Development endpoints
    async def dev_test(self, request):
        """Development test endpoint"""
        return web.json_response({
            'status': 'ok',
            'message': 'Neural API Server is running',
            'timestamp': datetime.now().isoformat(),
            'neural_engine_status': 'operational'
        })
    
    async def dev_simulate(self, request):
        """Simulate neural requests"""
        try:
            data = await request.json()
            count = data.get('count', 10)
            
            results = []
            for i in range(count):
                test_request = NeuralRequest(
                    request_id=f"sim_{i}",
                    path=f"/test/{i}",
                    method="GET",
                    headers={"Simulation": "true"},
                    timestamp=datetime.now()
                )
                
                response = await self.neural_engine.process_request_async(test_request)
                results.append({
                    'request_id': response.request_id,
                    'classification': response.classification,
                    'confidence': response.confidence,
                    'processing_time': response.processing_time
                })
            
            return web.json_response({
                'success': True,
                'simulated_requests': count,
                'results': results
            })
            
        except Exception as e:
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
    
    async def prometheus_metrics(self, request):
        """Prometheus metrics endpoint"""
        metrics_data = generate_latest()
        return web.Response(
            body=metrics_data,
            content_type=CONTENT_TYPE_LATEST
        )
    
    async def start(self):
        """Start the server"""
        await self.initialize()
        
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        
        self.logger.info(f"🚀 Neural API Server started on http://{self.host}:{self.port}")
        self.logger.info(f"📡 WebSocket endpoints: ws://{self.host}:{self.port}/api/ws/neural")
        self.logger.info(f"📊 Metrics: http://{self.host}:{self.port}/api/metrics")
        
    async def stop(self):
        """Stop the server"""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        if self.redis_client:
            await self.redis_client.close()
        
        self.logger.info("🛑 Neural API Server stopped")
    
    async def run_forever(self):
        """Run server forever"""
        try:
            await self.start()
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
        finally:
            await self.stop() 