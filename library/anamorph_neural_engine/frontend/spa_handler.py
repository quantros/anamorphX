"""
🌐 SPA Handler - Enterprise Frontend
Обработчик для Single Page Applications (React/Vue/Angular)
"""

import os
import json
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional, List
from aiohttp import web, hdrs
import aiofiles
import hashlib
from datetime import datetime, timedelta

class SPAHandler:
    """
    🌐 Enterprise SPA Handler
    Обслуживание современных frontend приложений
    """
    
    def __init__(self, 
                 static_dir: str = 'frontend/dist',
                 index_file: str = 'index.html',
                 api_prefix: str = '/api',
                 enable_caching: bool = True,
                 enable_compression: bool = True,
                 cache_max_age: int = 3600):
        
        self.static_dir = Path(static_dir)
        self.index_file = index_file
        self.api_prefix = api_prefix
        self.enable_caching = enable_caching
        self.enable_compression = enable_compression
        self.cache_max_age = cache_max_age
        
        # File cache for better performance
        self.file_cache = {}
        self.etag_cache = {}
        
        # MIME types configuration
        mimetypes.add_type('application/javascript', '.mjs')
        mimetypes.add_type('text/css', '.css')
        mimetypes.add_type('application/json', '.json')
        
    def setup_routes(self, app: web.Application):
        """Setup routes for SPA handling"""
        
        # Static files route
        app.router.add_get('/{path:.*}', self.handle_request)
        
    async def handle_request(self, request):
        """Handle all frontend requests"""
        
        # Skip API requests
        if request.path.startswith(self.api_prefix):
            raise web.HTTPNotFound()
        
        # Try to serve static file
        try:
            return await self.serve_static_file(request)
        except web.HTTPNotFound:
            # Fallback to index.html for SPA routing
            return await self.serve_index(request)
    
    async def serve_static_file(self, request):
        """Serve static files with caching"""
        
        path = request.path.lstrip('/')
        if not path:
            path = self.index_file
        
        file_path = self.static_dir / path
        
        # Security check - prevent directory traversal
        try:
            file_path = file_path.resolve()
            self.static_dir.resolve()
            file_path.relative_to(self.static_dir.resolve())
        except ValueError:
            raise web.HTTPForbidden()
        
        # Check if file exists
        if not file_path.exists() or not file_path.is_file():
            raise web.HTTPNotFound()
        
        # Get file stats
        stat = file_path.stat()
        last_modified = datetime.fromtimestamp(stat.st_mtime)
        file_size = stat.st_size
        
        # Generate ETag
        etag = self._generate_etag(file_path, stat)
        
        # Check cache headers
        if self.enable_caching:
            # Check If-None-Match (ETag)
            if_none_match = request.headers.get(hdrs.IF_NONE_MATCH)
            if if_none_match and etag in if_none_match:
                return web.Response(status=304)
            
            # Check If-Modified-Since
            if_modified_since = request.headers.get(hdrs.IF_MODIFIED_SINCE)
            if if_modified_since:
                try:
                    if_modified_since_date = datetime.strptime(
                        if_modified_since, '%a, %d %b %Y %H:%M:%S GMT'
                    )
                    if last_modified <= if_modified_since_date:
                        return web.Response(status=304)
                except ValueError:
                    pass
        
        # Read file content
        content = await self._read_file_cached(file_path)
        
        # Determine content type
        content_type, encoding = mimetypes.guess_type(str(file_path))
        if content_type is None:
            content_type = 'application/octet-stream'
        
        # Create response
        response = web.Response(
            body=content,
            content_type=content_type
        )
        
        # Add cache headers
        if self.enable_caching:
            response.headers[hdrs.ETAG] = etag
            response.headers[hdrs.LAST_MODIFIED] = last_modified.strftime(
                '%a, %d %b %Y %H:%M:%S GMT'
            )
            response.headers[hdrs.CACHE_CONTROL] = f'public, max-age={self.cache_max_age}'
        
        # Add content encoding if applicable
        if encoding:
            response.headers[hdrs.CONTENT_ENCODING] = encoding
        
        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        return response
    
    async def serve_index(self, request):
        """Serve index.html for SPA routing"""
        
        index_path = self.static_dir / self.index_file
        
        if not index_path.exists():
            # Generate default index.html if it doesn't exist
            return await self.serve_default_index(request)
        
        # Read index.html
        content = await self._read_file_cached(index_path)
        
        # Inject runtime configuration
        modified_content = await self._inject_config(content, request)
        
        response = web.Response(
            body=modified_content,
            content_type='text/html; charset=utf-8'
        )
        
        # No caching for index.html to ensure fresh config
        response.headers[hdrs.CACHE_CONTROL] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        # Security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "connect-src 'self' ws: wss:; "
            "img-src 'self' data: blob:; "
            "font-src 'self' data:;"
        )
        
        return response
    
    async def serve_default_index(self, request):
        """Generate and serve default index.html"""
        
        html_content = await self._generate_default_html(request)
        
        response = web.Response(
            body=html_content,
            content_type='text/html; charset=utf-8'
        )
        
        response.headers[hdrs.CACHE_CONTROL] = 'no-cache'
        return response
    
    async def _read_file_cached(self, file_path: Path) -> bytes:
        """Read file with caching"""
        
        file_key = str(file_path)
        stat = file_path.stat()
        
        # Check if cached version is still valid
        if file_key in self.file_cache:
            cached_stat, cached_content = self.file_cache[file_key]
            if cached_stat.st_mtime == stat.st_mtime:
                return cached_content
        
        # Read file
        async with aiofiles.open(file_path, mode='rb') as f:
            content = await f.read()
        
        # Cache the content
        self.file_cache[file_key] = (stat, content)
        
        return content
    
    def _generate_etag(self, file_path: Path, stat) -> str:
        """Generate ETag for file"""
        
        etag_key = f"{file_path}:{stat.st_mtime}:{stat.st_size}"
        
        if etag_key in self.etag_cache:
            return self.etag_cache[etag_key]
        
        etag = hashlib.md5(etag_key.encode()).hexdigest()
        etag = f'"{etag}"'
        
        self.etag_cache[etag_key] = etag
        return etag
    
    async def _inject_config(self, content: bytes, request) -> bytes:
        """Inject runtime configuration into HTML"""
        
        html_content = content.decode('utf-8')
        
        # Configuration object
        config = {
            'API_BASE_URL': f"http://{request.host}{self.api_prefix}",
            'WS_BASE_URL': f"ws://{request.host}{self.api_prefix}/ws",
            'VERSION': '1.0.0',
            'BUILD_TIME': datetime.now().isoformat(),
            'FEATURES': {
                'NEURAL_PROCESSING': True,
                'REAL_TIME_UPDATES': True,
                'AUTHENTICATION': True,
                'METRICS': True
            }
        }
        
        # Inject config script
        config_script = f"""
        <script>
            window.NEURAL_CONFIG = {json.dumps(config, indent=2)};
        </script>
        """
        
        # Insert before </head> or at the beginning
        if '</head>' in html_content:
            html_content = html_content.replace('</head>', f'{config_script}\n</head>')
        elif '<html>' in html_content:
            html_content = html_content.replace('<html>', f'<html>\n{config_script}')
        else:
            html_content = config_script + html_content
        
        return html_content.encode('utf-8')
    
    async def _generate_default_html(self, request) -> str:
        """Generate default HTML when no frontend is available"""
        
        config = {
            'API_BASE_URL': f"http://{request.host}{self.api_prefix}",
            'WS_BASE_URL': f"ws://{request.host}{self.api_prefix}/ws"
        }
        
        return f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 AnamorphX Neural Engine - Enterprise</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            flex: 1;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 3rem;
        }}
        
        .header h1 {{
            font-size: 3rem;
            margin-bottom: 1rem;
            background: linear-gradient(45deg, #fff, #f0f0f0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .header p {{
            font-size: 1.2rem;
            opacity: 0.9;
        }}
        
        .dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-bottom: 3rem;
        }}
        
        .card {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
        }}
        
        .card h3 {{
            font-size: 1.5rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
        }}
        
        .card .icon {{
            font-size: 2rem;
            margin-right: 0.5rem;
        }}
        
        .status {{
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        }}
        
        .status-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #4ade80;
            margin-right: 0.5rem;
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        
        .metric {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }}
        
        .metric-value {{
            font-weight: bold;
            color: #4ade80;
        }}
        
        .actions {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 3rem;
        }}
        
        .action-card {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.2);
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            color: white;
        }}
        
        .action-card:hover {{
            background: rgba(255, 255, 255, 0.2);
            transform: scale(1.05);
            color: white;
        }}
        
        .footer {{
            text-align: center;
            padding: 2rem;
            border-top: 1px solid rgba(255, 255, 255, 0.2);
            opacity: 0.8;
        }}
        
        .live-updates {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            padding: 1rem;
            min-width: 250px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        
        .update-item {{
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 0.9rem;
        }}
        
        .update-item:last-child {{
            border-bottom: none;
        }}
        
        .connection-status {{
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        }}
        
        .ws-status {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #ef4444;
            margin-right: 0.5rem;
        }}
        
        .ws-status.connected {{
            background: #4ade80;
            animation: pulse 2s infinite;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 AnamorphX Neural Engine</h1>
            <p>Enterprise Edition - Нейросетевой веб-сервер нового поколения</p>
        </div>
        
        <div class="dashboard">
            <div class="card">
                <h3><span class="icon">🧠</span>Neural Engine</h3>
                <div class="status">
                    <div class="status-dot"></div>
                    <span>Operational</span>
                </div>
                <div class="metric">
                    <span>Модель:</span>
                    <span class="metric-value">LSTM Classifier</span>
                </div>
                <div class="metric">
                    <span>Устройство:</span>
                    <span class="metric-value" id="device">Loading...</span>
                </div>
                <div class="metric">
                    <span>Параметры:</span>
                    <span class="metric-value" id="parameters">Loading...</span>
                </div>
            </div>
            
            <div class="card">
                <h3><span class="icon">📊</span>Performance</h3>
                <div class="metric">
                    <span>Запросов обработано:</span>
                    <span class="metric-value" id="requests">0</span>
                </div>
                <div class="metric">
                    <span>Среднее время:</span>
                    <span class="metric-value" id="avg-time">0ms</span>
                </div>
                <div class="metric">
                    <span>Активные соединения:</span>
                    <span class="metric-value" id="connections">0</span>
                </div>
            </div>
            
            <div class="card">
                <h3><span class="icon">🔐</span>Security</h3>
                <div class="status">
                    <div class="status-dot"></div>
                    <span>JWT Authentication</span>
                </div>
                <div class="metric">
                    <span>Rate Limiting:</span>
                    <span class="metric-value">Enabled</span>
                </div>
                <div class="metric">
                    <span>CORS:</span>
                    <span class="metric-value">Configured</span>
                </div>
            </div>
            
            <div class="card">
                <h3><span class="icon">🌐</span>API Status</h3>
                <div class="status">
                    <div class="status-dot"></div>
                    <span>REST API Active</span>
                </div>
                <div class="metric">
                    <span>WebSocket:</span>
                    <span class="metric-value">Ready</span>
                </div>
                <div class="metric">
                    <span>Version:</span>
                    <span class="metric-value">1.0.0</span>
                </div>
            </div>
        </div>
        
        <div class="actions">
            <a href="{self.api_prefix}/health" class="action-card">
                <div class="icon">❤️</div>
                <h4>Health Check</h4>
                <p>Проверка состояния системы</p>
            </a>
            
            <a href="{self.api_prefix}/neural/stats" class="action-card">
                <div class="icon">📈</div>
                <h4>Neural Stats</h4>
                <p>Статистика нейронной сети</p>
            </a>
            
            <a href="{self.api_prefix}/metrics" class="action-card">
                <div class="icon">📊</div>
                <h4>Prometheus Metrics</h4>
                <p>Метрики для мониторинга</p>
            </a>
            
            <div class="action-card" onclick="testNeural()">
                <div class="icon">🧪</div>
                <h4>Test Neural</h4>
                <p>Тестовый запрос к нейросети</p>
            </div>
        </div>
    </div>
    
    <div class="live-updates">
        <div class="connection-status">
            <div class="ws-status" id="ws-status"></div>
            <span>WebSocket: <span id="ws-text">Disconnected</span></span>
        </div>
        <div id="updates">
            <div class="update-item">Система запущена</div>
        </div>
    </div>
    
    <div class="footer">
        <p>&copy; 2024 AnamorphX Neural Engine - Enterprise Edition</p>
        <p>Powered by PyTorch • Built with ❤️ in Russia</p>
    </div>
    
    <script>
        window.NEURAL_CONFIG = {json.dumps(config, indent=8)};
        
        // WebSocket connection
        let ws = null;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        
        function connectWebSocket() {{
            try {{
                ws = new WebSocket(NEURAL_CONFIG.WS_BASE_URL + '/neural');
                
                ws.onopen = function() {{
                    console.log('WebSocket connected');
                    document.getElementById('ws-status').classList.add('connected');
                    document.getElementById('ws-text').textContent = 'Connected';
                    reconnectAttempts = 0;
                    addUpdate('🔗 WebSocket подключен');
                }};
                
                ws.onmessage = function(event) {{
                    const data = JSON.parse(event.data);
                    handleWebSocketMessage(data);
                }};
                
                ws.onclose = function() {{
                    console.log('WebSocket disconnected');
                    document.getElementById('ws-status').classList.remove('connected');
                    document.getElementById('ws-text').textContent = 'Disconnected';
                    
                    // Try to reconnect
                    if (reconnectAttempts < maxReconnectAttempts) {{
                        reconnectAttempts++;
                        setTimeout(connectWebSocket, 3000);
                        addUpdate(`🔄 Переподключение... (попытка ${{reconnectAttempts}})`);
                    }}
                }};
                
                ws.onerror = function(error) {{
                    console.error('WebSocket error:', error);
                    addUpdate('❌ Ошибка WebSocket');
                }};
            }} catch (error) {{
                console.error('Failed to connect WebSocket:', error);
            }}
        }}
        
        function handleWebSocketMessage(data) {{
            if (data.type === 'neural_broadcast') {{
                addUpdate(`🧠 Классификация: ${{data.classification.class}} (уверенность: ${{(data.confidence * 100).toFixed(1)}}%)`);
                updateMetrics();
            }} else if (data.type === 'connected') {{
                addUpdate('✅ Подключение установлено');
            }}
        }}
        
        function addUpdate(message) {{
            const updates = document.getElementById('updates');
            const item = document.createElement('div');
            item.className = 'update-item';
            item.textContent = `${{new Date().toLocaleTimeString()}}: ${{message}}`;
            updates.insertBefore(item, updates.firstChild);
            
            // Keep only last 5 updates
            while (updates.children.length > 5) {{
                updates.removeChild(updates.lastChild);
            }}
        }}
        
        async function updateMetrics() {{
            try {{
                const response = await fetch(NEURAL_CONFIG.API_BASE_URL + '/neural/stats');
                const stats = await response.json();
                
                document.getElementById('device').textContent = stats.device || 'CPU';
                document.getElementById('parameters').textContent = (stats.total_parameters || 0).toLocaleString();
                document.getElementById('requests').textContent = stats.total_requests || 0;
                document.getElementById('avg-time').textContent = ((stats.avg_processing_time || 0) * 1000).toFixed(1) + 'ms';
            }} catch (error) {{
                console.error('Failed to update metrics:', error);
            }}
        }}
        
        async function testNeural() {{
            try {{
                const response = await fetch(NEURAL_CONFIG.API_BASE_URL + '/neural/predict', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify({{
                        path: '/test/neural',
                        method: 'GET',
                        headers: {{ 'User-Agent': 'TestClient' }}
                    }})
                }});
                
                const result = await response.json();
                if (result.success) {{
                    addUpdate(`🧪 Тест: ${{result.classification.class}} (${{(result.confidence * 100).toFixed(1)}}%)`);
                }} else {{
                    addUpdate('❌ Ошибка теста: ' + result.error);
                }}
            }} catch (error) {{
                addUpdate('❌ Ошибка запроса: ' + error.message);
            }}
        }}
        
        // Initialize
        connectWebSocket();
        updateMetrics();
        setInterval(updateMetrics, 10000); // Update every 10 seconds
    </script>
</body>
</html>""" 