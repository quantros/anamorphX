#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 AnamorphX Neural Web Server Simulator (UTF-8 Fixed)
Реальный HTTP сервер на основе анализа AnamorphX кода с правильной кодировкой
"""

import sys
import os
import time
import re
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import socketserver

# Принудительная установка UTF-8
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

class NeuralWebServer:
    """Симулятор нейронного веб-сервера"""
    
    def __init__(self, anamorph_file="Project/web_server.anamorph"):
        self.anamorph_file = anamorph_file
        self.config = {}
        self.neural_network = {}
        self.api_endpoints = {}
        self.security_rules = {}
        self.running = False
        
        # Загрузка и анализ AnamorphX кода
        self._load_anamorph_config()
    
    def _load_anamorph_config(self):
        """Загрузка конфигурации из AnamorphX файла"""
        try:
            with open(self.anamorph_file, 'r', encoding='utf-8') as f:
                code = f.read()
            
            print(f"🧠 Загружен AnamorphX файл: {self.anamorph_file}")
            print(f"📄 Размер: {len(code):,} символов")
            
            # Извлечение конфигурации
            self._extract_server_config(code)
            self._extract_neural_network(code)
            self._extract_api_endpoints(code)
            self._extract_security_config(code)
            
        except Exception as e:
            print(f"❌ Ошибка загрузки AnamorphX: {e}")
            self._set_default_config()
    
    def _extract_server_config(self, code):
        """Извлечение конфигурации сервера"""
        self.config = {
            'host': 'localhost',
            'port': 8080,
            'debug': True,
            'neural_processing': True
        }
        
        # Поиск порта
        port_match = re.search(r'port[:\s]*(\d+)', code, re.IGNORECASE)
        if port_match:
            self.config['port'] = int(port_match.group(1))
        
        # Поиск хоста
        if 'localhost' in code.lower():
            self.config['host'] = 'localhost'
        elif '0.0.0.0' in code:
            self.config['host'] = '0.0.0.0'
    
    def _extract_neural_network(self, code):
        """Извлечение структуры нейронной сети"""
        self.neural_network = {
            'layers': [],
            'connections': [],
            'activations': []
        }
        
        # Улучшенный поиск нейронов
        neuron_patterns = [
            r'neuro\s+"([^"]+)"\s*\{([^}]+)\}',
            r'neuron\s+"([^"]+)"\s*\{([^}]+)\}',
            r'layer\s+"([^"]+)"\s*\{([^}]+)\}'
        ]
        
        for pattern in neuron_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE | re.DOTALL)
            for name, config in matches:
                layer = {'name': name}
                
                # Поиск параметров
                units_match = re.search(r'units[:\s]*(\d+)', config)
                if units_match:
                    layer['units'] = int(units_match.group(1))
                
                activation_match = re.search(r'activation[:\s]*["\']([^"\']+)["\']', config)
                if activation_match:
                    layer['activation'] = activation_match.group(1)
                    self.neural_network['activations'].append(activation_match.group(1))
                
                self.neural_network['layers'].append(layer)
        
        # Поиск связей
        synapse_patterns = [
            r'synap\s+"([^"]+)"\s*->\s*"([^"]+)"',
            r'connect\s+"([^"]+)"\s*->\s*"([^"]+)"',
            r'(\w+)\s*->\s*(\w+)'
        ]
        
        for pattern in synapse_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            for source, target in matches:
                self.neural_network['connections'].append({
                    'from': source,
                    'to': target
                })
    
    def _extract_api_endpoints(self, code):
        """Извлечение API эндпоинтов"""
        self.api_endpoints = {
            '/': 'index',
            '/api': 'api_info', 
            '/health': 'health_check',
            '/neural': 'neural_status',
            '/admin': 'admin_panel'
        }
        
        # Поиск дополнительных маршрутов
        route_matches = re.findall(r'route[:\s]*["\']([^"\']+)["\']', code, re.IGNORECASE)
        for route in route_matches:
            if route not in self.api_endpoints:
                self.api_endpoints[route] = 'custom_handler'
    
    def _extract_security_config(self, code):
        """Извлечение настроек безопасности"""
        self.security_rules = {
            'auth_required': 'auth' in code.lower(),
            'rate_limiting': 'throttle' in code.lower() or 'rate' in code.lower(),
            'encryption': 'encrypt' in code.lower() or 'ssl' in code.lower(),
            'audit_log': 'audit' in code.lower() or 'log' in code.lower()
        }
    
    def _set_default_config(self):
        """Настройки по умолчанию"""
        self.config = {'host': 'localhost', 'port': 8080}
        self.neural_network = {'layers': [], 'connections': []}
        self.api_endpoints = {'/': 'index', '/health': 'health'}
        self.security_rules = {}

class NeuralRequestHandler(BaseHTTPRequestHandler):
    """Обработчик HTTP запросов с нейронной логикой"""
    
    def __init__(self, server_instance, *args, **kwargs):
        self.server_instance = server_instance
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Обработка GET запросов"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # Логирование
        print(f"🌐 GET {path} от {self.client_address[0]}")
        
        # Маршрутизация
        if path in self.server_instance.api_endpoints:
            handler_name = self.server_instance.api_endpoints[path]
            response = self._handle_route(handler_name, path)
        else:
            response = self._handle_404()
        
        # Отправка ответа
        self._send_response(response)
    
    def do_POST(self):
        """Обработка POST запросов"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        print(f"🌐 POST {path} от {self.client_address[0]}")
        
        # Чтение данных
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b''
        
        response = self._handle_post(path, post_data)
        self._send_response(response)
    
    def _handle_route(self, handler_name, path):
        """Обработка конкретного маршрута"""
        if handler_name == 'index':
            return self._handle_index()
        elif handler_name == 'api_info':
            return self._handle_api_info()
        elif handler_name == 'health_check':
            return self._handle_health()
        elif handler_name == 'neural_status':
            return self._handle_neural_status()
        elif handler_name == 'admin_panel':
            return self._handle_admin()
        else:
            return self._handle_custom(path)
    
    def _handle_index(self):
        """Главная страница"""
        network = self.server_instance.neural_network
        html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 AnamorphX Neural Web Server</title>
    <style>
        body {{ font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
               min-height: 100vh; color: #333; }}
        .container {{ background: rgba(255,255,255,0.95); padding: 40px; border-radius: 20px; 
                     box-shadow: 0 20px 40px rgba(0,0,0,0.1); backdrop-filter: blur(10px); }}
        h1 {{ color: #4CAF50; text-align: center; font-size: 2.5em; margin-bottom: 10px; }}
        .neural {{ color: #4CAF50; text-align: center; font-size: 1.2em; margin-bottom: 30px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .card {{ background: white; padding: 20px; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
        .endpoint {{ background: linear-gradient(45deg, #e8f5e8, #f0f8f0); 
                    padding: 12px; margin: 8px 0; border-radius: 8px; 
                    border-left: 4px solid #4CAF50; }}
        .endpoint a {{ text-decoration: none; color: #2E7D32; font-weight: 500; }}
        .endpoint a:hover {{ color: #1B5E20; }}
        .stat {{ display: inline-block; background: #e3f2fd; padding: 8px 15px; 
                margin: 5px; border-radius: 20px; font-weight: bold; }}
        .security-ok {{ color: #4CAF50; }}
        .security-no {{ color: #f44336; }}
        .footer {{ text-align: center; margin-top: 30px; opacity: 0.7; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 AnamorphX Neural Web Server</h1>
        <p class="neural">Нейронный веб-сервер работает!</p>
        
        <div class="grid">
            <div class="card">
                <h2>📡 Доступные эндпоинты:</h2>
                {self._format_endpoints()}
            </div>
            
            <div class="card">
                <h2>🧠 Нейронная сеть:</h2>
                <div class="stat">Слоёв: {len(network['layers'])}</div>
                <div class="stat">Связей: {len(network['connections'])}</div>
                {self._format_neural_details()}
            </div>
            
            <div class="card">
                <h2>🔒 Безопасность:</h2>
                {self._format_security_info()}
            </div>
            
            <div class="card">
                <h2>📊 Статистика:</h2>
                <div class="stat">Файл: {os.path.basename(self.server_instance.anamorph_file)}</div>
                <div class="stat">Порт: {self.server_instance.config['port']}</div>
                <div class="stat">Запущен: {time.strftime('%H:%M:%S')}</div>
            </div>
        </div>
        
        <div class="footer">
            <p>🚀 Powered by AnamorphX Neural Engine | 
            <a href="/api">API Docs</a> | 
            <a href="/health">Health Check</a></p>
        </div>
    </div>
</body>
</html>"""
        return {'status': 200, 'content': html, 'content_type': 'text/html'}
    
    def _handle_api_info(self):
        """API информация"""
        api_info = {
            'server': 'AnamorphX Neural Web Server',
            'version': '1.0.0',
            'status': 'active',
            'neural_processing': True,
            'endpoints': list(self.server_instance.api_endpoints.keys()),
            'neural_network': {
                'layers': len(self.server_instance.neural_network['layers']),
                'connections': len(self.server_instance.neural_network['connections']),
                'activations': self.server_instance.neural_network['activations']
            },
            'security': self.server_instance.security_rules,
            'config': {
                'host': self.server_instance.config['host'],
                'port': self.server_instance.config['port']
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'uptime': time.time()
        }
        return {'status': 200, 'content': json.dumps(api_info, indent=2, ensure_ascii=False), 'content_type': 'application/json'}
    
    def _handle_health(self):
        """Проверка здоровья"""
        health = {
            'status': 'healthy',
            'neural_network': 'active',
            'layers_loaded': len(self.server_instance.neural_network['layers']),
            'connections_active': len(self.server_instance.neural_network['connections']),
            'uptime_seconds': int(time.time()),
            'memory': 'optimal',
            'connections': 'stable',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        return {'status': 200, 'content': json.dumps(health, indent=2, ensure_ascii=False), 'content_type': 'application/json'}
    
    def _handle_neural_status(self):
        """Статус нейронной сети"""
        neural_data = dict(self.server_instance.neural_network)
        neural_data['total_parameters'] = sum(layer.get('units', 0) for layer in neural_data['layers'])
        neural_data['unique_activations'] = list(set(neural_data['activations']))
        
        return {
            'status': 200,
            'content': json.dumps(neural_data, indent=2, ensure_ascii=False),
            'content_type': 'application/json'
        }
    
    def _handle_admin(self):
        """Админ панель"""
        network = self.server_instance.neural_network
        html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>🔧 Админ панель AnamorphX</title>
    <style>
        body {{ font-family: 'SF Mono', Monaco, monospace; margin: 40px; background: #1a1a1a; color: #00ff00; }}
        .panel {{ background: #000; padding: 30px; border-radius: 10px; border: 1px solid #00ff00; }}
        .status {{ margin: 10px 0; }}
        .ok {{ color: #00ff00; }}
        .warning {{ color: #ffaa00; }}
        .error {{ color: #ff0000; }}
    </style>
</head>
<body>
    <div class="panel">
        <h1>🔧 AnamorphX Neural Server Admin</h1>
        <div class="status ok">🧠 Нейронная сеть: АКТИВНА</div>
        <div class="status ok">📊 Слоёв загружено: {len(network['layers'])}</div>
        <div class="status ok">🔗 Связей активно: {len(network['connections'])}</div>
        <div class="status ok">🔒 Безопасность: ВКЛЮЧЕНА</div>
        <div class="status ok">💾 Память: ОПТИМАЛЬНА</div>
        <div class="status ok">🌐 Сеть: СТАБИЛЬНА</div>
        <hr>
        <p>Время работы: {time.strftime('%H:%M:%S')}</p>
        <p>Файл конфигурации: {self.server_instance.anamorph_file}</p>
    </div>
</body>
</html>"""
        return {'status': 200, 'content': html, 'content_type': 'text/html'}
    
    def _handle_custom(self, path):
        """Кастомный обработчик"""
        return {
            'status': 200,
            'content': f'<h1>🧠 Neural Route: {path}</h1><p>Обработано нейронной сетью AnamorphX</p>',
            'content_type': 'text/html'
        }
    
    def _handle_post(self, path, data):
        """Обработка POST запросов"""
        return {
            'status': 200,
            'content': json.dumps({'received': len(data), 'path': path, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}, ensure_ascii=False),
            'content_type': 'application/json'
        }
    
    def _handle_404(self):
        """404 ошибка"""
        return {
            'status': 404,
            'content': '<h1>404 - Нейронный путь не найден</h1><p>🧠 Данный маршрут не существует в нейронной сети</p>',
            'content_type': 'text/html'
        }
    
    def _format_endpoints(self):
        """Форматирование списка эндпоинтов"""
        html = ""
        for endpoint in self.server_instance.api_endpoints.keys():
            html += f'<div class="endpoint">📡 <a href="{endpoint}">{endpoint}</a></div>'
        return html
    
    def _format_neural_details(self):
        """Детальная информация о нейронной сети"""
        network = self.server_instance.neural_network
        if not network['layers']:
            return "<p>Нейроны загружаются...</p>"
        
        html = "<div style='margin-top: 15px;'>"
        for i, layer in enumerate(network['layers'][:3]):
            units = layer.get('units', 'N/A')
            activation = layer.get('activation', 'linear')
            html += f"<div style='margin: 5px 0;'>• <strong>{layer['name']}</strong>: {units} нейронов, {activation}</div>"
        
        if len(network['layers']) > 3:
            html += f"<div style='margin: 5px 0;'>... и ещё {len(network['layers']) - 3} слоёв</div>"
        
        html += "</div>"
        return html
    
    def _format_security_info(self):
        """Форматирование информации о безопасности"""
        security = self.server_instance.security_rules
        html = ""
        
        security_names = {
            'auth_required': 'Авторизация',
            'rate_limiting': 'Ограничение запросов', 
            'encryption': 'Шифрование',
            'audit_log': 'Журнал аудита'
        }
        
        for rule, enabled in security.items():
            name = security_names.get(rule, rule)
            status = "✅" if enabled else "❌"
            css_class = "security-ok" if enabled else "security-no"
            html += f"<div class='{css_class}'>{status} {name}</div>"
        
        return html
    
    def _send_response(self, response):
        """Отправка HTTP ответа с UTF-8"""
        self.send_response(response['status'])
        self.send_header('Content-type', response['content_type'] + '; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.end_headers()
        
        content = response['content']
        if isinstance(content, str):
            content = content.encode('utf-8')
        
        self.wfile.write(content)
    
    def log_message(self, format, *args):
        """Подавление стандартного логирования"""
        pass

def create_handler(server_instance):
    """Создание обработчика с привязкой к серверу"""
    def handler(*args, **kwargs):
        NeuralRequestHandler(server_instance, *args, **kwargs)
    return handler

def main():
    """Главная функция запуска сервера"""
    if len(sys.argv) > 1:
        anamorph_file = sys.argv[1]
    else:
        anamorph_file = "Project/web_server.anamorph"
    
    print("🧠 AnamorphX Neural Web Server (UTF-8)")
    print("=" * 50)
    
    # Создание экземпляра сервера
    neural_server = NeuralWebServer(anamorph_file)
    
    host = neural_server.config['host']
    port = neural_server.config['port']
    
    print(f"🚀 Запуск сервера...")
    print(f"📡 Хост: {host}")
    print(f"🔌 Порт: {port}")
    print(f"🧠 Нейронных слоев: {len(neural_server.neural_network['layers'])}")
    print(f"🔗 Связей: {len(neural_server.neural_network['connections'])}")
    print(f"📡 Эндпоинтов: {len(neural_server.api_endpoints)}")
    
    try:
        # Создание HTTP сервера
        handler = create_handler(neural_server)
        httpd = HTTPServer((host, port), handler)
        
        neural_server.running = True
        
        print(f"")
        print(f"✅ Сервер запущен!")
        print(f"🌐 URL: http://{host}:{port}")
        print(f"📡 API: http://{host}:{port}/api")
        print(f"💚 Health: http://{host}:{port}/health")
        print(f"🧠 Neural: http://{host}:{port}/neural")
        print(f"🔧 Admin: http://{host}:{port}/admin")
        print(f"")
        print(f"🛑 Для остановки нажмите Ctrl+C")
        print("=" * 50)
        
        # Запуск сервера
        httpd.serve_forever()
        
    except KeyboardInterrupt:
        print(f"\n🛑 Получен сигнал остановки")
        neural_server.running = False
        httpd.shutdown()
        print(f"✅ Сервер остановлен")
    except Exception as e:
        print(f"❌ Ошибка запуска сервера: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 