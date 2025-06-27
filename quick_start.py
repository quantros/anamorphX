#!/usr/bin/env python3
"""
🚀 AnamorphX Enterprise Neural Server - Quick Start
==================================================

Быстрый запуск enterprise нейронного сервера для демонстрации.
Этот скрипт автоматически создает минимальную конфигурацию и запускает сервер.
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path

def print_banner():
    """Красивый баннер"""
    banner = """
🚀 ════════════════════════════════════════════════════════════════════════
🧠   AnamorphX Enterprise Neural Server - Quick Start
🏢   Автоматический запуск с минимальной конфигурацией
🌐   Готов к работе за 30 секунд!
🚀 ════════════════════════════════════════════════════════════════════════
    """
    print(banner)

def check_python_version():
    """Проверка версии Python"""
    if sys.version_info < (3, 8):
        print("❌ Требуется Python 3.8 или выше")
        print(f"   Текущая версия: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} - OK")

def install_dependencies():
    """Установка зависимостей"""
    print("📦 Проверка зависимостей...")
    
    required_packages = [
        'torch',
        'aiohttp',
        'aiofiles', 
        'pyjwt',
        'pyyaml',
        'psutil'
    ]
    
    optional_packages = [
        'redis',
        'prometheus_client'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - не установлен")
    
    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} (optional)")
        except ImportError:
            print(f"⚠️  {package} (optional) - не установлен")
    
    if missing_packages:
        print(f"\n📦 Установка отсутствующих пакетов: {', '.join(missing_packages)}")
        
        for package in missing_packages:
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', package, '--quiet'
                ])
                print(f"✅ Установлен {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Ошибка установки {package}")
                return False
    
    return True

def create_minimal_structure():
    """Создание минимальной структуры файлов"""
    print("📁 Создание структуры файлов...")
    
    # Создаем необходимые директории
    directories = [
        'anamorph_neural_engine',
        'anamorph_neural_engine/core',
        'anamorph_neural_engine/backend', 
        'anamorph_neural_engine/frontend',
        'anamorph_neural_engine/monitoring',
        'anamorph_neural_engine/security',
        'anamorph_neural_engine/utils',
        'frontend/dist',
        'logs',
        'models'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 {directory}")
    
    # Создаем __init__.py файлы если их нет
    init_files = [
        'anamorph_neural_engine/__init__.py',
        'anamorph_neural_engine/core/__init__.py',
        'anamorph_neural_engine/backend/__init__.py',
        'anamorph_neural_engine/frontend/__init__.py',
        'anamorph_neural_engine/monitoring/__init__.py',
        'anamorph_neural_engine/security/__init__.py',
        'anamorph_neural_engine/utils/__init__.py'
    ]
    
    for init_file in init_files:
        if not Path(init_file).exists():
            Path(init_file).touch()

def create_quick_config():
    """Создание быстрой конфигурации"""
    print("⚙️  Создание минимальной конфигурации...")
    
    config_content = """# Quick Start Configuration for AnamorphX Enterprise Neural Server

server:
  host: "localhost"
  port: 8080
  redis_url: null  # Отключен для упрощения

neural:
  device: "auto"
  max_workers: 2
  model_config:
    vocab_size: 1000
    embedding_dim: 64
    hidden_dim: 128
    num_layers: 2
    num_classes: 5
    dropout: 0.2

auth:
  jwt_secret: "quick-start-secret-change-in-production"

security:
  cors_origins: ["*"]
  rate_limit:
    requests_per_minute: 120

frontend:
  static_dir: "frontend/dist"
  enable_caching: false

logging:
  level: "INFO"
  file: "logs/quick_start.log"

metrics:
  redis_url: null  # Отключен для упрощения
  enable_prometheus: false
"""
    
    with open('quick_start_config.yaml', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("✅ Конфигурация создана: quick_start_config.yaml")

def create_simple_frontend():
    """Создание простого frontend для демонстрации"""
    print("🌐 Создание демо frontend...")
    
    html_content = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 AnamorphX Enterprise - Quick Start</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; min-height: 100vh; padding: 2rem;
        }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { font-size: 2.5rem; margin-bottom: 1rem; text-align: center; }
        .card {
            background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
            border-radius: 15px; padding: 2rem; margin: 1rem 0;
            border: 1px solid rgba(255,255,255,0.2);
        }
        button {
            background: #4CAF50; color: white; border: none; padding: 12px 24px;
            border-radius: 8px; cursor: pointer; font-size: 16px; margin: 8px;
        }
        button:hover { background: #45a049; }
        .status { padding: 8px; border-radius: 4px; margin: 8px 0; }
        .success { background: #4CAF50; }
        .error { background: #f44336; }
        .info { background: #2196F3; }
        #output { 
            background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px;
            margin: 1rem 0; min-height: 200px; font-family: monospace;
            white-space: pre-wrap; overflow-y: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 AnamorphX Enterprise</h1>
        <div class="card">
            <h2>Quick Start Demo</h2>
            <p>Enterprise нейронный сервер успешно запущен!</p>
            
            <button onclick="testHealth()">🏥 Health Check</button>
            <button onclick="testNeural()">🧠 Neural Test</button>
            <button onclick="testAuth()">🔐 Auth Test</button>
            <button onclick="connectWS()">🔗 WebSocket</button>
            
            <div id="status"></div>
            <div id="output"></div>
        </div>
    </div>

    <script>
        let ws = null;
        
        function log(message, type = 'info') {
            const output = document.getElementById('output');
            const timestamp = new Date().toLocaleTimeString();
            output.textContent += `[${timestamp}] ${message}\\n`;
            output.scrollTop = output.scrollHeight;
            
            const status = document.getElementById('status');
            status.innerHTML = `<div class="status ${type}">${message}</div>`;
        }
        
        async function testHealth() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                log(`✅ Health Check: ${data.status}`, 'success');
                log(JSON.stringify(data, null, 2));
            } catch (error) {
                log(`❌ Health Check Error: ${error.message}`, 'error');
            }
        }
        
        async function testNeural() {
            try {
                const response = await fetch('/api/neural/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        path: '/test/neural',
                        method: 'GET',
                        headers: { 'User-Agent': 'QuickStartDemo' }
                    })
                });
                const data = await response.json();
                log(`🧠 Neural Prediction: ${data.classification.class} (${(data.confidence*100).toFixed(1)}%)`, 'success');
                log(JSON.stringify(data, null, 2));
            } catch (error) {
                log(`❌ Neural Test Error: ${error.message}`, 'error');
            }
        }
        
        async function testAuth() {
            try {
                const response = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        username: 'demo',
                        password: 'demo123'
                    })
                });
                const data = await response.json();
                if (data.success) {
                    log(`🔐 Auth Success: Welcome ${data.user.username}`, 'success');
                } else {
                    log(`🔐 Auth Demo: ${data.error}`, 'info');
                }
            } catch (error) {
                log(`❌ Auth Test Error: ${error.message}`, 'error');
            }
        }
        
        function connectWS() {
            try {
                if (ws) ws.close();
                ws = new WebSocket('ws://localhost:8080/api/ws/neural');
                
                ws.onopen = () => log('🔗 WebSocket Connected', 'success');
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    log(`📡 WebSocket: ${data.type}`, 'info');
                };
                ws.onclose = () => log('🔗 WebSocket Disconnected', 'info');
                ws.onerror = (error) => log(`❌ WebSocket Error: ${error}`, 'error');
            } catch (error) {
                log(`❌ WebSocket Connection Error: ${error.message}`, 'error');
            }
        }
        
        // Auto-test on load
        window.addEventListener('load', () => {
            log('🚀 AnamorphX Enterprise Quick Start Demo Ready');
            setTimeout(testHealth, 1000);
        });
    </script>
</body>
</html>"""
    
    frontend_dir = Path('frontend/dist')
    with open(frontend_dir / 'index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ Demo frontend создан: frontend/dist/index.html")

def run_server():
    """Запуск сервера"""
    print("🚀 Запуск Enterprise Neural Server...")
    print("=" * 60)
    
    try:
        # Импортируем и запускаем сервер
        import subprocess
        
        cmd = [
            sys.executable, 
            'enterprise_neural_server.py',
            '--config', 'quick_start_config.yaml',
            '--log-level', 'INFO'
        ]
        
        print(f"Команда: {' '.join(cmd)}")
        print("=" * 60)
        
        # Запускаем сервер
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Сервер остановлен пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка запуска сервера: {e}")

def main():
    """Главная функция"""
    
    print_banner()
    
    print("🔍 Шаг 1: Проверка системы...")
    check_python_version()
    
    print("\n📦 Шаг 2: Установка зависимостей...")
    if not install_dependencies():
        print("❌ Не удалось установить зависимости")
        sys.exit(1)
    
    print("\n📁 Шаг 3: Создание структуры...")
    create_minimal_structure()
    
    print("\n⚙️  Шаг 4: Создание конфигурации...")
    create_quick_config()
    
    print("\n🌐 Шаг 5: Создание frontend...")
    create_simple_frontend()
    
    print("\n✅ Подготовка завершена!")
    print("=" * 60)
    print("🌐 Сервер будет доступен по адресу: http://localhost:8080")
    print("📡 API: http://localhost:8080/api")
    print("❤️  Health: http://localhost:8080/api/health")
    print("🧠 Neural: http://localhost:8080/api/neural/stats")
    print("=" * 60)
    print("🛑 Для остановки нажмите Ctrl+C")
    print("=" * 60)
    
    input("\n▶️  Нажмите Enter для запуска сервера...")
    
    run_server()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 До свидания!")
    except Exception as e:
        print(f"\n💥 Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 