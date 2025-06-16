"""
Профайлер производительности для языка Anamorph

Анализирует:
- Время выполнения функций и методов
- Использование памяти
- Производительность нейронных операций
- Узкие места в коде
- Статистику выполнения
"""

import time
import threading
import psutil
import gc
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from functools import wraps
import json
import traceback
import sys
import os


@dataclass
class FunctionProfile:
    """Профиль функции"""
    name: str
    file_path: str
    line: int
    call_count: int = 0
    total_time: float = 0.0
    self_time: float = 0.0
    max_time: float = 0.0
    min_time: float = float('inf')
    memory_usage: List[float] = field(default_factory=list)
    child_calls: Dict[str, int] = field(default_factory=dict)
    
    @property
    def avg_time(self) -> float:
        return self.total_time / self.call_count if self.call_count > 0 else 0.0
    
    @property
    def avg_memory(self) -> float:
        return sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0.0


@dataclass
class NeuralProfile:
    """Профиль нейронных операций"""
    operation_type: str
    neural_object: str
    execution_count: int = 0
    total_time: float = 0.0
    total_activations: int = 0
    total_weight_updates: int = 0
    signal_throughput: List[float] = field(default_factory=list)
    computation_complexity: float = 0.0
    
    @property
    def avg_time_per_operation(self) -> float:
        return self.total_time / self.execution_count if self.execution_count > 0 else 0.0
    
    @property
    def avg_throughput(self) -> float:
        return sum(self.signal_throughput) / len(self.signal_throughput) if self.signal_throughput else 0.0


@dataclass
class MemorySnapshot:
    """Снимок состояния памяти"""
    timestamp: float
    total_memory: float
    heap_size: float
    objects_count: int
    neural_objects_count: int
    garbage_collected: int


@dataclass
class PerformanceEvent:
    """Событие производительности"""
    timestamp: float
    event_type: str
    function_name: str
    duration: float
    memory_delta: float
    details: Dict[str, Any] = field(default_factory=dict)


class CallStack:
    """Стек вызовов для профилирования"""
    
    def __init__(self):
        self.stack: List[Tuple[str, float, float]] = []  # (function, start_time, start_memory)
    
    def push(self, function_name: str):
        """Добавить функцию в стек"""
        current_time = time.perf_counter()
        current_memory = self._get_memory_usage()
        self.stack.append((function_name, current_time, current_memory))
    
    def pop(self) -> Optional[Tuple[str, float, float]]:
        """Убрать функцию из стека"""
        if self.stack:
            function_name, start_time, start_memory = self.stack.pop()
            current_time = time.perf_counter()
            current_memory = self._get_memory_usage()
            
            duration = current_time - start_time
            memory_delta = current_memory - start_memory
            
            return function_name, duration, memory_delta
        return None
    
    def current_function(self) -> Optional[str]:
        """Текущая функция"""
        return self.stack[-1][0] if self.stack else None
    
    def depth(self) -> int:
        """Глубина стека"""
        return len(self.stack)
    
    def _get_memory_usage(self) -> float:
        """Получить использование памяти в МБ"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0


class PerformanceAnalyzer:
    """Анализатор производительности"""
    
    def __init__(self):
        self.function_profiles: Dict[str, FunctionProfile] = {}
        self.neural_profiles: Dict[str, NeuralProfile] = {}
        self.memory_snapshots: List[MemorySnapshot] = []
        self.performance_events: deque = deque(maxlen=10000)
        
        self.call_stack = CallStack()
        self.is_profiling = False
        self.start_time = 0.0
        self.end_time = 0.0
        
        # Настройки
        self.memory_tracking = True
        self.neural_tracking = True
        self.detailed_analysis = True
        
        # Статистики
        self.total_function_calls = 0
        self.total_neural_operations = 0
        self.peak_memory_usage = 0.0
        
        # Потоки для мониторинга
        self.memory_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
    
    def start_profiling(self):
        """Начать профилирование"""
        self.is_profiling = True
        self.start_time = time.perf_counter()
        
        # Сброс статистик
        self.function_profiles.clear()
        self.neural_profiles.clear()
        self.memory_snapshots.clear()
        self.performance_events.clear()
        
        # Запуск мониторинга памяти
        if self.memory_tracking:
            self.monitoring_active = True
            self.memory_thread = threading.Thread(target=self._memory_monitor)
            self.memory_thread.daemon = True
            self.memory_thread.start()
        
        print("🎯 Профилирование начато")
    
    def stop_profiling(self):
        """Остановить профилирование"""
        self.is_profiling = False
        self.end_time = time.perf_counter()
        self.monitoring_active = False
        
        if self.memory_thread:
            self.memory_thread.join(timeout=1.0)
        
        print(f"⏱️ Профилирование завершено ({self.end_time - self.start_time:.3f}s)")
        return self.generate_report()
    
    def profile_function_call(self, function_name: str, file_path: str = "", line: int = 0):
        """Декоратор для профилирования функции"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.is_profiling:
                    return func(*args, **kwargs)
                
                # Начало профилирования
                self.call_stack.push(function_name)
                start_time = time.perf_counter()
                start_memory = self._get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    # Завершение профилирования
                    end_time = time.perf_counter()
                    end_memory = self._get_memory_usage()
                    
                    duration = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    self._record_function_call(function_name, file_path, line, duration, memory_delta)
                    self.call_stack.pop()
            
            return wrapper
        return decorator
    
    def profile_neural_operation(self, operation_type: str, neural_object: str):
        """Профилирование нейронной операции"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.is_profiling or not self.neural_tracking:
                    return func(*args, **kwargs)
                
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    
                    self._record_neural_operation(operation_type, neural_object, duration, kwargs)
            
            return wrapper
        return decorator
    
    def _record_function_call(self, function_name: str, file_path: str, line: int, 
                             duration: float, memory_delta: float):
        """Записать вызов функции"""
        if function_name not in self.function_profiles:
            self.function_profiles[function_name] = FunctionProfile(
                name=function_name,
                file_path=file_path,
                line=line
            )
        
        profile = self.function_profiles[function_name]
        profile.call_count += 1
        profile.total_time += duration
        profile.max_time = max(profile.max_time, duration)
        profile.min_time = min(profile.min_time, duration)
        
        if memory_delta != 0:
            profile.memory_usage.append(abs(memory_delta))
        
        # Связи между функциями
        parent_function = self.call_stack.current_function()
        if parent_function and parent_function != function_name:
            if parent_function not in profile.child_calls:
                profile.child_calls[parent_function] = 0
            profile.child_calls[parent_function] += 1
        
        # Событие производительности
        event = PerformanceEvent(
            timestamp=time.perf_counter(),
            event_type="function_call",
            function_name=function_name,
            duration=duration,
            memory_delta=memory_delta,
            details={
                'call_count': profile.call_count,
                'stack_depth': self.call_stack.depth()
            }
        )
        self.performance_events.append(event)
        
        self.total_function_calls += 1
    
    def _record_neural_operation(self, operation_type: str, neural_object: str, 
                                duration: float, operation_data: Dict):
        """Записать нейронную операцию"""
        key = f"{neural_object}::{operation_type}"
        
        if key not in self.neural_profiles:
            self.neural_profiles[key] = NeuralProfile(
                operation_type=operation_type,
                neural_object=neural_object
            )
        
        profile = self.neural_profiles[key]
        profile.execution_count += 1
        profile.total_time += duration
        
        # Анализ операции
        if operation_type in ['activate', 'forward']:
            profile.total_activations += 1
        elif operation_type in ['update_weights', 'backward']:
            profile.total_weight_updates += 1
        
        # Пропускная способность сигналов
        if 'signal_throughput' in operation_data:
            profile.signal_throughput.append(operation_data['signal_throughput'])
        
        # Сложность вычислений (приблизительная)
        if 'matrix_size' in operation_data:
            profile.computation_complexity += operation_data['matrix_size'] ** 2
        
        self.total_neural_operations += 1
    
    def _memory_monitor(self):
        """Мониторинг памяти в отдельном потоке"""
        while self.monitoring_active:
            try:
                snapshot = MemorySnapshot(
                    timestamp=time.perf_counter(),
                    total_memory=self._get_memory_usage(),
                    heap_size=self._get_heap_size(),
                    objects_count=len(gc.get_objects()),
                    neural_objects_count=self._count_neural_objects(),
                    garbage_collected=gc.get_count()[0]
                )
                
                self.memory_snapshots.append(snapshot)
                self.peak_memory_usage = max(self.peak_memory_usage, snapshot.total_memory)
                
                # Ограничиваем историю
                if len(self.memory_snapshots) > 1000:
                    self.memory_snapshots = self.memory_snapshots[-1000:]
                
                time.sleep(0.1)  # Обновление каждые 100мс
                
            except Exception:
                pass
    
    def _get_memory_usage(self) -> float:
        """Получить использование памяти в МБ"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _get_heap_size(self) -> float:
        """Получить размер кучи Python"""
        try:
            return sys.getsizeof(gc.get_objects()) / 1024 / 1024
        except:
            return 0.0
    
    def _count_neural_objects(self) -> int:
        """Подсчет нейронных объектов"""
        count = 0
        try:
            for obj in gc.get_objects():
                obj_type = type(obj).__name__
                if any(neural_type in obj_type.lower() 
                      for neural_type in ['neuron', 'synapse', 'signal', 'network', 'layer']):
                    count += 1
        except:
            pass
        return count
    
    def get_top_functions(self, n: int = 10, sort_by: str = 'total_time') -> List[FunctionProfile]:
        """Получить топ функций по времени выполнения"""
        profiles = list(self.function_profiles.values())
        
        if sort_by == 'total_time':
            profiles.sort(key=lambda p: p.total_time, reverse=True)
        elif sort_by == 'call_count':
            profiles.sort(key=lambda p: p.call_count, reverse=True)
        elif sort_by == 'avg_time':
            profiles.sort(key=lambda p: p.avg_time, reverse=True)
        elif sort_by == 'memory':
            profiles.sort(key=lambda p: p.avg_memory, reverse=True)
        
        return profiles[:n]
    
    def get_neural_bottlenecks(self, n: int = 10) -> List[NeuralProfile]:
        """Получить узкие места в нейронных операциях"""
        profiles = list(self.neural_profiles.values())
        profiles.sort(key=lambda p: p.total_time, reverse=True)
        return profiles[:n]
    
    def get_memory_timeline(self) -> List[Tuple[float, float]]:
        """Получить временную линию использования памяти"""
        return [(s.timestamp - self.start_time, s.total_memory) for s in self.memory_snapshots]
    
    def analyze_performance_hotspots(self) -> Dict[str, Any]:
        """Анализ узких мест производительности"""
        analysis = {
            'cpu_hotspots': [],
            'memory_hotspots': [],
            'neural_bottlenecks': [],
            'call_patterns': {},
            'recommendations': []
        }
        
        # CPU узкие места
        cpu_hotspots = self.get_top_functions(5, 'total_time')
        for profile in cpu_hotspots:
            analysis['cpu_hotspots'].append({
                'function': profile.name,
                'total_time': profile.total_time,
                'percentage': (profile.total_time / (self.end_time - self.start_time)) * 100,
                'call_count': profile.call_count,
                'avg_time': profile.avg_time
            })
        
        # Память узкие места
        memory_hotspots = self.get_top_functions(5, 'memory')
        for profile in memory_hotspots:
            if profile.avg_memory > 0:
                analysis['memory_hotspots'].append({
                    'function': profile.name,
                    'avg_memory': profile.avg_memory,
                    'total_allocations': len(profile.memory_usage),
                    'peak_memory': max(profile.memory_usage) if profile.memory_usage else 0
                })
        
        # Нейронные узкие места
        neural_hotspots = self.get_neural_bottlenecks(5)
        for profile in neural_hotspots:
            analysis['neural_bottlenecks'].append({
                'operation': f"{profile.neural_object}::{profile.operation_type}",
                'total_time': profile.total_time,
                'execution_count': profile.execution_count,
                'avg_time': profile.avg_time_per_operation,
                'throughput': profile.avg_throughput
            })
        
        # Рекомендации по оптимизации
        analysis['recommendations'] = self._generate_recommendations()
        
        return analysis
    
    def _generate_recommendations(self) -> List[str]:
        """Генерация рекомендаций по оптимизации"""
        recommendations = []
        
        # Анализ частых вызовов
        frequent_calls = [p for p in self.function_profiles.values() if p.call_count > 1000]
        if frequent_calls:
            recommendations.append(
                f"🔥 Найдено {len(frequent_calls)} функций с >1000 вызовов - рассмотрите кэширование"
            )
        
        # Анализ памяти
        if self.peak_memory_usage > 500:  # >500MB
            recommendations.append(
                f"💾 Пиковое использование памяти {self.peak_memory_usage:.1f}MB - оптимизируйте структуры данных"
            )
        
        # Анализ нейронных операций
        slow_neural_ops = [p for p in self.neural_profiles.values() if p.avg_time_per_operation > 0.01]
        if slow_neural_ops:
            recommendations.append(
                f"🧠 Найдено {len(slow_neural_ops)} медленных нейронных операций - используйте векторизацию"
            )
        
        # Анализ глубины стека
        deep_calls = [e for e in self.performance_events if e.details.get('stack_depth', 0) > 20]
        if deep_calls:
            recommendations.append(
                "📚 Обнаружены глубокие вызовы функций - избегайте глубокой рекурсии"
            )
        
        return recommendations
    
    def generate_report(self) -> Dict[str, Any]:
        """Генерация отчета о производительности"""
        total_time = self.end_time - self.start_time
        
        report = {
            'summary': {
                'total_execution_time': total_time,
                'total_function_calls': self.total_function_calls,
                'total_neural_operations': self.total_neural_operations,
                'peak_memory_usage': self.peak_memory_usage,
                'average_memory_usage': self._calculate_avg_memory(),
                'functions_profiled': len(self.function_profiles),
                'neural_operations_profiled': len(self.neural_profiles)
            },
            'top_functions': [
                {
                    'name': p.name,
                    'total_time': p.total_time,
                    'percentage': (p.total_time / total_time) * 100,
                    'call_count': p.call_count,
                    'avg_time': p.avg_time,
                    'file_path': p.file_path
                }
                for p in self.get_top_functions(10)
            ],
            'neural_performance': [
                {
                    'operation': f"{p.neural_object}::{p.operation_type}",
                    'total_time': p.total_time,
                    'execution_count': p.execution_count,
                    'avg_time': p.avg_time_per_operation,
                    'throughput': p.avg_throughput,
                    'activations': p.total_activations,
                    'weight_updates': p.total_weight_updates
                }
                for p in self.get_neural_bottlenecks(10)
            ],
            'memory_analysis': {
                'peak_usage': self.peak_memory_usage,
                'average_usage': self._calculate_avg_memory(),
                'snapshots_count': len(self.memory_snapshots),
                'timeline': self.get_memory_timeline()[-100:]  # Последние 100 точек
            },
            'performance_analysis': self.analyze_performance_hotspots(),
            'metadata': {
                'profiler_version': '1.0.0',
                'python_version': sys.version,
                'platform': sys.platform,
                'timestamp': time.time()
            }
        }
        
        return report
    
    def _calculate_avg_memory(self) -> float:
        """Расчет среднего использования памяти"""
        if not self.memory_snapshots:
            return 0.0
        return sum(s.total_memory for s in self.memory_snapshots) / len(self.memory_snapshots)
    
    def export_report(self, filename: str, format: str = 'json'):
        """Экспорт отчета в файл"""
        report = self.generate_report()
        
        if format == 'json':
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        elif format == 'html':
            html_content = self._generate_html_report(report)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        print(f"📊 Отчет сохранен: {filename}")
    
    def _generate_html_report(self, report: Dict) -> str:
        """Генерация HTML отчета"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AnamorphX Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .chart {{ width: 100%; height: 300px; background: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>🎯 AnamorphX Performance Report</h1>
            
            <div class="summary">
                <h2>📊 Сводка</h2>
                <p><strong>Общее время выполнения:</strong> {report['summary']['total_execution_time']:.3f}s</p>
                <p><strong>Всего вызовов функций:</strong> {report['summary']['total_function_calls']}</p>
                <p><strong>Нейронных операций:</strong> {report['summary']['total_neural_operations']}</p>
                <p><strong>Пиковая память:</strong> {report['summary']['peak_memory_usage']:.1f}MB</p>
            </div>
            
            <div class="section">
                <h2>🔥 Топ функций по времени</h2>
                <table>
                    <tr><th>Функция</th><th>Время (s)</th><th>%</th><th>Вызовов</th><th>Среднее (ms)</th></tr>
        """
        
        for func in report['top_functions'][:10]:
            html += f"""
                    <tr>
                        <td>{func['name']}</td>
                        <td>{func['total_time']:.3f}</td>
                        <td>{func['percentage']:.1f}%</td>
                        <td>{func['call_count']}</td>
                        <td>{func['avg_time']*1000:.2f}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>🧠 Нейронная производительность</h2>
                <table>
                    <tr><th>Операция</th><th>Время (s)</th><th>Выполнений</th><th>Среднее (ms)</th><th>Активации</th></tr>
        """
        
        for neural in report['neural_performance'][:10]:
            html += f"""
                    <tr>
                        <td>{neural['operation']}</td>
                        <td>{neural['total_time']:.3f}</td>
                        <td>{neural['execution_count']}</td>
                        <td>{neural['avg_time']*1000:.2f}</td>
                        <td>{neural['activations']}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>💡 Рекомендации</h2>
                <ul>
        """
        
        for rec in report['performance_analysis']['recommendations']:
            html += f"<li>{rec}</li>"
        
        html += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html


class AnamorphProfiler:
    """Главный класс профайлера для Anamorph"""
    
    def __init__(self):
        self.analyzer = PerformanceAnalyzer()
        self.active_sessions: Dict[str, PerformanceAnalyzer] = {}
    
    def create_session(self, session_name: str = "default") -> PerformanceAnalyzer:
        """Создать сессию профилирования"""
        analyzer = PerformanceAnalyzer()
        self.active_sessions[session_name] = analyzer
        return analyzer
    
    def start_session(self, session_name: str = "default"):
        """Запустить сессию профилирования"""
        if session_name in self.active_sessions:
            self.active_sessions[session_name].start_profiling()
        else:
            analyzer = self.create_session(session_name)
            analyzer.start_profiling()
    
    def stop_session(self, session_name: str = "default") -> Optional[Dict]:
        """Остановить сессию профилирования"""
        if session_name in self.active_sessions:
            return self.active_sessions[session_name].stop_profiling()
        return None
    
    def profile_function(self, session_name: str = "default"):
        """Декоратор для профилирования функции"""
        def decorator(func):
            function_name = func.__name__
            file_path = func.__code__.co_filename
            line = func.__code__.co_firstlineno
            
            if session_name not in self.active_sessions:
                self.create_session(session_name)
            
            return self.active_sessions[session_name].profile_function_call(
                function_name, file_path, line
            )(func)
        
        return decorator
    
    def profile_neural(self, operation_type: str, neural_object: str, session_name: str = "default"):
        """Декоратор для профилирования нейронных операций"""
        def decorator(func):
            if session_name not in self.active_sessions:
                self.create_session(session_name)
            
            return self.active_sessions[session_name].profile_neural_operation(
                operation_type, neural_object
            )(func)
        
        return decorator


# Глобальный профайлер
_global_profiler = AnamorphProfiler()

# Удобные функции
def start_profiling(session_name: str = "default"):
    """Начать глобальное профилирование"""
    _global_profiler.start_session(session_name)

def stop_profiling(session_name: str = "default") -> Optional[Dict]:
    """Остановить глобальное профилирование"""
    return _global_profiler.stop_session(session_name)

def profile(session_name: str = "default"):
    """Декоратор для профилирования функции"""
    return _global_profiler.profile_function(session_name)

def profile_neural(operation_type: str, neural_object: str = "unknown", session_name: str = "default"):
    """Декоратор для профилирования нейронных операций"""
    return _global_profiler.profile_neural(operation_type, neural_object, session_name)


if __name__ == "__main__":
    # Тестирование профайлера
    import random
    import numpy as np
    
    # Создание тестовых функций
    @profile()
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    @profile()
    def matrix_multiplication(size):
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        return np.dot(a, b)
    
    @profile_neural("activation", "test_neuron")
    def neural_activation(inputs):
        # Эмуляция нейронной активации
        weights = np.random.rand(len(inputs))
        activation = np.sum(inputs * weights)
        return 1 / (1 + np.exp(-activation))  # Sigmoid
    
    @profile_neural("forward_pass", "test_network")
    def forward_pass(network_size, input_data):
        # Эмуляция прямого прохода
        result = input_data
        for i in range(network_size):
            result = neural_activation(result)
            time.sleep(0.001)  # Эмуляция вычислений
        return result
    
    # Запуск профилирования
    print("🎯 Запуск тестового профилирования...")
    start_profiling("test_session")
    
    # Тестовые вычисления
    print("  Вычисление чисел Фибоначчи...")
    for i in range(1, 20):
        fibonacci(i)
    
    print("  Умножение матриц...")
    for size in [50, 100, 150]:
        matrix_multiplication(size)
    
    print("  Нейронные операции...")
    for i in range(100):
        input_data = np.random.rand(10)
        forward_pass(5, input_data)
    
    # Остановка и получение отчета
    print("⏱️ Завершение профилирования...")
    report = stop_profiling("test_session")
    
    if report:
        print(f"\n📊 Отчет о производительности:")
        print(f"  Общее время: {report['summary']['total_execution_time']:.3f}s")
        print(f"  Вызовов функций: {report['summary']['total_function_calls']}")
        print(f"  Нейронных операций: {report['summary']['total_neural_operations']}")
        print(f"  Пиковая память: {report['summary']['peak_memory_usage']:.1f}MB")
        
        print(f"\n🔥 Топ-5 функций по времени:")
        for i, func in enumerate(report['top_functions'][:5], 1):
            print(f"  {i}. {func['name']}: {func['total_time']:.3f}s ({func['percentage']:.1f}%)")
        
        print(f"\n🧠 Нейронная производительность:")
        for i, neural in enumerate(report['neural_performance'][:5], 1):
            print(f"  {i}. {neural['operation']}: {neural['total_time']:.3f}s")
        
        print(f"\n💡 Рекомендации:")
        for rec in report['performance_analysis']['recommendations']:
            print(f"  • {rec}")
        
        # Сохранение отчетов
        analyzer = _global_profiler.active_sessions["test_session"]
        analyzer.export_report("performance_report.json", "json")
        analyzer.export_report("performance_report.html", "html")
        print(f"\n💾 Отчеты сохранены в файлы")
    
    print("\n✅ Тестирование профайлера завершено") 