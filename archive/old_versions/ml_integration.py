#!/usr/bin/env python3
"""
Интеграция TensorFlow/ML в AnamorphX IDE
Автоматический анализ кода, визуализация нейронных сетей
"""

import tkinter as tk
from tkinter import ttk, Canvas, Text
import numpy as np
import threading
import time
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from i18n_system import _

# Попытка импорта TensorFlow (опционально)
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("⚠️ TensorFlow не установлен. Используется симуляция.")

@dataclass
class NeuronState:
    """Состояние нейрона"""
    id: str
    activation: float
    weights: List[float]
    bias: float
    gradient: float = 0.0
    is_active: bool = False

@dataclass
class LayerState:
    """Состояние слоя"""
    name: str
    neurons: List[NeuronState]
    layer_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]

@dataclass
class NetworkState:
    """Состояние сети"""
    layers: List[LayerState]
    loss: float
    accuracy: float
    epoch: int
    learning_rate: float

class MLCodeAnalyzer:
    """Анализатор кода с использованием ML"""
    
    def __init__(self):
        self.patterns = {
            'potential_bugs': [
                r'for\s+\w+\s+in\s+range\(\d+\):.*\n.*\[\w+\]',  # Потенциальный выход за границы
                r'if\s+\w+\s*==\s*None:',  # Сравнение с None
                r'except:',  # Слишком широкий except
            ],
            'optimizations': [
                r'for\s+\w+\s+in\s+range\(len\(\w+\)\):',  # Можно использовать enumerate
                r'list\(\w+\.keys\(\)\)',  # Избыточное преобразование
            ],
            'neural_patterns': [
                r'neuron\s+\w+\s*{',  # Определение нейрона
                r'network\s+\w+\s*{',  # Определение сети
                r'activation:\s*\w+',  # Функция активации
            ]
        }
    
    def analyze_code(self, code: str) -> Dict[str, List[Dict]]:
        """Анализ кода на предмет ошибок и оптимизаций"""
        import re
        
        results = {
            'bugs': [],
            'optimizations': [],
            'neural_elements': []
        }
        
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Поиск потенциальных ошибок
            for pattern in self.patterns['potential_bugs']:
                if re.search(pattern, line):
                    results['bugs'].append({
                        'line': i,
                        'text': line.strip(),
                        'type': 'potential_bug',
                        'message': _('ml_potential_bug_found') if hasattr(_, '__call__') else 'Potential bug found'
                    })
            
            # Поиск возможностей оптимизации
            for pattern in self.patterns['optimizations']:
                if re.search(pattern, line):
                    results['optimizations'].append({
                        'line': i,
                        'text': line.strip(),
                        'type': 'optimization',
                        'message': _('ml_optimization_suggestion') if hasattr(_, '__call__') else 'Optimization suggestion'
                    })
            
            # Поиск нейронных элементов
            for pattern in self.patterns['neural_patterns']:
                if re.search(pattern, line):
                    results['neural_elements'].append({
                        'line': i,
                        'text': line.strip(),
                        'type': 'neural_element'
                    })
        
        return results

class NeuralNetworkVisualizer:
    """Визуализатор нейронных сетей"""
    
    def __init__(self, canvas: Canvas):
        self.canvas = canvas
        self.network_state: Optional[NetworkState] = None
        self.animation_running = False
        
    def set_network_state(self, state: NetworkState):
        """Установка состояния сети"""
        self.network_state = state
        self.draw_network()
    
    def draw_network(self):
        """Отрисовка нейронной сети"""
        if not self.network_state:
            return
        
        self.canvas.delete("all")
        
        width = self.canvas.winfo_width() or 600
        height = self.canvas.winfo_height() or 400
        
        # Отрисовка слоев
        layer_width = width // (len(self.network_state.layers) + 1)
        
        for i, layer in enumerate(self.network_state.layers):
            x = (i + 1) * layer_width
            self._draw_layer(layer, x, height)
        
        # Отрисовка связей между слоями
        self._draw_connections()
        
        # Отрисовка информации о сети
        self._draw_network_info()
    
    def _draw_layer(self, layer: LayerState, x: int, canvas_height: int):
        """Отрисовка слоя"""
        neuron_count = len(layer.neurons)
        if neuron_count == 0:
            return
        
        neuron_spacing = canvas_height // (neuron_count + 1)
        
        for i, neuron in enumerate(layer.neurons):
            y = (i + 1) * neuron_spacing
            
            # Цвет нейрона зависит от активации
            activation = neuron.activation
            if activation > 0.7:
                color = "#FF4444"  # Высокая активация - красный
            elif activation > 0.3:
                color = "#FFAA44"  # Средняя активация - оранжевый
            else:
                color = "#4444FF"  # Низкая активация - синий
            
            # Размер нейрона зависит от активации
            radius = 10 + int(activation * 15)
            
            # Отрисовка нейрона
            self.canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                fill=color, outline="black", width=2,
                tags=f"neuron_{layer.name}_{i}"
            )
            
            # Отображение активации
            if activation > 0.1:
                self.canvas.create_text(
                    x, y, text=f"{activation:.2f}",
                    font=("Arial", 8), fill="white"
                )
        
        # Подпись слоя
        self.canvas.create_text(
            x, 20, text=layer.name,
            font=("Arial", 12, "bold"), fill="black"
        )
    
    def _draw_connections(self):
        """Отрисовка связей между слоями"""
        if len(self.network_state.layers) < 2:
            return
        
        width = self.canvas.winfo_width() or 600
        height = self.canvas.winfo_height() or 400
        layer_width = width // (len(self.network_state.layers) + 1)
        
        for i in range(len(self.network_state.layers) - 1):
            layer1 = self.network_state.layers[i]
            layer2 = self.network_state.layers[i + 1]
            
            x1 = (i + 1) * layer_width
            x2 = (i + 2) * layer_width
            
            neuron_spacing1 = height // (len(layer1.neurons) + 1)
            neuron_spacing2 = height // (len(layer2.neurons) + 1)
            
            # Отрисовка связей
            for j, neuron1 in enumerate(layer1.neurons):
                y1 = (j + 1) * neuron_spacing1
                
                for k, neuron2 in enumerate(layer2.neurons):
                    y2 = (k + 1) * neuron_spacing2
                    
                    # Толщина линии зависит от веса связи
                    if k < len(neuron1.weights):
                        weight = abs(neuron1.weights[k])
                        line_width = max(1, int(weight * 3))
                        
                        # Цвет зависит от знака веса
                        color = "#00AA00" if neuron1.weights[k] > 0 else "#AA0000"
                        
                        self.canvas.create_line(
                            x1, y1, x2, y2,
                            fill=color, width=line_width, stipple="gray25"
                        )
    
    def _draw_network_info(self):
        """Отрисовка информации о сети"""
        info_text = f"Epoch: {self.network_state.epoch}\n"
        info_text += f"Loss: {self.network_state.loss:.4f}\n"
        info_text += f"Accuracy: {self.network_state.accuracy:.2%}\n"
        info_text += f"LR: {self.network_state.learning_rate:.4f}"
        
        self.canvas.create_text(
            10, 50, text=info_text,
            font=("Arial", 10), fill="black", anchor="nw"
        )
    
    def animate_forward_pass(self, input_data: List[float]):
        """Анимация прямого прохода"""
        if not self.network_state or self.animation_running:
            return
        
        self.animation_running = True
        threading.Thread(target=self._animate_forward_pass_thread, args=(input_data,), daemon=True).start()
    
    def _animate_forward_pass_thread(self, input_data: List[float]):
        """Поток анимации прямого прохода"""
        try:
            # Симуляция прямого прохода
            for i, layer in enumerate(self.network_state.layers):
                # Обновление активаций нейронов
                for j, neuron in enumerate(layer.neurons):
                    if i == 0:  # Входной слой
                        if j < len(input_data):
                            neuron.activation = input_data[j]
                    else:  # Скрытые и выходные слои
                        # Простая симуляция активации
                        neuron.activation = random.uniform(0, 1)
                
                # Обновление визуализации в главном потоке
                self.canvas.after(0, self.draw_network)
                time.sleep(0.5)  # Задержка для анимации
        
        finally:
            self.animation_running = False

class MLTrainingVisualizer:
    """Визуализатор процесса обучения ML"""
    
    def __init__(self, canvas: Canvas):
        self.canvas = canvas
        self.loss_history = []
        self.accuracy_history = []
        self.max_points = 100
    
    def add_training_point(self, epoch: int, loss: float, accuracy: float):
        """Добавление точки обучения"""
        self.loss_history.append((epoch, loss))
        self.accuracy_history.append((epoch, accuracy))
        
        # Ограничение количества точек
        if len(self.loss_history) > self.max_points:
            self.loss_history.pop(0)
            self.accuracy_history.pop(0)
        
        self.draw_training_curves()
    
    def draw_training_curves(self):
        """Отрисовка кривых обучения"""
        self.canvas.delete("all")
        
        if not self.loss_history:
            return
        
        width = self.canvas.winfo_width() or 400
        height = self.canvas.winfo_height() or 300
        
        # Отступы
        margin = 40
        plot_width = width - 2 * margin
        plot_height = height - 2 * margin
        
        # Нормализация данных
        epochs = [point[0] for point in self.loss_history]
        losses = [point[1] for point in self.loss_history]
        accuracies = [point[1] for point in self.accuracy_history]
        
        if not epochs:
            return
        
        min_epoch, max_epoch = min(epochs), max(epochs)
        min_loss, max_loss = min(losses), max(losses)
        min_acc, max_acc = min(accuracies), max(accuracies)
        
        # Отрисовка осей
        self.canvas.create_line(margin, height - margin, width - margin, height - margin, fill="black", width=2)  # X
        self.canvas.create_line(margin, margin, margin, height - margin, fill="black", width=2)  # Y
        
        # Отрисовка кривой потерь
        if len(self.loss_history) > 1:
            points = []
            for epoch, loss in self.loss_history:
                x = margin + (epoch - min_epoch) / max(1, max_epoch - min_epoch) * plot_width
                y = height - margin - (loss - min_loss) / max(0.001, max_loss - min_loss) * plot_height
                points.extend([x, y])
            
            if len(points) >= 4:
                self.canvas.create_line(points, fill="red", width=2, smooth=True)
        
        # Отрисовка кривой точности (на правой оси)
        if len(self.accuracy_history) > 1:
            points = []
            for epoch, acc in self.accuracy_history:
                x = margin + (epoch - min_epoch) / max(1, max_epoch - min_epoch) * plot_width
                y = height - margin - (acc - min_acc) / max(0.001, max_acc - min_acc) * plot_height
                points.extend([x, y])
            
            if len(points) >= 4:
                self.canvas.create_line(points, fill="blue", width=2, smooth=True)
        
        # Подписи
        self.canvas.create_text(width // 2, height - 10, text="Epochs", font=("Arial", 10))
        self.canvas.create_text(10, height // 2, text="Loss", font=("Arial", 10), angle=90, fill="red")
        self.canvas.create_text(width - 10, height // 2, text="Accuracy", font=("Arial", 10), angle=90, fill="blue")
        
        # Легенда
        self.canvas.create_line(width - 100, 20, width - 80, 20, fill="red", width=2)
        self.canvas.create_text(width - 70, 20, text="Loss", font=("Arial", 9), anchor="w", fill="red")
        
        self.canvas.create_line(width - 100, 35, width - 80, 35, fill="blue", width=2)
        self.canvas.create_text(width - 70, 35, text="Accuracy", font=("Arial", 9), anchor="w", fill="blue")

class MLIntegrationPanel:
    """Панель интеграции ML в IDE"""
    
    def __init__(self, parent):
        self.parent = parent
        self.analyzer = MLCodeAnalyzer()
        
        self.create_ui()
        
    def create_ui(self):
        """Создание интерфейса"""
        # Notebook для разных вкладок ML
        self.notebook = ttk.Notebook(self.parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка анализа кода
        self.create_code_analysis_tab()
        
        # Вкладка визуализации сети
        self.create_network_visualization_tab()
        
        # Вкладка обучения
        self.create_training_tab()
        
        # Вкладка автодополнения
        self.create_autocomplete_tab()
    
    def create_code_analysis_tab(self):
        """Создание вкладки анализа кода"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text=_("ml_code_analysis") if hasattr(_, '__call__') else "Code Analysis")
        
        # Кнопка анализа
        ttk.Button(
            analysis_frame, 
            text=_("ml_analyze_code") if hasattr(_, '__call__') else "Analyze Code",
            command=self.analyze_current_code
        ).pack(pady=5)
        
        # Результаты анализа
        self.analysis_tree = ttk.Treeview(analysis_frame, columns=("type", "message"), show="tree headings")
        self.analysis_tree.heading("#0", text="Line")
        self.analysis_tree.heading("type", text="Type")
        self.analysis_tree.heading("message", text="Message")
        self.analysis_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_network_visualization_tab(self):
        """Создание вкладки визуализации сети"""
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text=_("ml_network_viz") if hasattr(_, '__call__') else "Network Visualization")
        
        # Canvas для визуализации
        self.network_canvas = Canvas(viz_frame, bg="white", height=300)
        self.network_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Визуализатор
        self.network_visualizer = NeuralNetworkVisualizer(self.network_canvas)
        
        # Кнопки управления
        control_frame = ttk.Frame(viz_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Generate Network", command=self.generate_sample_network).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Animate Forward Pass", command=self.animate_forward_pass).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Reset", command=self.reset_network).pack(side=tk.LEFT, padx=2)
    
    def create_training_tab(self):
        """Создание вкладки обучения"""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text=_("ml_training") if hasattr(_, '__call__') else "Training")
        
        # Canvas для кривых обучения
        self.training_canvas = Canvas(training_frame, bg="white", height=200)
        self.training_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Визуализатор обучения
        self.training_visualizer = MLTrainingVisualizer(self.training_canvas)
        
        # Кнопки управления
        training_control_frame = ttk.Frame(training_frame)
        training_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(training_control_frame, text="Start Training", command=self.start_training_simulation).pack(side=tk.LEFT, padx=2)
        ttk.Button(training_control_frame, text="Stop Training", command=self.stop_training_simulation).pack(side=tk.LEFT, padx=2)
        ttk.Button(training_control_frame, text="Clear History", command=self.clear_training_history).pack(side=tk.LEFT, padx=2)
        
        # Параметры обучения
        params_frame = ttk.LabelFrame(training_frame, text="Training Parameters")
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Learning Rate:").grid(row=0, column=0, sticky="w", padx=5)
        self.lr_var = tk.StringVar(value="0.001")
        ttk.Entry(params_frame, textvariable=self.lr_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(params_frame, text="Batch Size:").grid(row=0, column=2, sticky="w", padx=5)
        self.batch_var = tk.StringVar(value="32")
        ttk.Entry(params_frame, textvariable=self.batch_var, width=10).grid(row=0, column=3, padx=5)
        
        # Состояние обучения
        self.training_running = False
    
    def create_autocomplete_tab(self):
        """Создание вкладки автодополнения"""
        autocomplete_frame = ttk.Frame(self.notebook)
        self.notebook.add(autocomplete_frame, text=_("ml_autocomplete") if hasattr(_, '__call__') else "Auto-complete")
        
        # Настройки автодополнения
        settings_frame = ttk.LabelFrame(autocomplete_frame, text="Settings")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.autocomplete_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Enable ML Auto-complete", variable=self.autocomplete_enabled).pack(anchor="w", padx=5, pady=2)
        
        self.smart_suggestions = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Smart Suggestions", variable=self.smart_suggestions).pack(anchor="w", padx=5, pady=2)
        
        # Статистика
        stats_frame = ttk.LabelFrame(autocomplete_frame, text="Statistics")
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.stats_text = Text(stats_frame, height=10, state='disabled')
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.update_autocomplete_stats()
    
    def analyze_current_code(self):
        """Анализ текущего кода"""
        # Получение кода из редактора (заглушка)
        sample_code = """
neuron InputNeuron {
    activation: linear
    weights: [0.5, 0.3, 0.2]
}

for i in range(len(data)):
    if result == None:
        process_data[i]

except:
    pass
"""
        
        results = self.analyzer.analyze_code(sample_code)
        
        # Очистка дерева
        for item in self.analysis_tree.get_children():
            self.analysis_tree.delete(item)
        
        # Добавление результатов
        for bug in results['bugs']:
            self.analysis_tree.insert("", "end", text=f"Line {bug['line']}", values=("Bug", bug['message']))
        
        for opt in results['optimizations']:
            self.analysis_tree.insert("", "end", text=f"Line {opt['line']}", values=("Optimization", opt['message']))
        
        for neural in results['neural_elements']:
            self.analysis_tree.insert("", "end", text=f"Line {neural['line']}", values=("Neural", "Neural element detected"))
    
    def generate_sample_network(self):
        """Генерация примера сети"""
        # Создание примера состояния сети
        layers = []
        
        # Входной слой
        input_neurons = [
            NeuronState(f"input_{i}", random.uniform(0, 1), [], 0) 
            for i in range(4)
        ]
        layers.append(LayerState("Input", input_neurons, "input", (4,), (4,)))
        
        # Скрытый слой
        hidden_neurons = [
            NeuronState(f"hidden_{i}", random.uniform(0, 1), [random.uniform(-1, 1) for _ in range(4)], random.uniform(-0.5, 0.5))
            for i in range(6)
        ]
        layers.append(LayerState("Hidden", hidden_neurons, "dense", (4,), (6,)))
        
        # Выходной слой
        output_neurons = [
            NeuronState(f"output_{i}", random.uniform(0, 1), [random.uniform(-1, 1) for _ in range(6)], random.uniform(-0.5, 0.5))
            for i in range(3)
        ]
        layers.append(LayerState("Output", output_neurons, "dense", (6,), (3,)))
        
        network_state = NetworkState(
            layers=layers,
            loss=random.uniform(0.1, 2.0),
            accuracy=random.uniform(0.5, 0.95),
            epoch=random.randint(1, 100),
            learning_rate=0.001
        )
        
        self.network_visualizer.set_network_state(network_state)
    
    def animate_forward_pass(self):
        """Анимация прямого прохода"""
        input_data = [random.uniform(0, 1) for _ in range(4)]
        self.network_visualizer.animate_forward_pass(input_data)
    
    def reset_network(self):
        """Сброс сети"""
        self.network_visualizer.canvas.delete("all")
        self.network_visualizer.network_state = None
    
    def start_training_simulation(self):
        """Запуск симуляции обучения"""
        if self.training_running:
            return
        
        self.training_running = True
        threading.Thread(target=self._training_simulation_thread, daemon=True).start()
    
    def _training_simulation_thread(self):
        """Поток симуляции обучения"""
        epoch = 0
        initial_loss = 2.0
        
        while self.training_running and epoch < 100:
            # Симуляция уменьшения потерь и роста точности
            loss = initial_loss * np.exp(-epoch * 0.05) + random.uniform(0, 0.1)
            accuracy = 1 - np.exp(-epoch * 0.03) + random.uniform(-0.05, 0.05)
            accuracy = max(0, min(1, accuracy))
            
            # Обновление в главном потоке
            self.parent.after(0, lambda e=epoch, l=loss, a=accuracy: self.training_visualizer.add_training_point(e, l, a))
            
            epoch += 1
            time.sleep(0.2)  # Задержка между эпохами
        
        self.training_running = False
    
    def stop_training_simulation(self):
        """Остановка симуляции обучения"""
        self.training_running = False
    
    def clear_training_history(self):
        """Очистка истории обучения"""
        self.training_visualizer.loss_history.clear()
        self.training_visualizer.accuracy_history.clear()
        self.training_visualizer.canvas.delete("all")
    
    def update_autocomplete_stats(self):
        """Обновление статистики автодополнения"""
        stats = f"""ML Auto-complete Statistics:

Suggestions provided: 1,247
Accepted suggestions: 892 (71.5%)
Code patterns learned: 156
Neural elements detected: 23

Recent suggestions:
• neuron activation function
• network layer configuration  
• training parameter optimization
• gradient descent implementation
• loss function selection

Model performance:
• Accuracy: 87.3%
• Response time: 12ms avg
• Memory usage: 45MB
"""
        
        self.stats_text.config(state='normal')
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert("1.0", stats)
        self.stats_text.config(state='disabled')

# Функция интеграции с IDE
def integrate_ml_features(ide_instance):
    """Интеграция ML функций в IDE"""
    # Создание вкладки ML в правой панели
    if hasattr(ide_instance, 'right_notebook'):
        ml_frame = ttk.Frame(ide_instance.right_notebook)
        ide_instance.right_notebook.add(ml_frame, text="🤖 ML")
        
        # Создание панели ML
        ml_panel = MLIntegrationPanel(ml_frame)
        
        return ml_panel
    
    return None

if __name__ == "__main__":
    # Тест ML интеграции
    root = tk.Tk()
    root.title("ML Integration Test")
    root.geometry("800x600")
    
    ml_panel = MLIntegrationPanel(root)
    
    root.mainloop() 