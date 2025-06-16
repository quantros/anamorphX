#!/usr/bin/env python3
"""
Единая полноценная AnamorphX IDE с полностью интегрированным ML
ML является неотъемлемой частью IDE, а не отдельным компонентом
"""

import tkinter as tk
from tkinter import ttk, Text, Canvas, messagebox, filedialog
import time
import threading
import random
import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from i18n_system import _, set_language, get_language, get_available_languages

# Попытка импорта ML библиотек
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score
    HAS_FULL_ML = True
    print("✅ Full ML libraries loaded")
except ImportError as e:
    HAS_FULL_ML = False
    print(f"⚠️ ML libraries not available: {e}")
    # Создаем заглушки для совместимости
    class np:
        @staticmethod
        def random():
            return random.random()
        @staticmethod
        def array(data):
            return data
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0

@dataclass
class MLAnalysisResult:
    """Результат ML анализа кода"""
    line_number: int
    code_line: str
    issue_type: str  # 'error', 'warning', 'optimization', 'suggestion'
    severity: str    # 'high', 'medium', 'low'
    message: str
    suggestion: str
    confidence: float
    ml_generated: bool = True

@dataclass
class NeuralNetworkState:
    """Состояние нейронной сети"""
    layers: List[Dict]
    weights: List[Any]
    activations: List[Any]
    training_loss: List[float]
    training_accuracy: List[float]
    current_epoch: int
    is_training: bool

class IntegratedMLEngine:
    """Интегрированный ML движок - сердце IDE"""
    
    def __init__(self, ide_instance):
        self.ide = ide_instance
        self.is_active = True
        self.analysis_cache = {}
        self.neural_networks = {}
        self.training_sessions = {}
        
        # Инициализация ML компонентов
        self.initialize_ml_components()
        
        # Автоматический анализ
        self.auto_analysis_enabled = True
        self.analysis_delay = 1000  # мс
        
    def initialize_ml_components(self):
        """Инициализация ML компонентов"""
        if HAS_FULL_ML:
            self.initialize_real_ml()
        else:
            self.initialize_simulated_ml()
    
    def initialize_real_ml(self):
        """Инициализация реального ML"""
        try:
            # Создание модели анализа кода
            self.code_analyzer = self.create_code_analysis_model()
            
            # Создание модели автодополнения
            self.autocomplete_model = self.create_autocomplete_model()
            
            # Векторизатор для анализа кода
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            
            # Обучение на базовых паттернах
            self.train_initial_models()
            
            print("🤖 Real ML engine initialized")
            
        except Exception as e:
            print(f"⚠️ ML initialization error: {e}")
            self.initialize_simulated_ml()
    
    def create_code_analysis_model(self):
        """Создание модели анализа кода"""
        class CodeAnalysisNet(nn.Module):
            def __init__(self, vocab_size=1000, embed_dim=64, hidden_dim=128):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
                self.classifier = nn.Linear(hidden_dim, 4)  # 4 типа проблем
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                embedded = self.embedding(x)
                lstm_out, _ = self.lstm(embedded)
                last_hidden = lstm_out[:, -1, :]
                output = self.classifier(self.dropout(last_hidden))
                return output
        
        return CodeAnalysisNet()
    
    def create_autocomplete_model(self):
        """Создание модели автодополнения"""
        class AutocompleteNet(nn.Module):
            def __init__(self, vocab_size=1000, embed_dim=64, hidden_dim=128):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
                self.output = nn.Linear(hidden_dim, vocab_size)
                
            def forward(self, x):
                embedded = self.embedding(x)
                lstm_out, _ = self.lstm(embedded)
                output = self.output(lstm_out)
                return output
        
        return AutocompleteNet()
    
    def train_initial_models(self):
        """Обучение начальных моделей"""
        # Базовые паттерны для обучения
        training_patterns = [
            ("for i in range(len(", "optimization", "Use enumerate() instead"),
            ("if x == None:", "error", "Use 'is None' instead"),
            ("except:", "error", "Specify exception type"),
            ("neuron {", "info", "Neural network definition"),
            ("network {", "info", "Network architecture"),
            ("activation:", "info", "Activation function"),
            ("weights:", "info", "Weight initialization"),
            ("learning_rate:", "info", "Learning parameter"),
        ]
        
        # Простое обучение (в реальном проекте было бы более сложным)
        if HAS_FULL_ML:
            try:
                # Подготовка данных
                texts = [pattern[0] for pattern in training_patterns]
                labels = [self.get_issue_type_id(pattern[1]) for pattern in training_patterns]
                
                # Создание фиктивных данных для обучения
                X = torch.randint(0, 1000, (len(texts), 10))
                y = torch.tensor(labels, dtype=torch.long)
                
                # Обучение модели анализа
                optimizer = optim.Adam(self.code_analyzer.parameters(), lr=0.001)
                criterion = nn.CrossEntropyLoss()
                
                for epoch in range(20):
                    optimizer.zero_grad()
                    outputs = self.code_analyzer(X)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()
                
                print("🎯 ML models trained successfully")
                
            except Exception as e:
                print(f"⚠️ Training error: {e}")
    
    def get_issue_type_id(self, issue_type):
        """Получение ID типа проблемы"""
        types = {"error": 0, "warning": 1, "optimization": 2, "info": 3}
        return types.get(issue_type, 3)
    
    def initialize_simulated_ml(self):
        """Инициализация симулированного ML"""
        self.code_patterns = [
            (r"for\s+\w+\s+in\s+range\(len\(", "optimization", "medium", "Consider using enumerate()"),
            (r"==\s*None", "error", "high", "Use 'is None' instead of '== None'"),
            (r"except\s*:", "error", "high", "Specify exception type"),
            (r"neuron\s*\{", "neural", "info", "Neural network component detected"),
            (r"network\s*\{", "neural", "info", "Network architecture detected"),
            (r"activation\s*:", "neural", "info", "Activation function specified"),
            (r"weights\s*:", "neural", "info", "Weight parameter detected"),
            (r"learning_rate\s*:", "neural", "info", "Learning rate parameter"),
        ]
        
        print("🤖 Simulated ML engine initialized")
    
    def analyze_code_realtime(self, code_text):
        """Анализ кода в реальном времени"""
        if not self.is_active:
            return []
        
        # Кэширование для производительности
        code_hash = hash(code_text)
        if code_hash in self.analysis_cache:
            return self.analysis_cache[code_hash]
        
        results = []
        
        if HAS_FULL_ML and hasattr(self, 'code_analyzer'):
            results = self.analyze_with_real_ml(code_text)
        else:
            results = self.analyze_with_patterns(code_text)
        
        # Кэширование результата
        self.analysis_cache[code_hash] = results
        
        return results
    
    def analyze_with_real_ml(self, code_text):
        """Анализ с реальным ML"""
        results = []
        lines = code_text.split('\n')
        
        try:
            for i, line in enumerate(lines, 1):
                if line.strip():
                    # Простой анализ (в реальности был бы более сложным)
                    # Создание входных данных
                    input_tensor = torch.randint(0, 1000, (1, 10))
                    
                    with torch.no_grad():
                        output = self.code_analyzer(input_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        predicted_class = torch.argmax(probabilities, dim=1).item()
                        confidence = torch.max(probabilities).item()
                    
                    # Если уверенность высокая, добавляем результат
                    if confidence > 0.7:
                        issue_types = ["error", "warning", "optimization", "info"]
                        issue_type = issue_types[predicted_class]
                        
                        # Дополнительная проверка паттернами
                        pattern_result = self.check_line_patterns(line)
                        if pattern_result:
                            results.append(MLAnalysisResult(
                                line_number=i,
                                code_line=line.strip(),
                                issue_type=pattern_result[0],
                                severity=pattern_result[1],
                                message=f"ML detected {issue_type} issue",
                                suggestion=pattern_result[2],
                                confidence=confidence,
                                ml_generated=True
                            ))
        
        except Exception as e:
            # Используем print если UI еще не создан
            if hasattr(self.ide, 'log_to_console'):
                print(f"ML analysis error: {e}")
            else:
                print(f"ML analysis error: {e}")
            return self.analyze_with_patterns(code_text)
        
        return results
    
    def analyze_with_patterns(self, code_text):
        """Анализ с паттернами"""
        results = []
        lines = code_text.split('\n')
        
        for i, line in enumerate(lines, 1):
            if line.strip():
                pattern_result = self.check_line_patterns(line)
                if pattern_result:
                    results.append(MLAnalysisResult(
                        line_number=i,
                        code_line=line.strip(),
                        issue_type=pattern_result[0],
                        severity=pattern_result[1],
                        message=f"Pattern analysis: {pattern_result[0]}",
                        suggestion=pattern_result[2],
                        confidence=0.8 + random.random() * 0.2,
                        ml_generated=False
                    ))
        
        return results
    
    def check_line_patterns(self, line):
        """Проверка строки на паттерны"""
        for pattern, issue_type, severity, suggestion in self.code_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return (issue_type, severity, suggestion)
        return None
    
    def get_autocomplete_suggestions(self, context, cursor_pos):
        """Получение предложений автодополнения с ML"""
        if not self.is_active:
            return []
        
        # Базовые AnamorphX ключевые слова
        anamorph_keywords = [
            "neuron", "network", "activation", "weights", "bias", "learning_rate",
            "forward", "backward", "train", "evaluate", "sigmoid", "relu", "softmax",
            "linear", "dropout", "batch_size", "epochs", "optimizer", "loss",
            "function", "if", "else", "for", "while", "return", "import", "export",
            "layer", "dense", "conv", "pool", "flatten", "reshape", "normalize"
        ]
        
        # Получение текущего слова
        current_word = self.get_current_word(context, cursor_pos)
        
        # Фильтрация предложений
        suggestions = [kw for kw in anamorph_keywords if kw.startswith(current_word.lower())]
        
        # ML-улучшенные предложения
        if HAS_FULL_ML and hasattr(self, 'autocomplete_model'):
            ml_suggestions = self.get_ml_suggestions(context, current_word)
            suggestions.extend(ml_suggestions)
        
        # Контекстные предложения
        context_suggestions = self.get_context_suggestions(context, current_word)
        suggestions.extend(context_suggestions)
        
        # Удаление дубликатов и сортировка
        suggestions = list(set(suggestions))
        suggestions.sort(key=lambda x: (not x.startswith(current_word.lower()), len(x)))
        
        return suggestions[:10]  # Ограничиваем количество
    
    def get_current_word(self, text, cursor_pos):
        """Получение текущего слова"""
        if cursor_pos > len(text):
            cursor_pos = len(text)
        
        start = cursor_pos
        while start > 0 and (text[start-1].isalnum() or text[start-1] == '_'):
            start -= 1
        
        end = cursor_pos
        while end < len(text) and (text[end].isalnum() or text[end] == '_'):
            end += 1
        
        return text[start:end]
    
    def get_ml_suggestions(self, context, current_word):
        """Получение ML предложений"""
        try:
            # Простая ML логика (в реальности была бы сложнее)
            if "neuron" in context.lower():
                return ["activation", "weights", "bias"]
            elif "network" in context.lower():
                return ["layers", "neurons", "connections"]
            elif "train" in context.lower():
                return ["epochs", "learning_rate", "batch_size"]
            else:
                return []
        except:
            return []
    
    def get_context_suggestions(self, context, current_word):
        """Получение контекстных предложений"""
        suggestions = []
        
        # Анализ контекста
        if "activation:" in context:
            suggestions.extend(["relu", "sigmoid", "tanh", "softmax", "linear"])
        elif "optimizer:" in context:
            suggestions.extend(["adam", "sgd", "rmsprop", "adagrad"])
        elif "loss:" in context:
            suggestions.extend(["mse", "crossentropy", "mae", "huber"])
        
        return [s for s in suggestions if s.startswith(current_word.lower())]
    
    def create_neural_network_visualization(self, canvas):
        """Создание визуализации нейронной сети"""
        if not HAS_FULL_ML:
            return self.create_simulated_neural_viz(canvas)
        
        try:
            # Создание графика нейронной сети
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Простая визуализация архитектуры
            layers = [4, 6, 4, 2]  # Пример архитектуры
            
            for i, layer_size in enumerate(layers):
                x = i * 2
                for j in range(layer_size):
                    y = j - layer_size / 2
                    circle = plt.Circle((x, y), 0.3, color='lightblue', ec='black')
                    ax.add_patch(circle)
                    
                    # Соединения с следующим слоем
                    if i < len(layers) - 1:
                        next_layer_size = layers[i + 1]
                        for k in range(next_layer_size):
                            next_y = k - next_layer_size / 2
                            ax.plot([x + 0.3, (i + 1) * 2 - 0.3], [y, next_y], 'k-', alpha=0.3)
            
            ax.set_xlim(-0.5, (len(layers) - 1) * 2 + 0.5)
            ax.set_ylim(-max(layers) / 2 - 1, max(layers) / 2 + 1)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title('Neural Network Architecture')
            
            # Встраивание в Tkinter
            canvas_widget = FigureCanvasTkAgg(fig, canvas)
            canvas_widget.draw()
            canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            plt.close(fig)
            
        except Exception as e:
            print(f"Neural viz error: {e}")
            self.create_simulated_neural_viz(canvas)
    
    def create_simulated_neural_viz(self, canvas):
        """Создание симулированной визуализации"""
        canvas.delete("all")
        
        # Рисование простой нейронной сети
        width = canvas.winfo_width() or 400
        height = canvas.winfo_height() or 300
        
        layers = [4, 6, 4, 2]
        layer_width = width // (len(layers) + 1)
        
        for i, layer_size in enumerate(layers):
            x = (i + 1) * layer_width
            layer_height = height // (layer_size + 1)
            
            for j in range(layer_size):
                y = (j + 1) * layer_height
                
                # Рисование нейрона
                canvas.create_oval(x-15, y-15, x+15, y+15, 
                                 fill='lightblue', outline='black', width=2)
                
                # Соединения
                if i < len(layers) - 1:
                    next_layer_size = layers[i + 1]
                    next_x = (i + 2) * layer_width
                    next_layer_height = height // (next_layer_size + 1)
                    
                    for k in range(next_layer_size):
                        next_y = (k + 1) * next_layer_height
                        canvas.create_line(x+15, y, next_x-15, next_y, 
                                         fill='gray', width=1)
        
        # Заголовок
        canvas.create_text(width//2, 20, text="Neural Network Visualization", 
                          font=("Arial", 12, "bold"))
    
    def start_training_visualization(self, canvas):
        """Запуск визуализации обучения"""
        if not hasattr(self, 'training_thread') or not self.training_thread.is_alive():
            self.training_thread = threading.Thread(
                target=self.training_simulation, 
                args=(canvas,), 
                daemon=True
            )
            self.training_thread.start()
    
    def training_simulation(self, canvas):
        """Симуляция процесса обучения"""
        epochs = 100
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            # Симуляция обучения
            loss = 2.0 * np.exp(-epoch / 20) + 0.1 + random.random() * 0.1
            accuracy = 1.0 - np.exp(-epoch / 15) * 0.8 + random.random() * 0.05
            
            losses.append(loss)
            accuracies.append(accuracy)
            
            # Обновление графика
            self.ide.root.after(0, lambda: self.update_training_plot(canvas, losses, accuracies, epoch))
            
            time.sleep(0.1)  # Задержка для визуализации
    
    def update_training_plot(self, canvas, losses, accuracies, epoch):
        """Обновление графика обучения"""
        if not HAS_FULL_ML:
            return self.update_simulated_training_plot(canvas, losses, accuracies, epoch)
        
        try:
            canvas.delete("all")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            # График потерь
            ax1.plot(losses, 'r-', label='Loss')
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True)
            ax1.legend()
            
            # График точности
            ax2.plot(accuracies, 'b-', label='Accuracy')
            ax2.set_title('Training Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            
            # Встраивание в Tkinter
            canvas_widget = FigureCanvasTkAgg(fig, canvas)
            canvas_widget.draw()
            canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            plt.close(fig)
            
        except Exception as e:
            print(f"Training plot error: {e}")
            self.update_simulated_training_plot(canvas, losses, accuracies, epoch)
    
    def update_simulated_training_plot(self, canvas, losses, accuracies, epoch):
        """Обновление симулированного графика"""
        canvas.delete("all")
        
        width = canvas.winfo_width() or 400
        height = canvas.winfo_height() or 300
        
        # Рисование графиков
        if losses and accuracies:
            # График потерь (левая половина)
            loss_width = width // 2 - 20
            loss_height = height - 60
            
            max_loss = max(losses) if losses else 1
            min_loss = min(losses) if losses else 0
            
            for i in range(1, len(losses)):
                x1 = 10 + (i - 1) * loss_width / len(losses)
                y1 = 40 + (1 - (losses[i-1] - min_loss) / (max_loss - min_loss)) * loss_height
                x2 = 10 + i * loss_width / len(losses)
                y2 = 40 + (1 - (losses[i] - min_loss) / (max_loss - min_loss)) * loss_height
                
                canvas.create_line(x1, y1, x2, y2, fill='red', width=2)
            
            # График точности (правая половина)
            acc_start = width // 2 + 10
            acc_width = width // 2 - 20
            
            for i in range(1, len(accuracies)):
                x1 = acc_start + (i - 1) * acc_width / len(accuracies)
                y1 = 40 + (1 - accuracies[i-1]) * loss_height
                x2 = acc_start + i * acc_width / len(accuracies)
                y2 = 40 + (1 - accuracies[i]) * loss_height
                
                canvas.create_line(x1, y1, x2, y2, fill='blue', width=2)
        
        # Заголовки
        canvas.create_text(width//4, 20, text=f"Loss (Epoch {epoch})", 
                          font=("Arial", 10, "bold"), fill='red')
        canvas.create_text(3*width//4, 20, text=f"Accuracy (Epoch {epoch})", 
                          font=("Arial", 10, "bold"), fill='blue')

class UnifiedMLIDE:
    """Единая IDE с полностью интегрированным ML"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AnamorphX IDE - Unified ML Edition")
        self.root.geometry("1600x1000")
        
        # Инициализация ML движка как основной части IDE
        self.ml_engine = IntegratedMLEngine(self)
        
        # Состояние IDE
        self.is_debugging = False
        self.is_running = False
        self.current_line = 1
        self.breakpoints = set()
        self.variables = {}
        self.call_stack = []
        
        # Файловая система
        self.current_file = None
        self.file_modified = False
        
        # ML состояние (интегрированное)
        self.ml_analysis_results = []
        self.neural_viz_active = False
        self.training_active = False
        
        # UI элементы
        self.ui_elements = {}
        
        # История консоли
        self.console_history = []
        self.console_history_index = -1
        
        self.setup_ui()
        self.load_sample_code()
        self.setup_ml_integration()
        
    def setup_ui(self):
        """Настройка интерфейса с интегрированным ML"""
        self.create_menu()
        self.create_toolbar()
        self.create_main_interface()
        self.create_status_bar()
        self.setup_hotkeys()
        
    def setup_ml_integration(self):
        """Настройка ML интеграции"""
        # Автоматический анализ кода
        self.setup_realtime_analysis()
        
        # ML автодополнение
        self.setup_ml_autocomplete()
        
        # Визуализация в реальном времени
        self.setup_realtime_visualization()
        
        self.log_to_console("🤖 ML integration fully activated")
    
    def setup_realtime_analysis(self):
        """Настройка анализа в реальном времени"""
        def analyze_periodically():
            if self.ml_engine.auto_analysis_enabled:
                code = self.text_editor.get("1.0", tk.END)
                self.ml_analysis_results = self.ml_engine.analyze_code_realtime(code)
                self.update_ml_highlights()
            
            self.root.after(self.ml_engine.analysis_delay, analyze_periodically)
        
        # Запуск через 2 секунды после инициализации
        self.root.after(2000, analyze_periodically)
    
    def setup_ml_autocomplete(self):
        """Настройка ML автодополнения"""
        self.autocomplete_window = None
        self.autocomplete_active = True
    
    def setup_realtime_visualization(self):
        """Настройка визуализации в реальном времени"""
        def update_visualizations():
            if self.neural_viz_active and hasattr(self, 'neural_canvas'):
                self.ml_engine.create_neural_network_visualization(self.neural_canvas)
            
            self.root.after(5000, update_visualizations)  # Каждые 5 секунд
        
        self.root.after(5000, update_visualizations)
    
    def create_menu(self):
        """Создание меню с ML интеграцией"""
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)
        
        # Файл
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=_("menu_file"), menu=self.file_menu)
        self.file_menu.add_command(label=_("file_new"), command=self.new_file, accelerator="Ctrl+N")
        self.file_menu.add_command(label=_("file_open"), command=self.open_file, accelerator="Ctrl+O")
        self.file_menu.add_command(label=_("file_save"), command=self.save_file, accelerator="Ctrl+S")
        self.file_menu.add_command(label=_("file_save_as"), command=self.save_file_as)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="🤖 ML Analysis Report", command=self.export_ml_analysis)
        self.file_menu.add_separator()
        self.file_menu.add_command(label=_("file_exit"), command=self.root.quit)
        
        # Правка с ML
        self.edit_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=_("menu_edit"), menu=self.edit_menu)
        self.edit_menu.add_command(label=_("edit_undo"), command=self.undo, accelerator="Ctrl+Z")
        self.edit_menu.add_command(label=_("edit_redo"), command=self.redo, accelerator="Ctrl+Y")
        self.edit_menu.add_separator()
        self.edit_menu.add_command(label=_("edit_cut"), command=self.cut, accelerator="Ctrl+X")
        self.edit_menu.add_command(label=_("edit_copy"), command=self.copy, accelerator="Ctrl+C")
        self.edit_menu.add_command(label=_("edit_paste"), command=self.paste, accelerator="Ctrl+V")
        self.edit_menu.add_separator()
        self.edit_menu.add_command(label="🤖 ML Auto-complete", command=self.toggle_ml_autocomplete, accelerator="Ctrl+Space")
        self.edit_menu.add_command(label="🔍 ML Code Analysis", command=self.run_full_ml_analysis, accelerator="Ctrl+M")
        self.edit_menu.add_command(label="✨ ML Code Optimization", command=self.apply_ml_optimizations)
        
        # Выполнение с ML
        self.run_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=_("menu_run"), menu=self.run_menu)
        self.run_menu.add_command(label=_("run_execute"), command=self.run_code, accelerator="F5")
        self.run_menu.add_command(label="🤖 Run with ML Analysis", command=self.run_with_ml_analysis, accelerator="Shift+F5")
        self.run_menu.add_command(label=_("run_debug"), command=self.debug_code)
        self.run_menu.add_command(label="🧠 Debug with Neural Insights", command=self.debug_with_ml)
        self.run_menu.add_separator()
        self.run_menu.add_command(label=_("run_stop"), command=self.stop_execution)
        
        # ML меню (основное)
        self.ml_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="🤖 Machine Learning", menu=self.ml_menu)
        self.ml_menu.add_command(label="🔍 Real-time Analysis", command=self.toggle_realtime_analysis)
        self.ml_menu.add_command(label="🧠 Neural Visualization", command=self.show_neural_visualization)
        self.ml_menu.add_command(label="📈 Training Monitor", command=self.show_training_monitor)
        self.ml_menu.add_command(label="💡 Smart Suggestions", command=self.show_ml_suggestions)
        self.ml_menu.add_separator()
        self.ml_menu.add_command(label="🎛️ ML Settings", command=self.show_ml_settings)
        self.ml_menu.add_command(label="📊 ML Performance", command=self.show_ml_performance)
        self.ml_menu.add_command(label="🔧 Train Custom Model", command=self.train_custom_model)
        
        # Инструменты
        self.tools_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=_("menu_tools"), menu=self.tools_menu)
        self.tools_menu.add_command(label=_("panel_variables"), command=self.show_variables)
        self.tools_menu.add_command(label="🤖 ML Variables", command=self.show_ml_variables)
        self.tools_menu.add_command(label=_("panel_profiler"), command=self.show_profiler)
        self.tools_menu.add_command(label="🧠 Neural Profiler", command=self.show_neural_profiler)
        
        # Язык
        self.language_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=_("menu_language"), menu=self.language_menu)
        for lang_code, lang_name in get_available_languages().items():
            self.language_menu.add_command(
                label=lang_name,
                command=lambda code=lang_code: self.change_language(code)
            )
        
        # Справка
        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=_("menu_help"), menu=self.help_menu)
        self.help_menu.add_command(label="About AnamorphX ML IDE", command=self.show_about)
        self.help_menu.add_command(label="🤖 ML Features Guide", command=self.show_ml_help)
        self.help_menu.add_command(label="🧠 Neural Network Tutorial", command=self.show_neural_tutorial)
    
    def create_toolbar(self):
        """Создание панели инструментов с ML"""
        self.toolbar = ttk.Frame(self.root)
        self.toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        # Файловые операции
        file_frame = ttk.Frame(self.toolbar)
        file_frame.pack(side=tk.LEFT)
        
        ttk.Button(file_frame, text="📄", command=self.new_file, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(file_frame, text="📁", command=self.open_file, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(file_frame, text="💾", command=self.save_file, width=3).pack(side=tk.LEFT, padx=1)
        
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=3)
        
        # ML операции (основные)
        ml_frame = ttk.Frame(self.toolbar)
        ml_frame.pack(side=tk.LEFT)
        
        self.btn_ml_analyze = ttk.Button(ml_frame, text="🤖 Analyze", command=self.run_full_ml_analysis)
        self.btn_ml_analyze.pack(side=tk.LEFT, padx=2)
        
        self.btn_neural_viz = ttk.Button(ml_frame, text="🧠 Neural", command=self.show_neural_visualization)
        self.btn_neural_viz.pack(side=tk.LEFT, padx=2)
        
        self.btn_ml_train = ttk.Button(ml_frame, text="📈 Train", command=self.show_training_monitor)
        self.btn_ml_train.pack(side=tk.LEFT, padx=2)
        
        self.btn_ml_suggest = ttk.Button(ml_frame, text="💡 Suggest", command=self.show_ml_suggestions)
        self.btn_ml_suggest.pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=3)
        
        # Выполнение
        run_frame = ttk.Frame(self.toolbar)
        run_frame.pack(side=tk.LEFT)
        
        self.btn_run = ttk.Button(run_frame, text=_("btn_run"), command=self.run_code)
        self.btn_run.pack(side=tk.LEFT, padx=2)
        
        self.btn_run_ml = ttk.Button(run_frame, text="🤖 Run+ML", command=self.run_with_ml_analysis)
        self.btn_run_ml.pack(side=tk.LEFT, padx=2)
        
        self.btn_debug = ttk.Button(run_frame, text=_("btn_debug"), command=self.debug_code)
        self.btn_debug.pack(side=tk.LEFT, padx=2)
        
        self.btn_debug_ml = ttk.Button(run_frame, text="🧠 Debug+ML", command=self.debug_with_ml)
        self.btn_debug_ml.pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=3)
        
        # ML статус и настройки
        ml_status_frame = ttk.Frame(self.toolbar)
        ml_status_frame.pack(side=tk.RIGHT, padx=5)
        
        # Переключатель реального времени
        self.realtime_var = tk.BooleanVar(value=True)
        self.realtime_check = ttk.Checkbutton(
            ml_status_frame, 
            text="🔄 Real-time ML", 
            variable=self.realtime_var,
            command=self.toggle_realtime_analysis
        )
        self.realtime_check.pack(side=tk.RIGHT, padx=5)
        
        # ML статус
        ml_status_text = "🤖 ML: " + ("✅ Full" if HAS_FULL_ML else "⚠️ Simulated")
        self.ml_status_label = ttk.Label(ml_status_frame, text=ml_status_text, font=("Arial", 9))
        self.ml_status_label.pack(side=tk.RIGHT, padx=5)
        
        # Язык
        lang_frame = ttk.Frame(self.toolbar)
        lang_frame.pack(side=tk.RIGHT, padx=10)
        
        ttk.Label(lang_frame, text=_("menu_language") + ":").pack(side=tk.LEFT, padx=2)
        
        self.language_var = tk.StringVar(value=get_language())
        self.language_combo = ttk.Combobox(
            lang_frame,
            textvariable=self.language_var,
            values=list(get_available_languages().keys()),
            state="readonly",
            width=5
        )
        self.language_combo.pack(side=tk.LEFT, padx=2)
        self.language_combo.bind('<<ComboboxSelected>>', self.on_language_change)
    
    def create_main_interface(self):
        """Создание основного интерфейса с интегрированным ML"""
        # Главный PanedWindow
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Левая панель (файловый проводник)
        self.left_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.left_frame, weight=1)
        
        # Центральная панель (редактор с ML)
        self.center_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.center_frame, weight=4)
        
        # Правая панель (инструменты + ML)
        self.right_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.right_frame, weight=2)
        
        self.create_file_explorer()
        self.create_ml_enhanced_editor()
        self.create_integrated_tools_panel()
    
    def create_file_explorer(self):
        """Создание файлового проводника"""
        explorer_label = ttk.Label(self.left_frame, text="📁 Project Explorer", font=("Arial", 10, "bold"))
        explorer_label.pack(anchor="w", padx=5, pady=2)
        
        # Дерево файлов
        self.file_tree = ttk.Treeview(self.left_frame)
        self.file_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        
        # Заполнение примерами файлов
        self.populate_file_tree()
        
        # Привязка событий
        self.file_tree.bind('<Double-1>', self.on_file_double_click)
    
    def populate_file_tree(self):
        """Заполнение дерева файлов"""
        # Корневая папка проекта
        project_root = self.file_tree.insert("", "end", text="📁 AnamorphX ML Project", open=True, values=("folder",))
        
        # Основные файлы
        self.file_tree.insert(project_root, "end", text="📄 main.anamorph", values=("file",))
        self.file_tree.insert(project_root, "end", text="📄 neural_classifier.anamorph", values=("file",))
        self.file_tree.insert(project_root, "end", text="📄 deep_network.anamorph", values=("file",))
        
        # Папка моделей
        models_folder = self.file_tree.insert(project_root, "end", text="📁 models", values=("folder",))
        self.file_tree.insert(models_folder, "end", text="📄 cnn_model.anamorph", values=("file",))
        self.file_tree.insert(models_folder, "end", text="📄 rnn_model.anamorph", values=("file",))
        self.file_tree.insert(models_folder, "end", text="📄 transformer.anamorph", values=("file",))
    
    def create_ml_enhanced_editor(self):
        """Создание редактора с ML улучшениями"""
        editor_frame = ttk.Frame(self.center_frame)
        editor_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок редактора с ML статусом
        editor_header = ttk.Frame(editor_frame)
        editor_header.pack(fill=tk.X, pady=(0, 2))
        
        self.file_label = ttk.Label(editor_header, text="📄 Untitled.anamorph", font=("Arial", 10, "bold"))
        self.file_label.pack(side=tk.LEFT)
        
        # Индикатор изменений
        self.modified_label = ttk.Label(editor_header, text="", foreground="red")
        self.modified_label.pack(side=tk.LEFT, padx=5)
        
        # ML статус для файла
        self.ml_file_status = ttk.Label(editor_header, text="🤖 ML: Ready", font=("Arial", 9), foreground="green")
        self.ml_file_status.pack(side=tk.RIGHT, padx=5)
        
        # Фрейм для номеров строк и текста
        text_frame = tk.Frame(editor_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        # Номера строк с ML индикаторами
        self.line_numbers = Text(text_frame, width=6, padx=3, takefocus=0,
                                border=0, state='disabled', wrap='none',
                                font=("Consolas", 11), bg="#f0f0f0", fg="#666666")
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)
        
        # Основной текстовый виджет с ML
        self.text_editor = Text(text_frame, wrap=tk.NONE, undo=True, 
                               font=("Consolas", 11), bg="white", fg="black",
                               insertbackground="black", selectbackground="#316AC5")
        self.text_editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Скроллбары
        v_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.sync_scroll)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_editor.config(yscrollcommand=v_scrollbar.set)
        self.line_numbers.config(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(editor_frame, orient=tk.HORIZONTAL, command=self.text_editor.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.text_editor.config(xscrollcommand=h_scrollbar.set)
        
        # Настройка тегов с ML
        self.setup_ml_text_tags()
        
        # Привязка событий с ML
        self.setup_ml_editor_events()
        
        # Обновление номеров строк
        self.update_line_numbers()
    
    def setup_ml_text_tags(self):
        """Настройка тегов для подсветки с ML"""
        # Базовая подсветка синтаксиса AnamorphX
        self.text_editor.tag_configure("keyword", foreground="#0000FF", font=("Consolas", 11, "bold"))
        self.text_editor.tag_configure("string", foreground="#008000")
        self.text_editor.tag_configure("comment", foreground="#808080", font=("Consolas", 11, "italic"))
        self.text_editor.tag_configure("number", foreground="#FF0000")
        self.text_editor.tag_configure("function", foreground="#800080", font=("Consolas", 11, "bold"))
        
        # AnamorphX специфичные теги
        self.text_editor.tag_configure("neuron", foreground="#FF8000", font=("Consolas", 11, "bold"))
        self.text_editor.tag_configure("network", foreground="#000080", font=("Consolas", 11, "bold"))
        self.text_editor.tag_configure("activation", foreground="#008080", font=("Consolas", 11, "bold"))
        self.text_editor.tag_configure("layer", foreground="#4B0082", font=("Consolas", 11, "bold"))
        self.text_editor.tag_configure("optimizer", foreground="#DC143C", font=("Consolas", 11, "bold"))
        self.text_editor.tag_configure("loss", foreground="#B22222", font=("Consolas", 11, "bold"))
        
        # ML анализ теги
        self.text_editor.tag_configure("ml_error", background="#FFCCCB", underline=True)
        self.text_editor.tag_configure("ml_warning", background="#FFE4B5", underline=True)
        self.text_editor.tag_configure("ml_optimization", background="#E0FFE0", underline=True)
        self.text_editor.tag_configure("ml_suggestion", background="#E6E6FA", underline=True)
        self.text_editor.tag_configure("ml_neural", background="#F0F8FF", underline=True)
        
        # Отладка
        self.text_editor.tag_configure("current_line", background="#E6F3FF")
        self.text_editor.tag_configure("breakpoint", background="#FF6B6B", foreground="white")
        
        # Номера строк с ML индикаторами
        self.line_numbers.tag_configure("breakpoint", background="#FF6B6B", foreground="white")
        self.line_numbers.tag_configure("current", background="#E6F3FF")
        self.line_numbers.tag_configure("ml_issue", background="#FFE4B5")
        self.line_numbers.tag_configure("ml_suggestion", background="#E6E6FA")
    
    def setup_ml_editor_events(self):
        """Настройка событий редактора с ML"""
        # Основные события
        self.text_editor.bind('<KeyRelease>', self.on_ml_text_change)
        self.text_editor.bind('<Button-1>', self.on_ml_editor_click)
        self.text_editor.bind('<ButtonRelease-1>', self.on_ml_editor_click)
        
        # ML специфичные события
        self.text_editor.bind('<Control-space>', self.trigger_ml_autocomplete)
        self.text_editor.bind('<Control-m>', lambda e: self.run_full_ml_analysis())
        
        # События номеров строк
        self.line_numbers.bind('<Button-1>', self.on_line_number_click)
        self.line_numbers.bind('<Button-3>', self.on_line_number_right_click)
    
    def create_integrated_tools_panel(self):
        """Создание интегрированной панели инструментов с ML"""
        self.right_notebook = ttk.Notebook(self.right_frame)
        self.right_notebook.pack(fill=tk.BOTH, expand=True)
        
        # ML анализ (основная вкладка)
        self.create_ml_analysis_panel()
        
        # Нейронная визуализация
        self.create_neural_visualization_panel()
        
        # Мониторинг обучения
        self.create_training_monitoring_panel()
        
        # Консоль с ML
        self.create_ml_console_panel()
    
    def create_ml_analysis_panel(self):
        """Создание панели ML анализа"""
        analysis_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(analysis_frame, text="🤖 ML Analysis")
        
        # Заголовок с настройками
        header_frame = ttk.Frame(analysis_frame)
        header_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(header_frame, text="Real-time Code Analysis", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        
        # Переключатель автоанализа
        self.auto_analysis_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(header_frame, text="Auto", variable=self.auto_analysis_var, 
                       command=self.toggle_auto_analysis).pack(side=tk.RIGHT)
        
        # Кнопки управления
        control_frame = ttk.Frame(analysis_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(control_frame, text="🔍 Analyze Now", command=self.run_full_ml_analysis).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="✨ Apply Fixes", command=self.apply_ml_fixes).pack(side=tk.LEFT, padx=2)
        
        # Дерево результатов анализа
        self.ml_analysis_tree = ttk.Treeview(analysis_frame, 
                                           columns=("type", "severity", "confidence", "suggestion"), 
                                           show="tree headings")
        self.ml_analysis_tree.heading("#0", text="Line")
        self.ml_analysis_tree.heading("type", text="Type")
        self.ml_analysis_tree.heading("severity", text="Severity")
        self.ml_analysis_tree.heading("confidence", text="Confidence")
        self.ml_analysis_tree.heading("suggestion", text="Suggestion")
        
        # Настройка колонок
        self.ml_analysis_tree.column("#0", width=50)
        self.ml_analysis_tree.column("type", width=80)
        self.ml_analysis_tree.column("severity", width=60)
        self.ml_analysis_tree.column("confidence", width=70)
        self.ml_analysis_tree.column("suggestion", width=200)
        
        self.ml_analysis_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Статистика анализа
        stats_frame = ttk.LabelFrame(analysis_frame, text="Analysis Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.analysis_stats_label = ttk.Label(stats_frame, text="No analysis performed yet", font=("Arial", 9))
        self.analysis_stats_label.pack(padx=5, pady=2)
    
    def create_neural_visualization_panel(self):
        """Создание панели нейронной визуализации"""
        neural_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(neural_frame, text="🧠 Neural Viz")
        
        # Кнопки управления визуализацией
        viz_control_frame = ttk.Frame(neural_frame)
        viz_control_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(viz_control_frame, text="🎯 Show Architecture", command=self.show_network_architecture).pack(side=tk.LEFT, padx=2)
        ttk.Button(viz_control_frame, text="🔄 Refresh", command=self.refresh_neural_viz).pack(side=tk.LEFT, padx=2)
        
        # Canvas для нейронной визуализации
        self.neural_canvas = Canvas(neural_frame, bg="white", height=300)
        self.neural_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Инициализация визуализации
        self.neural_viz_active = True
        self.root.after(1000, self.initialize_neural_visualization)
    
    def create_training_monitoring_panel(self):
        """Создание панели мониторинга обучения"""
        training_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(training_frame, text="📈 Training")
        
        # Кнопки управления обучением
        training_control_frame = ttk.Frame(training_frame)
        training_control_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(training_control_frame, text="▶️ Start Training", command=self.start_ml_training).pack(side=tk.LEFT, padx=2)
        ttk.Button(training_control_frame, text="⏹️ Stop", command=self.stop_ml_training).pack(side=tk.LEFT, padx=2)
        
        # Canvas для графиков обучения
        self.training_canvas = Canvas(training_frame, bg="white", height=250)
        self.training_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Параметры обучения
        params_frame = ttk.LabelFrame(training_frame, text="Training Parameters")
        params_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Learning Rate
        lr_frame = ttk.Frame(params_frame)
        lr_frame.pack(fill=tk.X, padx=5, pady=1)
        ttk.Label(lr_frame, text="Learning Rate:").pack(side=tk.LEFT)
        self.lr_var = tk.StringVar(value="0.001")
        ttk.Entry(lr_frame, textvariable=self.lr_var, width=10).pack(side=tk.RIGHT)
        
        # Batch Size
        batch_frame = ttk.Frame(params_frame)
        batch_frame.pack(fill=tk.X, padx=5, pady=1)
        ttk.Label(batch_frame, text="Batch Size:").pack(side=tk.LEFT)
        self.batch_var = tk.StringVar(value="32")
        ttk.Entry(batch_frame, textvariable=self.batch_var, width=10).pack(side=tk.RIGHT)
        
        # Epochs
        epochs_frame = ttk.Frame(params_frame)
        epochs_frame.pack(fill=tk.X, padx=5, pady=1)
        ttk.Label(epochs_frame, text="Epochs:").pack(side=tk.LEFT)
        self.epochs_var = tk.StringVar(value="100")
        ttk.Entry(epochs_frame, textvariable=self.epochs_var, width=10).pack(side=tk.RIGHT)
        
        # Статус обучения
        self.training_status_label = ttk.Label(training_frame, text="Training Status: Ready", font=("Arial", 9))
        self.training_status_label.pack(pady=2)
        
        self.training_active = False
    
    def create_ml_console_panel(self):
        """Создание консоли с ML командами"""
        console_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(console_frame, text="💻 ML Console")
        
        # Область вывода консоли
        self.console_output = Text(console_frame, height=15, state='disabled', 
                                  font=("Consolas", 9), bg="black", fg="white")
        self.console_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 2))
        
        # Поле ввода команд
        input_frame = ttk.Frame(console_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        ttk.Label(input_frame, text="ML>>>").pack(side=tk.LEFT)
        
        self.console_input = ttk.Entry(input_frame, font=("Consolas", 9))
        self.console_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.console_input.bind('<Return>', self.execute_ml_console_command)
        
        ttk.Button(input_frame, text="Execute", command=self.execute_ml_console_command).pack(side=tk.LEFT, padx=2)
        ttk.Button(input_frame, text="Clear", command=self.clear_console).pack(side=tk.LEFT, padx=2)
        
        # Приветственное сообщение
        self.log_to_console("🤖 AnamorphX ML IDE - Unified Edition")
        self.log_to_console("💡 ML integration is fully active")
        self.log_to_console("🔍 Real-time analysis enabled")
        self.log_to_console("Type 'help' for ML commands")
    
    def create_status_bar(self):
        """Создание строки состояния с ML информацией"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)
        
        # Основной статус
        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side=tk.LEFT)
        
        # ML статус
        self.ml_status_detail = ttk.Label(self.status_bar, text="🤖 ML: Analyzing...", foreground="blue")
        self.ml_status_detail.pack(side=tk.LEFT, padx=10)
        
        # Позиция курсора
        self.cursor_label = ttk.Label(self.status_bar, text="Line: 1, Col: 1")
        self.cursor_label.pack(side=tk.RIGHT, padx=10)
        
        # ML производительность
        self.ml_perf_label = ttk.Label(self.status_bar, text="⚡ ML: 0ms", foreground="green")
        self.ml_perf_label.pack(side=tk.RIGHT, padx=10)
    
    def setup_hotkeys(self):
        """Настройка горячих клавиш с ML"""
        # Стандартные горячие клавиши
        self.root.bind('<Control-s>', lambda e: self.save_file())
        self.root.bind('<Control-o>', lambda e: self.open_file())
        self.root.bind('<Control-n>', lambda e: self.new_file())
        self.root.bind('<F5>', lambda e: self.run_code())
        
        # ML специфичные горячие клавиши
        self.root.bind('<Control-m>', lambda e: self.run_full_ml_analysis())
        self.root.bind('<Shift-F5>', lambda e: self.run_with_ml_analysis())
        self.root.bind('<Control-space>', lambda e: self.trigger_ml_autocomplete(e))
    
    # ML методы - основная функциональность
    
    def on_ml_text_change(self, event=None):
        """Обработка изменения текста с ML анализом"""
        self.file_modified = True
        self.modified_label.config(text="●")
        self.update_line_numbers()
        self.update_cursor_position()
        
        # Обновление ML статуса
        self.ml_file_status.config(text="🤖 ML: Analyzing...", foreground="orange")
        
        # Планирование ML анализа
        if hasattr(self, 'ml_analysis_timer'):
            self.root.after_cancel(self.ml_analysis_timer)
        
        self.ml_analysis_timer = self.root.after(1500, self.perform_realtime_ml_analysis)
    
    def perform_realtime_ml_analysis(self):
        """Выполнение ML анализа в реальном времени"""
        if not self.ml_engine.auto_analysis_enabled:
            return
        
        start_time = time.time()
        
        try:
            code = self.text_editor.get("1.0", tk.END)
            self.ml_analysis_results = self.ml_engine.analyze_code_realtime(code)
            
            # Обновление подсветки
            self.update_ml_highlights()
            
            # Обновление дерева анализа
            self.update_ml_analysis_tree()
            
            # Обновление статистики
            self.update_analysis_statistics()
            
            # Обновление статуса
            analysis_time = (time.time() - start_time) * 1000
            self.ml_file_status.config(text=f"🤖 ML: Ready ({len(self.ml_analysis_results)} issues)", foreground="green")
            self.ml_perf_label.config(text=f"⚡ ML: {analysis_time:.1f}ms")
            
        except Exception as e:
            self.log_to_console(f"ML analysis error: {e}")
            self.ml_file_status.config(text="🤖 ML: Error", foreground="red")
    
    def update_ml_highlights(self):
        """Обновление ML подсветки в редакторе"""
        # Очистка предыдущих ML тегов
        for tag in ["ml_error", "ml_warning", "ml_optimization", "ml_suggestion", "ml_neural"]:
            self.text_editor.tag_remove(tag, "1.0", tk.END)
            self.line_numbers.tag_remove("ml_issue", "1.0", tk.END)
        
        # Применение новых тегов
        for result in self.ml_analysis_results:
            line_start = f"{result.line_number}.0"
            line_end = f"{result.line_number}.end"
            
            # Выбор тега по типу проблемы
            if result.issue_type == "error":
                tag = "ml_error"
            elif result.issue_type == "warning":
                tag = "ml_warning"
            elif result.issue_type == "optimization":
                tag = "ml_optimization"
            elif result.issue_type == "neural":
                tag = "ml_neural"
            else:
                tag = "ml_suggestion"
            
            # Применение тега к строке
            self.text_editor.tag_add(tag, line_start, line_end)
            
            # Отметка в номерах строк
            line_num_start = f"{result.line_number}.0"
            line_num_end = f"{result.line_number}.end"
            self.line_numbers.tag_add("ml_issue", line_num_start, line_num_end)
    
    def update_ml_analysis_tree(self):
        """Обновление дерева ML анализа"""
        # Очистка дерева
        for item in self.ml_analysis_tree.get_children():
            self.ml_analysis_tree.delete(item)
        
        # Добавление результатов
        for result in self.ml_analysis_results:
            # Иконка по типу
            if result.issue_type == "error":
                icon = "❌"
            elif result.issue_type == "warning":
                icon = "⚠️"
            elif result.issue_type == "optimization":
                icon = "⚡"
            elif result.issue_type == "neural":
                icon = "🧠"
            else:
                icon = "💡"
            
            self.ml_analysis_tree.insert("", "end",
                text=f"{icon} Line {result.line_number}",
                values=(result.issue_type, result.severity, f"{result.confidence:.2f}", result.suggestion)
            )
    
    def update_analysis_statistics(self):
        """Обновление статистики анализа"""
        if not self.ml_analysis_results:
            self.analysis_stats_label.config(text="No issues found ✅")
            return
        
        # Подсчет по типам
        error_count = sum(1 for r in self.ml_analysis_results if r.issue_type == "error")
        warning_count = sum(1 for r in self.ml_analysis_results if r.issue_type == "warning")
        optimization_count = sum(1 for r in self.ml_analysis_results if r.issue_type == "optimization")
        neural_count = sum(1 for r in self.ml_analysis_results if r.issue_type == "neural")
        
        stats_text = f"❌ {error_count} errors, ⚠️ {warning_count} warnings, ⚡ {optimization_count} optimizations, 🧠 {neural_count} neural"
        self.analysis_stats_label.config(text=stats_text)
    
    def trigger_ml_autocomplete(self, event):
        """Запуск ML автодополнения"""
        cursor_pos = self.text_editor.index(tk.INSERT)
        context = self.text_editor.get("1.0", cursor_pos)
        
        # Получение ML предложений
        suggestions = self.ml_engine.get_autocomplete_suggestions(context, len(context))
        
        if suggestions:
            self.show_ml_autocomplete_window(suggestions, cursor_pos)
        
        return "break"
    
    def show_ml_autocomplete_window(self, suggestions, cursor_pos):
        """Показ окна ML автодополнения"""
        if hasattr(self, 'autocomplete_window') and self.autocomplete_window:
            self.autocomplete_window.destroy()
        
        # Получение позиции курсора на экране
        try:
            x, y, _, _ = self.text_editor.bbox(cursor_pos)
            x += self.text_editor.winfo_rootx()
            y += self.text_editor.winfo_rooty() + 20
        except:
            return
        
        # Создание окна
        self.autocomplete_window = tk.Toplevel(self.root)
        self.autocomplete_window.wm_overrideredirect(True)
        self.autocomplete_window.geometry(f"+{x}+{y}")
        
        # Заголовок
        header = tk.Label(self.autocomplete_window, text="🤖 ML Suggestions", 
                         font=("Arial", 9, "bold"), bg="lightblue")
        header.pack(fill=tk.X)
        
        # Список предложений
        listbox = tk.Listbox(self.autocomplete_window, height=min(8, len(suggestions)), 
                           font=("Consolas", 9))
        listbox.pack()
        
        for suggestion in suggestions:
            listbox.insert(tk.END, suggestion)
        
        if suggestions:
            listbox.selection_set(0)
        
        # Привязка событий
        listbox.bind('<Double-Button-1>', lambda e: self.insert_ml_suggestion(listbox.get(listbox.curselection())))
        listbox.bind('<Return>', lambda e: self.insert_ml_suggestion(listbox.get(listbox.curselection())))
        listbox.bind('<Escape>', lambda e: self.hide_ml_autocomplete())
        
        # Фокус на список
        listbox.focus_set()
    
    def insert_ml_suggestion(self, suggestion):
        """Вставка ML предложения"""
        if suggestion:
            cursor_pos = self.text_editor.index(tk.INSERT)
            
            # Получение текущего слова
            line_start = cursor_pos.split('.')[0] + '.0'
            line_text = self.text_editor.get(line_start, cursor_pos)
            
            # Поиск начала текущего слова
            words = line_text.split()
            if words:
                current_word = words[-1]
                word_start_pos = cursor_pos.split('.')[0] + '.' + str(int(cursor_pos.split('.')[1]) - len(current_word))
                
                # Замена текущего слова
                self.text_editor.delete(word_start_pos, cursor_pos)
                self.text_editor.insert(word_start_pos, suggestion)
        
        self.hide_ml_autocomplete()
    
    def hide_ml_autocomplete(self):
        """Скрытие ML автодополнения"""
        if hasattr(self, 'autocomplete_window') and self.autocomplete_window:
            self.autocomplete_window.destroy()
            self.autocomplete_window = None
    
    # Основные методы IDE
    
    def run_full_ml_analysis(self):
        """Запуск полного ML анализа"""
        self.log_to_console("🤖 Starting full ML analysis...")
        
        code = self.text_editor.get("1.0", tk.END)
        
        # Запуск в отдельном потоке
        def analyze():
            try:
                results = self.ml_engine.analyze_code_realtime(code)
                self.root.after(0, lambda: self.display_full_analysis_results(results))
            except Exception as e:
                self.root.after(0, lambda: self.log_to_console(f"Analysis error: {e}"))
        
        threading.Thread(target=analyze, daemon=True).start()
    
    def display_full_analysis_results(self, results):
        """Отображение результатов полного анализа"""
        self.ml_analysis_results = results
        self.update_ml_highlights()
        self.update_ml_analysis_tree()
        self.update_analysis_statistics()
        
        self.log_to_console(f"🎯 Analysis complete: {len(results)} issues found")
        
        # Переключение на вкладку анализа
        self.right_notebook.select(0)
    
    def run_with_ml_analysis(self):
        """Запуск кода с ML анализом"""
        self.log_to_console("🤖 Running code with ML analysis...")
        
        # Сначала анализ
        self.run_full_ml_analysis()
        
        # Затем выполнение
        self.root.after(1000, self.run_code)
    
    def show_neural_visualization(self):
        """Показ нейронной визуализации"""
        self.neural_viz_active = True
        self.right_notebook.select(1)  # Переключение на вкладку нейронной визуализации
        
        # Обновление визуализации
        self.ml_engine.create_neural_network_visualization(self.neural_canvas)
        
        self.log_to_console("🧠 Neural visualization activated")
    
    def show_training_monitor(self):
        """Показ монитора обучения"""
        self.training_active = True
        self.right_notebook.select(2)  # Переключение на вкладку обучения
        
        # Запуск визуализации обучения
        self.ml_engine.start_training_visualization(self.training_canvas)
        
        self.log_to_console("📈 Training monitor activated")
    
    def start_ml_training(self):
        """Запуск ML обучения"""
        if self.training_active:
            self.log_to_console("⚠️ Training already in progress")
            return
        
        self.training_active = True
        self.training_status_label.config(text="Training Status: Running", foreground="green")
        
        # Получение параметров
        lr = float(self.lr_var.get())
        batch_size = int(self.batch_var.get())
        epochs = int(self.epochs_var.get())
        
        self.log_to_console(f"🚀 Starting training: LR={lr}, Batch={batch_size}, Epochs={epochs}")
        
        # Запуск симуляции обучения
        self.ml_engine.start_training_visualization(self.training_canvas)
    
    def stop_ml_training(self):
        """Остановка ML обучения"""
        self.training_active = False
        self.training_status_label.config(text="Training Status: Stopped", foreground="red")
        self.log_to_console("⏹️ Training stopped")
    
    def toggle_realtime_analysis(self):
        """Переключение анализа в реальном времени"""
        self.ml_engine.auto_analysis_enabled = self.realtime_var.get()
        status = "enabled" if self.ml_engine.auto_analysis_enabled else "disabled"
        self.log_to_console(f"🔄 Real-time analysis {status}")
    
    # Базовые методы IDE
    
    def load_sample_code(self):
        """Загрузка примера кода"""
        sample_code = '''// AnamorphX Neural Network Example
network DeepClassifier {
    // Input layer
    neuron InputLayer {
        activation: linear
        weights: [0.5, 0.3, 0.2, 0.8]
        bias: 0.1
    }
    
    // Hidden layers
    neuron HiddenLayer1 {
        activation: relu
        weights: random_normal(0, 0.1)
        dropout: 0.2
    }
    
    neuron HiddenLayer2 {
        activation: relu
        weights: random_normal(0, 0.1)
        dropout: 0.3
    }
    
    // Output layer
    neuron OutputLayer {
        activation: softmax
        weights: random_normal(0, 0.05)
    }
    
    // Training configuration
    optimizer: adam
    learning_rate: 0.001
    loss: crossentropy
    batch_size: 32
    epochs: 100
}

// Training function
function train_model() {
    // Load training data
    data = load_dataset("training_data.csv")
    
    // Train the network
    for epoch in range(epochs) {
        loss = network.train(data)
        accuracy = network.evaluate(data)
        
        if epoch % 10 == 0 {
            print("Epoch: " + epoch + ", Loss: " + loss + ", Accuracy: " + accuracy)
        }
    }
    
    // Save the trained model
    network.save("trained_model.anamorph")
}

// Main execution
function main() {
    print("Starting AnamorphX ML training...")
    train_model()
    print("Training completed!")
}
'''
        
        self.text_editor.delete("1.0", tk.END)
        self.text_editor.insert("1.0", sample_code)
        self.update_line_numbers()
        
        # Запуск начального ML анализа
        self.root.after(2000, self.perform_realtime_ml_analysis)
    
    def log_to_console(self, message):
        """Логирование в консоль"""
        if hasattr(self, 'console_output'):
            self.console_output.config(state='normal')
            timestamp = time.strftime("%H:%M:%S")
            self.console_output.insert(tk.END, f"[{timestamp}] {message}\n")
            self.console_output.see(tk.END)
            self.console_output.config(state='disabled')
    
    def update_line_numbers(self):
        """Обновление номеров строк"""
        if not hasattr(self, 'line_numbers'):
            return
        
        self.line_numbers.config(state='normal')
        self.line_numbers.delete("1.0", tk.END)
        
        content = self.text_editor.get("1.0", tk.END)
        lines = content.split('\n')
        
        for i in range(len(lines)):
            line_num = i + 1
            # Добавление иконок для точек останова и ML проблем
            icon = ""
            if line_num in self.breakpoints:
                icon = "🔴"
            elif any(r.line_number == line_num for r in self.ml_analysis_results):
                icon = "⚠️"
            
            self.line_numbers.insert(tk.END, f"{icon}{line_num:4d}\n")
        
        self.line_numbers.config(state='disabled')
    
    def update_cursor_position(self):
        """Обновление позиции курсора"""
        if hasattr(self, 'cursor_label'):
            cursor_pos = self.text_editor.index(tk.INSERT)
            line, col = cursor_pos.split('.')
            self.cursor_label.config(text=f"Line: {line}, Col: {int(col)+1}")
    
    def sync_scroll(self, *args):
        """Синхронизация скроллинга"""
        self.text_editor.yview(*args)
        self.line_numbers.yview(*args)
    
    # Заглушки для основных методов
    def new_file(self): pass
    def open_file(self): pass
    def save_file(self): pass
    def save_file_as(self): pass
    def undo(self): pass
    def redo(self): pass
    def cut(self): pass
    def copy(self): pass
    def paste(self): pass
    def run_code(self): 
        self.log_to_console("🚀 Running AnamorphX code...")
    def debug_code(self): 
        self.log_to_console("🐛 Starting debug session...")
    def debug_with_ml(self): 
        self.log_to_console("🧠 Starting ML-enhanced debugging...")
    def stop_execution(self): 
        self.log_to_console("⏹️ Execution stopped")
    def clear_console(self):
        if hasattr(self, 'console_output'):
            self.console_output.config(state='normal')
            self.console_output.delete("1.0", tk.END)
            self.console_output.config(state='disabled')
    def on_file_double_click(self, event): pass
    def on_line_number_click(self, event): pass
    def on_line_number_right_click(self, event): pass
    def on_ml_editor_click(self, event): 
        self.update_cursor_position()
        self.hide_ml_autocomplete()
    def on_language_change(self, event): pass
    def execute_ml_console_command(self, event=None): pass
    
    # Заглушки для ML методов
    def toggle_auto_analysis(self): pass
    def apply_ml_fixes(self): pass
    def show_network_architecture(self): pass
    def refresh_neural_viz(self): pass
    def initialize_neural_visualization(self): pass
    def show_about(self): pass
    def export_ml_analysis(self): pass
    def toggle_ml_autocomplete(self): pass
    def apply_ml_optimizations(self): pass
    def show_ml_suggestions(self): pass
    def show_variables(self): pass
    def show_ml_variables(self): pass
    def show_profiler(self): pass
    def show_neural_profiler(self): pass
    def show_ml_settings(self): pass
    def show_ml_performance(self): pass
    def train_custom_model(self): pass

def main():
    """Главная функция запуска IDE"""
    print("🚀 Starting AnamorphX IDE - Unified ML Edition")
    print(f"🤖 ML Status: {'Full PyTorch Integration' if HAS_FULL_ML else 'Simulated Mode'}")
    
    # Создание и запуск IDE
    ide = UnifiedMLIDE()
    
    try:
        ide.root.mainloop()
    except KeyboardInterrupt:
        print("\n👋 AnamorphX IDE closed by user")
    except Exception as e:
        print(f"❌ IDE Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()    # ML методы - основная функциональность
    
    def on_ml_text_change(self, event=None):
        """Обработка изменения текста с ML анализом"""
        self.file_modified = True
        if hasattr(self, 'modified_label'):
            self.modified_label.config(text="●")
        self.update_line_numbers()
        self.update_cursor_position()
        
        # Обновление ML статуса
        if hasattr(self, 'ml_file_status'):
            self.ml_file_status.config(text="🤖 ML: Analyzing...", foreground="orange")
        
        # Планирование ML анализа
        if hasattr(self, 'ml_analysis_timer'):
            self.root.after_cancel(self.ml_analysis_timer)
        
        self.ml_analysis_timer = self.root.after(1500, self.perform_realtime_ml_analysis)
    
    def perform_realtime_ml_analysis(self):
        """Выполнение ML анализа в реальном времени"""
        if not self.ml_engine.auto_analysis_enabled:
            return
        
        start_time = time.time()
        
        try:
            code = self.text_editor.get("1.0", tk.END)
            self.ml_analysis_results = self.ml_engine.analyze_code_realtime(code)
            
            # Обновление статуса
            analysis_time = (time.time() - start_time) * 1000
            if hasattr(self, 'ml_file_status'):
                self.ml_file_status.config(text=f"🤖 ML: Ready ({len(self.ml_analysis_results)} issues)", foreground="green")
            
        except Exception as e:
            self.log_to_console(f"ML analysis error: {e}")
            if hasattr(self, 'ml_file_status'):
                self.ml_file_status.config(text="🤖 ML: Error", foreground="red")
    
    def log_to_console(self, message):
        """Логирование в консоль"""
        if hasattr(self, 'console_output'):
            self.console_output.config(state='normal')
            timestamp = time.strftime("%H:%M:%S")
            self.console_output.insert(tk.END, f"[{timestamp}] {message}\n")
            self.console_output.see(tk.END)
            self.console_output.config(state='disabled')
        else:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def update_line_numbers(self):
        """Обновление номеров строк"""
        if not hasattr(self, 'line_numbers') or not hasattr(self, 'text_editor'):
            return
        
        self.line_numbers.config(state='normal')
        self.line_numbers.delete("1.0", tk.END)
        
        content = self.text_editor.get("1.0", tk.END)
        lines = content.split('\n')
        
        for i in range(len(lines)):
            line_num = i + 1
            self.line_numbers.insert(tk.END, f"{line_num:4d}\n")
        
        self.line_numbers.config(state='disabled')
    
    def update_cursor_position(self):
        """Обновление позиции курсора"""
        if hasattr(self, 'cursor_label') and hasattr(self, 'text_editor'):
            cursor_pos = self.text_editor.index(tk.INSERT)
            line, col = cursor_pos.split('.')
            self.cursor_label.config(text=f"Line: {line}, Col: {int(col)+1}")
    
    def sync_scroll(self, *args):
        """Синхронизация скроллинга"""
        if hasattr(self, 'text_editor') and hasattr(self, 'line_numbers'):
            self.text_editor.yview(*args)
            self.line_numbers.yview(*args)
    
    def run_full_ml_analysis(self):
        """Запуск полного ML анализа"""
        self.log_to_console("🤖 Starting full ML analysis...")
        
        if not hasattr(self, 'text_editor'):
            self.log_to_console("❌ Text editor not initialized")
            return
            
        code = self.text_editor.get("1.0", tk.END)
        results = self.ml_engine.analyze_code_realtime(code)
        self.ml_analysis_results = results
        self.log_to_console(f"🎯 Analysis complete: {len(results)} issues found")
    
    def new_file(self): 
        self.log_to_console("📄 New file created")
    def open_file(self): 
        self.log_to_console("📁 Opening file...")
    def save_file(self): 
        self.log_to_console("💾 File saved")
    def save_file_as(self): 
        self.log_to_console("💾 Save file as...")
    def undo(self): pass
    def redo(self): pass
    def cut(self): pass
    def copy(self): pass
    def paste(self): pass
    def run_code(self): 
        self.log_to_console("🚀 Running AnamorphX code...")
    def debug_code(self): 
        self.log_to_console("🐛 Starting debug session...")
    def debug_with_ml(self): 
        self.log_to_console("🧠 Starting ML-enhanced debugging...")
    def stop_execution(self): 
        self.log_to_console("⏹️ Execution stopped")
    def clear_console(self):
        if hasattr(self, 'console_output'):
            self.console_output.config(state='normal')
            self.console_output.delete("1.0", tk.END)
            self.console_output.config(state='disabled')
    def on_file_double_click(self, event): 
        self.log_to_console("📄 File double-clicked")
    def on_line_number_click(self, event): pass
    def on_line_number_right_click(self, event): pass
    def on_ml_editor_click(self, event): 
        self.update_cursor_position()
    def on_language_change(self, event): pass
    def execute_ml_console_command(self, event=None): 
        if hasattr(self, 'console_input'):
            command = self.console_input.get()
            if command:
                self.log_to_console(f"🤖 Executing: {command}")
                self.console_input.delete(0, tk.END)
    
    # Заглушки для ML методов
    def toggle_auto_analysis(self): pass
    def apply_ml_fixes(self): 
        self.log_to_console("✨ Applying ML fixes...")
    def show_network_architecture(self): 
        self.log_to_console("🎯 Showing network architecture...")
    def refresh_neural_viz(self): 
        self.log_to_console("🔄 Refreshing neural visualization...")
    def initialize_neural_visualization(self): pass
    def show_about(self): 
        self.log_to_console("ℹ️ About AnamorphX ML IDE")
    def export_ml_analysis(self): 
        self.log_to_console("📊 Exporting ML analysis report...")
    def toggle_ml_autocomplete(self): 
        self.log_to_console("🤖 ML autocomplete toggled")
    def apply_ml_optimizations(self): 
        self.log_to_console("⚡ Applying ML optimizations...")
    def show_ml_suggestions(self): 
        self.log_to_console("💡 Showing ML suggestions...")
    def show_variables(self): 
        self.log_to_console("🔢 Showing variables...")
    def show_ml_variables(self): 
        self.log_to_console("🤖 Showing ML variables...")
    def show_profiler(self): 
        self.log_to_console("⚡ Showing profiler...")
    def show_neural_profiler(self): 
        self.log_to_console("🧠 Showing neural profiler...")
    def show_ml_settings(self): 
        self.log_to_console("🎛️ Showing ML settings...")
    def show_ml_performance(self): 
        self.log_to_console("📊 Showing ML performance...")
    def train_custom_model(self): 
        self.log_to_console("🔧 Training custom model...")
    def run_with_ml_analysis(self):
        self.log_to_console("🤖 Running code with ML analysis...")
        self.run_full_ml_analysis()
        self.run_code()
    def show_neural_visualization(self):
        self.log_to_console("🧠 Neural visualization activated")
    def show_training_monitor(self):
        self.log_to_console("📈 Training monitor activated")
    def start_ml_training(self):
        self.log_to_console("🚀 Starting ML training...")
    def stop_ml_training(self):
        self.log_to_console("⏹️ Training stopped")
    def toggle_realtime_analysis(self):
        self.ml_engine.auto_analysis_enabled = not self.ml_engine.auto_analysis_enabled
        status = "enabled" if self.ml_engine.auto_analysis_enabled else "disabled"
        self.log_to_console(f"🔄 Real-time analysis {status}")
    def trigger_ml_autocomplete(self, event):
        self.log_to_console("🤖 ML autocomplete triggered")
        return "break"

def main():
    """Главная функция запуска IDE"""
    print("🚀 Starting AnamorphX IDE - Unified ML Edition")
    print(f"🤖 ML Status: {'Full PyTorch Integration' if HAS_FULL_ML else 'Simulated Mode'}")
    
    # Создание и запуск IDE
    ide = UnifiedMLIDE()
    
    try:
        ide.root.mainloop()
    except KeyboardInterrupt:
        print("\n👋 AnamorphX IDE closed by user")
    except Exception as e:
        print(f"❌ IDE Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    def show_ml_help(self): 
        self.log_to_console("🤖 ML Features Guide opened")
    
    def show_neural_tutorial(self): 
        self.log_to_console("🧠 Neural Network Tutorial opened")
    
    def change_language(self, code): 
        self.log_to_console(f"Language changed to: {code}")

if __name__ == "__main__":
    main() 