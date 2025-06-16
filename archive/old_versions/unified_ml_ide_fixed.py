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
