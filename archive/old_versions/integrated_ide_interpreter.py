#!/usr/bin/env python3
"""
AnamorphX IDE - Integrated Interpreter Edition
Полнофункциональная IDE с интегрированным интерпретатором AnamorphX
Объединяет complete_ml_ide.py + demo_working_interpreter.py + src/interpreter/
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Text, Canvas, Scrollbar
import os
import sys
import time
import threading
import random
import re
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import traceback

# Настройка путей для импорта интерпретатора
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# ML библиотеки
HAS_FULL_ML = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_FULL_ML = True
    print("✅ Full ML libraries loaded")
except ImportError as e:
    print(f"⚠️ ML libraries not available: {e}")
    # Заглушки для ML
    class torch:
        class nn:
            class Module: pass
            class LSTM: pass
            class Linear: pass
        @staticmethod
        def tensor(*args): return None
    
    class TfidfVectorizer:
        def __init__(self, **kwargs): pass
    
    class np:
        @staticmethod
        def random(): return [0.1, 0.2, 0.3]
        @staticmethod
        def array(data): return data

# Импорт компонентов интерпретатора
INTERPRETER_READY = False
interpreter_components = {}

try:
    from interpreter.execution_engine import ExecutionEngine
    interpreter_components["ExecutionEngine"] = ExecutionEngine
    print("✅ Execution Engine imported")
except Exception as e:
    print(f"⚠️ Execution Engine: {e}")

try:
    from interpreter.ast_interpreter import ASTInterpreter
    interpreter_components["ASTInterpreter"] = ASTInterpreter
    print("✅ AST Interpreter imported")
except Exception as e:
    print(f"⚠️ AST Interpreter: {e}")

try:
    from interpreter.type_system import TypeSystem
    interpreter_components["TypeSystem"] = TypeSystem
    print("✅ Type System imported")
except Exception as e:
    print(f"⚠️ Type System: {e}")

try:
    from interpreter.error_handler import ErrorHandler
    interpreter_components["ErrorHandler"] = ErrorHandler
    print("✅ Error Handler imported")
except Exception as e:
    print(f"⚠️ Error Handler: {e}")

try:
    from interpreter.enhanced_memory_manager import EnhancedMemoryManager
    interpreter_components["MemoryManager"] = EnhancedMemoryManager
    print("✅ Memory Manager imported")
except Exception as e:
    print(f"⚠️ Memory Manager: {e}")

try:
    from interpreter.commands import CommandRegistry
    interpreter_components["Commands"] = CommandRegistry
    print("✅ Commands imported")
except Exception as e:
    print(f"⚠️ Commands: {e}")

# Проверяем готовность интерпретатора
INTERPRETER_READY = len(interpreter_components) >= 3
print(f"🤖 Interpreter status: {'✅ READY' if INTERPRETER_READY else '⚠️ PARTIAL'} ({len(interpreter_components)}/6 components)")

class AnamorphXInterpreter:
    """Интегрированный интерпретатор AnamorphX"""
    
    def __init__(self):
        self.is_ready = INTERPRETER_READY
        self.components = interpreter_components.copy()
        self.current_program = ""
        self.execution_state = "idle"  # idle, running, error
        self.variables = {}
        self.output_buffer = []
        
        # Инициализация компонентов
        self.initialize_components()
    
    def initialize_components(self):
        """Инициализация компонентов интерпретатора"""
        try:
            if "TypeSystem" in self.components:
                self.type_system = self.components["TypeSystem"]()
                print("🎯 Type System initialized")
            
            if "ErrorHandler" in self.components:
                self.error_handler = self.components["ErrorHandler"]()
                print("🛡️ Error Handler initialized")
            
            if "MemoryManager" in self.components:
                self.memory_manager = self.components["MemoryManager"]()
                print("💾 Memory Manager initialized")
            
            if "ExecutionEngine" in self.components:
                self.execution_engine = self.components["ExecutionEngine"]()
                print("⚡ Execution Engine initialized")
            
            if "ASTInterpreter" in self.components:
                self.ast_interpreter = self.components["ASTInterpreter"]()
                print("🌳 AST Interpreter initialized")
            
            if "Commands" in self.components:
                self.command_registry = self.components["Commands"]()
                print("📋 Command Registry initialized")
                
        except Exception as e:
            print(f"⚠️ Component initialization error: {e}")
            self.is_ready = False
    
    def execute_code(self, code_text):
        """Выполнение кода AnamorphX"""
        if not self.is_ready:
            return self.simulate_execution(code_text)
        
        try:
            self.current_program = code_text
            self.execution_state = "running"
            self.output_buffer = []
            
            # Реальное выполнение через компоненты
            if hasattr(self, 'ast_interpreter'):
                result = self.ast_interpreter.interpret(code_text)
                self.execution_state = "completed"
                return {
                    "success": True,
                    "result": result,
                    "output": self.output_buffer,
                    "variables": self.get_variables(),
                    "execution_time": 0.1
                }
            else:
                return self.simulate_execution(code_text)
                
        except Exception as e:
            self.execution_state = "error"
            error_msg = f"Execution error: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "output": self.output_buffer
            }
    
    def simulate_execution(self, code_text):
        """Симуляция выполнения кода"""
        lines = code_text.strip().split('\n')
        output = []
        variables = {}
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            
            # Простая симуляция выполнения
            if 'synap' in line and '=' in line:
                # Переменная
                parts = line.split('=')
                if len(parts) == 2:
                    var_name = parts[0].replace('synap', '').strip()
                    var_value = parts[1].strip()
                    variables[var_name] = var_value
                    output.append(f"✅ Line {i}: Created variable {var_name} = {var_value}")
            
            elif 'print' in line:
                # Вывод
                output.append(f"📄 Line {i}: Print statement executed")
            
            elif 'network' in line:
                # Нейронная сеть
                output.append(f"🧠 Line {i}: Neural network definition")
            
            elif 'function' in line:
                # Функция
                output.append(f"⚙️ Line {i}: Function definition")
            
            else:
                output.append(f"⚡ Line {i}: Statement executed")
        
        return {
            "success": True,
            "result": "Program executed successfully",
            "output": output,
            "variables": variables,
            "execution_time": len(lines) * 0.05,
            "simulated": True
        }
    
    def get_variables(self):
        """Получение текущих переменных"""
        if hasattr(self, 'memory_manager'):
            try:
                return self.memory_manager.get_all_variables()
            except:
                pass
        return self.variables
    
    def get_status(self):
        """Получение статуса интерпретатора"""
        return {
            "ready": self.is_ready,
            "state": self.execution_state,
            "components": len(self.components),
            "has_real_interpreter": INTERPRETER_READY
        }

class IntegratedMLIDE:
    """Интегрированная IDE с ML и интерпретатором"""
    
    def __init__(self):
        print("🚀 Starting AnamorphX IDE - Integrated Interpreter Edition")
        
        # Инициализация интерпретатора
        self.interpreter = AnamorphXInterpreter()
        print(f"🤖 Interpreter status: {self.interpreter.get_status()}")
        
        # Основное окно
        self.root = tk.Tk()
        self.root.title("AnamorphX IDE - Integrated Interpreter Edition")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Состояние
        self.current_file = None
        self.is_modified = False
        self.execution_thread = None
        
        # ML интеграция (упрощенная) - ПЕРЕД setup_ui
        self.setup_basic_ml()
        
        # Настройка UI
        self.setup_ui()
        
        print("✅ IDE initialized successfully")
    
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Меню
        self.create_menu()
        
        # Тулбар
        self.create_toolbar()
        
        # Основной интерфейс
        self.create_main_interface()
        
        # Статус бар
        self.create_status_bar()
        
        # Горячие клавиши
        self.setup_hotkeys()
    
    def create_menu(self):
        """Создание меню"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.new_file, accelerator="Ctrl+N")
        file_menu.add_command(label="Open", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As", command=self.save_file_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Run menu
        run_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Run", menu=run_menu)
        run_menu.add_command(label="Run Code", command=self.run_code, accelerator="F5")
        run_menu.add_command(label="Debug", command=self.debug_code, accelerator="F9")
        run_menu.add_command(label="Stop", command=self.stop_execution, accelerator="Ctrl+C")
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="ML Analysis", command=self.run_ml_analysis)
        tools_menu.add_command(label="Variables", command=self.show_variables)
        tools_menu.add_command(label="Interpreter Status", command=self.show_interpreter_status)
    
    def create_toolbar(self):
        """Создание тулбара"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)
        
        # Кнопки файлов
        ttk.Button(toolbar, text="📄 New", command=self.new_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📂 Open", command=self.open_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="💾 Save", command=self.save_file).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Кнопки выполнения
        ttk.Button(toolbar, text="▶️ Run", command=self.run_code).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🐛 Debug", command=self.debug_code).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="⏹️ Stop", command=self.stop_execution).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # ML кнопки
        ttk.Button(toolbar, text="🤖 ML Analysis", command=self.run_ml_analysis).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📊 Variables", command=self.show_variables).pack(side=tk.LEFT, padx=2)
        
        # Статус интерпретатора
        status = self.interpreter.get_status()
        status_text = "✅ READY" if status["ready"] else "⚠️ PARTIAL"
        self.interpreter_status_label = ttk.Label(toolbar, text=f"Interpreter: {status_text}")
        self.interpreter_status_label.pack(side=tk.RIGHT, padx=10)
    
    def create_main_interface(self):
        """Создание основного интерфейса"""
        # Главный контейнер
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Левая панель - файловый проводник
        left_frame = ttk.Frame(main_paned, width=250)
        main_paned.add(left_frame, weight=1)
        self.create_file_explorer(left_frame)
        
        # Центральная панель - редактор
        center_paned = ttk.PanedWindow(main_paned, orient=tk.VERTICAL)
        main_paned.add(center_paned, weight=3)
        
        # Редактор
        editor_frame = ttk.Frame(center_paned)
        center_paned.add(editor_frame, weight=2)
        self.create_editor(editor_frame)
        
        # Консоль вывода
        console_frame = ttk.Frame(center_paned)
        center_paned.add(console_frame, weight=1)
        self.create_console(console_frame)
        
        # Правая панель - инструменты
        right_frame = ttk.Frame(main_paned, width=300)
        main_paned.add(right_frame, weight=1)
        self.create_tools_panel(right_frame)
    
    def create_file_explorer(self, parent):
        """Создание файлового проводника"""
        ttk.Label(parent, text="📁 Project Explorer").pack(pady=5)
        
        # Дерево файлов
        self.file_tree = ttk.Treeview(parent)
        self.file_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Заполнение дерева
        self.populate_file_tree()
        
        # Обработчик двойного клика
        self.file_tree.bind("<Double-1>", self.on_file_double_click)
    
    def create_editor(self, parent):
        """Создание редактора кода"""
        ttk.Label(parent, text="📝 AnamorphX Editor").pack(pady=5)
        
        # Фрейм для редактора с номерами строк
        editor_container = ttk.Frame(parent)
        editor_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Номера строк
        self.line_numbers = Text(editor_container, width=4, padx=3, takefocus=0,
                                border=0, state='disabled', wrap='none',
                                bg='#3c3c3c', fg='#858585', font=('Consolas', 10))
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)
        
        # Основной редактор
        self.editor = Text(editor_container, wrap='none', undo=True,
                          bg='#2b2b2b', fg='#f8f8f2', insertbackground='white',
                          font=('Consolas', 11), selectbackground='#44475a')
        self.editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Скроллбары
        v_scrollbar = ttk.Scrollbar(editor_container, orient=tk.VERTICAL, command=self.sync_scroll)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.editor.config(yscrollcommand=v_scrollbar.set)
        
        # События редактора
        self.editor.bind('<KeyRelease>', self.on_text_change)
        self.editor.bind('<Button-1>', self.on_text_change)
        self.editor.bind('<MouseWheel>', self.sync_scroll)
        
        # Загрузка примера кода
        self.load_sample_code()
    
    def create_console(self, parent):
        """Создание консоли вывода"""
        ttk.Label(parent, text="💻 Output Console").pack(pady=5)
        
        # Консоль
        self.console = Text(parent, height=10, bg='#1e1e1e', fg='#00ff00',
                           font=('Consolas', 10), state='disabled')
        self.console.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Скроллбар для консоли
        console_scroll = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.console.yview)
        console_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.console.config(yscrollcommand=console_scroll.set)
    
    def create_tools_panel(self, parent):
        """Создание панели инструментов"""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Вкладка переменных
        vars_frame = ttk.Frame(notebook)
        notebook.add(vars_frame, text="📊 Variables")
        self.create_variables_panel(vars_frame)
        
        # Вкладка ML анализа
        ml_frame = ttk.Frame(notebook)
        notebook.add(ml_frame, text="🤖 ML Analysis")
        self.create_ml_panel(ml_frame)
        
        # Вкладка статуса
        status_frame = ttk.Frame(notebook)
        notebook.add(status_frame, text="⚙️ Status")
        self.create_status_panel(status_frame)
    
    def create_variables_panel(self, parent):
        """Создание панели переменных"""
        ttk.Label(parent, text="Program Variables").pack(pady=5)
        
        # Дерево переменных
        self.vars_tree = ttk.Treeview(parent, columns=('Value', 'Type'), show='tree headings')
        self.vars_tree.heading('#0', text='Variable')
        self.vars_tree.heading('Value', text='Value')
        self.vars_tree.heading('Type', text='Type')
        self.vars_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_ml_panel(self, parent):
        """Создание ML панели"""
        ttk.Label(parent, text="ML Analysis Results").pack(pady=5)
        
        # Результаты ML анализа
        self.ml_results = Text(parent, height=15, bg='#2b2b2b', fg='#f8f8f2',
                              font=('Consolas', 9), state='disabled')
        self.ml_results.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_status_panel(self, parent):
        """Создание панели статуса"""
        ttk.Label(parent, text="System Status").pack(pady=5)
        
        # Статус системы
        self.status_text = Text(parent, height=15, bg='#2b2b2b', fg='#f8f8f2',
                               font=('Consolas', 9), state='disabled')
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Обновление статуса
        self.update_status_panel()
    
    def create_status_bar(self):
        """Создание статус бара"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Элементы статус бара
        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.cursor_label = ttk.Label(self.status_bar, text="Line: 1, Col: 1")
        self.cursor_label.pack(side=tk.RIGHT, padx=5)
    
    def setup_hotkeys(self):
        """Настройка горячих клавиш"""
        self.root.bind('<Control-n>', lambda e: self.new_file())
        self.root.bind('<Control-o>', lambda e: self.open_file())
        self.root.bind('<Control-s>', lambda e: self.save_file())
        self.root.bind('<F5>', lambda e: self.run_code())
        self.root.bind('<F9>', lambda e: self.debug_code())
        self.root.bind('<Control-c>', lambda e: self.stop_execution())
    
    def setup_basic_ml(self):
        """Настройка базового ML"""
        self.ml_enabled = HAS_FULL_ML
        if self.ml_enabled:
            print("🤖 ML integration enabled")
        else:
            print("⚠️ ML integration disabled (libraries not available)")
    
    # Методы файловых операций
    def new_file(self):
        """Создание нового файла"""
        if self.ask_save_changes():
            self.editor.delete(1.0, tk.END)
            self.current_file = None
            self.is_modified = False
            self.update_title()
            self.log_to_console("📄 New file created")
    
    def open_file(self):
        """Открытие файла"""
        if self.ask_save_changes():
            file_path = filedialog.askopenfilename(
                title="Open AnamorphX File",
                filetypes=[("AnamorphX files", "*.anamorph"), ("All files", "*.*")]
            )
            if file_path:
                self.load_file(file_path)
    
    def save_file(self):
        """Сохранение файла"""
        if self.current_file:
            self.save_to_file(self.current_file)
        else:
            self.save_file_as()
    
    def save_file_as(self):
        """Сохранение файла как"""
        file_path = filedialog.asksaveasfilename(
            title="Save AnamorphX File",
            defaultextension=".anamorph",
            filetypes=[("AnamorphX files", "*.anamorph"), ("All files", "*.*")]
        )
        if file_path:
            self.save_to_file(file_path)
    
    def load_file(self, file_path):
        """Загрузка файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.editor.delete(1.0, tk.END)
            self.editor.insert(1.0, content)
            self.current_file = file_path
            self.is_modified = False
            self.update_title()
            self.update_line_numbers()
            self.log_to_console(f"📂 Loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
    
    def save_to_file(self, file_path):
        """Сохранение в файл"""
        try:
            content = self.editor.get(1.0, tk.END + '-1c')
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.current_file = file_path
            self.is_modified = False
            self.update_title()
            self.log_to_console(f"💾 Saved: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")
    
    def ask_save_changes(self):
        """Запрос сохранения изменений"""
        if self.is_modified:
            result = messagebox.askyesnocancel("Save Changes", "Do you want to save changes?")
            if result is True:
                self.save_file()
                return True
            elif result is False:
                return True
            else:
                return False
        return True
    
    # Методы выполнения кода
    def run_code(self):
        """Запуск кода"""
        code = self.editor.get(1.0, tk.END + '-1c')
        if not code.strip():
            self.log_to_console("⚠️ No code to execute")
            return
        
        self.log_to_console("🚀 Starting execution...")
        self.status_label.config(text="Running...")
        
        # Запуск в отдельном потоке
        self.execution_thread = threading.Thread(target=self.execute_code_thread, args=(code,))
        self.execution_thread.daemon = True
        self.execution_thread.start()
    
    def execute_code_thread(self, code):
        """Выполнение кода в отдельном потоке"""
        try:
            start_time = time.time()
            result = self.interpreter.execute_code(code)
            execution_time = time.time() - start_time
            
            # Обновление UI в главном потоке
            self.root.after(0, self.handle_execution_result, result, execution_time)
            
        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            self.root.after(0, self.handle_execution_error, error_msg)
    
    def handle_execution_result(self, result, execution_time):
        """Обработка результата выполнения"""
        if result["success"]:
            self.log_to_console(f"✅ Execution completed in {execution_time:.3f}s")
            
            # Вывод результатов
            if "output" in result and result["output"]:
                for line in result["output"]:
                    self.log_to_console(line)
            
            if "result" in result:
                self.log_to_console(f"📄 Result: {result['result']}")
            
            # Обновление переменных
            if "variables" in result:
                self.update_variables_display(result["variables"])
            
            # Отметка о симуляции
            if result.get("simulated", False):
                self.log_to_console("ℹ️ Note: Execution was simulated")
        else:
            self.log_to_console(f"❌ Execution failed: {result.get('error', 'Unknown error')}")
            if "traceback" in result:
                self.log_to_console("📋 Traceback:")
                for line in result["traceback"].split('\n'):
                    if line.strip():
                        self.log_to_console(f"  {line}")
        
        self.status_label.config(text="Ready")
    
    def handle_execution_error(self, error_msg):
        """Обработка ошибки выполнения"""
        self.log_to_console(f"❌ {error_msg}")
        self.status_label.config(text="Error")
    
    def debug_code(self):
        """Отладка кода"""
        self.log_to_console("🐛 Debug mode not implemented yet")
    
    def stop_execution(self):
        """Остановка выполнения"""
        if self.execution_thread and self.execution_thread.is_alive():
            self.log_to_console("⏹️ Stopping execution...")
            # В реальной реализации здесь была бы остановка потока
        else:
            self.log_to_console("ℹ️ No execution to stop")
    
    # ML методы
    def run_ml_analysis(self):
        """Запуск ML анализа"""
        code = self.editor.get(1.0, tk.END + '-1c')
        if not code.strip():
            self.log_to_console("⚠️ No code to analyze")
            return
        
        self.log_to_console("🤖 Running ML analysis...")
        
        # Простой анализ
        analysis_results = self.analyze_code_simple(code)
        self.display_ml_results(analysis_results)
    
    def analyze_code_simple(self, code):
        """Простой анализ кода"""
        results = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            # Простые проверки
            if 'synap' in line and '=' in line:
                results.append(f"Line {i}: Variable declaration detected")
            elif 'network' in line:
                results.append(f"Line {i}: Neural network definition detected")
            elif 'function' in line:
                results.append(f"Line {i}: Function definition detected")
            elif 'print' in line:
                results.append(f"Line {i}: Output statement detected")
        
        return results
    
    def display_ml_results(self, results):
        """Отображение результатов ML анализа"""
        self.ml_results.config(state='normal')
        self.ml_results.delete(1.0, tk.END)
        
        if results:
            for result in results:
                self.ml_results.insert(tk.END, f"🔍 {result}\n")
        else:
            self.ml_results.insert(tk.END, "No analysis results available.\n")
        
        self.ml_results.config(state='disabled')
        self.log_to_console(f"🤖 ML analysis completed: {len(results)} findings")
    
    # Вспомогательные методы
    def show_variables(self):
        """Показать переменные"""
        variables = self.interpreter.get_variables()
        self.update_variables_display(variables)
        self.log_to_console(f"📊 Variables updated: {len(variables)} items")
    
    def show_interpreter_status(self):
        """Показать статус интерпретатора"""
        status = self.interpreter.get_status()
        self.log_to_console("⚙️ Interpreter Status:")
        self.log_to_console(f"  Ready: {status['ready']}")
        self.log_to_console(f"  State: {status['state']}")
        self.log_to_console(f"  Components: {status['components']}/6")
        self.log_to_console(f"  Real Interpreter: {status['has_real_interpreter']}")
    
    def update_variables_display(self, variables):
        """Обновление отображения переменных"""
        # Очистка дерева
        for item in self.vars_tree.get_children():
            self.vars_tree.delete(item)
        
        # Добавление переменных
        for name, value in variables.items():
            var_type = type(value).__name__
            self.vars_tree.insert('', 'end', text=name, values=(str(value), var_type))
    
    def update_status_panel(self):
        """Обновление панели статуса"""
        status_info = [
            "🚀 AnamorphX IDE - Integrated Interpreter Edition",
            "",
            "📊 System Status:",
            f"  Interpreter Ready: {self.interpreter.is_ready}",
            f"  Components Loaded: {len(self.interpreter.components)}/6",
            f"  ML Integration: {self.ml_enabled}",
            "",
            "🔧 Available Components:",
        ]
        
        for name, component in self.interpreter.components.items():
            status_info.append(f"  ✅ {name}")
        
        missing_components = set(["ExecutionEngine", "ASTInterpreter", "TypeSystem", 
                                "ErrorHandler", "MemoryManager", "Commands"]) - set(self.interpreter.components.keys())
        
        for name in missing_components:
            status_info.append(f"  ❌ {name}")
        
        # Обновление текста
        self.status_text.config(state='normal')
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(1.0, '\n'.join(status_info))
        self.status_text.config(state='disabled')
    
    def populate_file_tree(self):
        """Заполнение дерева файлов"""
        # Очистка дерева
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        
        # Добавление примеров файлов
        examples = [
            "example1.anamorph",
            "neural_network.anamorph", 
            "simple_program.anamorph",
            "ml_training.anamorph"
        ]
        
        for example in examples:
            self.file_tree.insert('', 'end', text=example, values=(example,))
    
    def on_file_double_click(self, event):
        """Обработчик двойного клика по файлу"""
        selection = self.file_tree.selection()
        if selection:
            item = self.file_tree.item(selection[0])
            filename = item['text']
            self.log_to_console(f"📂 Selected: {filename}")
            # В реальной реализации здесь была бы загрузка файла
    
    def load_sample_code(self):
        """Загрузка примера кода"""
        sample_code = '''// AnamorphX Neural Network Example
network SimpleClassifier {
    neuron InputLayer {
        activation: linear
        weights: [0.5, 0.3, 0.2]
        bias: 0.1
    }
    
    neuron HiddenLayer {
        activation: relu
        weights: random_normal(0, 0.1)
        dropout: 0.2
    }
    
    neuron OutputLayer {
        activation: softmax
        weights: random_normal(0, 0.05)
    }
    
    optimizer: adam
    learning_rate: 0.001
    loss: crossentropy
}

function main() {
    synap x = 42
    synap y = 3.14
    synap result = x + y
    
    print("Hello from AnamorphX!")
    print("Result:", result)
    
    // Train the network
    network.train()
}
'''
        self.editor.insert(1.0, sample_code)
        self.update_line_numbers()
    
    def on_text_change(self, event=None):
        """Обработчик изменения текста"""
        self.is_modified = True
        self.update_title()
        self.update_line_numbers()
        self.update_cursor_position()
    
    def update_title(self):
        """Обновление заголовка окна"""
        title = "AnamorphX IDE - Integrated Interpreter Edition"
        if self.current_file:
            title += f" - {os.path.basename(self.current_file)}"
        if self.is_modified:
            title += " *"
        self.root.title(title)
    
    def update_line_numbers(self):
        """Обновление номеров строк"""
        content = self.editor.get(1.0, tk.END)
        lines = content.split('\n')
        line_numbers = '\n'.join(str(i) for i in range(1, len(lines)))
        
        self.line_numbers.config(state='normal')
        self.line_numbers.delete(1.0, tk.END)
        self.line_numbers.insert(1.0, line_numbers)
        self.line_numbers.config(state='disabled')
    
    def update_cursor_position(self):
        """Обновление позиции курсора"""
        cursor_pos = self.editor.index(tk.INSERT)
        line, col = cursor_pos.split('.')
        self.cursor_label.config(text=f"Line: {line}, Col: {int(col)+1}")
    
    def sync_scroll(self, *args):
        """Синхронизация скроллинга"""
        self.line_numbers.yview_moveto(args[0])
    
    def log_to_console(self, message):
        """Вывод сообщения в консоль"""
        self.console.config(state='normal')
        timestamp = time.strftime("%H:%M:%S")
        self.console.insert(tk.END, f"[{timestamp}] {message}\n")
        self.console.see(tk.END)
        self.console.config(state='disabled')
    
    def run(self):
        """Запуск IDE"""
        self.log_to_console("🚀 AnamorphX IDE started successfully!")
        self.log_to_console(f"🤖 Interpreter status: {'✅ READY' if self.interpreter.is_ready else '⚠️ PARTIAL'}")
        self.root.mainloop()

def main():
    """Главная функция"""
    print("🚀 Starting AnamorphX IDE - Integrated Interpreter Edition")
    print("=" * 60)
    
    try:
        ide = IntegratedMLIDE()
        ide.run()
    except Exception as e:
        print(f"❌ Failed to start IDE: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 