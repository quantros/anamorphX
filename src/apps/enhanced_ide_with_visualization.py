#!/usr/bin/env python3
"""
Enhanced AnamorphX IDE with Command Visualization
Интегрированная IDE с визуализацией выполнения команд
"""

import tkinter as tk
from tkinter import ttk, Canvas, messagebox, filedialog
import time
import math
import threading
from typing import Dict, List, Any, Optional

# Импорт базовой IDE
try:
    from full_ml_interpreter_ide import UnifiedMLIDE
    BASE_IDE_AVAILABLE = True
except ImportError:
    BASE_IDE_AVAILABLE = False
    print("Warning: Base IDE not available, creating minimal version")

# Импорт визуализации команд
try:
    from visual_command_integration import VisualCommandIntegrator, integrate_with_ide
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: Command visualization not available")


class EnhancedAnamorphXIDE:
    """Улучшенная IDE с визуализацией команд"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AnamorphX Enhanced IDE - Commands + Visualization")
        self.root.geometry("1400x900")
        
        # Компоненты
        self.command_visualizer = None
        self.neural_canvas = None
        self.base_ide = None
        
        # Состояние
        self.current_file = None
        self.file_modified = False
        self.is_running = False
        
        # Логирование (должно быть до setup_command_integration)
        self.log_messages = []
        
        # Инициализация
        self.setup_enhanced_ui()
        self.setup_command_integration()
        
    def setup_enhanced_ui(self):
        """Настройка улучшенного интерфейса"""
        # Главное меню
        self.create_enhanced_menu()
        
        # Панель инструментов
        self.create_enhanced_toolbar()
        
        # Основной интерфейс
        self.create_enhanced_main_interface()
        
        # Статусная строка
        self.create_enhanced_status_bar()
    
    def create_enhanced_menu(self):
        """Создание улучшенного меню"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Файл
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.new_file, accelerator="Ctrl+N")
        file_menu.add_command(label="Open", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Выполнение
        run_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Run", menu=run_menu)
        run_menu.add_command(label="Execute Code", command=self.run_code, accelerator="F5")
        run_menu.add_command(label="Execute with Visualization", command=self.run_with_visualization)
        run_menu.add_separator()
        run_menu.add_command(label="Stop", command=self.stop_execution)
        
        # Команды
        commands_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Commands", menu=commands_menu)
        commands_menu.add_command(label="Show Command Palette", command=self.show_command_palette)
        commands_menu.add_command(label="Neural Commands", command=self.show_neural_commands)
        commands_menu.add_separator()
        commands_menu.add_command(label="Clear Visualization", command=self.clear_visualization)
        
        # Визуализация
        viz_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Visualization", menu=viz_menu)
        viz_menu.add_command(label="Show Neural Network", command=self.show_neural_visualization)
        viz_menu.add_command(label="Export Visualization", command=self.export_visualization)
        viz_menu.add_separator()
        viz_menu.add_command(label="Animation Settings", command=self.show_animation_settings)
    
    def create_enhanced_toolbar(self):
        """Создание улучшенной панели инструментов"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        # Кнопки файлов
        ttk.Button(toolbar, text="📄 New", command=self.new_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📂 Open", command=self.open_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="💾 Save", command=self.save_file).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        # Кнопки выполнения
        ttk.Button(toolbar, text="▶️ Run", command=self.run_code).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🎬 Run+Viz", command=self.run_with_visualization).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="⏹️ Stop", command=self.stop_execution).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        # Кнопки команд
        ttk.Button(toolbar, text="🧠 Neural", command=self.show_neural_commands).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🎯 Commands", command=self.show_command_palette).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        # Кнопки визуализации
        ttk.Button(toolbar, text="🌐 Network", command=self.show_neural_visualization).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🧹 Clear", command=self.clear_visualization).pack(side=tk.LEFT, padx=2)
    
    def create_enhanced_main_interface(self):
        """Создание основного интерфейса"""
        # Главный контейнер
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Левая панель (файлы + команды)
        left_panel = ttk.Frame(main_paned)
        main_paned.add(left_panel, weight=1)
        
        # Центральная панель (редактор)
        center_panel = ttk.Frame(main_paned)
        main_paned.add(center_panel, weight=3)
        
        # Правая панель (визуализация)
        right_panel = ttk.Frame(main_paned)
        main_paned.add(right_panel, weight=2)
        
        # Настройка панелей
        self.setup_left_panel(left_panel)
        self.setup_center_panel(center_panel)
        self.setup_right_panel(right_panel)
    
    def setup_left_panel(self, parent):
        """Настройка левой панели"""
        # Notebook для вкладок
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка файлов
        files_frame = ttk.Frame(notebook)
        notebook.add(files_frame, text="📁 Files")
        
        # Простое дерево файлов
        self.file_tree = ttk.Treeview(files_frame, columns=('type',), show='tree headings')
        self.file_tree.heading('#0', text='Name')
        self.file_tree.heading('type', text='Type')
        self.file_tree.pack(fill=tk.BOTH, expand=True)
        
        # Заполняем примерами
        self.populate_file_tree()
        
        # Привязываем события
        self.file_tree.bind('<Double-1>', self.on_file_double_click)
        
        # Вкладка команд
        commands_frame = ttk.Frame(notebook)
        notebook.add(commands_frame, text="⚡ Commands")
        
        # Список команд
        self.commands_listbox = tk.Listbox(commands_frame)
        self.commands_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Заполняем командами
        self.populate_commands_list()
        
        # Привязываем события
        self.commands_listbox.bind('<Double-1>', self.on_command_double_click)
    
    def setup_center_panel(self, parent):
        """Настройка центральной панели"""
        # Верхняя панель (информация о файле)
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.file_label = ttk.Label(info_frame, text="📄 No file loaded")
        self.file_label.pack(side=tk.LEFT)
        
        self.modified_label = ttk.Label(info_frame, text="", foreground="red")
        self.modified_label.pack(side=tk.RIGHT)
        
        # Редактор кода
        editor_frame = ttk.Frame(parent)
        editor_frame.pack(fill=tk.BOTH, expand=True)
        
        # Номера строк
        self.line_numbers = tk.Text(editor_frame, width=4, padx=3, takefocus=0,
                                   border=0, state='disabled', wrap='none')
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)
        
        # Основной редактор
        self.text_editor = tk.Text(editor_frame, wrap='none', undo=True)
        self.text_editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Скроллбары
        v_scrollbar = ttk.Scrollbar(editor_frame, orient=tk.VERTICAL, command=self.text_editor.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_editor.config(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self.text_editor.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.text_editor.config(xscrollcommand=h_scrollbar.set)
        
        # События редактора
        self.text_editor.bind('<KeyRelease>', self.on_text_change)
        self.text_editor.bind('<Button-1>', self.on_text_change)
        
        # Консоль внизу
        console_frame = ttk.LabelFrame(parent, text="Console Output")
        console_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.console_text = tk.Text(console_frame, height=8, state='disabled')
        self.console_text.pack(fill=tk.BOTH, expand=True)
        
        console_scrollbar = ttk.Scrollbar(console_frame, orient=tk.VERTICAL, command=self.console_text.yview)
        console_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.console_text.config(yscrollcommand=console_scrollbar.set)
    
    def setup_right_panel(self, parent):
        """Настройка правой панели"""
        # Notebook для визуализации
        viz_notebook = ttk.Notebook(parent)
        viz_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка нейронной сети
        neural_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(neural_frame, text="🧠 Neural Network")
        
        # Canvas для визуализации
        self.neural_canvas = Canvas(neural_frame, bg='white', width=400, height=300)
        self.neural_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка команд
        command_viz_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(command_viz_frame, text="⚡ Command Trace")
        
        # Список выполненных команд
        self.command_trace = ttk.Treeview(command_viz_frame, columns=('time', 'status'), show='tree headings')
        self.command_trace.heading('#0', text='Command')
        self.command_trace.heading('time', text='Time')
        self.command_trace.heading('status', text='Status')
        self.command_trace.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка статистики
        stats_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(stats_frame, text="📊 Statistics")
        
        self.stats_text = tk.Text(stats_frame, state='disabled')
        self.stats_text.pack(fill=tk.BOTH, expand=True)
    
    def setup_command_integration(self):
        """Настройка интеграции команд"""
        if VISUALIZATION_AVAILABLE and self.neural_canvas:
            self.command_visualizer = integrate_with_ide(self, self.neural_canvas)
            self.log_to_console("✅ Command visualization integrated")
        else:
            self.log_to_console("⚠️ Command visualization not available")
    
    def create_enhanced_status_bar(self):
        """Создание улучшенной статусной строки"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.cursor_label = ttk.Label(status_frame, text="Line: 1, Col: 1")
        self.cursor_label.pack(side=tk.RIGHT, padx=5)
    
    def populate_file_tree(self):
        """Заполнение дерева файлов"""
        # Примеры файлов
        files = [
            ("main.anamorph", "file"),
            ("neural_classifier.anamorph", "file"),
            ("deep_network.anamorph", "file"),
            ("sample.anamorph", "file")
        ]
        
        for filename, file_type in files:
            self.file_tree.insert('', 'end', text=filename, values=(file_type,))
    
    def populate_commands_list(self):
        """Заполнение списка команд"""
        commands = [
            "neuro - Create neural node",
            "synap - Create synapse",
            "pulse - Send signal",
            "bind - Bind data to node",
            "cluster - Create cluster",
            "expand - Expand cluster",
            "contract - Contract cluster",
            "morph - Transform node",
            "evolve - Evolve node",
            "prune - Remove inactive nodes",
            "forge - Create structure",
            "drift - Move signal",
            "echo - Echo signal",
            "reflect - Reflect signal",
            "absorb - Absorb signal",
            "diffuse - Diffuse signal",
            "merge - Merge nodes",
            "split - Split node",
            "loop - Create loop",
            "halt - Stop execution"
        ]
        
        for command in commands:
            self.commands_listbox.insert(tk.END, command)
    
    def on_file_double_click(self, event):
        """Обработка двойного клика по файлу"""
        selection = self.file_tree.selection()
        if selection:
            item = selection[0]
            filename = self.file_tree.item(item, 'text')
            self.load_sample_file(filename)
    
    def on_command_double_click(self, event):
        """Обработка двойного клика по команде"""
        selection = self.commands_listbox.curselection()
        if selection:
            command_text = self.commands_listbox.get(selection[0])
            command_name = command_text.split(' - ')[0]
            self.insert_command_template(command_name)
    
    def load_sample_file(self, filename):
        """Загрузка примера файла"""
        # Базовый шаблон
        content = '''// AnamorphX Neural Network Example
network BasicNetwork {
    neuron InputLayer {
        activation: linear
        units: 10
    }
    
    neuron HiddenLayer {
        activation: relu
        units: 64
        dropout: 0.3
    }
    
    neuron OutputLayer {
        activation: softmax
        units: 3
    }
}

// Commands
neuro "input_processor" {
    activation: "linear"
    units: 10
}

neuro "hidden_processor" {
    activation: "relu"
    units: 64
}

synap "input_processor" -> "hidden_processor" {
    weight: 0.5
}

pulse {
    target: "input_processor"
    intensity: 1.0
}'''
        
        if 'neural_classifier' in filename:
            content = '''// Neural Classifier Example
network ImageClassifier {
    neuron ConvLayer1 {
        activation: relu
        filters: 32
    }
    
    neuron ConvLayer2 {
        activation: relu
        filters: 64
    }
    
    neuron OutputLayer {
        activation: softmax
        units: 10
    }
}

// Commands
neuro "conv1" {
    activation: "relu"
    filters: 32
}

neuro "conv2" {
    activation: "relu"
    filters: 64
}

synap "conv1" -> "conv2" {
    weight: 0.8
}

pulse {
    target: "conv1"
    signal: "image_data"
}'''
        
        # Загружаем в редактор
        self.text_editor.delete("1.0", tk.END)
        self.text_editor.insert("1.0", content)
        
        # Обновляем состояние
        self.current_file = None
        self.file_modified = False
        self.file_label.config(text=f"📄 {filename}")
        self.modified_label.config(text="")
        
        self.log_to_console(f"📄 Loaded: {filename}")
    
    def insert_command_template(self, command_name):
        """Вставка шаблона команды"""
        templates = {
            'neuro': 'neuro "node_name" {\n    activation: "relu"\n    units: 64\n}',
            'synap': 'synap "source" -> "target" {\n    weight: 1.0\n}',
            'pulse': 'pulse {\n    target: "node_name"\n    intensity: 1.0\n}',
            'bind': 'bind "node_name" {\n    data: "value"\n}',
            'cluster': 'cluster "cluster_name" {\n    nodes: ["node1", "node2"]\n}'
        }
        
        template = templates.get(command_name, f'{command_name} {{\n    // parameters\n}}')
        
        # Вставляем в текущую позицию курсора
        self.text_editor.insert(tk.INSERT, template)
        self.log_to_console(f"📝 Inserted template: {command_name}")
    
    def run_code(self):
        """Выполнение кода"""
        code = self.text_editor.get("1.0", tk.END).strip()
        if not code:
            self.log_to_console("⚠️ No code to execute")
            return
        
        self.log_to_console("🚀 Executing code...")
        self.is_running = True
        
        # Простая симуляция выполнения
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if line and not line.startswith('//'):
                self.log_to_console(f"  Line {i}: {line}")
                
                # Добавляем в трассировку команд
                self.add_command_trace(line, "executed")
        
        self.log_to_console("✅ Execution completed")
        self.is_running = False
    
    def run_with_visualization(self):
        """Выполнение кода с визуализацией"""
        code = self.text_editor.get("1.0", tk.END).strip()
        if not code:
            self.log_to_console("⚠️ No code to execute")
            return
        
        self.log_to_console("🎬 Executing with visualization...")
        
        if self.command_visualizer:
            # Парсим и выполняем команды с визуализацией
            lines = code.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('//'):
                    self.execute_command_with_visualization(line)
        else:
            self.log_to_console("⚠️ Visualization not available")
            self.run_code()
    
    def execute_command_with_visualization(self, command_line):
        """Выполнение команды с визуализацией"""
        # Простой парсер команд
        if command_line.startswith('neuro'):
            # Извлекаем имя узла
            import re
            match = re.search(r'neuro\s+"([^"]+)"', command_line)
            if match:
                node_name = match.group(1)
                if self.command_visualizer:
                    # Создаем узел в визуализации
                    self.command_visualizer.create_node_at_position(
                        200 + len(self.command_visualizer.visual_nodes) * 50,
                        150 + len(self.command_visualizer.visual_nodes) * 30
                    )
                self.log_to_console(f"🧠 Created neuron: {node_name}")
                self.add_command_trace(f"neuro {node_name}", "success")
        
        elif command_line.startswith('synap'):
            # Создание синапса
            self.log_to_console(f"🔗 Created synapse")
            self.add_command_trace("synap", "success")
        
        elif command_line.startswith('pulse'):
            # Отправка импульса
            if self.command_visualizer:
                # Анимируем пульс по всем узлам
                for node in self.command_visualizer.visual_nodes.values():
                    self.command_visualizer.animate_node_pulse(node, 1.0)
            self.log_to_console(f"⚡ Pulse sent")
            self.add_command_trace("pulse", "success")
        
        else:
            self.log_to_console(f"📝 Executed: {command_line}")
            self.add_command_trace(command_line, "executed")
    
    def add_command_trace(self, command, status):
        """Добавление команды в трассировку"""
        timestamp = time.strftime("%H:%M:%S")
        self.command_trace.insert('', 'end', text=command, values=(timestamp, status))
        
        # Автопрокрутка
        children = self.command_trace.get_children()
        if children:
            self.command_trace.see(children[-1])
    
    def show_command_palette(self):
        """Показ палитры команд"""
        palette = tk.Toplevel(self.root)
        palette.title("Command Palette")
        palette.geometry("400x300")
        
        # Поиск
        search_frame = ttk.Frame(palette)
        search_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        search_entry = ttk.Entry(search_frame)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Список команд
        commands_frame = ttk.Frame(palette)
        commands_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        commands_list = tk.Listbox(commands_frame)
        commands_list.pack(fill=tk.BOTH, expand=True)
        
        # Заполняем команды
        all_commands = [
            "neuro - Create neural node",
            "synap - Create synapse connection",
            "pulse - Send neural pulse",
            "bind - Bind data to node",
            "cluster - Create node cluster",
            "expand - Expand cluster",
            "contract - Contract cluster",
            "morph - Transform node type",
            "evolve - Evolve node properties",
            "prune - Remove inactive nodes"
        ]
        
        for cmd in all_commands:
            commands_list.insert(tk.END, cmd)
        
        # Кнопки
        button_frame = ttk.Frame(palette)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        def insert_selected():
            selection = commands_list.curselection()
            if selection:
                cmd_text = commands_list.get(selection[0])
                cmd_name = cmd_text.split(' - ')[0]
                self.insert_command_template(cmd_name)
                palette.destroy()
        
        ttk.Button(button_frame, text="Insert", command=insert_selected).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=palette.destroy).pack(side=tk.RIGHT)
    
    def show_neural_commands(self):
        """Показ нейронных команд"""
        neural_window = tk.Toplevel(self.root)
        neural_window.title("Neural Commands")
        neural_window.geometry("500x400")
        
        # Создаем интерфейс для быстрого создания нейронных структур
        ttk.Label(neural_window, text="Quick Neural Commands", font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Кнопки команд
        commands_frame = ttk.Frame(neural_window)
        commands_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Создание узлов
        nodes_frame = ttk.LabelFrame(commands_frame, text="Create Nodes")
        nodes_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(nodes_frame, text="🧠 Input Node", 
                  command=lambda: self.quick_command('neuro "input_node" { activation: "linear" }')).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(nodes_frame, text="🔄 Hidden Node", 
                  command=lambda: self.quick_command('neuro "hidden_node" { activation: "relu" }')).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(nodes_frame, text="📤 Output Node", 
                  command=lambda: self.quick_command('neuro "output_node" { activation: "softmax" }')).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Создание связей
        connections_frame = ttk.LabelFrame(commands_frame, text="Create Connections")
        connections_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(connections_frame, text="🔗 Connect Nodes", 
                  command=lambda: self.quick_command('synap "node1" -> "node2" { weight: 1.0 }')).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Сигналы
        signals_frame = ttk.LabelFrame(commands_frame, text="Send Signals")
        signals_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(signals_frame, text="⚡ Pulse All", 
                  command=lambda: self.quick_command('pulse { target: "broadcast", intensity: 1.0 }')).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(signals_frame, text="🎯 Pulse Target", 
                  command=lambda: self.quick_command('pulse { target: "node_name", intensity: 0.8 }')).pack(side=tk.LEFT, padx=5, pady=5)
    
    def quick_command(self, command):
        """Быстрое выполнение команды"""
        self.text_editor.insert(tk.INSERT, command + '\n')
        self.log_to_console(f"📝 Added: {command}")
    
    def show_neural_visualization(self):
        """Показ визуализации нейронной сети"""
        if self.command_visualizer:
            stats = self.command_visualizer.get_visualization_stats()
            self.update_stats_display(stats)
            self.log_to_console("🌐 Neural visualization updated")
        else:
            self.log_to_console("⚠️ Visualization not available")
    
    def clear_visualization(self):
        """Очистка визуализации"""
        if self.command_visualizer:
            self.command_visualizer.clear_visualization()
        
        # Очищаем трассировку команд
        for item in self.command_trace.get_children():
            self.command_trace.delete(item)
        
        self.log_to_console("🧹 Visualization cleared")
    
    def export_visualization(self):
        """Экспорт визуализации"""
        if self.command_visualizer:
            filename = filedialog.asksaveasfilename(
                defaultextension=".ps",
                filetypes=[("PostScript files", "*.ps"), ("All files", "*.*")]
            )
            if filename:
                if self.command_visualizer.export_visualization(filename):
                    self.log_to_console(f"📸 Visualization exported to {filename}")
                else:
                    self.log_to_console("❌ Export failed")
    
    def show_animation_settings(self):
        """Показ настроек анимации"""
        settings = tk.Toplevel(self.root)
        settings.title("Animation Settings")
        settings.geometry("300x200")
        
        ttk.Label(settings, text="Animation Speed:").pack(pady=5)
        speed_var = tk.IntVar(value=50)
        speed_scale = ttk.Scale(settings, from_=10, to=200, variable=speed_var, orient=tk.HORIZONTAL)
        speed_scale.pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Label(settings, text="Node Size:").pack(pady=5)
        size_var = tk.IntVar(value=20)
        size_scale = ttk.Scale(settings, from_=10, to=50, variable=size_var, orient=tk.HORIZONTAL)
        size_scale.pack(fill=tk.X, padx=20, pady=5)
        
        def apply_settings():
            if self.command_visualizer:
                self.command_visualizer.animation_speed = speed_var.get()
                # Обновляем размеры узлов
                for node in self.command_visualizer.visual_nodes.values():
                    node.radius = size_var.get()
                    self.command_visualizer.draw_node(node)
            settings.destroy()
        
        ttk.Button(settings, text="Apply", command=apply_settings).pack(pady=10)
    
    def update_stats_display(self, stats):
        """Обновление отображения статистики"""
        self.stats_text.config(state='normal')
        self.stats_text.delete("1.0", tk.END)
        
        stats_text = f"""Neural Network Statistics:

Nodes: {stats['nodes_count']}
Connections: {stats['connections_count']}
Active Nodes: {stats['active_nodes']}
Commands Executed: {stats['commands_executed']}
Canvas Size: {stats['canvas_size'][0]}x{stats['canvas_size'][1]}

Total Connections: {stats['total_connections']}
"""
        
        self.stats_text.insert("1.0", stats_text)
        self.stats_text.config(state='disabled')
    
    def on_text_change(self, event=None):
        """Обработка изменения текста"""
        if not self.file_modified:
            self.file_modified = True
            self.modified_label.config(text="●")
        
        # Обновляем номера строк
        self.update_line_numbers()
        
        # Обновляем позицию курсора
        self.update_cursor_position()
    
    def update_line_numbers(self):
        """Обновление номеров строк"""
        self.line_numbers.config(state='normal')
        self.line_numbers.delete("1.0", tk.END)
        
        # Получаем количество строк
        line_count = int(self.text_editor.index('end-1c').split('.')[0])
        
        # Добавляем номера строк
        for i in range(1, line_count + 1):
            self.line_numbers.insert(tk.END, f"{i:3d}\n")
        
        self.line_numbers.config(state='disabled')
    
    def update_cursor_position(self):
        """Обновление позиции курсора"""
        cursor_pos = self.text_editor.index(tk.INSERT)
        line, col = cursor_pos.split('.')
        self.cursor_label.config(text=f"Line: {line}, Col: {int(col)+1}")
    
    def log_to_console(self, message):
        """Логирование в консоль"""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        self.console_text.config(state='normal')
        self.console_text.insert(tk.END, log_message + '\n')
        self.console_text.see(tk.END)
        self.console_text.config(state='disabled')
        
        # Сохраняем в истории
        self.log_messages.append(log_message)
    
    def new_file(self):
        """Создание нового файла"""
        if self.file_modified:
            if not messagebox.askyesno("Unsaved Changes", "Discard unsaved changes?"):
                return
        
        self.text_editor.delete("1.0", tk.END)
        self.current_file = None
        self.file_modified = False
        self.file_label.config(text="📄 New file")
        self.modified_label.config(text="")
        self.log_to_console("📄 New file created")
    
    def open_file(self):
        """Открытие файла"""
        filename = filedialog.askopenfilename(
            filetypes=[("AnamorphX files", "*.anamorph"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.text_editor.delete("1.0", tk.END)
                self.text_editor.insert("1.0", content)
                
                self.current_file = filename
                self.file_modified = False
                self.file_label.config(text=f"📄 {filename}")
                self.modified_label.config(text="")
                
                self.log_to_console(f"📂 Opened: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file: {e}")
    
    def save_file(self):
        """Сохранение файла"""
        if self.current_file:
            try:
                content = self.text_editor.get("1.0", tk.END)
                with open(self.current_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.file_modified = False
                self.modified_label.config(text="")
                self.log_to_console(f"💾 Saved: {self.current_file}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {e}")
        else:
            self.save_file_as()
    
    def save_file_as(self):
        """Сохранение файла как"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".anamorph",
            filetypes=[("AnamorphX files", "*.anamorph"), ("All files", "*.*")]
        )
        if filename:
            try:
                content = self.text_editor.get("1.0", tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.current_file = filename
                self.file_modified = False
                self.file_label.config(text=f"📄 {filename}")
                self.modified_label.config(text="")
                self.log_to_console(f"💾 Saved as: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {e}")
    
    def stop_execution(self):
        """Остановка выполнения"""
        self.is_running = False
        self.log_to_console("⏹️ Execution stopped")
    
    def run(self):
        """Запуск IDE"""
        self.log_to_console("🚀 AnamorphX Enhanced IDE started")
        self.log_to_console("💡 Use toolbar buttons or menu for commands")
        
        # Загружаем пример по умолчанию
        self.load_sample_file("main.anamorph")
        
        # Запускаем главный цикл
        self.root.mainloop()


def main():
    """Главная функция"""
    print("🚀 Starting AnamorphX Enhanced IDE with Command Visualization")
    print("=" * 70)
    
    try:
        ide = EnhancedAnamorphXIDE()
        ide.run()
    except KeyboardInterrupt:
        print("\n👋 IDE closed by user")
    except Exception as e:
        print(f"❌ Error starting IDE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 